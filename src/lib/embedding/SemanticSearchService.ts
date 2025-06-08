import { embeddingService } from './EmbeddingService';
import { Block } from '@blocknote/core';
import { vecToBlob, blobToVec } from './binaryUtils';
import { tables, events } from '../../livestore/schema';
import { toast } from 'sonner';
import { HNSW } from './hnsw';
import { hnswPersistence } from './hnsw/persistence';
import { embeddingCleanupService } from './CleanupService';
import { generateChunkId } from '../utils/ids';

interface NoteEmbedding {
  chunkId: string;
  parentNoteId: string;
  title: string;
  content: string;
  embedding: Float32Array;
}

interface SearchResult {
  noteId: string;
  title: string;
  content: string;
  score: number;
}

interface BuildProgress {
  phase: 'cleanup' | 'sync' | 'build' | 'persist' | 'complete';
  current: number;
  total: number;
  message: string;
}

// Helper to convert BlockNote content to plain text
function blocksToText(blocks: Block[] | undefined): string {
  if (!blocks) return '';
  
  return blocks.map(block => {
    if (!block.content) return '';
    
    // Handle different content types safely
    if (Array.isArray(block.content)) {
      return block.content.map(inline => {
        if (typeof inline === 'object' && inline !== null && 'type' in inline && inline.type === 'text') {
          return (inline as any).text || '';
        }
        return '';
      }).join('');
    }
    
    // Handle non-array content (like tables)
    return '';
  }).join('\n');
}

// L2 normalize a vector to unit length for vector hygiene
function l2Normalize(v: Float32Array): Float32Array {
  let norm = 0; 
  for (const x of v) norm += x * x;
  norm = 1 / Math.sqrt(norm || 1e-9);
  return v.map(x => x * norm) as Float32Array;
}

class SemanticSearchService {
  private embeddings = new Map<string, NoteEmbedding>();
  private hnswIndex: HNSW;
  private noteIdToHnswId = new Map<string, number>();
  private hnswIdToNoteId = new Map<number, string>();
  private nextHnswId = 0;
  private isReady = false;
  private isInitialized = false;
  private buildProgressCallback: ((progress: BuildProgress) => void) | null = null;

  // Store reference will be injected by the hooks
  private storeRef: any = null;

  constructor() {
    // Initialize HNSW with cosine similarity
    this.hnswIndex = new HNSW(16, 200, null, 'cosine');
  }

  async initialize() {
    if (this.isReady) return;
    try {
      await embeddingService.ready();
      this.isReady = true;
      console.log('SemanticSearchService initialized with HNSW');
    } catch (error) {
      console.error('Failed to initialize SemanticSearchService:', error);
    }
  }

  // Method to inject store reference from React hooks
  setStore(store: any) {
    this.storeRef = store;
    embeddingCleanupService.setStore(store);
    
    if (!this.isInitialized && store) {
      this.loadEmbeddingsFromStore();
      this.isInitialized = true;
    }
  }

  // NEW: Getter for external access to HNSW index
  getHnswIndex(): HNSW {
    return this.hnswIndex;
  }

  // NEW: Incremental addition method for delta orchestrator
  async addPointToIndex(embeddingRow: any): Promise<void> {
    if (!this.isReady || !this.hnswIndex) {
      console.warn('SemanticSearchService: Cannot add point - service not ready');
      return;
    }

    try {
      const existingHnswId = this.noteIdToHnswId.get(embeddingRow.parentNoteId);
      if (existingHnswId !== undefined) {
        // This is an update, remove old point first
        this.removePointFromIndex(embeddingRow.parentNoteId);
      }

      const embeddingVector = blobToVec(embeddingRow.vecData, embeddingRow.vecDim);
      const normalizedVector = l2Normalize(embeddingVector);
      const newHnswId = this.nextHnswId++;

      await this.hnswIndex.addPoint(newHnswId, normalizedVector);
      
      this.noteIdToHnswId.set(embeddingRow.parentNoteId, newHnswId);
      this.hnswIdToNoteId.set(newHnswId, embeddingRow.parentNoteId);
      this.embeddings.set(embeddingRow.parentNoteId, {
        chunkId: embeddingRow.chunkId,
        parentNoteId: embeddingRow.parentNoteId,
        title: embeddingRow.title || '',
        content: embeddingRow.content || '',
        embedding: normalizedVector
      });

      console.log(`HNSW: Added point for note ${embeddingRow.parentNoteId}`);
    } catch (error) {
      console.error(`Failed to add point to HNSW index for note ${embeddingRow.parentNoteId}:`, error);
    }
  }

  // NEW: Incremental removal method for delta orchestrator
  removePointFromIndex(noteId: string): void {
    if (!this.hnswIndex) {
      console.warn('SemanticSearchService: Cannot remove point - no HNSW index');
      return;
    }

    const hnswId = this.noteIdToHnswId.get(noteId);
    if (hnswId !== undefined) {
      // Note: True HNSW removal is complex. We mark as deleted by removing from mappings.
      // The graph node remains but won't be discoverable in searches.
      // Full rebuild during next snapshot cycle will properly remove it.
      this.noteIdToHnswId.delete(noteId);
      this.hnswIdToNoteId.delete(hnswId);
      this.embeddings.delete(noteId);
      console.log(`HNSW: Marked point for removal for note ${noteId}`);
    }
  }

  // NEW: Initialize from snapshot (for warm boot)
  async initializeFromSnapshot(): Promise<boolean> {
    try {
      console.log('SemanticSearchService: Attempting to load from snapshot...');
      const persistedGraph = await hnswPersistence.loadGraph('latest');
      
      if (persistedGraph && persistedGraph.nodes.size > 0) {
        this.hnswIndex = persistedGraph;
        console.log(`SemanticSearchService: Loaded HNSW graph with ${persistedGraph.nodes.size} nodes`);
        
        // Rebuild mappings from LiveStore state
        await this.rebuildMappingsFromStore();
        return true;
      }
    } catch (error) {
      console.warn('Failed to load from snapshot:', error);
    }
    
    return false;
  }

  // NEW: Rebuild mappings after loading snapshot
  private async rebuildMappingsFromStore(): Promise<void> {
    if (!this.storeRef) {
      console.warn('Cannot rebuild mappings - no store reference');
      return;
    }

    try {
      const embeddingRows = this.storeRef.query(tables.embeddings.select());
      
      this.embeddings.clear();
      this.noteIdToHnswId.clear();
      this.hnswIdToNoteId.clear();
      this.nextHnswId = 0;

      if (Array.isArray(embeddingRows)) {
        let maxHnswId = -1;
        
        // For now, we'll rebuild the index from scratch since mapping HNSW IDs 
        // to note IDs from a snapshot is complex. The snapshot serves as backup.
        embeddingRows.forEach((row: any, index: number) => {
          try {
            const embedding = blobToVec(row.vecData, row.vecDim);
            const hnswId = index; // Simple mapping for now
            
            this.noteIdToHnswId.set(row.parentNoteId, hnswId);
            this.hnswIdToNoteId.set(hnswId, row.parentNoteId);
            this.embeddings.set(row.parentNoteId, {
              chunkId: row.chunkId,
              parentNoteId: row.parentNoteId,
              title: row.title || '',
              content: row.content || '',
              embedding
            });
            
            maxHnswId = Math.max(maxHnswId, hnswId);
          } catch (error) {
            console.error(`Failed to rebuild mapping for note ${row?.parentNoteId}:`, error);
          }
        });
        
        this.nextHnswId = maxHnswId + 1;
        console.log(`SemanticSearchService: Rebuilt mappings for ${this.embeddings.size} embeddings`);
      }
    } catch (error) {
      console.error('Failed to rebuild mappings from store:', error);
    }
  }

  // Set progress callback for UI updates
  setBuildProgressCallback(callback: (progress: BuildProgress) => void) {
    this.buildProgressCallback = callback;
  }

  private reportProgress(phase: BuildProgress['phase'], current: number, total: number, message: string) {
    if (this.buildProgressCallback) {
      this.buildProgressCallback({ phase, current, total, message });
    }
  }

  // Enhanced cleanup with detailed reporting
  async forceCleanupStaleEmbeddings() {
    console.log('SemanticSearchService: Starting force cleanup of stale embeddings');
    
    try {
      const result = await embeddingCleanupService.forceCleanupStaleEmbeddings();
      
      // Update in-memory cache after cleanup
      this.loadEmbeddingsFromStore();
      
      return result;
    } catch (error) {
      console.error('SemanticSearchService: Force cleanup failed:', error);
      throw error;
    }
  }

  // Load all embeddings from LiveStore into memory for fast search
  private async loadEmbeddingsFromStore() {
    if (!this.storeRef) {
      console.warn('SemanticSearchService: Cannot load embeddings - no store reference');
      return;
    }

    try {
      // Query all embeddings from the database
      const embeddingRows = this.storeRef.query(tables.embeddings.select());
      
      console.log(`SemanticSearchService: Loading ${embeddingRows?.length || 0} embeddings from LiveStore`);
      
      // Reset embeddings map and HNSW index before loading
      this.embeddings.clear();
      this.hnswIndex = new HNSW(16, 200, null, 'cosine');
      this.noteIdToHnswId.clear();
      this.hnswIdToNoteId.clear();
      this.nextHnswId = 0;
      
      // Convert each row back to in-memory format and build HNSW index
      if (Array.isArray(embeddingRows)) {
        const hnswData: { id: number; vector: Float32Array }[] = [];
        
        embeddingRows.forEach((row: any) => {
          try {
            if (!row || !row.vecData) {
              console.warn(`Skipping invalid embedding row:`, row);
              return;
            }
            
            const embedding = blobToVec(row.vecData, row.vecDim);
            const hnswId = this.nextHnswId++;
            
            // Store mapping between noteId and HNSW ID
            this.noteIdToHnswId.set(row.parentNoteId, hnswId);
            this.hnswIdToNoteId.set(hnswId, row.parentNoteId);
            
            this.embeddings.set(row.parentNoteId, {
              chunkId: row.chunkId,
              parentNoteId: row.parentNoteId,
              title: row.title || '',
              content: row.content || '',
              embedding
            });
            
            // Prepare data for HNSW index
            hnswData.push({ id: hnswId, vector: embedding });
          } catch (error) {
            console.error(`Failed to load embedding for note ${row?.parentNoteId}:`, error);
          }
        });
        
        // Build HNSW index with all data at once for better performance
        if (hnswData.length > 0) {
          await this.hnswIndex.buildIndex(hnswData);
          console.log(`SemanticSearchService: Built HNSW index with ${hnswData.length} vectors`);
        }
      }

      console.log(`SemanticSearchService: Loaded ${this.embeddings.size} embeddings into memory`);
    } catch (error) {
      console.error('Failed to load embeddings from LiveStore:', error);
    }
  }

  // ENHANCED: Explicit HNSW persistence with event tracking
  async persistHnswIndex() {
    if (!this.storeRef) {
      throw new Error('Store reference not set');
    }

    try {
      const now = new Date().toISOString();
      const fileName = `hnsw-index-${now.replace(/[:.]/g, '-')}`;
      
      // Persist the graph using the persistence service
      await hnswPersistence.persistGraph(this.hnswIndex, fileName);
      
      // Get persistence metadata
      const snapshotInfo = await hnswPersistence.getSnapshotInfo();
      const latestSnapshot = snapshotInfo.snapshots[0]; // Most recent
      
      if (latestSnapshot) {
        // Commit the successful persistence as an event
        this.storeRef.commit(events.hnswGraphSnapshotCreated({
          fileName: latestSnapshot.fileName,
          checksum: latestSnapshot.checksum,
          size: latestSnapshot.size,
          nodeCount: this.embeddings.size,
          embeddingModel: 'Snowflake/snowflake-arctic-embed-s',
          createdAt: now
        }));
        
        console.log(`HNSW graph persisted and tracked: ${fileName}`);
      }
      
      return { fileName, success: true };
    } catch (error) {
      console.error('Failed to persist HNSW index:', error);
      throw error;
    }
  }

  // Enhanced addOrUpdateNote with better error handling
  async addOrUpdateNote(noteId: string, title: string, content: Block[]) {
    try {
      await this.initialize();
      
      const textContent = blocksToText(content);
      if (!textContent.trim()) {
        // Remove from both memory, HNSW index, and LiveStore
        this.removePointFromIndex(noteId);
        if (this.storeRef) {
          // Find the chunk ID for this note and remove it
          const existingChunks = this.storeRef.query(
            tables.embeddings.select().where({ parentNoteId: noteId })
          );
          if (Array.isArray(existingChunks)) {
            existingChunks.forEach((chunk: any) => {
              this.storeRef.commit(events.chunkRemoved({ chunkId: chunk.chunkId }));
            });
          }
        }
        return;
      }

      const { vector } = await embeddingService.embed([textContent]);
      
      // Apply additional L2 normalization for vector hygiene
      const normalizedVector = l2Normalize(vector);
      
      // Remove old entry if it exists
      this.removePointFromIndex(noteId);
      
      // Add to HNSW index
      const hnswId = this.nextHnswId++;
      await this.hnswIndex.addPoint(hnswId, normalizedVector);
      
      // Generate a new chunk ID
      const chunkId = generateChunkId();
      
      // Store mapping between noteId and HNSW ID
      this.noteIdToHnswId.set(noteId, hnswId);
      this.hnswIdToNoteId.set(hnswId, noteId);
      
      // Update in-memory cache immediately for fast search
      this.embeddings.set(noteId, {
        chunkId,
        parentNoteId: noteId,
        title,
        content: textContent,
        embedding: normalizedVector
      });

      // Persist to LiveStore with chunk-based event
      if (this.storeRef) {
        const now = new Date().toISOString();
        this.storeRef.commit(events.chunkEmbedded({
          chunkId,
          parentNoteId: noteId,
          startIndex: 0,
          endIndex: textContent.length,
          tokenCount: textContent.split(/\s+/).length,
          vecData: vecToBlob(normalizedVector),
          vecDim: normalizedVector.length,
          embeddingModel: 'Snowflake/snowflake-arctic-embed-s',
          createdAt: now,
          updatedAt: now
        }));
      } else {
        console.warn('SemanticSearchService: Cannot commit embedding - no store reference');
      }
    } catch (error) {
      console.error('Failed to embed note:', error);
      toast.error('Failed to generate embedding for note');
    }
  }

  removeNote(noteId: string) {
    try {
      // Remove from both memory and LiveStore
      this.removePointFromIndex(noteId);
      if (this.storeRef) {
        // Find all chunks for this note and remove them
        const existingChunks = this.storeRef.query(
          tables.embeddings.select().where({ parentNoteId: noteId })
        );
        if (Array.isArray(existingChunks)) {
          existingChunks.forEach((chunk: any) => {
            this.storeRef.commit(events.chunkRemoved({ chunkId: chunk.chunkId }));
          });
        }
      } else {
        console.warn('SemanticSearchService: Cannot remove embedding - no store reference');
      }
    } catch (error) {
      console.error('Failed to remove embedding:', error);
    }
  }

  // Enhanced buildIndex with three-phase process
  async syncAllNotes(notes: Array<{ id: string; title: string; content: Block[] }>) {
    try {
      await this.initialize();
      
      console.log(`SemanticSearchService: Starting enhanced sync of ${notes.length} notes`);
      
      // Phase 1: Enhanced stale embedding cleanup
      this.reportProgress('cleanup', 0, 4, 'Cleaning up stale embeddings...');
      const cleanupResult = await this.forceCleanupStaleEmbeddings();
      console.log(`Cleanup summary - ${cleanupResult.summary}`);
      
      if (cleanupResult.remainingStale > 0) {
        console.warn(`Warning - ${cleanupResult.remainingStale} stale embeddings still remain`);
      }
      
      // Phase 2: Sync current notes
      this.reportProgress('sync', 1, 4, 'Syncing note embeddings...');
      
      // Clear in-memory cache and HNSW index
      this.embeddings.clear();
      this.hnswIndex = new HNSW(16, 200, null, 'cosine');
      this.noteIdToHnswId.clear();
      this.hnswIdToNoteId.clear();
      this.nextHnswId = 0;
      
      let successCount = 0;
      let errorCount = 0;
      
      // Process each note and generate embeddings
      for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        try {
          await this.addOrUpdateNote(note.id, note.title, note.content);
          successCount++;
          
          if (i % 10 === 0) {
            this.reportProgress('sync', 1, 4, `Processing note ${i + 1}/${notes.length}...`);
          }
        } catch (error) {
          errorCount++;
          console.error(`Failed to sync note ${note.id}:`, error);
        }
      }
      
      // Phase 3: Rebuild HNSW index with persistence
      this.reportProgress('build', 2, 4, 'Building search index...');
      
      // Reload from store to ensure consistency
      await this.loadEmbeddingsFromStore();
      
      // Phase 4: Persistence and cleanup
      this.reportProgress('persist', 3, 4, 'Persisting index and cleanup...');
      
      try {
        // Validate vectors before persistence
        let vectorValidationPassed = true;
        for (const [nodeId, node] of (this.hnswIndex as any).nodes || new Map()) {
          if (!node.vector || node.vector.length === 0) {
            console.error(`Built index has invalid vector for node ${nodeId}`);
            vectorValidationPassed = false;
          }
        }
        
        if (vectorValidationPassed && this.embeddings.size > 0) {
          await this.persistHnswIndex();
          console.log('HNSW index persisted successfully with event tracking');
        } else {
          console.warn('Cannot persist index - vector validation failed or no embeddings');
        }
        
        // Garbage collection of old snapshots
        await hnswPersistence.gcOldSnapshots(2);
      } catch (error) {
        console.error('Failed to persist HNSW graph:', error);
      }
      
      this.reportProgress('complete', 4, 4, 'Sync complete');
      
      console.log(`SemanticSearchService: Sync complete, ${successCount} embeddings created, ${errorCount} errors`);
      
      return this.embeddings.size;
    } catch (error) {
      console.error('Failed to sync notes:', error);
      toast.error('Failed to synchronize embeddings');
      return 0;
    }
  }

  async search(query: string, limit = 10): Promise<SearchResult[]> {
    try {
      await this.initialize();
      
      if (!query.trim() || this.embeddings.size === 0) {
        return [];
      }

      const { vector: queryVector } = await embeddingService.embed([`search_query: ${query}`]);
      
      // Apply L2 normalization to query vector for vector hygiene
      const normalizedQueryVector = l2Normalize(queryVector);
      
      // Use HNSW for fast approximate nearest neighbor search
      const hnswResults = this.hnswIndex.searchKNN(normalizedQueryVector, Math.min(limit * 2, 50));
      
      const results: SearchResult[] = [];
      
      // Convert HNSW results back to our format
      for (const hnswResult of hnswResults) {
        const noteId = this.hnswIdToNoteId.get(hnswResult.id);
        if (noteId && this.embeddings.has(noteId)) {
          const embedding = this.embeddings.get(noteId)!;
          results.push({
            noteId: embedding.parentNoteId,
            title: embedding.title,
            content: embedding.content,
            score: hnswResult.score
          });
        }
      }

      return results
        .sort((a, b) => b.score - a.score)
        .slice(0, limit);
    } catch (error) {
      console.error('Search failed:', error);
      toast.error('Search failed');
      return [];
    }
  }

  async getSnapshotInfo() {
    return await hnswPersistence.getSnapshotInfo();
  }

  async detectStaleData() {
    return await embeddingCleanupService.detectStaleData();
  }

  getEmbeddingCount(): number {
    return this.embeddings.size;
  }

  getStoredEmbeddingCount(): number {
    if (!this.storeRef) return 0;
    
    try {
      const countResult = this.storeRef.query(tables.embeddings.count());
      return countResult || 0;
    } catch (error) {
      console.error('Failed to get stored embedding count:', error);
      return 0;
    }
  }
}

export const semanticSearchService = new SemanticSearchService();
