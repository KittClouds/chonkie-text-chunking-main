
import { embeddingService } from './EmbeddingService';
import { Block } from '@blocknote/core';
import { vecToBlob } from './binaryUtils';
import { tables, events } from '../../livestore/schema';
import { toast } from 'sonner';
import { hnswPersistence } from './hnsw/persistence';
import { embeddingCleanupService } from './CleanupService';
import { SearchEngine } from './SearchEngine';

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
  private searchEngine: SearchEngine;
  private isReady = false;
  private isInitialized = false;
  private buildProgressCallback: ((progress: BuildProgress) => void) | null = null;

  // Store reference will be injected by the hooks
  private storeRef: any = null;

  constructor() {
    this.searchEngine = new SearchEngine();
  }

  async initialize() {
    if (this.isReady) return;
    try {
      await embeddingService.ready();
      await this.searchEngine.initialize();
      this.isReady = true;
      console.log('SemanticSearchService initialized');
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

  // Getter for external access to HNSW index
  getHnswIndex() {
    return this.searchEngine.getHnswIndex();
  }

  // Incremental addition method for delta orchestrator
  async addPointToIndex(embeddingRow: any): Promise<void> {
    if (!this.isReady) {
      console.warn('SemanticSearchService: Cannot add point - service not ready');
      return;
    }

    await this.searchEngine.addPoint(embeddingRow);
  }

  // Incremental removal method for delta orchestrator
  removePointFromIndex(noteId: string): void {
    this.searchEngine.removePoint(noteId);
  }

  // Initialize from snapshot (for warm boot)
  async initializeFromSnapshot(): Promise<boolean> {
    try {
      console.log('SemanticSearchService: Attempting to load from snapshot...');
      const persistedGraph = await hnswPersistence.loadGraph('latest');
      
      if (persistedGraph && persistedGraph.nodes.size > 0) {
        this.searchEngine.setHnswIndex(persistedGraph);
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

  // Rebuild mappings after loading snapshot
  private async rebuildMappingsFromStore(): Promise<void> {
    if (!this.storeRef) {
      console.warn('Cannot rebuild mappings - no store reference');
      return;
    }

    try {
      const embeddingRows = this.storeRef.query(tables.embeddings.select());
      await this.searchEngine.rebuildMappings(embeddingRows);
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
      
      // Update search engine after cleanup
      this.loadEmbeddingsFromStore();
      
      return result;
    } catch (error) {
      console.error('SemanticSearchService: Force cleanup failed:', error);
      throw error;
    }
  }

  // Load all embeddings from LiveStore into search engine
  private async loadEmbeddingsFromStore() {
    if (!this.storeRef) {
      console.warn('SemanticSearchService: Cannot load embeddings - no store reference');
      return;
    }

    try {
      // Query all embeddings from the database
      const embeddingRows = this.storeRef.query(tables.embeddings.select());
      console.log(`SemanticSearchService: Loading ${embeddingRows?.length || 0} embeddings from LiveStore`);
      
      // Load embeddings into search engine
      await this.searchEngine.loadEmbeddings(embeddingRows);
      console.log(`SemanticSearchService: Loaded ${this.searchEngine.getEmbeddingCount()} embeddings into search engine`);
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
      await hnswPersistence.persistGraph(this.searchEngine.getHnswIndex(), fileName);
      
      // Get persistence metadata
      const snapshotInfo = await hnswPersistence.getSnapshotInfo();
      const latestSnapshot = snapshotInfo.snapshots[0]; // Most recent
      
      if (latestSnapshot) {
        // Commit the successful persistence as an event
        this.storeRef.commit(events.hnswGraphSnapshotCreated({
          fileName: latestSnapshot.fileName,
          checksum: latestSnapshot.checksum,
          size: latestSnapshot.size,
          nodeCount: this.searchEngine.getEmbeddingCount(),
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
        // Remove from both search engine and LiveStore
        this.searchEngine.removePoint(noteId);
        if (this.storeRef) {
          this.storeRef.commit(events.embeddingRemoved({ noteId }));
        }
        return;
      }

      const { vector } = await embeddingService.embed([textContent]);
      
      // Apply additional L2 normalization for vector hygiene
      const normalizedVector = l2Normalize(vector);
      
      // Create embedding row data for search engine
      const embeddingRow = {
        noteId,
        title,
        content: textContent,
        vecData: vecToBlob(normalizedVector),
        vecDim: normalizedVector.length
      };
      
      // Add to search engine
      await this.searchEngine.addPoint(embeddingRow);

      // Persist to LiveStore with enhanced tracking
      if (this.storeRef) {
        const now = new Date().toISOString();
        this.storeRef.commit(events.noteEmbedded({
          noteId,
          title,
          content: textContent,
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
      // Remove from both search engine and LiveStore
      this.searchEngine.removePoint(noteId);
      if (this.storeRef) {
        this.storeRef.commit(events.embeddingRemoved({ noteId }));
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
      
      // Clear search engine
      this.searchEngine.clear();
      
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
      
      // Phase 3: Rebuild search index
      this.reportProgress('build', 2, 4, 'Building search index...');
      
      // Reload from store to ensure consistency
      await this.loadEmbeddingsFromStore();
      
      // Phase 4: Persistence and cleanup
      this.reportProgress('persist', 3, 4, 'Persisting index and cleanup...');
      
      try {
        // Validate vectors before persistence
        let vectorValidationPassed = true;
        const hnswIndex = this.searchEngine.getHnswIndex();
        for (const [nodeId, node] of (hnswIndex as any).nodes || new Map()) {
          if (!node.vector || node.vector.length === 0) {
            console.error(`Built index has invalid vector for node ${nodeId}`);
            vectorValidationPassed = false;
          }
        }
        
        if (vectorValidationPassed && this.searchEngine.getEmbeddingCount() > 0) {
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
      
      return this.searchEngine.getEmbeddingCount();
    } catch (error) {
      console.error('Failed to sync notes:', error);
      toast.error('Failed to synchronize embeddings');
      return 0;
    }
  }

  async search(query: string, limit = 10): Promise<SearchResult[]> {
    try {
      return await this.searchEngine.search(query, limit);
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
    return this.searchEngine.getEmbeddingCount();
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
