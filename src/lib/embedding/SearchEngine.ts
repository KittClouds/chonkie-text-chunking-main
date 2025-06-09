
import { embeddingService } from './EmbeddingService';
import { HNSW } from './hnsw';
import { blobToVec } from './binaryUtils';

interface NoteEmbedding {
  noteId: string;
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

// L2 normalize a vector to unit length for vector hygiene
function l2Normalize(v: Float32Array): Float32Array {
  let norm = 0; 
  for (const x of v) norm += x * x;
  norm = 1 / Math.sqrt(norm || 1e-9);
  return v.map(x => x * norm) as Float32Array;
}

export class SearchEngine {
  private embeddings = new Map<string, NoteEmbedding>();
  private hnswIndex: HNSW;
  private noteIdToHnswId = new Map<string, number>();
  private hnswIdToNoteId = new Map<number, string>();
  private nextHnswId = 0;
  private isReady = false;

  constructor() {
    // Initialize HNSW with cosine similarity
    this.hnswIndex = new HNSW(16, 200, null, 'cosine');
  }

  async initialize() {
    if (this.isReady) return;
    try {
      await embeddingService.ready();
      this.isReady = true;
      console.log('SearchEngine initialized with HNSW');
    } catch (error) {
      console.error('Failed to initialize SearchEngine:', error);
    }
  }

  // Getter for external access to HNSW index
  getHnswIndex(): HNSW {
    return this.hnswIndex;
  }

  // Add or update a point in the search index
  async addPoint(embeddingRow: any): Promise<void> {
    if (!this.isReady || !this.hnswIndex) {
      console.warn('SearchEngine: Cannot add point - service not ready');
      return;
    }

    try {
      const existingHnswId = this.noteIdToHnswId.get(embeddingRow.noteId);
      if (existingHnswId !== undefined) {
        // This is an update, remove old point first
        this.removePoint(embeddingRow.noteId);
      }

      const embeddingVector = blobToVec(embeddingRow.vecData, embeddingRow.vecDim);
      const normalizedVector = l2Normalize(embeddingVector);
      const newHnswId = this.nextHnswId++;

      await this.hnswIndex.addPoint(newHnswId, normalizedVector);
      
      this.noteIdToHnswId.set(embeddingRow.noteId, newHnswId);
      this.hnswIdToNoteId.set(newHnswId, embeddingRow.noteId);
      this.embeddings.set(embeddingRow.noteId, {
        noteId: embeddingRow.noteId,
        title: embeddingRow.title,
        content: embeddingRow.content,
        embedding: normalizedVector
      });

      console.log(`SearchEngine: Added point for note ${embeddingRow.noteId}`);
    } catch (error) {
      console.error(`Failed to add point to search index for note ${embeddingRow.noteId}:`, error);
    }
  }

  // Remove a point from the search index
  removePoint(noteId: string): void {
    if (!this.hnswIndex) {
      console.warn('SearchEngine: Cannot remove point - no HNSW index');
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
      console.log(`SearchEngine: Marked point for removal for note ${noteId}`);
    }
  }

  // Build the search index from embedding data
  async buildIndex(embeddingData: Array<{ id: number; vector: Float32Array }>): Promise<void> {
    if (embeddingData.length > 0) {
      await this.hnswIndex.buildIndex(embeddingData);
      console.log(`SearchEngine: Built HNSW index with ${embeddingData.length} vectors`);
    }
  }

  // Clear all data from the search engine
  clear(): void {
    this.embeddings.clear();
    this.hnswIndex = new HNSW(16, 200, null, 'cosine');
    this.noteIdToHnswId.clear();
    this.hnswIdToNoteId.clear();
    this.nextHnswId = 0;
  }

  // Load embeddings from database rows into the search index
  async loadEmbeddings(embeddingRows: any[]): Promise<void> {
    console.log(`SearchEngine: Loading ${embeddingRows?.length || 0} embeddings`);
    
    // Reset search engine before loading
    this.clear();
    
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
          this.noteIdToHnswId.set(row.noteId, hnswId);
          this.hnswIdToNoteId.set(hnswId, row.noteId);
          
          this.embeddings.set(row.noteId, {
            noteId: row.noteId,
            title: row.title,
            content: row.content,
            embedding
          });
          
          // Prepare data for HNSW index
          hnswData.push({ id: hnswId, vector: embedding });
        } catch (error) {
          console.error(`Failed to load embedding for note ${row?.noteId}:`, error);
        }
      });
      
      // Build HNSW index with all data at once for better performance
      await this.buildIndex(hnswData);
    }

    console.log(`SearchEngine: Loaded ${this.embeddings.size} embeddings into search index`);
  }

  // Rebuild mappings after loading from snapshot
  async rebuildMappings(embeddingRows: any[]): Promise<void> {
    try {
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
            
            this.noteIdToHnswId.set(row.noteId, hnswId);
            this.hnswIdToNoteId.set(hnswId, row.noteId);
            this.embeddings.set(row.noteId, {
              noteId: row.noteId,
              title: row.title,
              content: row.content,
              embedding
            });
            
            maxHnswId = Math.max(maxHnswId, hnswId);
          } catch (error) {
            console.error(`Failed to rebuild mapping for note ${row?.noteId}:`, error);
          }
        });
        
        this.nextHnswId = maxHnswId + 1;
        console.log(`SearchEngine: Rebuilt mappings for ${this.embeddings.size} embeddings`);
      }
    } catch (error) {
      console.error('Failed to rebuild mappings:', error);
    }
  }

  // Perform semantic search
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
            noteId: embedding.noteId,
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
      throw error;
    }
  }

  // Get embedding count
  getEmbeddingCount(): number {
    return this.embeddings.size;
  }

  // Initialize from snapshot
  setHnswIndex(index: HNSW): void {
    this.hnswIndex = index;
  }
}
