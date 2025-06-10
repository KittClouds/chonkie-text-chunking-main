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

// Helper to parse chunk IDs and extract parent information
function parseNodeId(id: string): { isChunk: boolean; parentId: string; chunkIndex?: number } {
  if (id.includes(':')) {
    const [parentId, chunkIndexStr] = id.split(':');
    return {
      isChunk: true,
      parentId,
      chunkIndex: parseInt(chunkIndexStr, 10)
    };
  }
  return {
    isChunk: false,
    parentId: id
  };
}

export class SearchEngine {
  private embeddings = new Map<string, NoteEmbedding>();
  private hnswIndex: HNSW;
  private noteIdToHnswId = new Map<string, number>();
  private hnswIdToNoteId = new Map<number, string>();
  private nextHnswId = 0;
  private isReady = false;

  // New fields for enhanced functionality
  private tombstones = new Set<number>();
  private queryCache = new Map<string, Float32Array>();
  private resultsCache = new Map<string, SearchResult[]>();
  private bm25Provider?: (query: string) => Map<string, number>;
  private config = {
    efSearch: 50,
    efConstruction: 200,
    alpha: 1.0, // 1.0 = pure vector, 0.0 = pure BM25
    cacheSize: 128
  };

  constructor(config: Partial<SearchEngine['config']> = {}) {
    this.config = { ...this.config, ...config };
    this.hnswIndex = new HNSW(16, this.config.efConstruction, 'cosine');
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

  // New method to add chunk-specific points
  async addChunkPoint(chunkId: string, vector: Float32Array, metadata?: { parentId: string; chunkIndex: number }): Promise<void> {
    if (!this.isReady || !this.hnswIndex) {
      console.warn('SearchEngine: Cannot add chunk point - service not ready');
      return;
    }

    try {
      const normalizedVector = l2Normalize(vector);
      const newHnswId = this.nextHnswId++;

      await this.hnswIndex.addPoint(newHnswId, normalizedVector);
      
      this.noteIdToHnswId.set(chunkId, newHnswId);
      this.hnswIdToNoteId.set(newHnswId, chunkId);
      
      if (metadata) {
        this.embeddings.set(chunkId, {
          noteId: chunkId,
          title: `Chunk ${metadata.chunkIndex} of ${metadata.parentId}`,
          content: '', // Chunks don't store full content in embeddings map
          embedding: normalizedVector
        });
      }

      console.log(`SearchEngine: Added chunk point ${chunkId}`);
    } catch (error) {
      console.error(`Failed to add chunk point ${chunkId}:`, error);
    }
  }

  // Remove a point from the search index using tombstones
  removePoint(noteId: string): void {
    const hnswId = this.noteIdToHnswId.get(noteId);
    if (hnswId !== undefined) {
      this.tombstones.add(hnswId);
      // Note: We no longer delete from the main maps, as tombstones handle it.
      // The entries will be fully purged on the next snapshot rebuild.
      console.log(`SearchEngine: Added tombstone for HNSW ID ${hnswId} (Note: ${noteId})`);
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
    this.hnswIndex = new HNSW(16, this.config.efConstruction, 'cosine');
    this.noteIdToHnswId.clear();
    this.hnswIdToNoteId.clear();
    this.nextHnswId = 0;
    this.tombstones.clear();
    this.queryCache.clear();
    this.resultsCache.clear();
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

  // New private method for adaptive search
  private adaptiveSearch(queryVector: Float32Array, limit: number) {
    let efSearch = this.config.efSearch;
    let results = this.hnswIndex.searchKNN(queryVector, limit, efSearch);

    // Filter out tombstoned results
    results = results.filter(result => !this.tombstones.has(result.id));

    // If recall looks low (top score is weak or not enough results), double ef and retry once.
    if (results.length > 0 && (results[0].score < 0.65 || results.length < limit)) {
      efSearch *= 2;
      const retryResults = this.hnswIndex.searchKNN(queryVector, limit * 2, efSearch);
      results = retryResults.filter(result => !this.tombstones.has(result.id));
    }
    return results;
  }

  // New private method for precise cosine calculation
  private exactCosine(v1: Float32Array, v2: Float32Array): number {
    let dotProduct = 0;
    for (let i = 0; i < v1.length; i++) {
      dotProduct += v1[i] * v2[i];
    }
    return dotProduct; // Assumes vectors are pre-normalized
  }

  // New public method to set BM25 provider
  public setBm25Provider(provider: (query: string) => Map<string, number>): void {
    this.bm25Provider = provider;
  }

  // Enhanced search method with all new features
  async search(query: string, limit = 10): Promise<SearchResult[]> {
    try {
      await this.initialize();
      const trimmedQuery = query.trim();
      if (!trimmedQuery || this.embeddings.size === 0) return [];

      // 1. Check results cache (for identical queries)
      if (this.resultsCache.has(trimmedQuery)) {
        return this.resultsCache.get(trimmedQuery)!.slice(0, limit);
      }

      // 2. Get query vector (from cache or by embedding)
      let queryVector = this.queryCache.get(trimmedQuery);
      if (!queryVector) {
        const { vector } = await embeddingService.embed([`search_query: ${trimmedQuery}`]);
        queryVector = l2Normalize(vector);
        this.queryCache.set(trimmedQuery, queryVector);
        if (this.queryCache.size > this.config.cacheSize) {
          this.queryCache.delete(this.queryCache.keys().next().value); // Evict oldest
        }
      }

      // 3. Adaptive HNSW search (with tombstone filter)
      const kForRerank = limit * 5;
      const hnswResults = this.adaptiveSearch(queryVector, kForRerank);

      // 4. Two-Stage Exact Reranking
      const candidates = hnswResults.slice(0, kForRerank).map(hnswResult => {
        const noteId = this.hnswIdToNoteId.get(hnswResult.id);
        if (!noteId) return null;
        const embedding = this.embeddings.get(noteId);
        if (!embedding) return null;
        
        const vectorScore = this.exactCosine(queryVector!, embedding.embedding);
        return { noteId, embedding, vectorScore };
      }).filter(c => c !== null) as { noteId: string; embedding: NoteEmbedding; vectorScore: number }[];

      // 5. Score Fusion (with BM25 stub)
      const bm25Scores = this.bm25Provider ? this.bm25Provider(trimmedQuery) : new Map<string, number>();
      
      const finalResults = candidates.map(candidate => {
        const { isChunk, parentId } = parseNodeId(candidate.noteId);
        const noteId = isChunk ? parentId : candidate.noteId;
        
        const bm25Score = bm25Scores.get(noteId) || 0;
        const finalScore = this.config.alpha * candidate.vectorScore + (1 - this.config.alpha) * bm25Score;
        
        return {
          noteId: noteId,
          title: candidate.embedding.title,
          content: candidate.embedding.content,
          score: finalScore
        };
      });

      // Deduplicate results based on parent note ID, keeping the best score
      const uniqueResults = Array.from(
        finalResults.reduce((map, item) => {
          if (!map.has(item.noteId) || item.score > map.get(item.noteId)!.score) {
            map.set(item.noteId, item);
          }
          return map;
        }, new Map<string, SearchResult>()).values()
      );

      const sortedResults = uniqueResults.sort((a, b) => b.score - a.score).slice(0, limit);
      
      // 6. Update results cache
      this.resultsCache.set(trimmedQuery, sortedResults);
      if (this.resultsCache.size > this.config.cacheSize) {
        this.resultsCache.delete(this.resultsCache.keys().next().value); // Evict oldest
      }
      
      return sortedResults;
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
