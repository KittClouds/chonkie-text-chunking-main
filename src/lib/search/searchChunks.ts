
import { embeddingService } from '@/lib/embedding/EmbeddingService';
import { semanticSearchService } from '@/lib/embedding/SemanticSearchService';

interface ChunkSearchResult {
  id: string;
  parentId: string;
  score: number;
  chunkIndex?: number;
}

interface ParentSearchResult {
  parentId: string;
  bestChunkId: string;
  bestScore: number;
  chunkIndex?: number;
}

export class ChunkSearchService {
  async searchChunks(query: string, k = 10): Promise<ChunkSearchResult[]> {
    try {
      // Generate query embedding
      const { vector: queryVector } = await embeddingService.embed([query]);
      
      // Get HNSW index from semantic search service
      const hnswIndex = semanticSearchService.getHnswIndex();
      
      // Over-fetch to allow for parent deduplication
      const hits = hnswIndex.searchKNN(queryVector, k * 4);
      
      // Convert HNSW results to chunk results
      return hits.map(hit => {
        const idStr = hit.id.toString();
        const isChunk = idStr.includes(':');
        
        if (isChunk) {
          const [parentId, chunkIndexStr] = idStr.split(':');
          return {
            id: idStr,
            parentId,
            score: hit.score,
            chunkIndex: parseInt(chunkIndexStr, 10)
          };
        } else {
          return {
            id: idStr,
            parentId: idStr,
            score: hit.score
          };
        }
      }).slice(0, k);
    } catch (error) {
      console.error('Chunk search failed:', error);
      throw error;
    }
  }

  async searchParents(query: string, k = 10): Promise<ParentSearchResult[]> {
    try {
      // Get chunk-level results
      const chunkResults = await this.searchChunks(query, k * 4);
      
      // Group by parent and find best chunk per parent
      const parentMap = new Map<string, ParentSearchResult>();
      
      for (const chunk of chunkResults) {
        const existing = parentMap.get(chunk.parentId);
        
        if (!existing || chunk.score > existing.bestScore) {
          parentMap.set(chunk.parentId, {
            parentId: chunk.parentId,
            bestChunkId: chunk.id,
            bestScore: chunk.score,
            chunkIndex: chunk.chunkIndex
          });
        }
      }
      
      // Sort by best score and limit results
      return Array.from(parentMap.values())
        .sort((a, b) => b.bestScore - a.bestScore)
        .slice(0, k);
    } catch (error) {
      console.error('Parent search failed:', error);
      throw error;
    }
  }

  async hybridSearch(query: string, k = 10): Promise<{
    chunks: ChunkSearchResult[];
    parents: ParentSearchResult[];
  }> {
    try {
      const [chunks, parents] = await Promise.all([
        this.searchChunks(query, k),
        this.searchParents(query, k)
      ]);
      
      return { chunks, parents };
    } catch (error) {
      console.error('Hybrid search failed:', error);
      throw error;
    }
  }
}

export const chunkSearchService = new ChunkSearchService();

// Convenience functions
export const searchChunks = (query: string, k = 10) => 
  chunkSearchService.searchChunks(query, k);

export const searchParents = (query: string, k = 10) => 
  chunkSearchService.searchParents(query, k);

export const hybridSearch = (query: string, k = 10) => 
  chunkSearchService.hybridSearch(query, k);
