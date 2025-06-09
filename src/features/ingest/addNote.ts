
import { chunk } from '@/lib/transform/chonkie';
import { embeddingService } from '@/lib/embedding/EmbeddingService';
import { nodeId, chunkId } from '@/lib/utils/ids';
import { semanticSearchService } from '@/lib/embedding/SemanticSearchService';
import { Block } from '@blocknote/core';

interface IngestResult {
  nodeId: string;
  chunkCount: number;
  pooledVector: Float32Array;
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

export async function ingestNote(
  noteId: string,
  title: string, 
  content: Block[] | string
): Promise<IngestResult> {
  try {
    // Convert content to text
    const raw = typeof content === 'string' ? content : blocksToText(content);
    
    if (!raw.trim()) {
      throw new Error('No text content to ingest');
    }

    // Generate stable node ID
    const node = nodeId();
    
    // Chunk the text using recursive chunker for better semantic boundaries
    const chunks = await chunk(raw, true);
    console.log(`Chunked text into ${chunks.length} chunks`);
    
    if (chunks.length === 0) {
      throw new Error('No chunks generated from text');
    }

    // Extract text from chunks for embedding
    const chunkTexts = chunks.map(c => c.text);
    
    // Generate embeddings for all chunks
    const { vector: vectors, dim } = await embeddingService.embed(chunkTexts);
    
    // Get the search engine from semantic search service
    const searchEngine = semanticSearchService.getHnswIndex();
    
    // Store each chunk with its embedding
    for (let i = 0; i < chunks.length; i++) {
      const chunkVector = vectors.slice(i * dim, (i + 1) * dim);
      const chunkNodeId = chunkId(node, i);
      
      // Add chunk to HNSW index
      await searchEngine.addPoint(parseInt(chunkNodeId.replace(/[^0-9]/g, ''), 36), chunkVector);
    }
    
    // Create mean-pooled vector for legacy cluster logic
    const pooled = new Float32Array(dim);
    for (let d = 0; d < dim; d++) {
      pooled[d] = chunkTexts.reduce((acc, _, i) => acc + vectors[i * dim + d], 0) / chunkTexts.length;
    }
    
    // Add pooled vector to search engine and LiveStore
    await searchEngine.addPoint(parseInt(node.replace(/[^0-9]/g, ''), 36), pooled);
    
    // Use existing semantic search service to persist to LiveStore
    await semanticSearchService.addOrUpdateNote(noteId, title, typeof content === 'string' ? [] : content);
    
    console.log(`Ingested note ${noteId} with ${chunks.length} chunks`);
    
    return {
      nodeId: node,
      chunkCount: chunks.length,
      pooledVector: pooled
    };
  } catch (error) {
    console.error('Failed to ingest note:', error);
    throw error;
  }
}

export async function ingestBatch(
  notes: Array<{ id: string; title: string; content: Block[] | string }>
): Promise<IngestResult[]> {
  const results: IngestResult[] = [];
  
  for (const note of notes) {
    try {
      const result = await ingestNote(note.id, note.title, note.content);
      results.push(result);
    } catch (error) {
      console.error(`Failed to ingest note ${note.id}:`, error);
    }
  }
  
  return results;
}
