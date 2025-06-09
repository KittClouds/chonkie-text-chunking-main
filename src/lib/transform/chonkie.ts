
import { TokenChunker, RecursiveChunker, RecursiveRules } from 'chonkie';

interface ChunkResult {
  text: string;
  startIndex: number;
  endIndex: number;
  tokenCount: number;
  level?: number;
}

class ChunkingService {
  private tokenChunker: TokenChunker | null = null;
  private recursiveChunker: RecursiveChunker | null = null;
  private isInitialized = false;

  async initialize() {
    if (this.isInitialized) return;

    try {
      // Initialize token chunker with default settings
      this.tokenChunker = await TokenChunker.create({
        tokenizer: "Xenova/gpt2",
        chunkSize: 512,
        returnType: 'chunks'
      });

      // Initialize recursive chunker with hierarchical rules
      const rules = new RecursiveRules([
        { delimiters: ["\n\n"], includeDelim: "prev" }, // Paragraphs
        { delimiters: [". ", "! ", "? "], includeDelim: "prev" }, // Sentences
        { whitespace: true } // Words as fallback
      ]);

      this.recursiveChunker = await RecursiveChunker.create({
        tokenizer: "Xenova/gpt2",
        chunkSize: 256,
        rules,
        minCharactersPerChunk: 24,
        returnType: 'chunks'
      });

      this.isInitialized = true;
      console.log('ChunkingService initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ChunkingService:', error);
      throw error;
    }
  }

  async chunkWithToken(text: string): Promise<ChunkResult[]> {
    await this.initialize();
    if (!this.tokenChunker) throw new Error('Token chunker not initialized');

    const chunks = await this.tokenChunker.chunk(text);
    return chunks.map(chunk => ({
      text: chunk.text,
      startIndex: chunk.startIndex,
      endIndex: chunk.endIndex,
      tokenCount: chunk.tokenCount
    }));
  }

  async chunkWithRecursive(text: string): Promise<ChunkResult[]> {
    await this.initialize();
    if (!this.recursiveChunker) throw new Error('Recursive chunker not initialized');

    const chunks = await this.recursiveChunker.chunk(text);
    return chunks.map(chunk => ({
      text: chunk.text,
      startIndex: chunk.startIndex,
      endIndex: chunk.endIndex,
      tokenCount: chunk.tokenCount,
      level: chunk.level
    }));
  }

  async chunkBatch(texts: string[], useRecursive = true): Promise<ChunkResult[][]> {
    await this.initialize();
    
    const chunker = useRecursive ? this.recursiveChunker : this.tokenChunker;
    if (!chunker) throw new Error('Chunker not initialized');

    const batchChunks = await chunker.chunkBatch(texts);
    return batchChunks.map(docChunks => 
      docChunks.map(chunk => ({
        text: chunk.text,
        startIndex: chunk.startIndex,
        endIndex: chunk.endIndex,
        tokenCount: chunk.tokenCount,
        level: (chunk as any).level
      }))
    );
  }
}

export const chunkingService = new ChunkingService();

// Convenience functions
export const chunk = (text: string, useRecursive = true): Promise<ChunkResult[]> => {
  return useRecursive 
    ? chunkingService.chunkWithRecursive(text)
    : chunkingService.chunkWithToken(text);
};

export const chunkBatch = (texts: string[], useRecursive = true): Promise<ChunkResult[][]> => {
  return chunkingService.chunkBatch(texts, useRecursive);
};
