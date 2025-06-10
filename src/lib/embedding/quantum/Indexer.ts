
import { QPSConfig, TokenMetadata } from './types';
import { TextProcessor } from './TextProcessor';
import { IndexStore } from './IndexStore';

export class Indexer {
  private config: Required<QPSConfig>;
  private textProcessor: TextProcessor;
  private store: IndexStore;

  constructor(config: Required<QPSConfig>, textProcessor: TextProcessor, store: IndexStore) {
    this.config = config;
    this.textProcessor = textProcessor;
    this.store = store;
  }

  public async index(docId: string, content: string): Promise<void> {
    const { sentences, totalTokenCount } = await this.textProcessor.processDocument(content);
    
    let segmentIndex = 0;
    const tokenMetadataForDoc: Map<string, TokenMetadata> = new Map();

    for (const sentence of sentences) {
      const tokensInSentence = sentence.tokens;
      const isSubstantiveSegment = tokensInSentence.length > 1;
      
      const bitIndex = Math.min(segmentIndex, this.config.maxSegments - 1);
      const segmentBit = 1 << bitIndex;

      for (const token of tokensInSentence) {
        if (!this.store.invertedIndex.has(token)) {
          this.store.invertedIndex.set(token, new Set());
        }
        this.store.invertedIndex.get(token)!.add(docId);

        if (!tokenMetadataForDoc.has(token)) {
          tokenMetadataForDoc.set(token, { freq: 0, segmentMask: 0 });
        }
        const metadata = tokenMetadataForDoc.get(token)!;
        metadata.freq += 1;
        metadata.segmentMask |= segmentBit;
      }
      
      if (isSubstantiveSegment) {
        segmentIndex++;
      }
    }
    
    this.store.docTokenMetadata.set(docId, tokenMetadataForDoc);
    this.store.docStore.set(docId, {
      originalContent: content,
      tokenCount: totalTokenCount,
    });
  }
}
