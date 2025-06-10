
import { QPSConfig, SearchResult } from './types';
import { TextProcessor } from './TextProcessor';
import { IndexStore } from './IndexStore';

export class Searcher {
  private config: Required<QPSConfig>;
  private textProcessor: TextProcessor;
  private store: IndexStore;

  constructor(config: Required<QPSConfig>, textProcessor: TextProcessor, store: IndexStore) {
    this.config = config;
    this.textProcessor = textProcessor;
    this.store = store;
  }

  public async search(query: string): Promise<SearchResult[]> {
    const queryTokens = await this.textProcessor.tokenizeQuery(query);
    const scoreMap = new Map<string, { score: number, combinedMask: number }>();

    for (const token of queryTokens) {
      const matchingDocIds = this.store.getMatchingDocIds(token);
      if (!matchingDocIds) continue;

      for (const docId of matchingDocIds) {
        const docMetadata = this.store.getDocMetadata(docId)!;
        const tokenMetadata = docMetadata.get(token)!;
        const docData = this.store.getDoc(docId)!;
        
        const tokenScore = (Math.pow(tokenMetadata.freq, 2) / docData.tokenCount) + 1;

        if (!scoreMap.has(docId)) {
          scoreMap.set(docId, { score: tokenScore, combinedMask: tokenMetadata.segmentMask });
        } else {
          const currentEntry = scoreMap.get(docId)!;
          const commonMask = currentEntry.combinedMask & tokenMetadata.segmentMask;
          const overlapCount = this.countSetBits(commonMask);
          const proximityBonus = overlapCount * this.config.proximityBonus;
          
          currentEntry.score += tokenScore + proximityBonus;
          currentEntry.combinedMask |= tokenMetadata.segmentMask;
        }
      }
    }
    
    const results: SearchResult[] = Array.from(scoreMap.entries()).map(([docId, data]) => ({
      docId,
      score: data.score,
      content: this.store.getDoc(docId)!.originalContent,
    }));
    
    return results.sort((a, b) => b.score - a.score);
  }
  
  private countSetBits(n: number): number {
    let count = 0;
    while (n > 0) {
      n &= (n - 1);
      count++;
    }
    return count;
  }
}
