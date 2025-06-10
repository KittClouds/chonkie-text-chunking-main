
import { TokenMetadata, DocumentData } from './types';

export class IndexStore {
  // Maps token to a set of document IDs that contain the token.
  public readonly invertedIndex: Map<string, Set<string>> = new Map();
  // Maps docId -> (maps token -> metadata for that token in the doc).
  public readonly docTokenMetadata: Map<string, Map<string, TokenMetadata>> = new Map();
  // Maps docId -> stored data about the document.
  public readonly docStore: Map<string, DocumentData> = new Map();

  public getDoc(docId: string): DocumentData | undefined {
    return this.docStore.get(docId);
  }

  public getDocMetadata(docId: string): Map<string, TokenMetadata> | undefined {
    return this.docTokenMetadata.get(docId);
  }

  public getTokenMapSizeForDoc(docId: string): number | undefined {
    const tokenMap = this.docTokenMetadata.get(docId);
    return tokenMap ? tokenMap.size : undefined;
  }

  public getMatchingDocIds(token: string): Set<string> | undefined {
    return this.invertedIndex.get(token);
  }

  public clear(): void {
    this.invertedIndex.clear();
    this.docTokenMetadata.clear();
    this.docStore.clear();
  }

  public removeDocument(docId: string): void {
    // Remove from doc store
    this.docStore.delete(docId);
    
    // Remove from doc token metadata
    const tokenMetadata = this.docTokenMetadata.get(docId);
    this.docTokenMetadata.delete(docId);
    
    // Remove from inverted index
    if (tokenMetadata) {
      for (const token of tokenMetadata.keys()) {
        const docSet = this.invertedIndex.get(token);
        if (docSet) {
          docSet.delete(docId);
          if (docSet.size === 0) {
            this.invertedIndex.delete(token);
          }
        }
      }
    }
  }
}
