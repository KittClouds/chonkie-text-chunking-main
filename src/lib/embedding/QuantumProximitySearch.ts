
import { QPSConfig, SearchResult, DocumentData, TokenMetadata } from './quantum/types';
import { TextProcessor } from './quantum/TextProcessor';
import { IndexStore } from './quantum/IndexStore';
import { Indexer } from './quantum/Indexer';
import { Searcher } from './quantum/Searcher';

export class QuantumProximitySearch {
  private config: Required<QPSConfig>;
  private nextDocId = 0;

  // Encapsulated modules
  private textProcessor: TextProcessor;
  private store: IndexStore;
  private indexer: Indexer;
  private searcher: Searcher;

  constructor(config: QPSConfig = {}) {
    this.config = {
      maxSegments: config.maxSegments ?? 21,
      proximityBonus: config.proximityBonus ?? 2,
    };
    
    // Instantiate and wire up the modules
    this.textProcessor = new TextProcessor();
    this.store = new IndexStore();
    this.indexer = new Indexer(this.config, this.textProcessor, this.store);
    this.searcher = new Searcher(this.config, this.textProcessor, this.store);
  }
  
  public async indexDocument(content: string, customDocId?: string): Promise<string> {
    const docId = customDocId || (this.nextDocId++).toString();
    await this.indexer.index(docId, content);
    return docId;
  }
  
  public async search(query: string): Promise<SearchResult[]> {
    return await this.searcher.search(query);
  }

  public removeDocument(docId: string): void {
    this.store.removeDocument(docId);
  }

  public clear(): void {
    this.store.clear();
    this.nextDocId = 0;
  }

  public getDocumentCount(): number {
    return this.store.docStore.size;
  }

  // --- Methods for Testability ---
  public getTestDocData(docId: string): DocumentData | undefined {
    return this.store.getDoc(docId);
  }

  public getTestTokenMetadata(docId: string, token: string): TokenMetadata | undefined {
    const docMeta = this.store.getDocMetadata(docId);
    return docMeta ? docMeta.get(token) : undefined;
  }

  public getTestTokenMapSizeForDoc(docId: string): number | undefined {
    return this.store.getTokenMapSizeForDoc(docId);
  }
}

// Re-export types for external use
export type { SearchResult, QPSConfig } from './quantum/types';
