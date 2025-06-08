
import { Store } from '@livestore/livestore';
import { tables } from '../livestore/schema';
import { queryDb } from '@livestore/livestore';
import { semanticSearchService } from '../lib/embedding/SemanticSearchService';
import { hnswPersistence } from '../lib/embedding/hnsw/persistence';

class HnswSyncOrchestrator {
  private store: Store<any> | null = null;
  private unsubscribe: (() => void) | null = null;
  private knownEmbeddingIds = new Set<string>();
  private knownEmbeddingHashes = new Map<string, string>(); // noteId -> hash of embedding data
  private snapshotInterval: NodeJS.Timeout | null = null;
  private isInitialized = false;

  // This is the query we watch
  private readonly allEmbeddingsQuery$ = queryDb(tables.embeddings, { label: 'allEmbeddingsForHnswSync$' });

  public initialize(store: Store<any>) {
    if (this.store || this.isInitialized) return;
    this.store = store;
    this.isInitialized = true;
    
    console.log("HnswSyncOrchestrator: Initializing...");
    
    // Initial Full Sync on boot
    this.performFullSync();

    // Subscribe to changes with debounced delta sync
    this.unsubscribe = store.subscribe(this.allEmbeddingsQuery$, () => {
      // Use a short timeout to batch rapid changes
      setTimeout(() => this.performDeltaSync(), 100);
    });
    
    // Set up periodic snapshots every 5 minutes
    this.snapshotInterval = setInterval(() => this.triggerSnapshot(), 5 * 60 * 1000);

    console.log("HnswSyncOrchestrator initialized.");
  }

  private performFullSync() {
    console.log("HNSW Orchestrator: Performing initial full sync.");
    if (!this.store) return;

    const allEmbeddings = this.store.query(this.allEmbeddingsQuery$);
    this.knownEmbeddingIds.clear();
    this.knownEmbeddingHashes.clear();

    if (Array.isArray(allEmbeddings)) {
      for (const embedding of allEmbeddings) {
        semanticSearchService.addPointToIndex(embedding);
        this.knownEmbeddingIds.add(embedding.noteId);
        // Create a simple hash of the embedding data for change detection
        const hash = this.createEmbeddingHash(embedding);
        this.knownEmbeddingHashes.set(embedding.noteId, hash);
      }
    }
    
    console.log(`HNSW Orchestrator: Full sync complete. Index has ${this.knownEmbeddingIds.size} items.`);
  }
  
  private performDeltaSync() {
    console.log("HNSW Orchestrator: Performing delta sync.");
    if (!this.store) return;

    const currentEmbeddings = this.store.query(this.allEmbeddingsQuery$);
    if (!Array.isArray(currentEmbeddings)) return;

    const currentIds = new Set(currentEmbeddings.map(e => e.noteId));
    const currentHashes = new Map<string, string>();

    // Find new/updated embeddings
    for (const embedding of currentEmbeddings) {
      const currentHash = this.createEmbeddingHash(embedding);
      currentHashes.set(embedding.noteId, currentHash);
      
      const knownHash = this.knownEmbeddingHashes.get(embedding.noteId);
      
      if (!this.knownEmbeddingIds.has(embedding.noteId) || knownHash !== currentHash) {
        // This is either new or updated
        console.log(`HNSW Orchestrator: Adding/updating note ${embedding.noteId}`);
        semanticSearchService.addPointToIndex(embedding);
        this.knownEmbeddingIds.add(embedding.noteId);
      }
    }
    
    // Find removed embeddings
    const removedIds = [...this.knownEmbeddingIds].filter(id => !currentIds.has(id));
    for (const noteId of removedIds) {
      console.log(`HNSW Orchestrator: Removing note ${noteId}`);
      semanticSearchService.removePointFromIndex(noteId);
      this.knownEmbeddingIds.delete(noteId);
      this.knownEmbeddingHashes.delete(noteId);
    }

    // Update our known hashes
    this.knownEmbeddingHashes = currentHashes;

    if (removedIds.length > 0 || currentEmbeddings.some(e => {
      const currentHash = this.createEmbeddingHash(e);
      const knownHash = this.knownEmbeddingHashes.get(e.noteId);
      return !this.knownEmbeddingIds.has(e.noteId) || knownHash !== currentHash;
    })) {
      console.log(`HNSW Orchestrator: Delta sync detected changes. Index now has ${this.knownEmbeddingIds.size} items.`);
    }
  }

  // Create a simple hash of embedding data to detect changes
  private createEmbeddingHash(embedding: any): string {
    // Create a hash based on content that matters for the search index
    const hashData = `${embedding.title}|${embedding.content}|${embedding.updatedAt}|${embedding.embeddingModel}`;
    return btoa(hashData).slice(0, 16); // Simple hash using base64 encoding
  }

  public async triggerSnapshot() {
    console.log("HNSW Orchestrator: Triggering periodic snapshot...");
    const indexToPersist = semanticSearchService.getHnswIndex();
    
    if (indexToPersist && this.knownEmbeddingIds.size > 0) {
      try {
        // This logic ensures we always have a backup.
        // Move latest to backup, then save new latest
        await hnswPersistence.removeFile('backup').catch(() => {}); // Ignore if backup doesn't exist
        await hnswPersistence.renameFile('latest', 'backup').catch(() => {}); // Ignore if latest doesn't exist
        await hnswPersistence.persistGraph(indexToPersist, 'latest');
        console.log("HNSW Orchestrator: Snapshot completed successfully.");
      } catch (error) {
        console.error("HNSW Orchestrator: Snapshot failed.", error);
        // Attempt to restore backup if latest failed mid-write
        await hnswPersistence.renameFile('backup', 'latest').catch(() => {
          console.warn("HNSW Orchestrator: Could not restore backup after failed snapshot.");
        });
      }
    } else {
      console.log("HNSW Orchestrator: Skipping snapshot - no data to persist.");
    }
  }

  // Manual trigger for immediate sync (useful for debugging)
  public async forceSync() {
    console.log("HNSW Orchestrator: Force sync requested.");
    this.performDeltaSync();
  }

  // Manual trigger for immediate snapshot
  public async forceSnapshot() {
    console.log("HNSW Orchestrator: Force snapshot requested.");
    await this.triggerSnapshot();
  }

  public getStats() {
    return {
      knownEmbeddingCount: this.knownEmbeddingIds.size,
      isInitialized: this.isInitialized,
      hasStore: !!this.store
    };
  }

  public shutdown() {
    console.log("HnswSyncOrchestrator: Shutting down...");
    
    this.unsubscribe?.();
    this.unsubscribe = null;
    
    if (this.snapshotInterval) {
      clearInterval(this.snapshotInterval);
      this.snapshotInterval = null;
    }
    
    this.store = null;
    this.isInitialized = false;
    this.knownEmbeddingIds.clear();
    this.knownEmbeddingHashes.clear();
    
    console.log("HnswSyncOrchestrator shut down.");
  }
}

export const hnswSyncOrchestrator = new HnswSyncOrchestrator();
