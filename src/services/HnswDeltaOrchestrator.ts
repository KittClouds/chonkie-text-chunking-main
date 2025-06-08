import { queryDb } from '@livestore/livestore';
import { tables } from '../livestore/schema';
import { 
  notesRequiringEmbedding$, 
  orphanedEmbeddings$ 
} from '../livestore/queries/sync';
import { semanticSearchService } from '../lib/embedding/SemanticSearchService';
import { hnswPersistence } from '../lib/embedding/hnsw/persistence';

interface DeltaState {
  knownEmbeddingIds: Set<string>;
  lastProcessedTime: number;
  snapshotCounter: number;
}

class HnswDeltaOrchestrator {
  private store: any = null;
  private unsubscribeCallbacks: (() => void)[] = [];
  private deltaState: DeltaState = {
    knownEmbeddingIds: new Set(),
    lastProcessedTime: 0,
    snapshotCounter: 0
  };
  private snapshotInterval: NodeJS.Timeout | null = null;
  private isProcessing = false;
  private pendingDeltas = false;

  // Configuration
  private readonly SNAPSHOT_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes
  private readonly CHANGES_THRESHOLD = 50; // Snapshot after 50 changes
  private readonly DEBOUNCE_MS = 1000; // 1 second debounce

  async initialize(store: any): Promise<void> {
    if (this.store) {
      console.log('HnswDeltaOrchestrator: Already initialized');
      return;
    }

    this.store = store;
    console.log('HnswDeltaOrchestrator: Initializing...');

    try {
      // Try warm boot first (load from snapshot)
      const warmBootSuccess = await this.attemptWarmBoot();
      
      if (!warmBootSuccess) {
        // Fall back to cold boot (full rebuild)
        await this.performColdBoot();
      }

      // Start delta sync subscriptions
      this.startDeltaSync();
      
      // Start periodic snapshots
      this.startPeriodicSnapshots();
      
      console.log('HnswDeltaOrchestrator: Initialization complete');
    } catch (error) {
      console.error('HnswDeltaOrchestrator: Initialization failed:', error);
      // Fall back to cold boot on any error
      await this.performColdBoot();
      this.startDeltaSync();
      this.startPeriodicSnapshots();
    }
  }

  private async attemptWarmBoot(): Promise<boolean> {
    console.log('HnswDeltaOrchestrator: Attempting warm boot from snapshot...');
    
    try {
      const snapshotLoaded = await semanticSearchService.initializeFromSnapshot();
      
      if (snapshotLoaded) {
        // Initialize known embeddings from current store state
        await this.syncKnownEmbeddings();
        console.log('HnswDeltaOrchestrator: Warm boot successful');
        return true;
      }
    } catch (error) {
      console.warn('HnswDeltaOrchestrator: Warm boot failed:', error);
    }
    
    return false;
  }

  private async performColdBoot(): Promise<void> {
    console.log('HnswDeltaOrchestrator: Performing cold boot (full rebuild)...');
    
    try {
      // Get all current embeddings and initialize the service
      const allEmbeddings = this.store.query(tables.embeddings.select());
      
      if (Array.isArray(allEmbeddings)) {
        // Add each embedding to the index
        for (const embedding of allEmbeddings) {
          await semanticSearchService.addPointToIndex(embedding);
          this.deltaState.knownEmbeddingIds.add(embedding.noteId);
        }
      }
      
      console.log(`HnswDeltaOrchestrator: Cold boot complete with ${this.deltaState.knownEmbeddingIds.size} embeddings`);
    } catch (error) {
      console.error('HnswDeltaOrchestrator: Cold boot failed:', error);
      throw error;
    }
  }

  private async syncKnownEmbeddings(): Promise<void> {
    try {
      const allEmbeddings = this.store.query(tables.embeddings.select());
      this.deltaState.knownEmbeddingIds.clear();
      
      if (Array.isArray(allEmbeddings)) {
        allEmbeddings.forEach(embedding => {
          this.deltaState.knownEmbeddingIds.add(embedding.noteId);
        });
      }
      
      console.log(`HnswDeltaOrchestrator: Synced ${this.deltaState.knownEmbeddingIds.size} known embeddings`);
    } catch (error) {
      console.error('HnswDeltaOrchestrator: Failed to sync known embeddings:', error);
    }
  }

  private startDeltaSync(): void {
    console.log('HnswDeltaOrchestrator: Starting delta sync subscriptions...');
    
    // Subscribe to notes requiring embedding updates
    const unsubscribeStale = this.store.subscribe(notesRequiringEmbedding$, {
      onUpdate: () => {
        this.scheduleDeltaProcessing();
      }
    });

    // Subscribe to orphaned embeddings for cleanup
    const unsubscribeOrphaned = this.store.subscribe(orphanedEmbeddings$, {
      onUpdate: () => {
        this.scheduleDeltaProcessing();
      }
    });

    this.unsubscribeCallbacks.push(unsubscribeStale, unsubscribeOrphaned);
  }

  private scheduleDeltaProcessing(): void {
    if (this.isProcessing) {
      this.pendingDeltas = true;
      return;
    }

    // Debounced processing
    setTimeout(() => {
      this.processDelta();
    }, this.DEBOUNCE_MS);
  }

  private async processDelta(): Promise<void> {
    if (this.isProcessing) return;
    
    this.isProcessing = true;
    this.pendingDeltas = false;

    try {
      console.log('HnswDeltaOrchestrator: Processing delta updates...');
      
      // Get current state
      const currentEmbeddings = this.store.query(tables.embeddings.select());
      const orphanedEmbeddings = this.store.query(orphanedEmbeddings$);
      
      let changesCount = 0;
      
      // Process new/updated embeddings
      if (Array.isArray(currentEmbeddings)) {
        for (const embedding of currentEmbeddings) {
          if (!this.deltaState.knownEmbeddingIds.has(embedding.noteId)) {
            await semanticSearchService.addPointToIndex(embedding);
            this.deltaState.knownEmbeddingIds.add(embedding.noteId);
            changesCount++;
          }
        }
      }
      
      // Process orphaned embeddings (removals)
      if (Array.isArray(orphanedEmbeddings)) {
        for (const orphaned of orphanedEmbeddings) {
          if (this.deltaState.knownEmbeddingIds.has(orphaned.noteId)) {
            semanticSearchService.removePointFromIndex(orphaned.noteId);
            this.deltaState.knownEmbeddingIds.delete(orphaned.noteId);
            changesCount++;
          }
        }
      }
      
      // Update counters
      this.deltaState.snapshotCounter += changesCount;
      this.deltaState.lastProcessedTime = Date.now();
      
      if (changesCount > 0) {
        console.log(`HnswDeltaOrchestrator: Processed ${changesCount} delta changes`);
      }
      
      // Trigger snapshot if threshold reached
      if (this.deltaState.snapshotCounter >= this.CHANGES_THRESHOLD) {
        await this.triggerSnapshot('changes_threshold');
      }
      
    } catch (error) {
      console.error('HnswDeltaOrchestrator: Delta processing failed:', error);
    } finally {
      this.isProcessing = false;
      
      // Process any pending deltas
      if (this.pendingDeltas) {
        this.scheduleDeltaProcessing();
      }
    }
  }

  private startPeriodicSnapshots(): void {
    if (this.snapshotInterval) {
      clearInterval(this.snapshotInterval);
    }
    
    this.snapshotInterval = setInterval(() => {
      this.triggerSnapshot('periodic');
    }, this.SNAPSHOT_INTERVAL_MS);
    
    console.log('HnswDeltaOrchestrator: Started periodic snapshots');
  }

  public async triggerSnapshot(reason: 'periodic' | 'changes_threshold' | 'manual'): Promise<void> {
    try {
      console.log(`HnswDeltaOrchestrator: Triggering snapshot (reason: ${reason})...`);
      
      const hnswIndex = semanticSearchService.getHnswIndex();
      if (!hnswIndex || hnswIndex.nodes.size === 0) {
        console.log('HnswDeltaOrchestrator: No index to snapshot');
        return;
      }
      
      // Implement latest + backup strategy
      try {
        // First, move current latest to backup (if it exists)
        await hnswPersistence.renameFile('latest', 'backup');
      } catch (error) {
        // Latest might not exist yet, that's okay
        console.log('HnswDeltaOrchestrator: No existing latest to backup');
      }
      
      try {
        // Save new latest
        await hnswPersistence.persistGraph(hnswIndex, 'latest');
        console.log('HnswDeltaOrchestrator: Snapshot saved as latest');
        
        // Reset change counter
        this.deltaState.snapshotCounter = 0;
        
        // Cleanup old snapshots (keep only latest + backup)
        await hnswPersistence.gcOldSnapshots(0); // Keep only latest and backup
        
      } catch (error) {
        console.error('HnswDeltaOrchestrator: Failed to save latest snapshot:', error);
        
        // Attempt to restore backup
        try {
          await hnswPersistence.renameFile('backup', 'latest');
          console.log('HnswDeltaOrchestrator: Restored backup after snapshot failure');
        } catch (restoreError) {
          console.error('HnswDeltaOrchestrator: Failed to restore backup:', restoreError);
        }
        
        throw error;
      }
      
    } catch (error) {
      console.error('HnswDeltaOrchestrator: Snapshot failed:', error);
    }
  }

  public async forceFullRebuild(): Promise<void> {
    console.log('HnswDeltaOrchestrator: Starting forced full rebuild...');
    
    try {
      // Clear current state
      this.deltaState.knownEmbeddingIds.clear();
      this.deltaState.snapshotCounter = 0;
      
      // Perform cold boot
      await this.performColdBoot();
      
      // Trigger snapshot
      await this.triggerSnapshot('manual');
      
      console.log('HnswDeltaOrchestrator: Forced full rebuild complete');
    } catch (error) {
      console.error('HnswDeltaOrchestrator: Forced full rebuild failed:', error);
      throw error;
    }
  }

  public getStatus() {
    return {
      isInitialized: !!this.store,
      knownEmbeddingCount: this.deltaState.knownEmbeddingIds.size,
      snapshotCounter: this.deltaState.snapshotCounter,
      lastProcessedTime: this.deltaState.lastProcessedTime,
      isProcessing: this.isProcessing
    };
  }

  public shutdown(): void {
    console.log('HnswDeltaOrchestrator: Shutting down...');
    
    // Clear subscriptions
    this.unsubscribeCallbacks.forEach(unsubscribe => unsubscribe());
    this.unsubscribeCallbacks = [];
    
    // Clear intervals
    if (this.snapshotInterval) {
      clearInterval(this.snapshotInterval);
      this.snapshotInterval = null;
    }
    
    // Reset state
    this.store = null;
    this.isProcessing = false;
    this.pendingDeltas = false;
    
    console.log('HnswDeltaOrchestrator: Shutdown complete');
  }
}

export const hnswDeltaOrchestrator = new HnswDeltaOrchestrator();
