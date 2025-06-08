
import { useEffect } from 'react';
import { useStore } from '@livestore/react';
import { hnswSyncOrchestrator } from '../services/HnswSyncOrchestrator';

export function useHnswOrchestrator() {
  const { store } = useStore();

  useEffect(() => {
    if (store) {
      console.log('[useHnswOrchestrator] Initializing HNSW Sync Orchestrator...');
      hnswSyncOrchestrator.initialize(store);

      // Cleanup on unmount
      return () => {
        hnswSyncOrchestrator.shutdown();
      };
    }
  }, [store]);

  return {
    forceSync: () => hnswSyncOrchestrator.forceSync(),
    forceSnapshot: () => hnswSyncOrchestrator.forceSnapshot(),
    getStats: () => hnswSyncOrchestrator.getStats()
  };
}
