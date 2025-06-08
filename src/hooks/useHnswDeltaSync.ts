
import { useEffect } from 'react';
import { useStore } from '@livestore/react';
import { hnswDeltaOrchestrator } from '../services/HnswDeltaOrchestrator';
import { semanticSearchService } from '../lib/embedding/SemanticSearchService';

/**
 * Hook to initialize and manage the HNSW delta sync system
 * This should be used once in your main App component
 */
export function useHnswDeltaSync() {
  const { store } = useStore();

  useEffect(() => {
    if (!store) return;

    console.log('useHnswDeltaSync: Initializing HNSW delta sync system...');

    // Set store reference for semantic search service
    semanticSearchService.setStore(store);

    // Initialize the delta orchestrator
    const initializeOrchestrator = async () => {
      try {
        await hnswDeltaOrchestrator.initialize(store);
        console.log('useHnswDeltaSync: HNSW delta sync system initialized successfully');
      } catch (error) {
        console.error('useHnswDeltaSync: Failed to initialize HNSW delta sync system:', error);
      }
    };

    initializeOrchestrator();

    // Cleanup on unmount
    return () => {
      console.log('useHnswDeltaSync: Cleaning up HNSW delta sync system...');
      hnswDeltaOrchestrator.shutdown();
    };
  }, [store]);

  // Return orchestrator methods for external use
  return {
    triggerSnapshot: () => hnswDeltaOrchestrator.triggerSnapshot('manual'),
    forceFullRebuild: () => hnswDeltaOrchestrator.forceFullRebuild(),
    getStatus: () => hnswDeltaOrchestrator.getStatus()
  };
}
