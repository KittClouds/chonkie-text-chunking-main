
import { useStore } from '@livestore/react';
import { useEffect, useState, useCallback } from 'react';
import { semanticSearchService } from '@/lib/embedding/SemanticSearchService';
import { embeddingCleanupService } from '@/lib/embedding/CleanupService';

interface EmbeddingHealth {
  isHealthy: boolean;
  hasStaleData: boolean;
  staleCount: number;
  noteCount: number;
  embeddingCount: number;
  snapshotCount: number;
  totalSnapshotSize: number;
  lastUpdate: Date;
}

interface BuildProgress {
  phase: 'cleanup' | 'sync' | 'build' | 'persist' | 'complete';
  current: number;
  total: number;
  message: string;
  isActive: boolean;
}

export function useEmbeddingHealth() {
  const { store } = useStore();
  const [health, setHealth] = useState<EmbeddingHealth>({
    isHealthy: true,
    hasStaleData: false,
    staleCount: 0,
    noteCount: 0,
    embeddingCount: 0,
    snapshotCount: 0,
    totalSnapshotSize: 0,
    lastUpdate: new Date()
  });

  const [buildProgress, setBuildProgress] = useState<BuildProgress>({
    phase: 'complete',
    current: 0,
    total: 0,
    message: '',
    isActive: false
  });

  // Set up progress callback
  useEffect(() => {
    semanticSearchService.setBuildProgressCallback((progress) => {
      setBuildProgress({
        ...progress,
        isActive: progress.phase !== 'complete'
      });
    });
  }, []);

  const checkHealth = useCallback(async () => {
    try {
      // Inject store reference if needed
      semanticSearchService.setStore(store);
      embeddingCleanupService.setStore(store);

      // Get stale data info
      const staleInfo = await semanticSearchService.detectStaleData();
      
      // Get snapshot info
      const snapshotInfo = await semanticSearchService.getSnapshotInfo();

      const newHealth: EmbeddingHealth = {
        isHealthy: !staleInfo.hasStaleData,
        hasStaleData: staleInfo.hasStaleData,
        staleCount: staleInfo.staleCount,
        noteCount: staleInfo.noteCount,
        embeddingCount: staleInfo.embeddingCount,
        snapshotCount: snapshotInfo.count,
        totalSnapshotSize: snapshotInfo.totalSize,
        lastUpdate: new Date()
      };

      setHealth(newHealth);
      return newHealth;
    } catch (error) {
      console.error('Failed to check embedding health:', error);
      return health;
    }
  }, [store, health]);

  const forceCleanup = useCallback(async () => {
    try {
      const result = await semanticSearchService.forceCleanupStaleEmbeddings();
      await checkHealth(); // Refresh health after cleanup
      return result;
    } catch (error) {
      console.error('Force cleanup failed:', error);
      throw error;
    }
  }, [checkHealth]);

  // Auto-check health on mount and store changes
  useEffect(() => {
    checkHealth();
  }, [store]);

  return {
    health,
    buildProgress,
    checkHealth,
    forceCleanup,
    isBuilding: buildProgress.isActive
  };
}
