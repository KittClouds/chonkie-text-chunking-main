
import { useStore } from '@livestore/react';
import { queryDb } from '@livestore/livestore';
import { tables } from '../livestore/schema';
import { semanticSearchService } from '@/lib/embedding/SemanticSearchService';
import { useEffect } from 'react';

// Query for all embeddings
const embeddings$ = queryDb(
  () => tables.embeddings.select(),
  { label: 'embeddings' }
);

// Query for embedding count
const embeddingCount$ = queryDb(
  () => tables.embeddings.count(),
  { label: 'embeddingCount' }
);

// Reactive hook for embeddings data
export function useEmbeddings() {
  const { store } = useStore();
  const embeddings = store.useQuery(embeddings$);
  
  // Inject store reference into semantic search service
  useEffect(() => {
    semanticSearchService.setStore(store);
  }, [store]);
  
  return Array.isArray(embeddings) ? embeddings : [];
}

// Hook for embedding count
export function useEmbeddingCount() {
  const { store } = useStore();
  return store.useQuery(embeddingCount$) || 0;
}

// Hook for paginated embeddings - fix the query builder API
export function usePaginatedEmbeddings(limit: number = 50, page: number = 0) {
  const { store } = useStore();
  
  const paginatedEmbeddings$ = queryDb(
    () => tables.embeddings
      .select('noteId', 'title', 'createdAt', 'updatedAt')
      .orderBy('updatedAt', 'desc')
      .limit(limit),
    { 
      label: 'paginatedEmbeddings',
      deps: [limit, page]
    }
  );
  
  const embeddings = store.useQuery(paginatedEmbeddings$);
  return Array.isArray(embeddings) ? embeddings : [];
}
