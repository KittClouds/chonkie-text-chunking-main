import { useStore } from '@livestore/react';
import { 
  activeNoteId$, 
  activeClusterId$, 
  activeThreadId$,
  activeNote$,
  notes$,
  clusters$,
  activeClusterNotes$,
  standardNotes$,
  activeNoteConnections$,
  entityAttributes$,
  blueprints$,
  backlinksToActiveNote$,
  backlinkStats$,
  // New entity queries
  globalEntities$,
  clusterEntitiesMap$,
  folderEntitiesMap$,
  activeNoteEntities$,
  entitiesByType$,
  entityReferenceCounts$,
  recentEntities$,
  orphanedEntities$,
  allEntitiesArray$
} from '../livestore/queries';
// Import new cross-note relation queries
import { 
  entityCoOccurrences$, 
  globalTriples$, 
  createEntityGlobalRelationsQuery,
  topCoOccurrences$,
  topGlobalTriples$
} from '../livestore/queries/derived';
import { events } from '../livestore/schema';

// Custom hooks that wrap LiveStore usage with proper typing
export function useActiveNoteId() {
  const { store } = useStore();
  const activeNoteId = store.useQuery(activeNoteId$);
  
  const setActiveNoteId = (id: string | null) => {
    store.commit(events.uiStateSet({ 
      activeNoteId: id,
      activeClusterId: store.query(activeClusterId$),
      activeThreadId: store.query(activeThreadId$),
      graphInitialized: false,
      graphLayout: 'dagre'
    }));
  };

  return [activeNoteId, setActiveNoteId] as const;
}

export function useActiveClusterId() {
  const { store } = useStore();
  const activeClusterId = store.useQuery(activeClusterId$);
  
  const setActiveClusterId = (id: string) => {
    store.commit(events.uiStateSet({ 
      activeNoteId: store.query(activeNoteId$),
      activeClusterId: id,
      activeThreadId: store.query(activeThreadId$),
      graphInitialized: false,
      graphLayout: 'dagre'
    }));
  };

  return [activeClusterId, setActiveClusterId] as const;
}

export function useActiveNote() {
  const { store } = useStore();
  return store.useQuery(activeNote$);
}

export function useNotes() {
  const { store } = useStore();
  const notes = store.useQuery(notes$);
  return Array.isArray(notes) ? notes : [];
}

export function useClusters() {
  const { store } = useStore();
  const clusters = store.useQuery(clusters$);
  return Array.isArray(clusters) ? clusters : [];
}

export function useActiveClusterNotes() {
  const { store } = useStore();
  const notes = store.useQuery(activeClusterNotes$);
  return Array.isArray(notes) ? notes : [];
}

export function useStandardNotes() {
  const { store } = useStore();
  const notes = store.useQuery(standardNotes$);
  return Array.isArray(notes) ? notes : [];
}

export function useActiveNoteConnections() {
  const { store } = useStore();
  return store.useQuery(activeNoteConnections$);
}

export function useEntityAttributes() {
  const { store } = useStore();
  const attrs = store.useQuery(entityAttributes$);
  return Array.isArray(attrs) ? attrs : [];
}

export function useBlueprintsArray() {
  const { store } = useStore();
  const blueprints = store.useQuery(blueprints$);
  return Array.isArray(blueprints) ? blueprints : [];
}

// NEW: Backlink hooks
export function useActiveNoteBacklinks() {
  const { store } = useStore();
  return store.useQuery(backlinksToActiveNote$);
}

export function useBacklinkStats() {
  const { store } = useStore();
  return store.useQuery(backlinkStats$);
}

// New entity-specific hooks
export function useGlobalEntities() {
  const { store } = useStore();
  return store.useQuery(globalEntities$);
}

export function useAllEntitiesArray() {
  const { store } = useStore();
  return store.useQuery(allEntitiesArray$);
}

export function useClusterEntitiesMap() {
  const { store } = useStore();
  return store.useQuery(clusterEntitiesMap$);
}

export function useFolderEntitiesMap() {
  const { store } = useStore();
  return store.useQuery(folderEntitiesMap$);
}

export function useActiveNoteEntities() {
  const { store } = useStore();
  return store.useQuery(activeNoteEntities$);
}

export function useEntitiesByType() {
  const { store } = useStore();
  return store.useQuery(entitiesByType$);
}

export function useEntityReferenceCounts() {
  const { store } = useStore();
  return store.useQuery(entityReferenceCounts$);
}

export function useRecentEntities() {
  const { store } = useStore();
  return store.useQuery(recentEntities$);
}

export function useOrphanedEntities() {
  const { store } = useStore();
  return store.useQuery(orphanedEntities$);
}

// NEW: Cross-note relation hooks
export function useEntityCoOccurrences() {
  const { store } = useStore();
  return store.useQuery(entityCoOccurrences$);
}

export function useGlobalTriples() {
  const { store } = useStore();
  return store.useQuery(globalTriples$);
}

export function useEntityGlobalRelations(entityId: string) {
  const { store } = useStore();
  const query = createEntityGlobalRelationsQuery(entityId);
  return store.useQuery(query);
}

export function useTopCoOccurrences() {
  const { store } = useStore();
  return store.useQuery(topCoOccurrences$);
}

export function useTopGlobalTriples() {
  const { store } = useStore();
  return store.useQuery(topGlobalTriples$);
}

// Enhanced helper to commit note updates with embedding integration
export function useNoteActions() {
  const { store } = useStore();
  
  const updateNote = (id: string, updates: any) => {
    store.commit(events.noteUpdated({
      id,
      updates,
      updatedAt: new Date().toISOString()
    }));
    
    // Background embedding update - semantic search service will handle LiveStore events
    queueMicrotask(async () => {
      try {
        const { semanticSearchService } = await import('@/lib/embedding/SemanticSearchService');
        // Get updated note to regenerate embedding
        const updatedNote = store.query(tables.notes.select().where({ id }));
        if (updatedNote && updatedNote.length > 0) {
          const note = updatedNote[0];
          await semanticSearchService.addOrUpdateNote(note.id, note.title, note.content);
        }
      } catch (error) {
        console.error('Failed to update note embedding:', error);
      }
    });
  };

  const createNote = (note: any) => {
    store.commit(events.noteCreated(note));
    
    // Background embedding creation - semantic search service will handle LiveStore events
    queueMicrotask(async () => {
      try {
        const { semanticSearchService } = await import('@/lib/embedding/SemanticSearchService');
        await semanticSearchService.addOrUpdateNote(note.id, note.title, note.content);
      } catch (error) {
        console.error('Failed to create note embedding:', error);
      }
    });
  };

  const deleteNote = (id: string) => {
    store.commit(events.noteDeleted({ id }));
    
    // Remove from semantic search - service will handle LiveStore events
    queueMicrotask(async () => {
      try {
        const { semanticSearchService } = await import('@/lib/embedding/SemanticSearchService');
        semanticSearchService.removeNote(id);
      } catch (error) {
        console.error('Failed to remove note embedding:', error);
      }
    });
  };

  const createCluster = (cluster: any) => {
    store.commit(events.clusterCreated(cluster));
  };

  const updateCluster = (id: string, updates: any) => {
    store.commit(events.clusterUpdated({
      id,
      updates,
      updatedAt: new Date().toISOString()
    }));
  };

  const deleteCluster = (id: string) => {
    store.commit(events.clusterDeleted({ id }));
  };

  // Enhanced note sync with progress tracking
  const syncAllNotesEmbeddings = async (progressCallback?: (progress: any) => void) => {
    try {
      const { semanticSearchService } = await import('@/lib/embedding/SemanticSearchService');
      const notes = store.query(tables.notes.select()) || [];
      
      if (progressCallback) {
        semanticSearchService.setBuildProgressCallback(progressCallback);
      }
      
      return await semanticSearchService.syncAllNotes(notes);
    } catch (error) {
      console.error('Failed to sync note embeddings:', error);
      throw error;
    }
  };

  return { 
    updateNote, 
    createNote, 
    deleteNote,
    createCluster,
    updateCluster,
    deleteCluster,
    syncAllNotesEmbeddings
  };
}
