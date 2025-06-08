
import { queryDb, computed } from '@livestore/livestore';
import { tables } from '../schema';

// Core reactive query for detecting notes that need new or updated embeddings
export const notesRequiringEmbedding$ = computed((get) => {
  const allNotes = get(queryDb(tables.notes, { label: 'allNotesForSync$' }));
  const allEmbeddings = get(queryDb(tables.embeddings, { label: 'allEmbeddingsForSync$' }));

  if (!Array.isArray(allNotes) || !Array.isArray(allEmbeddings)) {
    return [];
  }

  // Create a quick lookup map for embeddings
  const embeddingsMap = new Map(allEmbeddings.map(e => [e.noteId, e]));

  const staleNotes = allNotes.filter(note => {
    const embedding = embeddingsMap.get(note.id);
    if (!embedding) {
      return true; // Note exists, but embedding does not. It's stale.
    }
    // Note was updated after its embedding was created. It's stale.
    if (new Date(note.updatedAt) > new Date(embedding.updatedAt)) {
      return true;
    }
    return false;
  });

  console.log(`Sync Query: Found ${staleNotes.length} notes requiring embedding updates`);
  return staleNotes;
}, { label: 'notesRequiringEmbedding$' });

// Query for embedding sync statistics
export const embeddingSyncStats$ = computed((get) => {
  const allNotes = get(queryDb(tables.notes, { label: 'allNotesForStats$' }));
  const allEmbeddings = get(queryDb(tables.embeddings, { label: 'allEmbeddingsForStats$' }));
  const staleNotes = get(notesRequiringEmbedding$);

  if (!Array.isArray(allNotes) || !Array.isArray(allEmbeddings)) {
    return {
      totalNotes: 0,
      totalEmbeddings: 0,
      staleCount: 0,
      syncPercentage: 0,
      lastSyncCheck: new Date().toISOString()
    };
  }

  const syncPercentage = allNotes.length > 0 
    ? Math.round(((allNotes.length - staleNotes.length) / allNotes.length) * 100)
    : 100;

  return {
    totalNotes: allNotes.length,
    totalEmbeddings: allEmbeddings.length,
    staleCount: staleNotes.length,
    syncPercentage,
    lastSyncCheck: new Date().toISOString()
  };
}, { label: 'embeddingSyncStats$' });

// Query for notes that have embeddings but were recently updated
export const recentlyUpdatedNotes$ = computed((get) => {
  const cutoffTime = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(); // 24 hours ago
  
  const recentNotes = get(queryDb(
    tables.notes.where(note => note.updatedAt > cutoffTime).orderBy('updatedAt', 'desc'),
    { label: 'recentNotesForSync$' }
  ));

  const staleNotes = get(notesRequiringEmbedding$);
  const staleNoteIds = new Set(staleNotes.map(n => n.id));

  if (!Array.isArray(recentNotes)) {
    return [];
  }

  return recentNotes.filter(note => staleNoteIds.has(note.id));
}, { label: 'recentlyUpdatedNotes$' });

// Query for orphaned embeddings (embeddings without corresponding notes)
export const orphanedEmbeddings$ = computed((get) => {
  const allNotes = get(queryDb(tables.notes, { label: 'allNotesForOrphanCheck$' }));
  const allEmbeddings = get(queryDb(tables.embeddings, { label: 'allEmbeddingsForOrphanCheck$' }));

  if (!Array.isArray(allNotes) || !Array.isArray(allEmbeddings)) {
    return [];
  }

  const noteIds = new Set(allNotes.map(n => n.id));
  const orphanedEmbeddings = allEmbeddings.filter(embedding => !noteIds.has(embedding.noteId));

  if (orphanedEmbeddings.length > 0) {
    console.log(`Sync Query: Found ${orphanedEmbeddings.length} orphaned embeddings`);
  }

  return orphanedEmbeddings;
}, { label: 'orphanedEmbeddings$' });

// Query for embedding model version consistency
export const embeddingModelConsistency$ = computed((get) => {
  const allEmbeddings = get(queryDb(tables.embeddings, { label: 'allEmbeddingsForModelCheck$' }));

  if (!Array.isArray(allEmbeddings)) {
    return {
      isConsistent: true,
      currentModel: 'Snowflake/snowflake-arctic-embed-s',
      modelCounts: {},
      outdatedEmbeddings: []
    };
  }

  const modelCounts: Record<string, number> = {};
  const currentModel = 'Snowflake/snowflake-arctic-embed-s';
  
  allEmbeddings.forEach(embedding => {
    const model = embedding.embeddingModel || 'unknown';
    modelCounts[model] = (modelCounts[model] || 0) + 1;
  });

  const outdatedEmbeddings = allEmbeddings.filter(
    embedding => embedding.embeddingModel !== currentModel
  );

  const isConsistent = Object.keys(modelCounts).length <= 1 && 
                      (Object.keys(modelCounts)[0] === currentModel || Object.keys(modelCounts).length === 0);

  return {
    isConsistent,
    currentModel,
    modelCounts,
    outdatedEmbeddings
  };
}, { label: 'embeddingModelConsistency$' });
