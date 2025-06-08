
import { queryDb, computed } from '@livestore/livestore';
import { tables } from './schema';
import { parseAllNotes } from '@/lib/utils/parsingUtils';
import { parseNoteConnectionsFromDocument } from '@/lib/utils/documentParser';
import { crossNoteRelationService } from '@/services/CrossNoteRelationService';
import { backlinkService } from '@/services/BacklinkService';

// Re-export entity queries
export * from './queries/entities';

// NEW: Re-export sync tracking queries
export * from './queries/sync';

// Basic entity queries with proper typing
export const clusters$ = queryDb(
  tables.clusters.orderBy('createdAt', 'desc'),
  { label: 'clusters$' }
);

export const notes$ = queryDb(
  tables.notes.orderBy('createdAt', 'desc'),
  { label: 'notes$' }
);

export const threads$ = queryDb(
  tables.threads.orderBy('createdAt', 'desc'),
  { label: 'threads$' }
);

export const threadMessages$ = queryDb(
  tables.threadMessages.orderBy('createdAt', 'asc'),
  { label: 'threadMessages$' }
);

export const blueprints$ = queryDb(
  tables.blueprints.orderBy('createdAt', 'desc'),
  { label: 'blueprints$' }
);

export const entityAttributes$ = queryDb(
  tables.entityAttributes.orderBy('entityKind', 'asc'),
  { label: 'entityAttributes$' }
);

// UI state queries - use the client document pattern correctly
export const uiState$ = computed((get) => {
  // For client documents, we need to query the table and get the first result
  const results = get(queryDb(tables.uiState, { label: 'uiStateRaw$' }));
  return Array.isArray(results) && results.length > 0 ? results[0].value : {
    activeNoteId: null,
    activeClusterId: 'cluster-default',
    activeThreadId: null,
    graphInitialized: false,
    graphLayout: 'dagre'
  };
}, { label: 'uiState$' });

export const activeNoteId$ = computed((get) => {
  const ui = get(uiState$);
  return ui?.activeNoteId || null;
}, { label: 'activeNoteId$' });

export const activeClusterId$ = computed((get) => {
  const ui = get(uiState$);
  return ui?.activeClusterId || 'cluster-default';
}, { label: 'activeClusterId$' });

export const activeThreadId$ = computed((get) => {
  const ui = get(uiState$);
  return ui?.activeThreadId || null;
}, { label: 'activeThreadId$' });

// Filtered queries
export const activeClusterNotes$ = queryDb((get) => {
  const activeClusterId = get(activeClusterId$);
  return tables.notes.where({ clusterId: activeClusterId }).orderBy('createdAt', 'desc');
}, { label: 'activeClusterNotes$' });

export const standardNotes$ = queryDb(
  tables.notes.where({ clusterId: null }).orderBy('createdAt', 'desc'),
  { label: 'standardNotes$' }
);

export const activeNote$ = computed((get) => {
  const activeId = get(activeNoteId$);
  const allNotes = get(notes$);
  if (!activeId || !Array.isArray(allNotes)) return null;
  return allNotes.find((note: any) => note.id === activeId) || null;
}, { label: 'activeNote$' });

export const activeThreadMessages$ = queryDb((get) => {
  const activeThreadId = get(activeThreadId$);
  if (!activeThreadId) return tables.threadMessages.where({ threadId: '__nonexistent__' });
  return tables.threadMessages.where({ threadId: activeThreadId }).orderBy('createdAt', 'asc');
}, { label: 'activeThreadMessages$' });

// Legacy parsing for backward compatibility
export const noteConnections$ = computed((get) => {
  const notes = get(notes$);
  if (!Array.isArray(notes)) {
    return {
      tagsMap: new Map(),
      mentionsMap: new Map(),
      linksMap: new Map(),
      entitiesMap: new Map(),
      triplesMap: new Map(),
      backlinksMap: new Map()
    };
  }
  console.log("LiveStore: Recalculating all parsed connections (legacy)");
  return parseAllNotes(notes);
}, { label: 'noteConnections$' });

export const noteTagsMap$ = computed((get) => {
  return get(noteConnections$).tagsMap;
}, { label: 'noteTagsMap$' });

export const noteMentionsMap$ = computed((get) => {
  return get(noteConnections$).mentionsMap;
}, { label: 'noteMentionsMap$' });

export const noteLinksMap$ = computed((get) => {
  return get(noteConnections$).linksMap;
}, { label: 'noteLinksMap$' });

export const noteEntitiesMap$ = computed((get) => {
  return get(noteConnections$).entitiesMap;
}, { label: 'noteEntitiesMap$' });

export const noteTriplesMap$ = computed((get) => {
  return get(noteConnections$).triplesMap;
}, { label: 'noteTriplesMap$' });

export const noteBacklinksMap$ = computed((get) => {
  return get(noteConnections$).backlinksMap;
}, { label: 'noteBacklinksMap$' });

// New document-based parsing for canonical data
export const activeNoteConnections$ = computed((get) => {
  const activeNote = get(activeNote$);
  if (!activeNote || !activeNote.content) {
    return { tags: [], mentions: [], links: [], entities: [], triples: [], backlinks: [] };
  }

  // Use document-based parsing for canonical data
  console.log("LiveStore: Parsing active note connections from document structure");
  const connections = parseNoteConnectionsFromDocument(activeNote.content);
  
  return {
    tags: connections.tags,
    mentions: connections.mentions,
    links: connections.links,
    entities: connections.entities,
    triples: connections.triples,
    backlinks: connections.backlinks
  };
}, { label: 'activeNoteConnections$' });

// NEW: Backlink queries using BacklinkService
export const backlinksToActiveNote$ = computed((get) => {
  const activeNote = get(activeNote$);
  const notes = get(notes$);
  
  if (!activeNote || !Array.isArray(notes)) {
    return [];
  }
  
  console.log("LiveStore: Computing backlinks to active note:", activeNote.title);
  
  // Update backlink service with current notes
  backlinkService.updateFromNotes(notes);
  
  // Get backlinks to this note title
  const backlinkRefs = backlinkService.getBacklinksTo(activeNote.title);
  
  // Convert to the format expected by UI components
  return backlinkRefs.map(ref => ({
    id: ref.noteId,
    title: ref.noteTitle,
    backlinkTitle: ref.backlinkTitle
  }));
}, { label: 'backlinksToActiveNote$' });

export const backlinkStats$ = computed((get) => {
  const notes = get(notes$);
  
  if (!Array.isArray(notes)) {
    return { totalBacklinkTargets: 0, totalBacklinkReferences: 0, notesWithBacklinks: 0 };
  }
  
  // Update backlink service with current notes
  backlinkService.updateFromNotes(notes);
  
  return backlinkService.getStats();
}, { label: 'backlinkStats$' });

// NEW: Cross-Note Relation Engine queries
// These provide reactive access to the computed cross-note relationships

export const entityCoOccurrences$ = computed((get) => {
  const entitiesMap = get(noteEntitiesMap$);
  const triplesMap = get(noteTriplesMap$);
  
  console.log("LiveStore: Computing entity co-occurrences");
  
  // Update the service with current data
  crossNoteRelationService.updateFromNoteData(entitiesMap, triplesMap);
  
  // Get all co-occurrences and structure for UI consumption
  const allCoOccurrences = crossNoteRelationService.getAllCoOccurrences();
  const coOccurrenceMap = new Map<string, Array<{
    entity: { kind: string; label: string };
    count: number;
    noteIds: string[];
  }>>();
  
  // Group by entity for easy lookup
  allCoOccurrences.forEach(coOcc => {
    const entity1Key = `${coOcc.entity1.kind}::${coOcc.entity1.label}`;
    const entity2Key = `${coOcc.entity2.kind}::${coOcc.entity2.label}`;
    
    // Add to entity1's co-occurrences
    if (!coOccurrenceMap.has(entity1Key)) {
      coOccurrenceMap.set(entity1Key, []);
    }
    coOccurrenceMap.get(entity1Key)!.push({
      entity: coOcc.entity2,
      count: coOcc.count,
      noteIds: coOcc.noteIds
    });
    
    // Add to entity2's co-occurrences
    if (!coOccurrenceMap.has(entity2Key)) {
      coOccurrenceMap.set(entity2Key, []);
    }
    coOccurrenceMap.get(entity2Key)!.push({
      entity: coOcc.entity1,
      count: coOcc.count,
      noteIds: coOcc.noteIds
    });
  });
  
  return coOccurrenceMap;
}, { label: 'entityCoOccurrences$' });

export const globalTriples$ = computed((get) => {
  const entitiesMap = get(noteEntitiesMap$);
  const triplesMap = get(noteTriplesMap$);
  
  console.log("LiveStore: Computing global triples");
  
  // Update the service with current data
  crossNoteRelationService.updateFromNoteData(entitiesMap, triplesMap);
  
  // Return the global triple index
  return crossNoteRelationService.getAllGlobalTriples();
}, { label: 'globalTriples$' });

export const entityGlobalRelations$ = (entityId: string) => computed((get) => {
  const entitiesMap = get(noteEntitiesMap$);
  const triplesMap = get(noteTriplesMap$);
  
  console.log(`LiveStore: Computing global relations for entity ${entityId}`);
  
  // Update the service with current data
  crossNoteRelationService.updateFromNoteData(entitiesMap, triplesMap);
  
  // Get both co-occurrences and global triples for this entity
  const coOccurrences = crossNoteRelationService.getCoOccurrences(entityId);
  const globalTriples = crossNoteRelationService.getGlobalTriples(entityId);
  
  return {
    coOccurrences,
    globalTriples,
    entityId
  };
}, { label: `entityGlobalRelations$_${entityId}` });
