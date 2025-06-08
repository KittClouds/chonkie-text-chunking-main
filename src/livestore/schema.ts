import {
  Events,
  makeSchema,
  Schema,
  SessionIdSymbol,
  State
} from '@livestore/livestore';

// Define SQLite tables for our app state
export const tables = {
  clusters: State.SQLite.table({
    name: 'clusters',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      title: State.SQLite.text(),
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text()
    }
  }),

  notes: State.SQLite.table({
    name: 'notes',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      parentId: State.SQLite.text({ nullable: true }),
      clusterId: State.SQLite.text({ nullable: true }),
      title: State.SQLite.text(),
      content: State.SQLite.json({ schema: Schema.Array(Schema.Any) }),
      type: State.SQLite.text({ default: 'note' }),
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text(),
      path: State.SQLite.text({ nullable: true }),
      tags: State.SQLite.json({ schema: Schema.Array(Schema.String), nullable: true }),
      mentions: State.SQLite.json({ schema: Schema.Array(Schema.String), nullable: true })
    }
  }),

  threads: State.SQLite.table({
    name: 'threads',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      title: State.SQLite.text(),
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text()
    }
  }),

  threadMessages: State.SQLite.table({
    name: 'threadMessages',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      threadId: State.SQLite.text(),
      role: State.SQLite.text(),
      content: State.SQLite.text(),
      createdAt: State.SQLite.text(),
      parentId: State.SQLite.text({ nullable: true })
    }
  }),

  entityAttributes: State.SQLite.table({
    name: 'entityAttributes',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      entityKind: State.SQLite.text(),
      entityLabel: State.SQLite.text(),
      attributes: State.SQLite.json({ schema: Schema.Any }),
      metadata: State.SQLite.json({ schema: Schema.Any })
    }
  }),

  blueprints: State.SQLite.table({
    name: 'blueprints',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      entityKind: State.SQLite.text(),
      name: State.SQLite.text(),
      description: State.SQLite.text({ nullable: true }),
      templates: State.SQLite.json({ schema: Schema.Array(Schema.Any) }),
      isDefault: State.SQLite.boolean({ default: false }),
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text()
    }
  }),

  // ENHANCED: Embeddings table with better sync tracking
  embeddings: State.SQLite.table({
    name: 'embeddings',
    columns: {
      noteId: State.SQLite.text({ primaryKey: true }),
      title: State.SQLite.text(),
      content: State.SQLite.text(),
      vecDim: State.SQLite.integer({ default: 384 }),
      vecData: State.SQLite.blob(), // Binary embedding data
      embeddingModel: State.SQLite.text({ default: 'Snowflake/snowflake-arctic-embed-s' }), // Track model version
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text() // Crucial for sync logic
    }
  }),

  // NEW: Graph persistence tables
  graphNodes: State.SQLite.table({
    name: 'graphNodes',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      nodeType: State.SQLite.text(), // 'note', 'entity', 'cluster', 'tag', 'triple'
      entityKind: State.SQLite.text({ nullable: true }), // for entity nodes
      entityLabel: State.SQLite.text({ nullable: true }), // for entity nodes
      position: State.SQLite.json({ schema: Schema.Struct({ x: Schema.Number, y: Schema.Number }) }),
      style: State.SQLite.json({ schema: Schema.Any, nullable: true }), // cytoscape styling
      metadata: State.SQLite.json({ schema: Schema.Any, nullable: true }), // additional node data
      clusterId: State.SQLite.text({ nullable: true }),
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text()
    }
  }),

  graphEdges: State.SQLite.table({
    name: 'graphEdges',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      sourceId: State.SQLite.text(),
      targetId: State.SQLite.text(),
      edgeType: State.SQLite.text(), // 'contains', 'mentions', 'has_tag', etc.
      weight: State.SQLite.real({ default: 1 }),
      style: State.SQLite.json({ schema: Schema.Any, nullable: true }),
      metadata: State.SQLite.json({ schema: Schema.Any, nullable: true }), // relationship metadata
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text()
    }
  }),

  graphLayouts: State.SQLite.table({
    name: 'graphLayouts',
    columns: {
      id: State.SQLite.text({ primaryKey: true }),
      name: State.SQLite.text(),
      layoutType: State.SQLite.text(), // 'dagre', 'circle', 'grid', etc.
      viewport: State.SQLite.json({ 
        schema: Schema.Struct({ 
          zoom: Schema.Number, 
          pan: Schema.Struct({ x: Schema.Number, y: Schema.Number }) 
        }) 
      }),
      nodePositions: State.SQLite.json({ schema: Schema.Any }), // serialized position data
      isDefault: State.SQLite.boolean({ default: false }),
      clusterId: State.SQLite.text({ nullable: true }),
      createdAt: State.SQLite.text(),
      updatedAt: State.SQLite.text()
    }
  }),

  // Client-only UI state (like atoms, doesn't sync)
  uiState: State.SQLite.clientDocument({
    name: 'uiState',
    schema: Schema.Struct({
      activeNoteId: Schema.NullOr(Schema.String),
      activeClusterId: Schema.String,
      activeThreadId: Schema.NullOr(Schema.String),
      graphInitialized: Schema.Boolean,
      graphLayout: Schema.String
    }),
    default: { 
      id: SessionIdSymbol, 
      value: { 
        activeNoteId: null,
        activeClusterId: 'cluster-default',
        activeThreadId: null,
        graphInitialized: false,
        graphLayout: 'dagre'
      } 
    }
  })
};

// Define events for state changes
export const events = {
  // Cluster events
  clusterCreated: Events.synced({
    name: 'v1.ClusterCreated',
    schema: Schema.Struct({
      id: Schema.String,
      title: Schema.String,
      createdAt: Schema.String,
      updatedAt: Schema.String
    })
  }),

  clusterUpdated: Events.synced({
    name: 'v1.ClusterUpdated',
    schema: Schema.Struct({
      id: Schema.String,
      updates: Schema.Any,
      updatedAt: Schema.String
    })
  }),

  clusterDeleted: Events.synced({
    name: 'v1.ClusterDeleted',
    schema: Schema.Struct({
      id: Schema.String
    })
  }),

  // Note events
  noteCreated: Events.synced({
    name: 'v1.NoteCreated',
    schema: Schema.Struct({
      id: Schema.String,
      parentId: Schema.NullOr(Schema.String),
      clusterId: Schema.NullOr(Schema.String),
      title: Schema.String,
      content: Schema.Array(Schema.Any),
      type: Schema.String,
      createdAt: Schema.String,
      updatedAt: Schema.String,
      path: Schema.NullOr(Schema.String),
      tags: Schema.NullOr(Schema.Array(Schema.String)),
      mentions: Schema.NullOr(Schema.Array(Schema.String))
    })
  }),

  noteUpdated: Events.synced({
    name: 'v1.NoteUpdated',
    schema: Schema.Struct({
      id: Schema.String,
      updates: Schema.Any,
      updatedAt: Schema.String
    })
  }),

  noteDeleted: Events.synced({
    name: 'v1.NoteDeleted',
    schema: Schema.Struct({
      id: Schema.String
    })
  }),

  // ENHANCED: Enhanced embedding events with model tracking
  noteEmbedded: Events.synced({
    name: 'v1.NoteEmbedded',
    schema: Schema.Struct({
      noteId: Schema.String,
      title: Schema.String,
      content: Schema.String,
      vecData: Schema.Uint8ArrayFromSelf,
      vecDim: Schema.Number,
      embeddingModel: Schema.String,
      createdAt: Schema.String,
      updatedAt: Schema.String
    })
  }),

  embeddingRemoved: Events.synced({
    name: 'v1.EmbeddingRemoved',
    schema: Schema.Struct({
      noteId: Schema.String
    })
  }),

  // NEW: Enhanced embedding management events
  embeddingIndexCleared: Events.synced({
    name: 'v1.EmbeddingIndexCleared',
    schema: Schema.Struct({
      clearedAt: Schema.String,
      reason: Schema.String
    })
  }),

  embeddingIndexRebuilt: Events.synced({
    name: 'v1.EmbeddingIndexRebuilt',
    schema: Schema.Struct({
      indexId: Schema.String,
      nodeCount: Schema.Number,
      rebuiltAt: Schema.String
    })
  }),

  // ENHANCED: HNSW Graph persistence events with more tracking
  hnswGraphSnapshotCreated: Events.synced({
    name: 'v1.HnswGraphSnapshotCreated',
    schema: Schema.Struct({
      fileName: Schema.String,
      checksum: Schema.String,
      size: Schema.Number,
      nodeCount: Schema.Number,
      embeddingModel: Schema.String,
      createdAt: Schema.String
    })
  }),

  // Thread events
  threadCreated: Events.synced({
    name: 'v1.ThreadCreated',
    schema: Schema.Struct({
      id: Schema.String,
      title: Schema.String,
      createdAt: Schema.String,
      updatedAt: Schema.String
    })
  }),

  threadMessageCreated: Events.synced({
    name: 'v1.ThreadMessageCreated',
    schema: Schema.Struct({
      id: Schema.String,
      threadId: Schema.String,
      role: Schema.String,
      content: Schema.String,
      createdAt: Schema.String,
      parentId: Schema.NullOr(Schema.String)
    })
  }),

  threadMessageUpdated: Events.synced({
    name: 'v1.ThreadMessageUpdated',
    schema: Schema.Struct({
      id: Schema.String,
      updates: Schema.Any
    })
  }),

  threadMessageDeleted: Events.synced({
    name: 'v1.ThreadMessageDeleted',
    schema: Schema.Struct({
      id: Schema.String
    })
  }),

  // Entity events
  entityAttributesUpdated: Events.synced({
    name: 'v1.EntityAttributesUpdated',
    schema: Schema.Struct({
      id: Schema.String,
      entityKind: Schema.String,
      entityLabel: Schema.String,
      attributes: Schema.Any,
      metadata: Schema.Any
    })
  }),

  // Blueprint events
  blueprintCreated: Events.synced({
    name: 'v1.BlueprintCreated',
    schema: Schema.Struct({
      id: Schema.String,
      entityKind: Schema.String,
      name: Schema.String,
      description: Schema.NullOr(Schema.String),
      templates: Schema.Array(Schema.Any),
      isDefault: Schema.Boolean,
      createdAt: Schema.String,
      updatedAt: Schema.String
    })
  }),

  blueprintUpdated: Events.synced({
    name: 'v1.BlueprintUpdated',
    schema: Schema.Struct({
      id: Schema.String,
      updates: Schema.Any,
      updatedAt: Schema.String
    })
  }),

  // NEW: Graph persistence events
  graphNodeCreated: Events.synced({
    name: 'v1.GraphNodeCreated',
    schema: Schema.Struct({
      id: Schema.String,
      nodeType: Schema.String,
      entityKind: Schema.NullOr(Schema.String),
      entityLabel: Schema.NullOr(Schema.String),
      position: Schema.Struct({ x: Schema.Number, y: Schema.Number }),
      style: Schema.NullOr(Schema.Any),
      metadata: Schema.NullOr(Schema.Any),
      clusterId: Schema.NullOr(Schema.String),
      createdAt: Schema.String,
      updatedAt: Schema.String
    })
  }),

  graphNodeUpdated: Events.synced({
    name: 'v1.GraphNodeUpdated',
    schema: Schema.Struct({
      id: Schema.String,
      updates: Schema.Any,
      updatedAt: Schema.String
    })
  }),

  graphNodeDeleted: Events.synced({
    name: 'v1.GraphNodeDeleted',
    schema: Schema.Struct({
      id: Schema.String
    })
  }),

  graphNodeMoved: Events.synced({
    name: 'v1.GraphNodeMoved',
    schema: Schema.Struct({
      id: Schema.String,
      position: Schema.Struct({ x: Schema.Number, y: Schema.Number }),
      updatedAt: Schema.String
    })
  }),

  graphNodeStyled: Events.synced({
    name: 'v1.GraphNodeStyled',
    schema: Schema.Struct({
      id: Schema.String,
      style: Schema.Any,
      updatedAt: Schema.String
    })
  }),

  graphEdgeCreated: Events.synced({
    name: 'v1.GraphEdgeCreated',
    schema: Schema.Struct({
      id: Schema.String,
      sourceId: Schema.String,
      targetId: Schema.String,
      edgeType: Schema.String,
      weight: Schema.Number,
      style: Schema.NullOr(Schema.Any),
      metadata: Schema.NullOr(Schema.Any),
      createdAt: Schema.String,
      updatedAt: Schema.String
    })
  }),

  graphEdgeUpdated: Events.synced({
    name: 'v1.GraphEdgeUpdated',
    schema: Schema.Struct({
      id: Schema.String,
      updates: Schema.Any,
      updatedAt: Schema.String
    })
  }),

  graphEdgeDeleted: Events.synced({
    name: 'v1.GraphEdgeDeleted',
    schema: Schema.Struct({
      id: Schema.String
    })
  }),

  graphLayoutSaved: Events.synced({
    name: 'v1.GraphLayoutSaved',
    schema: Schema.Struct({
      id: Schema.String,
      name: Schema.String,
      layoutType: Schema.String,
      viewport: Schema.Struct({ 
        zoom: Schema.Number, 
        pan: Schema.Struct({ x: Schema.Number, y: Schema.Number }) 
      }),
      nodePositions: Schema.Any,
      isDefault: Schema.Boolean,
      clusterId: Schema.NullOr(Schema.String),
      createdAt: Schema.String,
      updatedAt: Schema.String
    })
  }),

  graphLayoutLoaded: Events.synced({
    name: 'v1.GraphLayoutLoaded',
    schema: Schema.Struct({
      id: Schema.String,
      loadedAt: Schema.String
    })
  }),

  graphLayoutDeleted: Events.synced({
    name: 'v1.GraphLayoutDeleted',
    schema: Schema.Struct({
      id: Schema.String
    })
  }),

  graphViewportChanged: Events.synced({
    name: 'v1.GraphViewportChanged',
    schema: Schema.Struct({
      layoutId: Schema.String,
      viewport: Schema.Struct({ 
        zoom: Schema.Number, 
        pan: Schema.Struct({ x: Schema.Number, y: Schema.Number }) 
      }),
      updatedAt: Schema.String
    })
  }),

  // UI state events (client-only)
  uiStateSet: tables.uiState.set
};

// Enhanced materializers with new events - FIXED to remove log-only materializers
const materializers = State.SQLite.materializers(events, {
  // Cluster materializers
  'v1.ClusterCreated': ({ id, title, createdAt, updatedAt }) =>
    tables.clusters.insert({ id, title, createdAt, updatedAt }),

  'v1.ClusterUpdated': ({ id, updates, updatedAt }) =>
    tables.clusters.update({ ...updates, updatedAt }).where({ id }),

  'v1.ClusterDeleted': ({ id }) =>
    tables.clusters.delete().where({ id }),

  // Note materializers
  'v1.NoteCreated': ({ id, parentId, clusterId, title, content, type, createdAt, updatedAt, path, tags, mentions }) =>
    tables.notes.insert({ id, parentId, clusterId, title, content, type, createdAt, updatedAt, path, tags, mentions }),

  'v1.NoteUpdated': ({ id, updates, updatedAt }) =>
    tables.notes.update({ ...updates, updatedAt }).where({ id }),

  'v1.NoteDeleted': ({ id }) =>
    tables.notes.delete().where({ id }),

  // ENHANCED: Embedding materializers with model tracking
  'v1.NoteEmbedded': ({ noteId, title, content, vecData, vecDim, embeddingModel, createdAt, updatedAt }) => [
    tables.embeddings.delete().where({ noteId }),
    tables.embeddings.insert({ noteId, title, content, vecData, vecDim, embeddingModel, createdAt, updatedAt })
  ],

  'v1.EmbeddingRemoved': ({ noteId }) =>
    tables.embeddings.delete().where({ noteId }),

  // Thread materializers
  'v1.ThreadCreated': ({ id, title, createdAt, updatedAt }) =>
    tables.threads.insert({ id, title, createdAt, updatedAt }),

  'v1.ThreadMessageCreated': ({ id, threadId, role, content, createdAt, parentId }) =>
    tables.threadMessages.insert({ id, threadId, role, content, createdAt, parentId }),

  'v1.ThreadMessageUpdated': ({ id, updates }) =>
    tables.threadMessages.update(updates).where({ id }),

  'v1.ThreadMessageDeleted': ({ id }) =>
    tables.threadMessages.delete().where({ id }),

  // Entity and blueprint materializers
  'v1.EntityAttributesUpdated': ({ id, entityKind, entityLabel, attributes, metadata }) => [
    tables.entityAttributes.insert({ id, entityKind, entityLabel, attributes, metadata }),
    tables.entityAttributes.update({ attributes, metadata }).where({ id })
  ],

  'v1.BlueprintCreated': ({ id, entityKind, name, description, templates, isDefault, createdAt, updatedAt }) =>
    tables.blueprints.insert({ id, entityKind, name, description, templates, isDefault, createdAt, updatedAt }),

  'v1.BlueprintUpdated': ({ id, updates, updatedAt }) =>
    tables.blueprints.update({ ...updates, updatedAt }).where({ id }),

  // NEW: Graph persistence materializers
  'v1.GraphNodeCreated': ({ id, nodeType, entityKind, entityLabel, position, style, metadata, clusterId, createdAt, updatedAt }) =>
    tables.graphNodes.insert({ id, nodeType, entityKind, entityLabel, position, style, metadata, clusterId, createdAt, updatedAt }),

  'v1.GraphNodeUpdated': ({ id, updates, updatedAt }) =>
    tables.graphNodes.update({ ...updates, updatedAt }).where({ id }),

  'v1.GraphNodeDeleted': ({ id }) =>
    tables.graphNodes.delete().where({ id }),

  'v1.GraphNodeMoved': ({ id, position, updatedAt }) =>
    tables.graphNodes.update({ position, updatedAt }).where({ id }),

  'v1.GraphNodeStyled': ({ id, style, updatedAt }) =>
    tables.graphNodes.update({ style, updatedAt }).where({ id }),

  'v1.GraphEdgeCreated': ({ id, sourceId, targetId, edgeType, weight, style, metadata, createdAt, updatedAt }) =>
    tables.graphEdges.insert({ id, sourceId, targetId, edgeType, weight, style, metadata, createdAt, updatedAt }),

  'v1.GraphEdgeUpdated': ({ id, updates, updatedAt }) =>
    tables.graphEdges.update({ ...updates, updatedAt }).where({ id }),

  'v1.GraphEdgeDeleted': ({ id }) =>
    tables.graphEdges.delete().where({ id }),

  'v1.GraphLayoutSaved': ({ id, name, layoutType, viewport, nodePositions, isDefault, clusterId, createdAt, updatedAt }) =>
    tables.graphLayouts.insert({ id, name, layoutType, viewport, nodePositions, isDefault, clusterId, createdAt, updatedAt }),

  'v1.GraphLayoutDeleted': ({ id }) =>
    tables.graphLayouts.delete().where({ id }),

  'v1.GraphViewportChanged': ({ layoutId, viewport, updatedAt }) =>
    tables.graphLayouts.update({ viewport, updatedAt }).where({ id: layoutId })
});

// Create the state with tables and materializers
const state = State.SQLite.makeState({ tables, materializers });

// Export the complete schema
export const schema = makeSchema({ events, state });
