
import { events } from './schema';
import { Store } from '@livestore/livestore';

export interface LegacyData {
  notes?: any[];
  clusters?: any[];
  threads?: any[];
  threadMessages?: any[];
  blueprints?: any[];
}

export function migrateLegacyData(store: Store<any>): boolean {
  try {
    // Check if migration has already been completed
    const migrationFlag = localStorage.getItem('livestore-migration-completed');
    const embeddingMigrationFlag = localStorage.getItem('livestore-embedding-migration-completed');
    
    if (migrationFlag === 'true' && embeddingMigrationFlag === 'true') {
      console.log('LiveStore: Migration already completed, skipping');
      return true;
    }

    console.log('LiveStore: Starting migration from localStorage...');

    // First, handle the embedding schema migration
    if (embeddingMigrationFlag !== 'true') {
      migrateEmbeddingSchema(store);
      localStorage.setItem('livestore-embedding-migration-completed', 'true');
    }

    // Only run main migration if not already completed
    if (migrationFlag !== 'true') {
      // Read existing data from localStorage
      const legacyNotes = localStorage.getItem('galaxy-notes');
      const legacyClusters = localStorage.getItem('galaxy-notes-clusters');
      const legacyThreads = localStorage.getItem('galaxy-notes-threads');
      const legacyThreadMessages = localStorage.getItem('galaxy-notes-thread-messages');
      const legacyBlueprints = localStorage.getItem('galaxy-blueprint-storage');

      let migratedCount = 0;

      // Migrate clusters
      if (legacyClusters) {
        const clusters = JSON.parse(legacyClusters);
        if (Array.isArray(clusters)) {
          clusters.forEach(cluster => {
            store.commit(events.clusterCreated({
              id: cluster.id,
              title: cluster.title,
              createdAt: cluster.createdAt,
              updatedAt: cluster.updatedAt
            }));
            migratedCount++;
          });
          console.log(`LiveStore: Migrated ${clusters.length} clusters`);
        }
      }

      // Migrate notes
      if (legacyNotes) {
        const notes = JSON.parse(legacyNotes);
        if (Array.isArray(notes)) {
          notes.forEach(note => {
            store.commit(events.noteCreated({
              id: note.id,
              parentId: note.parentId,
              clusterId: note.clusterId,
              title: note.title,
              content: note.content || [],
              type: note.type || 'note',
              createdAt: note.createdAt,
              updatedAt: note.updatedAt,
              path: note.path || null,
              tags: note.tags || null,
              mentions: note.mentions || null
            }));
            migratedCount++;
          });
          console.log(`LiveStore: Migrated ${notes.length} notes`);
        }
      }

      // Migrate threads
      if (legacyThreads) {
        const threads = JSON.parse(legacyThreads);
        if (Array.isArray(threads)) {
          threads.forEach(thread => {
            store.commit(events.threadCreated({
              id: thread.id,
              title: thread.title,
              createdAt: thread.createdAt,
              updatedAt: thread.updatedAt
            }));
            migratedCount++;
          });
          console.log(`LiveStore: Migrated ${threads.length} threads`);
        }
      }

      // Migrate thread messages
      if (legacyThreadMessages) {
        const messages = JSON.parse(legacyThreadMessages);
        if (Array.isArray(messages)) {
          messages.forEach(message => {
            store.commit(events.threadMessageCreated({
              id: message.id,
              threadId: message.threadId,
              role: message.role,
              content: message.content,
              createdAt: message.createdAt,
              parentId: message.parentId
            }));
            migratedCount++;
          });
          console.log(`LiveStore: Migrated ${messages.length} thread messages`);
        }
      }

      // Migrate blueprints
      if (legacyBlueprints) {
        const blueprintData = JSON.parse(legacyBlueprints);
        if (blueprintData.blueprints && Array.isArray(blueprintData.blueprints)) {
          blueprintData.blueprints.forEach((blueprint: any) => {
            store.commit(events.blueprintCreated({
              id: blueprint.id,
              entityKind: blueprint.entityKind,
              name: blueprint.name,
              description: blueprint.description || null,
              templates: blueprint.templates || [],
              isDefault: blueprint.isDefault || false,
              createdAt: blueprint.createdAt,
              updatedAt: blueprint.updatedAt
            }));
            migratedCount++;
          });
          console.log(`LiveStore: Migrated ${blueprintData.blueprints.length} blueprints`);
        }
      }

      // Set initial UI state based on existing localStorage values
      const activeNoteId = localStorage.getItem('galaxy-notes-active-note-id');
      const activeClusterId = localStorage.getItem('galaxy-notes-active-cluster-id');
      
      store.commit(events.uiStateSet({
        activeNoteId: activeNoteId ? JSON.parse(activeNoteId) : null,
        activeClusterId: activeClusterId ? JSON.parse(activeClusterId) : 'cluster-default',
        activeThreadId: null,
        graphInitialized: false,
        graphLayout: 'dagre'
      }));

      // Mark main migration as completed
      localStorage.setItem('livestore-migration-completed', 'true');
      
      console.log(`LiveStore: Migration completed successfully! Migrated ${migratedCount} items`);
    }

    return true;
  } catch (error) {
    console.error('LiveStore: Migration failed:', error);
    return false;
  }
}

/**
 * Handle the embedding schema migration from note-based to chunk-based
 */
function migrateEmbeddingSchema(store: Store<any>): void {
  try {
    console.log('LiveStore: Starting embedding schema migration...');
    
    // Clear any existing embedding data that might be in the old format
    // Since embeddings can be regenerated, it's safer to clear and rebuild
    // rather than attempt complex data migration
    
    // Commit an embedding index clear event to log this migration
    store.commit(events.embeddingIndexCleared({
      clearedAt: new Date().toISOString(),
      reason: 'Schema migration from note-based to chunk-based embeddings'
    }));
    
    console.log('LiveStore: Embedding schema migration completed - old embeddings cleared');
    console.log('LiveStore: Embeddings will be regenerated when needed');
    
  } catch (error) {
    console.error('LiveStore: Embedding schema migration failed:', error);
    // Don't throw - let the app continue and embeddings will be rebuilt naturally
  }
}
