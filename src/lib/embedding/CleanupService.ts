
import { tables, events } from '../../livestore/schema';

interface CleanupResult {
  removed: number;
  remainingStale: number;
  errors: string[];
  summary: string;
}

export class EmbeddingCleanupService {
  private storeRef: any = null;

  setStore(store: any) {
    this.storeRef = store;
  }

  async forceCleanupStaleEmbeddings(): Promise<CleanupResult> {
    if (!this.storeRef) {
      throw new Error('Store reference not set');
    }

    const result: CleanupResult = {
      removed: 0,
      remainingStale: 0,
      errors: [],
      summary: ''
    };

    try {
      // Get current notes as source of truth
      const allNotesResult = this.storeRef.query(tables.notes.select());
      const allNotes = Array.isArray(allNotesResult) ? allNotesResult : [];
      const currentNoteIds = new Set(allNotes.map((n: any) => n.id));

      console.log(`Cleanup: Found ${allNotes.length} current notes`);

      // Get all embeddings
      const allEmbeddingRows = this.storeRef.query(tables.embeddings.select());
      const embeddings = Array.isArray(allEmbeddingRows) ? allEmbeddingRows : [];

      console.log(`Cleanup: Found ${embeddings.length} stored embeddings`);

      // Find stale embeddings
      const staleEmbeddings = embeddings.filter((embedding: any) => {
        const noteId = embedding.noteId;
        const isValid = typeof noteId === 'string' && noteId.length > 0;
        const noteExists = isValid && currentNoteIds.has(noteId);
        return !isValid || !noteExists;
      });

      console.log(`Cleanup: Found ${staleEmbeddings.length} stale embeddings`);

      // Remove stale embeddings
      for (const embedding of staleEmbeddings) {
        try {
          this.storeRef.commit(events.embeddingRemoved({ noteId: embedding.noteId }));
          result.removed++;
          console.log(`Cleanup: Removed stale embedding for note ${embedding.noteId}`);
        } catch (error) {
          const errorMsg = `Failed to remove embedding for note ${embedding.noteId}: ${error}`;
          result.errors.push(errorMsg);
          console.error(errorMsg);
        }
      }

      // Verify cleanup
      const remainingEmbeddings = this.storeRef.query(tables.embeddings.select());
      const remainingCount = Array.isArray(remainingEmbeddings) ? remainingEmbeddings.length : 0;
      result.remainingStale = Math.max(0, remainingCount - allNotes.length);

      result.summary = `Removed ${result.removed} stale embeddings. ${result.remainingStale} stale entries remain. ${result.errors.length} errors occurred.`;
      
      if (result.errors.length > 0) {
        console.warn('Cleanup completed with errors:', result.errors);
      } else {
        console.log('Cleanup completed successfully:', result.summary);
      }

      return result;
    } catch (error) {
      const errorMsg = `Cleanup failed: ${error}`;
      result.errors.push(errorMsg);
      result.summary = errorMsg;
      console.error(errorMsg);
      throw error;
    }
  }

  async detectStaleData(): Promise<{ hasStaleData: boolean; staleCount: number; noteCount: number; embeddingCount: number }> {
    if (!this.storeRef) {
      return { hasStaleData: false, staleCount: 0, noteCount: 0, embeddingCount: 0 };
    }

    try {
      const noteCount = this.storeRef.query(tables.notes.count()) || 0;
      const embeddingCount = this.storeRef.query(tables.embeddings.count()) || 0;
      const staleCount = Math.max(0, embeddingCount - noteCount);
      
      return {
        hasStaleData: staleCount > 0,
        staleCount,
        noteCount,
        embeddingCount
      };
    } catch (error) {
      console.error('Failed to detect stale data:', error);
      return { hasStaleData: false, staleCount: 0, noteCount: 0, embeddingCount: 0 };
    }
  }
}

export const embeddingCleanupService = new EmbeddingCleanupService();
