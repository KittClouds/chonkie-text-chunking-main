
import { HNSW } from './main';

interface GraphSnapshot {
  fileName: string;
  checksum: string;
  createdAt: Date;
  size: number;
}

interface SerializedGraph {
  M: number;
  efConstruction: number;
  levelMax: number;
  entryPointId: number;
  nodes: any[];
  metadata: {
    version: string;
    createdAt: string;
    nodeCount: number;
  };
}

export class HNSWPersistence {
  private readonly GRAPH_DIR = 'hnsw-graphs';
  private readonly VERSION = '1.0.0';

  async persistGraph(index: HNSW, fileName: string = 'main-index'): Promise<void> {
    try {
      // 1. Serialize the graph
      const serialized = this.serializeGraph(index);
      const data = new TextEncoder().encode(JSON.stringify(serialized));
      
      // 2. Write to OPFS
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(this.GRAPH_DIR, { create: true });
      const file = await dir.getFileHandle(`${fileName}.json`, { create: true });
      
      const writable = await file.createWritable();
      await writable.write(data);
      await writable.close();
      
      // 3. Create checksum
      const checksum = await this.createChecksum(data);
      
      console.log(`HNSW graph persisted: ${fileName}, size: ${data.length} bytes, checksum: ${checksum}`);
    } catch (error) {
      console.error('Failed to persist HNSW graph:', error);
      throw error;
    }
  }

  async loadGraph(fileName: string = 'main-index'): Promise<HNSW | null> {
    try {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(this.GRAPH_DIR, { create: false });
      const file = await dir.getFileHandle(`${fileName}.json`);
      
      const fileData = await file.getFile();
      const text = await fileData.text();
      const serialized: SerializedGraph = JSON.parse(text);
      
      // Validate and deserialize
      if (serialized.metadata?.version !== this.VERSION) {
        console.warn(`Version mismatch: expected ${this.VERSION}, got ${serialized.metadata?.version}`);
      }
      
      const hnsw = HNSW.fromJSON(serialized);
      console.log(`HNSW graph loaded: ${fileName}, nodes: ${serialized.metadata?.nodeCount}`);
      
      return hnsw;
    } catch (error) {
      console.warn(`Failed to load HNSW graph: ${fileName}`, error);
      return null;
    }
  }

  // NEW: Remove a snapshot file
  async removeFile(fileName: string): Promise<void> {
    try {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(this.GRAPH_DIR, { create: false });
      await dir.removeEntry(`${fileName}.json`);
      console.log(`HNSW: Removed file ${fileName}.json`);
    } catch (error: any) {
      if (error.name !== 'NotFoundError') {
        console.error(`Failed to remove file ${fileName}.json:`, error);
        throw error;
      }
      // If not found, it's a success in our context.
    }
  }

  // NEW: Rename a snapshot file
  async renameFile(oldName: string, newName: string): Promise<void> {
    try {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(this.GRAPH_DIR, { create: false });
      const fileHandle = await dir.getFileHandle(`${oldName}.json`);

      // OPFS doesn't have a direct 'rename'. We read and write to a new file, then delete the old one.
      const fileData = await fileHandle.getFile();
      const newFileHandle = await dir.getFileHandle(`${newName}.json`, { create: true });
      const writable = await newFileHandle.createWritable();
      await writable.write(await fileData.arrayBuffer());
      await writable.close();

      await dir.removeEntry(`${oldName}.json`);
      console.log(`HNSW: Renamed ${oldName}.json to ${newName}.json`);
    } catch (error: any) {
      if (error.name !== 'NotFoundError') {
        console.error(`Failed to rename ${oldName}.json to ${newName}.json:`, error);
        throw error;
      }
    }
  }

  async getSnapshotInfo(): Promise<{ count: number; totalSize: number; snapshots: GraphSnapshot[] }> {
    try {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(this.GRAPH_DIR, { create: false });
      
      const snapshots: GraphSnapshot[] = [];
      let totalSize = 0;
      
      // Use a more compatible approach with async iteration
      try {
        // Try to use values() method which is more widely supported
        for await (const handle of (dir as any).values()) {
          if (handle.kind === 'file' && handle.name.endsWith('.json')) {
            try {
              const file = await handle.getFile();
              const size = file.size;
              totalSize += size;
              
              snapshots.push({
                fileName: handle.name.replace('.json', ''),
                checksum: '',
                createdAt: new Date(file.lastModified),
                size
              });
            } catch (error) {
              console.warn(`Failed to get info for file: ${handle.name}`, error);
            }
          }
        }
      } catch (iterationError) {
        // Fallback: if iteration fails, return empty results
        console.warn('Failed to iterate directory:', iterationError);
        return { count: 0, totalSize: 0, snapshots: [] };
      }
      
      return {
        count: snapshots.length,
        totalSize,
        snapshots: snapshots.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
      };
    } catch (error) {
      return { count: 0, totalSize: 0, snapshots: [] };
    }
  }

  async gcOldSnapshots(keepCount: number = 2): Promise<number> {
    try {
      const { snapshots } = await this.getSnapshotInfo();
      if (snapshots.length <= keepCount) return 0;
      
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(this.GRAPH_DIR, { create: false });
      
      const toDelete = snapshots.slice(keepCount);
      let deletedCount = 0;
      
      for (const snapshot of toDelete) {
        try {
          await dir.removeEntry(`${snapshot.fileName}.json`);
          deletedCount++;
        } catch (error) {
          console.warn(`Failed to delete snapshot: ${snapshot.fileName}`, error);
        }
      }
      
      console.log(`Garbage collected ${deletedCount} old HNSW snapshots`);
      return deletedCount;
    } catch (error) {
      console.error('Failed to garbage collect snapshots:', error);
      return 0;
    }
  }

  private serializeGraph(index: HNSW): SerializedGraph {
    const json = index.toJSON();
    return {
      ...json,
      metadata: {
        version: this.VERSION,
        createdAt: new Date().toISOString(),
        nodeCount: json.nodes.length
      }
    };
  }

  private async createChecksum(data: Uint8Array): Promise<string> {
    const hashBuffer = await crypto.subtle.digest('SHA-256', data);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    return hashArray.map(b => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
  }
}

export const hnswPersistence = new HNSWPersistence();
