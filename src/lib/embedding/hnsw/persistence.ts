
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

  async getSnapshotInfo(): Promise<{ count: number; totalSize: number; snapshots: GraphSnapshot[] }> {
    try {
      const root = await navigator.storage.getDirectory();
      const dir = await root.getDirectoryHandle(this.GRAPH_DIR, { create: false });
      
      const snapshots: GraphSnapshot[] = [];
      let totalSize = 0;
      
      for await (const [name, handle] of dir.entries()) {
        if (handle.kind === 'file' && name.endsWith('.json')) {
          const file = await handle.getFile();
          const size = file.size;
          totalSize += size;
          
          snapshots.push({
            fileName: name.replace('.json', ''),
            checksum: '',
            createdAt: new Date(file.lastModified),
            size
          });
        }
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
