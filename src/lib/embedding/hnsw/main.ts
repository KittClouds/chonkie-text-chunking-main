import { PriorityQueue } from './pqueue';
import { Node } from './node';
import { cosineSimilarity, euclideanSimilarity } from './similarity';

type Metric = 'cosine' | 'euclidean';

interface SerializedNodeData {
  id: number;
  level: number;
  vector: number[];
  neighbors: number[][];
  deleted: boolean;
}

export interface SerializedHNSWData {
  M: number;
  efConstruction: number;
  levelMax: number;
  entryPointId: number;
  nodes: [number, SerializedNodeData][];
  metric: Metric;
  d: number | null;
  // probs and levelMult are derived in constructor or by methods, so not typically part of minimal serialization
}

export class HNSW {
  metric: Metric; // Metric to use
  similarityFunction: (a: number[] | Float32Array, b: number[] | Float32Array) => number;
  d: number | null = null; // Dimension of the vectors
  M: number; // Max number of neighbors
  efConstruction: number; // Max number of nodes to visit during construction
  levelMax: number; // Max level of the graph
  entryPointId: number; // Id of the entry point
  nodes: Map<number, Node>; // Map of nodes
  probs: number[]; // Probabilities for the levels
  levelMult: number; // Multiplier for level generation probability

  constructor(
    M = 16, 
    efConstruction = 200, 
    metric: Metric = 'cosine', 
    levelMult?: number
  ) {
    this.metric = metric;
    this.d = null; // Dimensionality is set by the first vector added
    this.M = M;
    this.efConstruction = efConstruction;
    this.levelMult = levelMult === undefined ? 1 / Math.log(M) : levelMult;
    this.entryPointId = -1;
    this.nodes = new Map<number, Node>();
    // Initialize probs using the M and determined levelMult
    this.probs = this.set_probs(this.M, this.levelMult);
    this.levelMax = this.probs.length - 1;
    this.similarityFunction = this.getMetric(metric);
  }

  private getMetric(metric: Metric): (a: number[] | Float32Array, b: number[] | Float32Array) => number {
    if (metric === 'cosine') {
      return cosineSimilarity;
    } else if (metric === 'euclidean') {
      return euclideanSimilarity;
    } else {
      throw new Error('Invalid metric');
    }
  }

  private set_probs(M: number, levelMult: number): number[] {
    let level = 0;
    const probs = [];
    while (true) {
      const prob = Math.exp(-level / levelMult) * (1 - Math.exp(-1 / levelMult));
      if (prob < 1e-9) break;
      probs.push(prob);
      level++;
    }
    return probs;
  }

  private selectLevel(): number {
    let r = Math.random();
    this.probs.forEach((p, i) => {
      if (r < p) {
        return i;
      }
      r -= p;
    });
    return this.probs.length - 1;
  }

  private _searchLayer(
    queryVector: Float32Array | number[],
    entryPointIds: number[],
    layerNumber: number,
    numCandidatesToKeep: number
  ): { candidates: PriorityQueue<number>; terminatedEarly: boolean } {
    const visited = new Set<number>();
    // Worklist: Max-heap of {id, similarity} to explore most promising first
    const W = new PriorityQueue<{ id: number; similarity: number }>(
        (a, b) => b.similarity - a.similarity
    );
    // Results: Min-heap of {id, similarity} to keep top N, peek() is worst of top N
    const C = new PriorityQueue<{ id: number; similarity: number }>(
        (a, b) => a.similarity - b.similarity
    );
    let terminatedEarly = false;

    for (const epId of entryPointIds) {
      const epNode = this.nodes.get(epId);
      // Ignore if node doesn't exist, is deleted, or already visited
      if (!epNode || epNode.deleted || visited.has(epId)) continue;
      
      // epNode is guaranteed to exist and not be deleted here
      const sim = this.similarityFunction(queryVector, epNode.vector);
      
      W.push({ id: epId, similarity: sim });
      visited.add(epId);

      if (C.size() < numCandidatesToKeep) {
        C.push({ id: epId, similarity: sim });
      } else if (sim > C.peek()!.similarity) {
        C.pop();
        C.push({ id: epId, similarity: sim });
      }
    }

    while (!W.isEmpty()) {
      const current = W.pop()!; // Pop best candidate to explore

      // Early termination check: if C is full and current's similarity is worse than the worst in C
      if (C.size() === numCandidatesToKeep && current.similarity < C.peek()!.similarity) {
        terminatedEarly = true;
        break; // Exit the while (!W.isEmpty()) loop
      }

      const currentNodeObject = this.nodes.get(current.id);
      // Ignore if node doesn't exist, is deleted, or has no neighbors at this level
      if (!currentNodeObject || currentNodeObject.deleted || !currentNodeObject.neighbors[layerNumber]) continue;

      for (const neighborId of currentNodeObject.neighbors[layerNumber]) {
        if (neighborId === -1) continue; // Skip placeholder
        
        const neighborNode = this.nodes.get(neighborId);
        // Ignore if neighbor doesn't exist, is deleted, or already visited
        if (!neighborNode || neighborNode.deleted || visited.has(neighborId)) continue;

        visited.add(neighborId); // Add to visited only if valid and not deleted
        // neighborNode is guaranteed to exist and not be deleted here
        const simToQuery = this.similarityFunction(queryVector, neighborNode.vector);

        if (C.size() < numCandidatesToKeep || simToQuery > C.peek()!.similarity) {
          if (C.size() === numCandidatesToKeep) {
            C.pop(); // Remove worst from C if full
          }
          C.push({ id: neighborId, similarity: simToQuery });
        }
        W.push({ id: neighborId, similarity: simToQuery }); // Add to worklist
      }
    }

    // Convert C (min-heap of {id, similarity}) to a PQ of just IDs, ordered best first for popping.
    const returnQueue = new PriorityQueue<number>(
      (id_a, id_b) => 
        this.similarityFunction(queryVector, this.nodes.get(id_b)!.vector) -
        this.similarityFunction(queryVector, this.nodes.get(id_a)!.vector)
    );
    const tempArray: { id: number; similarity: number }[] = [];
    while (!C.isEmpty()) tempArray.push(C.pop()!); 
    // tempArray is now sorted worst first (due to min-heap pop order)
    tempArray.sort((a, b) => b.similarity - a.similarity); // Sort best first by similarity
    tempArray.forEach(item => returnQueue.push(item.id));
    
    return { candidates: returnQueue, terminatedEarly: terminatedEarly }; // 4. Return object
  }

  private async addNodeToGraph(node: Node) {
    if (this.entryPointId === -1) {
      this.entryPointId = node.id;
      return;
    }

    const nodeVector = node.vector;
    let currentGlobalEntryPointIds: number[] = [this.entryPointId];

    // 1. Search layers above node.level to find the best entry point(s) for node.level
    for (let searchLayer = this.levelMax; searchLayer > node.level; searchLayer--) {
      if (currentGlobalEntryPointIds.length === 0) break; // Stop if no entry points
      const bestEntryPointQueue = this._searchLayer(nodeVector, currentGlobalEntryPointIds, searchLayer, 1);
      if (!bestEntryPointQueue.candidates.isEmpty()) {
        currentGlobalEntryPointIds = [bestEntryPointQueue.candidates.peek()!];
      } else {
        currentGlobalEntryPointIds = []; // No path found at this level
      }
    }

    // 2. Search in node's layers (from node.level down to 0) to find efConstruction neighbors
    for (let connectionLayer = Math.min(node.level, this.levelMax); connectionLayer >= 0; connectionLayer--) {
      if (currentGlobalEntryPointIds.length === 0 && connectionLayer < Math.min(node.level, this.levelMax)) {
          // If we lost all entry points and we are not in the very first connection layer search,
          // try to reset to global entry point to prevent complete disconnection if possible.
          // This is a heuristic and might need refinement.
          currentGlobalEntryPointIds = [this.entryPointId]; 
      }
      if (currentGlobalEntryPointIds.length === 0 && this.nodes.size > 1) {
        // If still no entry points (e.g. global entry point is not suitable or graph is tiny),
        // we might not be able to connect this node at this layer.
        // This situation should be rare in a healthy graph construction.
        // For now, we'll let _searchLayer handle empty entryPointIds (it will return empty queue).
      }

      const neighborCandidateQueue = this._searchLayer(nodeVector, currentGlobalEntryPointIds, connectionLayer, this.efConstruction);
      
      const neighborsFoundAtThisLayer: number[] = [];
      while(!neighborCandidateQueue.candidates.isEmpty()){
          neighborsFoundAtThisLayer.push(neighborCandidateQueue.candidates.pop()!); // Pop gives best due to returnQueue's comparator
      }

      for (const neighborId of neighborsFoundAtThisLayer) {
        if (neighborId === node.id) continue; 

        const neighborNode = this.nodes.get(neighborId);
        if (neighborNode) {
          // Connect the new node (node) to its selected neighbor (neighborId).
          // _updateNeighborsAndPrune will handle if neighborId points to a (now) deleted node internally.
          this._updateNeighborsAndPrune(node, neighborId, connectionLayer);
          
          // Connect the selected neighbor (neighborNode) back to the new node (node.id),
          // only if neighborNode has not been marked as deleted.
          if (!neighborNode.deleted) {
            this._updateNeighborsAndPrune(neighborNode, node.id, connectionLayer);
          }
        }
      }
      
      if (neighborsFoundAtThisLayer.length > 0) {
        currentGlobalEntryPointIds = neighborsFoundAtThisLayer;
      } 
      // If neighborsFoundAtThisLayer is empty, currentGlobalEntryPointIds from previous layer search persists,
      // or the global entry point if it was reset.
    }
  }

  private _updateNeighborsAndPrune(baseNode: Node, newNeighborId: number, level: number): void {
    const M = this.M; // Max neighbors for this level

    // Get current valid neighbors and add the new one
    let candidates = baseNode.neighbors[level].filter(id => id !== -1);
    if (!candidates.includes(newNeighborId) && newNeighborId !== baseNode.id) {
        candidates.push(newNeighborId);
    }

    if (candidates.length > M) {
      // Need to prune. Calculate similarities and pick the M closest.
      const candidateSims: { id: number; similarity: number }[] = [];
      for (const neighborId of candidates) {
        const neighborNode = this.nodes.get(neighborId);
        // Only consider existing, non-deleted nodes for similarity calculation
        if (neighborNode && !neighborNode.deleted) {
          candidateSims.push({
            id: neighborId,
            similarity: this.similarityFunction(baseNode.vector, neighborNode.vector)
          });
        }
      }

      // Sort by similarity (descending) to get the closest ones
      candidateSims.sort((a, b) => b.similarity - a.similarity);
      
      // Update neighbors with the top M
      for (let i = 0; i < M; i++) {
        baseNode.neighbors[level][i] = candidateSims[i] ? candidateSims[i].id : -1;
      }
      // Fill remaining slots if fewer than M actual neighbors
      for (let i = candidateSims.length; i < M; i++) {
          baseNode.neighbors[level][i] = -1;
      }

    } else {
      // No pruning needed. Filter out deleted nodes from candidates.
      const validCandidates = candidates.filter(candId => {
        const candNode = this.nodes.get(candId);
        return candNode && !candNode.deleted;
      });

      // Update baseNode's neighbors for the current level
      for (let i = 0; i < M; i++) {
        baseNode.neighbors[level][i] = i < validCandidates.length ? validCandidates[i] : -1;
      }

    }
  }

  addPoint(id: number, vector: Float32Array | number[]) {
    if (this.nodes.has(id)) {
      throw new Error(`Node with id ${id} already exists.`);
    }

    if (this.d === null) {
      this.d = vector.length;
    } else if (vector.length !== this.d) {
      throw new Error(`Vector dimensionality ${vector.length} does not match index dimensionality ${this.d}`);
    }

    const level = this.selectLevel();
    const node = new Node(id, vector, level, this.M);
    this.nodes.set(id, node);

    if (this.entryPointId === -1) {
      this.entryPointId = id;
      this.levelMax = node.level;
      return;
    }
    
    this.levelMax = Math.max(this.levelMax, node.level); 

    this.addNodeToGraph(node);
  }

  public deletePoint(id: number): void {
    const nodeToDelete = this.nodes.get(id);
    if (nodeToDelete) {
        nodeToDelete.deleted = true;
        // Note: This simple implementation doesn't remove the node from neighbors' lists.
        // A more advanced version would require a "cleanup" process or more complex logic here.
    } else {
        throw new Error(`Node with id ${id} not found.`);
    }
  }

  searchKNN(
    query: Float32Array | number[],
    k: number,
    efSearch?: number
  ): { id: number; score: number }[] {
    if (this.nodes.size === 0) {
      return [];
    }
    // Attempt to recover if entryPointId is -1 but nodes exist (e.g. after all nodes deleted then new ones added)
    if (this.entryPointId === -1 && this.nodes.size > 0) {
        let foundRecoveryEP = false;
        for (const [id, node] of this.nodes) {
            if (!node.deleted) {
                this.entryPointId = id;
                // Ensure levelMax is sensible if we had to recover entryPointId
                // This might involve checking node.level, but addPoint should manage levelMax generally.
                // For now, we assume levelMax is either correct or will be less critical than finding an EP.
                console.warn(`HNSW.searchKNN: Recovered missing entryPointId to ${id}.`);
                foundRecoveryEP = true;
                break;
            }
        }
        if (!foundRecoveryEP) {
            console.error("HNSW.searchKNN: entryPointId is -1 and no non-deleted nodes found to recover for search.");
            return [];
        }
    }
    
    let effectiveEntryPointId = this.entryPointId;
    let currentGlobalEntryPointNode = this.nodes.get(effectiveEntryPointId);

    // Check if the determined entry point (either original or recovered) is valid and not deleted.
    if (!currentGlobalEntryPointNode || currentGlobalEntryPointNode.deleted) {
      console.warn(`HNSW.searchKNN: Entry point ID ${effectiveEntryPointId} (original or recovered) is invalid or deleted. Attempting to find a new one for this search.`);
      let foundNewFallback = false;
      for (const [id, node] of this.nodes) {
        if (!node.deleted) {
          effectiveEntryPointId = id;
          currentGlobalEntryPointNode = node;
          foundNewFallback = true;
          console.log(`HNSW.searchKNN: Using fallback entry point ${id} for this search as ${this.entryPointId} was unusable.`);
          break;
        }
      }
      if (!foundNewFallback) {
        console.error(`HNSW.searchKNN: No valid non-deleted entry points found in the graph to conduct search.`);
        return [];
      }
    }
    // At this point, 'currentGlobalEntryPointNode' is a non-deleted node if any exist and 'effectiveEntryPointId' is its ID.
    // If no non-deleted nodes exist, we would have returned an empty array.

    if (this.d !== null && query.length !== this.d) {
        throw new Error(`Query vector dimensionality ${query.length} does not match index dimensionality ${this.d}`);
    }
    
    // Handle case with only one valid node
    // currentGlobalEntryPointNode is guaranteed to be non-null and non-deleted here if nodes.size >= 1
    if (this.nodes.size === 1 && currentGlobalEntryPointNode) { 
        const similarity = this.similarityFunction(currentGlobalEntryPointNode.vector, query);
        return [{ id: effectiveEntryPointId, score: similarity }];
    }
    // If nodes map is empty (should have been caught by initial check, but as a safeguard)
    if (this.nodes.size === 0) return []; 
    if (!currentGlobalEntryPointNode) { // Should not happen if nodes.size > 0 and entry point logic is correct
        console.error("HNSW.searchKNN: currentGlobalEntryPointNode is unexpectedly null before dual-branch logic.");
        return [];
    }

    let currentEntryPoints: number[] = [effectiveEntryPointId];
    // At this point, currentGlobalEntryPointNode is the validated, non-deleted entry point.
    const mainEntryPointNode = currentGlobalEntryPointNode; 

    // --- DIVERSITY-AWARE MULTI-BRANCH LOGIC ---
    const MAX_BRANCHES = 3; // Configurable number of diverse entry points
    const DIVERSITY_SIMILARITY_THRESHOLD = 0.9; // How similar candidates can be to be considered non-diverse

    // 1. Perform a cheap search at a high level to find diverse candidates, only if the graph is large enough
    if (this.levelMax > 0 && this.nodes.size > this.M * 2) {
        const presearchResult = this._searchLayer(
            query,
            [effectiveEntryPointId],
            this.levelMax, // Search at the highest layer
            this.M // Look for a number of candidates related to M
        );

        const candidatePool: number[] = [effectiveEntryPointId];
        const tempPQ = presearchResult.candidates.clone();
        while (!tempPQ.isEmpty() && candidatePool.length < this.M) {
            candidatePool.push(tempPQ.pop()!); // Pop returns highest similarity first
        }

        // 2. Select a few candidates from the pool that are dissimilar from each other
        const diverseEntryPoints: number[] = [effectiveEntryPointId];
        for (const candidateId of candidatePool) {
            if (diverseEntryPoints.length >= MAX_BRANCHES) break;
            if (diverseEntryPoints.includes(candidateId)) continue;

            const candidateNode = this.nodes.get(candidateId);
            if (!candidateNode || candidateNode.deleted) continue;

            let isDiverseEnough = true;
            for (const depId of diverseEntryPoints) {
                const depNode = this.nodes.get(depId)!; // Should exist
                // Check if the new candidate is too similar to any already in our diverse set
                if (this.similarityFunction(candidateNode.vector, depNode.vector) > DIVERSITY_SIMILARITY_THRESHOLD) {
                    isDiverseEnough = false;
                    break;
                }
            }

            if (isDiverseEnough) {
                diverseEntryPoints.push(candidateId);
            }
        }
        currentEntryPoints = diverseEntryPoints;
    }
    // --- END DUAL-BRANCH LOGIC ---

    // Phase 1: Search from top layer down to layer 1
    for (let currentLayer = this.levelMax; currentLayer >= 1; currentLayer--) {
      currentEntryPoints = currentEntryPoints.filter(epId => {
          const node = this.nodes.get(epId);
          return node && !node.deleted;
      });
      if (currentEntryPoints.length === 0) {
          const fallbackEntryPoint = this.nodes.get(this.entryPointId); // Use original global entry
          if (fallbackEntryPoint && !fallbackEntryPoint.deleted) {
            currentEntryPoints = [this.entryPointId];
          } else { // Should not happen if global entry point was validated initially
            return []; 
          }
      }

      const numCandidatesToKeep = currentEntryPoints.length;
      const { candidates } = this._searchLayer(query, currentEntryPoints, currentLayer, numCandidatesToKeep);
      
      if (!candidates.isEmpty()) {
          const topCandidates: number[] = [];
          while(!candidates.isEmpty()) {
              topCandidates.push(candidates.pop()!); 
          }
          currentEntryPoints = topCandidates.reverse(); 
      } else {
        const fallbackEntryPoint = this.nodes.get(this.entryPointId);
        if (fallbackEntryPoint && !fallbackEntryPoint.deleted) {
            currentEntryPoints = [this.entryPointId];
        } else {
            return []; 
        }
      }
    }
    
    currentEntryPoints = currentEntryPoints.filter(epId => {
        const node = this.nodes.get(epId);
        return node && !node.deleted;
    });
    if (currentEntryPoints.length === 0) {
        const fallbackEntryPoint = this.nodes.get(this.entryPointId);
        if (fallbackEntryPoint && !fallbackEntryPoint.deleted) {
            currentEntryPoints = [this.entryPointId];
        } else {
            return []; 
        }
    }

    // Phase 2: Adaptive search at layer 0
    let finalCandidateIdsQueue: PriorityQueue<number> | undefined;
    let efSearchCurrent = Math.max(k, efSearch !== undefined ? efSearch : 32); 
    const efSearchMax = efSearch !== undefined ? Math.max(k, efSearch) : Math.max(k, this.efConstruction);

    for (let attempt = 0; attempt < 2; attempt++) { 
        const result = this._searchLayer(query, currentEntryPoints, 0, efSearchCurrent);
        finalCandidateIdsQueue = result.candidates;

        if (result.terminatedEarly || (efSearch !== undefined && finalCandidateIdsQueue && finalCandidateIdsQueue.size() >= k)) {
            break;
        }

        efSearchCurrent = Math.min(efSearchCurrent * 2, efSearchMax); 
        
        if (attempt === 1 || efSearchCurrent >= efSearchMax) {
            break;
        }
        
        if (finalCandidateIdsQueue && !finalCandidateIdsQueue.isEmpty()){ 
            const bestSoFar: number[] = [];
            const tempPQ = finalCandidateIdsQueue.clone(); 
            while(!tempPQ.isEmpty() && bestSoFar.length < Math.max(1, currentEntryPoints.length)) { 
                bestSoFar.push(tempPQ.pop()!); // Pop gives best due to returnQueue's comparator
            }
            currentEntryPoints = bestSoFar.length > 0 ? bestSoFar : [this.entryPointId]; 
            currentEntryPoints = currentEntryPoints.filter(epId => { 
                const node = this.nodes.get(epId);
                return node && !node.deleted;
            });
            if (currentEntryPoints.length === 0) { 
                const fallbackEntryPoint = this.nodes.get(this.entryPointId);
                if (fallbackEntryPoint && !fallbackEntryPoint.deleted) currentEntryPoints = [this.entryPointId]; else return [];
            }
        } else { 
            const fallbackEntryPoint = this.nodes.get(this.entryPointId);
            if (fallbackEntryPoint && !fallbackEntryPoint.deleted) currentEntryPoints = [this.entryPointId]; else return [];
        }
    }

    // Phase 3: Extract top K results
    const allCandidatesWithScores: { id: number; score: number }[] = [];
    if (!finalCandidateIdsQueue) {
        console.warn("HNSW.searchKNN: finalCandidateIdsQueue is undefined after adaptive search. No results.");
        return [];
    }

    const seenIds = new Set<number>(); // To handle potential duplicates from PQ if any
    while (!finalCandidateIdsQueue.isEmpty()) {
        const candidateId = finalCandidateIdsQueue.pop()!;
        if (seenIds.has(candidateId)) continue;
        seenIds.add(candidateId);

        const candidateNode = this.nodes.get(candidateId);
        if (candidateNode && !candidateNode.deleted) {
            const score = this.similarityFunction(candidateNode.vector, query);
            allCandidatesWithScores.push({ id: candidateId, score: score });
        }
    }

    // Sort by score in descending order (highest similarity first)
    allCandidatesWithScores.sort((a, b) => b.score - a.score);

    // Return the top K results
    return allCandidatesWithScores.slice(0, k);
  }

  buildIndex(data: { id: number; vector: Float32Array | number[] }[]) {
    // Clear existing index
    this.nodes.clear();
    this.levelMax = 0;
    this.entryPointId = -1;
    this.d = null; // Reset dimensionality

    // Add points to the index
    for (const item of data) {
      this.addPoint(item.id, item.vector);
    }
  }

  toJSON(): SerializedHNSWData {
    const entries = Array.from(this.nodes.entries());
    return {
      M: this.M,
      efConstruction: this.efConstruction,
      levelMax: this.levelMax,
      entryPointId: this.entryPointId,
      nodes: entries.map(([id, node]) => {
        return [
          id,
          {
            id: node.id,
            level: node.level,
            vector: Array.from(node.vector),
            neighbors: node.neighbors.map((level) => Array.from(level)),
            deleted: node.deleted,
          } as SerializedNodeData,
        ];
      }),
      metric: this.metric,
      d: this.d,
    };
  }

  static fromJSON(jsonData: SerializedHNSWData): HNSW {
    const hnsw = new HNSW(jsonData.M, jsonData.efConstruction, jsonData.metric);
    hnsw.levelMax = jsonData.levelMax;
    hnsw.entryPointId = jsonData.entryPointId;
    hnsw.d = jsonData.d; // Restore dimensionality
    hnsw.nodes = new Map(
      jsonData.nodes.map(([id, nodeData]: [number, SerializedNodeData]) => {
        const newNodeInstance = new Node(nodeData.id, new Float32Array(nodeData.vector), nodeData.level, hnsw.M);
        newNodeInstance.neighbors = nodeData.neighbors; // Assign deserialized neighbors
        newNodeInstance.deleted = nodeData.deleted;     // Assign deserialized deleted status
        return [id, newNodeInstance];
      }),
    );
    return hnsw;
  }

  public pruneDeletedNodes(): void {
    const deletedNodeIds = new Set<number>();
    // First, find all nodes marked for deletion
    for (const [id, node] of this.nodes.entries()) {
        if (node.deleted) {
            deletedNodeIds.add(id);
        }
    }

    if (deletedNodeIds.size === 0) return; // Nothing to do

    // For every non-deleted node, remove links to deleted nodes
    for (const node of this.nodes.values()) {
        if (node.deleted) continue;

        for (let level = 0; level <= node.level; level++) {
            const originalNeighbors = node.neighbors[level];
            if (!originalNeighbors) continue; 

            const cleanedNeighbors = originalNeighbors.filter(id => id !== -1 && !deletedNodeIds.has(id));
            
            // Re-pad the array. Assuming neighbor arrays are of size this.M for all levels.
            const newNeighbors = new Array(this.M).fill(-1);
            cleanedNeighbors.forEach((id, i) => {
                if (i < newNeighbors.length) { // Ensure we don't write out of bounds
                    newNeighbors[i] = id;
                }
            });
            node.neighbors[level] = newNeighbors;
        }
    }

    // Finally, remove the deleted nodes from the main map
    for (const id of deletedNodeIds) {
        this.nodes.delete(id);
    }

    // Optional: Reset entry point if it was deleted
    if (this.entryPointId !== -1 && deletedNodeIds.has(this.entryPointId)) {
        // After deletion, this.nodes contains only non-deleted nodes.
        // Pick the first available node as the new entry point.
        const firstRemainingNodeId = this.nodes.keys().next().value;
        this.entryPointId = firstRemainingNodeId !== undefined ? firstRemainingNodeId : -1;
    }
  }
}