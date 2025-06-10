export class Node {
  id: number;
  level: number;
  vector: Float32Array | number[];
  neighbors: number[][];
  public deleted: boolean; // Flag to mark node as deleted

  constructor(id: number, vector: Float32Array | number[], level: number, M: number) {
    this.id = id;
    this.vector = vector;
    this.level = level;
    this.neighbors = Array.from({ length: level + 1 }, () => new Array(M).fill(-1));
    this.deleted = false; // Initialize as not deleted
  }
}