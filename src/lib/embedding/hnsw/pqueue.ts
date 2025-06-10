export class PriorityQueue<T> {
  private heap: T[] = [];

  constructor(private compare: (a: T, b: T) => number) {}

  public push(item: T): void {
    this.heap.push(item);
    this.siftUp();
  }

  public pop(): T | undefined {
    if (this.isEmpty()) {
      return undefined;
    }
    const item = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0 && last !== undefined) {
      this.heap[0] = last;
      this.siftDown();
    }
    return item;
  }

  public peek(): T | undefined {
    return this.heap[0];
  }

  public size(): number {
    return this.heap.length;
  }

  public isEmpty(): boolean {
    return this.heap.length === 0;
  }

  public clone(): PriorityQueue<T> {
    const newQueue = new PriorityQueue<T>(this.compare);
    newQueue.heap = [...this.heap]; // Shallow copy of the heap
    return newQueue;
  }

  private siftUp(): void {
    let nodeIndex = this.heap.length - 1;
    while (nodeIndex > 0 && this.compareWithParent(nodeIndex) < 0) {
      this.swap(nodeIndex, this.getParentIndex(nodeIndex));
      nodeIndex = this.getParentIndex(nodeIndex);
    }
  }

  private siftDown(): void {
    let nodeIndex = 0;
    while (
      (this.getLeftChildIndex(nodeIndex) < this.heap.length &&
        this.compareWithLeftChild(nodeIndex) > 0) ||
      (this.getRightChildIndex(nodeIndex) < this.heap.length &&
        this.compareWithRightChild(nodeIndex) > 0)
    ) {
      const greaterChildIndex =
        this.getRightChildIndex(nodeIndex) < this.heap.length &&
        this.compareWithRightChild(nodeIndex) < this.compareWithLeftChild(nodeIndex)
          ? this.getRightChildIndex(nodeIndex)
          : this.getLeftChildIndex(nodeIndex);
      this.swap(nodeIndex, greaterChildIndex);
      nodeIndex = greaterChildIndex;
    }
  }

  private getParentIndex(nodeIndex: number): number {
    return Math.floor((nodeIndex - 1) / 2);
  }

  private getLeftChildIndex(nodeIndex: number): number {
    return 2 * nodeIndex + 1;
  }

  private getRightChildIndex(nodeIndex: number): number {
    return 2 * nodeIndex + 2;
  }

  private compareWithParent(nodeIndex: number): number {
    return this.compare(this.heap[nodeIndex], this.heap[this.getParentIndex(nodeIndex)]);
  }

  private compareWithLeftChild(nodeIndex: number): number {
    return this.compare(this.heap[nodeIndex], this.heap[this.getLeftChildIndex(nodeIndex)]);
  }

  private compareWithRightChild(nodeIndex: number): number {
    return this.compare(this.heap[nodeIndex], this.heap[this.getRightChildIndex(nodeIndex)]);
  }

  private swap(i: number, j: number): void {
    const temp = this.heap[i];
    this.heap[i] = this.heap[j];
    this.heap[j] = temp;
  }
}