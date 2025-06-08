
import { generateNodeId, generateDocumentId } from '../../utils/ids';
import { NodeRelationship, ObjectType } from './types';
import type { Metadata, RelatedNodeInfo, RelatedNodeType, BaseNodeParams, TextNodeParams } from './types';

/**
 * A helper function to create a SHA-256 hash using the browser's Web Crypto API.
 * This is an asynchronous operation.
 * @param data The string data to hash.
 * @returns A promise that resolves to a Base64 encoded hash string.
 */
async function createSHA256Async(data: string): Promise<string> {
  const textAsBuffer = new TextEncoder().encode(data);
  const hashBuffer = await crypto.subtle.digest('SHA-256', textAsBuffer);
  
  // Convert ArrayBuffer to Base64 string in a browser-safe way
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const binaryString = hashArray.map(b => String.fromCharCode(b)).join('');
  return btoa(binaryString);
}

/**
 * Generic abstract class for retrievable nodes.
 * The hashing mechanism has been updated to be async and browser-compatible.
 */
export abstract class BaseNode<T extends Metadata = Metadata> {
  id_: string;
  metadata: T;
  relationships: Partial<Record<NodeRelationship, RelatedNodeType<T>>>;

  private _hash: string = '';
  private _hashPromise: Promise<string> | null = null;

  protected constructor(init?: BaseNodeParams<T>) {
    const { id_, metadata, relationships } = init || {};
    // Use the nanoid-based generator for default IDs
    this.id_ = id_ ?? generateNodeId(); 
    this.metadata = metadata ?? ({} as T);
    this.relationships = relationships ?? {};
  }

  /**
   * Asynchronously gets the hash of the node.
   * The hash is computed lazily on the first call and cached.
   */
  get hash(): Promise<string> {
    if (this._hash) {
      return Promise.resolve(this._hash);
    }
    if (!this._hashPromise) {
      this._hashPromise = this.generateHash().then(h => {
        this._hash = h;
        return h;
      });
    }
    return this._hashPromise;
  }

  abstract get type(): ObjectType;

  abstract getContent(): string;

  abstract getMetadataStr(): string;

  get sourceNode(): RelatedNodeInfo<T> | undefined {
    const relationship = this.relationships[NodeRelationship.SOURCE];
    if (Array.isArray(relationship)) {
      throw new Error('Source object must be a single RelatedNodeInfo object');
    }
    return relationship;
  }

  get prevNode(): RelatedNodeInfo<T> | undefined {
    const relationship = this.relationships[NodeRelationship.PREVIOUS];
    if (Array.isArray(relationship)) {
      throw new Error('Previous object must be a single RelatedNodeInfo object');
    }
    return relationship;
  }

  get nextNode(): RelatedNodeInfo<T> | undefined {
    const relationship = this.relationships[NodeRelationship.NEXT];
    if (Array.isArray(relationship)) {
      throw new Error('Next object must be a single RelatedNodeInfo object');
    }
    return relationship;
  }

  get parentNode(): RelatedNodeInfo<T> | undefined {
    const relationship = this.relationships[NodeRelationship.PARENT];
    if (Array.isArray(relationship)) {
      throw new Error('Parent object must be a single RelatedNodeInfo object');
    }
    return relationship;
  }

  get childNodes(): RelatedNodeInfo<T>[] | undefined {
    const relationship = this.relationships[NodeRelationship.CHILD];
    if (relationship === undefined) {
      return undefined;
    }
    if (Array.isArray(relationship)) {
      return relationship;
    }
    // If it's a single RelatedNodeInfo, wrap it in an array
    return [relationship];
  }
  
  /**
   * The method to generate the node's hash is now asynchronous.
   */
  abstract generateHash(): Promise<string>;
}

/**
 * TextNode is the default node type for text.
 */
export class TextNode<T extends Metadata = Metadata> extends BaseNode<T> {
  text: string;
  startCharIdx?: number;
  endCharIdx?: number;
  metadataSeparator: string;

  constructor(init: TextNodeParams<T> = {}) {
    super(init);
    const { text, startCharIdx, endCharIdx, metadataSeparator } = init;
    this.text = text ?? '';
    if (startCharIdx) {
      this.startCharIdx = startCharIdx;
    }
    if (endCharIdx) {
      this.endCharIdx = endCharIdx;
    }
    this.metadataSeparator = metadataSeparator ?? '\n';
  }

  /**
   * Asynchronously generate a hash of the text node's content.
   * The ID is not part of the hash as it can change independent of content.
   */
  async generateHash(): Promise<string> {
    const contentToHash = [
      `type=${this.type}`,
      `startCharIdx=${this.startCharIdx ?? ''}`,
      `endCharIdx=${this.endCharIdx ?? ''}`,
      this.getContent()
    ].join('');
    return createSHA256Async(contentToHash);
  }

  get type() {
    return ObjectType.TEXT;
  }

  getContent(): string {
    const metadataStr = this.getMetadataStr().trim();
    if (metadataStr) {
      return `${metadataStr}\n\n${this.text}`.trim();
    }
    return this.text.trim();
  }

  getMetadataStr(): string {
    const usableMetadataKeys = Object.keys(this.metadata).sort();
    return usableMetadataKeys.map(key => `${key}: ${this.metadata[key]}`).join(this.metadataSeparator);
  }

  getNodeInfo() {
    return { start: this.startCharIdx, end: this.endCharIdx };
  }

  getText() {
    return this.text;
  }
}

/**
 * A document is a special text node with a document-specific ID format.
 */
export class Document<T extends Metadata = Metadata> extends TextNode<T> {
  constructor(init: TextNodeParams<T> = {}) {
    // If no ID is provided, generate a document-specific one before calling super
    const id_ = init.id_ ?? generateDocumentId();
    super({ ...init, id_ });
  }

  get type() {
    return ObjectType.DOCUMENT;
  }
}
