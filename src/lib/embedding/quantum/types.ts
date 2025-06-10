
export interface TokenMetadata {
  freq: number;
  segmentMask: number;
}

export interface DocumentData {
  originalContent: string;
  tokenCount: number;
}

export interface SearchResult {
  docId: string;
  score: number;
  content: string;
}

export interface QPSConfig {
  maxSegments?: number;
  proximityBonus?: number;
}

export interface ProcessedDocument {
  sentences: { tokens: string[] }[];
  totalTokenCount: number;
}
