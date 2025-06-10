
import { ProcessedDocument } from './types';

// Dynamic imports for wink-nlp to handle browser compatibility
let winkNLP: any;
let model: any;
let nlp: any;
let its: any;

async function initializeWink() {
  if (!nlp) {
    try {
      // Use dynamic import with string concatenation to avoid TS resolution
      const winkModule = await import('wink' + '-nlp');
      const modelModule = await import('wink-eng-lite' + '-web-model');
      
      winkNLP = winkModule.default || winkModule;
      model = modelModule.default || modelModule;
      nlp = winkNLP(model);
      its = nlp.its;
    } catch (error) {
      console.warn('Failed to load wink-nlp, falling back to simple tokenization:', error);
    }
  }
}

export class TextProcessor {
  private initialized = false;

  private async ensureInitialized() {
    if (!this.initialized) {
      await initializeWink();
      this.initialized = true;
    }
  }

  /**
   * Processes a document's text to segment it into sentences and tokens.
   */
  public async processDocument(content: string): Promise<ProcessedDocument> {
    await this.ensureInitialized();
    
    if (!nlp) {
      // Fallback to simple processing if wink-nlp is not available
      return this.fallbackProcessDocument(content);
    }

    try {
      const winkDoc = nlp.readDoc(content);
      const sentences: { tokens: string[] }[] = [];
      let totalTokenCount = 0;

      winkDoc.sentences().each((sentence: any) => {
        const tokens = sentence.tokens().out(its.normal);
        sentences.push({ tokens });
        totalTokenCount += tokens.length;
      });

      return { sentences, totalTokenCount };
    } catch (error) {
      console.warn('Error processing document with wink-nlp, using fallback:', error);
      return this.fallbackProcessDocument(content);
    }
  }

  /**
   * Tokenizes a simple query string.
   */
  public async tokenizeQuery(query: string): Promise<string[]> {
    await this.ensureInitialized();
    
    if (!nlp) {
      return this.fallbackTokenizeQuery(query);
    }

    try {
      return nlp.readDoc(query).tokens().out(its.normal);
    } catch (error) {
      console.warn('Error tokenizing query with wink-nlp, using fallback:', error);
      return this.fallbackTokenizeQuery(query);
    }
  }

  private fallbackProcessDocument(content: string): ProcessedDocument {
    // Simple sentence splitting on periods, exclamation marks, and question marks
    const sentences = content
      .split(/[.!?]+/)
      .filter(s => s.trim().length > 0)
      .map(sentence => ({
        tokens: this.fallbackTokenizeQuery(sentence.trim())
      }));

    const totalTokenCount = sentences.reduce((count, sentence) => count + sentence.tokens.length, 0);
    
    return { sentences, totalTokenCount };
  }

  private fallbackTokenizeQuery(query: string): string[] {
    return query
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(token => token.length > 0);
  }
}
