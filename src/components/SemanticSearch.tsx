
import React, { useState, useCallback, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Search, Loader2, Zap, Database, AlertTriangle, CheckCircle, Trash2, Info } from 'lucide-react';
import { useDebouncedCallback } from '@/hooks/useDebouncedCallback';
import { semanticSearchService } from '@/lib/embedding/SemanticSearchService';
import { useActiveNoteId, useNotes, useNoteActions } from '@/hooks/useLiveStore';
import { useEmbeddings, useEmbeddingCount } from '@/hooks/useEmbeddings';
import { useEmbeddingHealth } from '@/hooks/useEmbeddingHealth';
import { toast } from 'sonner';

interface SearchResult {
  noteId: string;
  title: string;
  content: string;
  score: number;
}

export function SemanticSearch() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [, setActiveNoteId] = useActiveNoteId();
  const notes = useNotes();
  const { syncAllNotesEmbeddings } = useNoteActions();
  
  // Use LiveStore reactive queries
  const embeddings = useEmbeddings();
  const embeddingCount = useEmbeddingCount();
  
  // Enhanced health monitoring
  const { health, buildProgress, checkHealth, forceCleanup, isBuilding } = useEmbeddingHealth();

  const handleSearch = useDebouncedCallback(async (searchQuery: string) => {
    if (!searchQuery.trim()) {
      setResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const searchResults = await semanticSearchService.search(searchQuery, 10);
      setResults(searchResults);
    } catch (error) {
      console.error('Search failed:', error);
      toast.error('Search failed');
    } finally {
      setIsSearching(false);
    }
  }, 500);

  const handleSyncEmbeddings = useCallback(async () => {
    try {
      const count = await syncAllNotesEmbeddings();
      toast.success(`Synchronized ${count} note embeddings`);
      await checkHealth();
    } catch (error) {
      console.error('Sync failed:', error);
      toast.error('Failed to sync embeddings');
    }
  }, [syncAllNotesEmbeddings, checkHealth]);

  const handleForceCleanup = useCallback(async () => {
    try {
      const result = await forceCleanup();
      toast.success(result.summary);
    } catch (error) {
      console.error('Force cleanup failed:', error);
      toast.error('Failed to cleanup stale embeddings');
    }
  }, [forceCleanup]);

  const handleSelectNote = (noteId: string) => {
    setActiveNoteId(noteId);
  };

  // Auto-refresh health periodically
  useEffect(() => {
    const interval = setInterval(checkHealth, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, [checkHealth]);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="p-4 flex flex-col h-full space-y-4">
      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Semantic search..."
          className="pl-8"
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            handleSearch(e.target.value);
          }}
        />
      </div>

      {/* Health Status Card */}
      <Card className="text-xs">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center">
            {health.isHealthy ? (
              <CheckCircle className="h-4 w-4 mr-2 text-green-500" />
            ) : (
              <AlertTriangle className="h-4 w-4 mr-2 text-yellow-500" />
            )}
            Embedding Health
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="flex items-center">
              <Database className="h-3 w-3 mr-1" />
              Embeddings: {embeddingCount}
            </span>
            <span>Notes: {health.noteCount}</span>
          </div>
          
          {health.hasStaleData && (
            <div className="flex items-center justify-between">
              <Badge variant="destructive" className="text-xs">
                {health.staleCount} Stale
              </Badge>
              <Button
                variant="outline"
                size="sm"
                onClick={handleForceCleanup}
                className="h-6 text-xs px-2"
                disabled={isBuilding}
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Cleanup
              </Button>
            </div>
          )}
          
          {health.snapshotCount > 0 && (
            <div className="flex items-center justify-between text-muted-foreground">
              <span className="flex items-center">
                <Info className="h-3 w-3 mr-1" />
                {health.snapshotCount} snapshots
              </span>
              <span>{formatBytes(health.totalSnapshotSize)}</span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Sync Button */}
      <Button
        variant="outline"
        size="sm"
        className="w-full"
        onClick={handleSyncEmbeddings}
        disabled={isBuilding}
      >
        {isBuilding ? (
          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
        ) : (
          <Zap className="h-4 w-4 mr-2" />
        )}
        {isBuilding ? buildProgress.message : 'Sync Embeddings'}
      </Button>

      {/* Build Progress */}
      {isBuilding && (
        <div className="space-y-2">
          <Progress value={(buildProgress.current / buildProgress.total) * 100} className="h-2" />
          <div className="text-xs text-muted-foreground text-center">
            {buildProgress.phase} • {buildProgress.current}/{buildProgress.total}
          </div>
        </div>
      )}

      {/* Search Loading */}
      {isSearching && (
        <div className="flex justify-center py-4">
          <Loader2 className="h-5 w-5 animate-spin" />
        </div>
      )}

      {/* Search Results */}
      <ScrollArea className="flex-1">
        <div className="space-y-2">
          {results.map((result) => (
            <div
              key={result.noteId}
              className="p-3 rounded-md border border-transparent hover:border-primary/20 hover:bg-primary/10 cursor-pointer transition-colors"
              onClick={() => handleSelectNote(result.noteId)}
            >
              <div className="flex justify-between items-center mb-1">
                <div className="font-medium text-sm truncate">{result.title}</div>
                <Badge variant="secondary" className="text-xs">
                  {Math.round(result.score * 100)}%
                </Badge>
              </div>
              <p className="text-xs text-muted-foreground line-clamp-2">
                {result.content.slice(0, 100)}...
              </p>
            </div>
          ))}
          {query && !isSearching && results.length === 0 && (
            <div className="text-center text-muted-foreground py-4">
              <p className="text-sm">No results found</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}
