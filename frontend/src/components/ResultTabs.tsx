import { Loader2, CheckCircle2, XCircle, Clock, Ban } from 'lucide-react';
import { MODELS, type ModelResult } from '../types';

interface Props {
  results: ModelResult[];
  activeModelId: string | null;
  onSelect: (modelId: string) => void;
}

export default function ResultTabs({ results, activeModelId, onSelect }: Props) {
  if (results.length === 0) return null;

  return (
    <div className="flex gap-1 bg-bg-secondary rounded-lg p-1 border border-border">
      {results.map((mr) => {
        const model = MODELS.find((m) => m.id === mr.modelId);
        if (!model) return null;

        const isActive = activeModelId === mr.modelId;
        const StatusIcon = {
          queued: Clock,
          processing: Loader2,
          extracting: Loader2,
          completed: CheckCircle2,
          failed: XCircle,
          cancelled: Ban,
        }[mr.status] || Clock;

        const isLoading = mr.status === 'processing' || mr.status === 'extracting';

        return (
          <button
            key={mr.modelId}
            onClick={() => onSelect(mr.modelId)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors ${
              isActive ? 'text-white' : 'text-text-muted hover:text-text-secondary'
            }`}
            style={{
              backgroundColor: isActive ? `${model.color}22` : undefined,
              color: isActive ? model.color : undefined,
            }}
          >
            <StatusIcon className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
            <span className="font-medium">{model.name}</span>
            {mr.status === 'processing' && mr.progress && (
              <span className="opacity-60">{Math.round(mr.progress.progress * 100)}%</span>
            )}
          </button>
        );
      })}
    </div>
  );
}
