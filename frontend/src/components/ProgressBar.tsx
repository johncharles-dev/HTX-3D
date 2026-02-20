import type { ProgressUpdate, TaskStatus } from '../types';
import { Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react';

interface Props {
  progress: ProgressUpdate | null;
  status: TaskStatus | null;
}

const STATUS_ICONS: Record<string, React.ReactNode> = {
  queued: <Clock className="w-4 h-4 text-text-muted animate-pulse" />,
  processing: <Loader2 className="w-4 h-4 text-accent animate-spin" />,
  extracting: <Loader2 className="w-4 h-4 text-warning animate-spin" />,
  completed: <CheckCircle2 className="w-4 h-4 text-success" />,
  failed: <XCircle className="w-4 h-4 text-error" />,
};

const STATUS_COLORS: Record<string, string> = {
  queued: 'bg-text-muted',
  processing: 'bg-accent',
  extracting: 'bg-warning',
  completed: 'bg-success',
  failed: 'bg-error',
};

export default function ProgressBar({ progress, status }: Props) {
  if (!status) return null;

  const pct = progress ? Math.round(progress.progress * 100) : 0;
  const stage = progress?.stage || status;
  const message = progress?.message || '';

  return (
    <div className="bg-bg-tertiary rounded-lg p-3 border border-border">
      <div className="flex items-center gap-2 mb-2">
        {STATUS_ICONS[status] || null}
        <span className="text-sm font-medium text-text-primary">{stage}</span>
        <span className="text-xs text-text-muted ml-auto">{pct}%</span>
      </div>
      <div className="h-1.5 bg-bg-secondary rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${STATUS_COLORS[status] || 'bg-accent'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {message && <p className="text-xs text-text-muted mt-1.5">{message}</p>}
    </div>
  );
}
