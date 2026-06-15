import { Ruler, Check, AlertTriangle, Loader2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import type { AutoScaleMetadata } from '../types';
import { rescaleTask } from '../api/client';

interface Props {
  autoScale: AutoScaleMetadata | null | undefined;
  taskId: string | null;
  /** Called after a successful rescale so the parent can reload the GLB to pick up the new scale. */
  onRescaled?: (newAutoScale: AutoScaleMetadata) => void;
}

const CONFIDENCE_STYLE: Record<string, { dot: string; label: string }> = {
  high:   { dot: 'bg-emerald-400', label: 'high confidence' },
  medium: { dot: 'bg-amber-400',   label: 'medium confidence' },
  low:    { dot: 'bg-red-400',     label: 'low confidence' },
};

function fmt(n: number): string {
  return n >= 10 ? n.toFixed(1) : n.toFixed(2);
}

export default function AutoScalePanel({ autoScale, taskId, onRescaled }: Props) {
  const dims = autoScale?.dimensions_m;
  const [target, setTarget] = useState<string>(dims ? fmt(dims.longest_m) : '');
  const [applying, setApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Sync input when active item changes
  useEffect(() => {
    if (dims) setTarget(fmt(dims.longest_m));
  }, [dims?.longest_m]);

  if (!autoScale) return null;

  const failed = !autoScale.auto_scaled;
  const conf = CONFIDENCE_STYLE[autoScale.confidence || 'low'] || CONFIDENCE_STYLE.low;
  const manual = autoScale.scale_source === 'manual';

  const handleApply = async () => {
    if (!taskId) return;
    const t = parseFloat(target);
    if (!isFinite(t) || t <= 0) {
      setError('Enter a positive number');
      return;
    }
    if (dims && Math.abs(t - dims.longest_m) < 1e-3) return;  // no-op
    setApplying(true);
    setError(null);
    try {
      const res = await rescaleTask(taskId, t);
      onRescaled?.(res.auto_scale);
    } catch (e: any) {
      setError(e?.message || 'Rescale failed');
    } finally {
      setApplying(false);
    }
  };

  return (
    <div className="space-y-2.5">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-text-primary flex items-center gap-1.5">
          <Ruler className="w-4 h-4" />
          Real-world size
        </h3>
        {!failed && (
          <span className="text-[10px] text-text-muted flex items-center gap-1">
            <span className={`w-1.5 h-1.5 rounded-full ${conf.dot}`} />
            {manual ? 'manual' : conf.label}
          </span>
        )}
      </div>

      {failed ? (
        <div className="text-xs text-text-muted flex items-start gap-1.5 p-2 rounded bg-bg-tertiary border border-border">
          <AlertTriangle className="w-3.5 h-3.5 text-amber-400 mt-0.5 shrink-0" />
          <span>Auto-scale unavailable{autoScale.reason ? `: ${autoScale.reason}` : '. Override below.'}</span>
        </div>
      ) : dims && (
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="p-2 rounded bg-bg-tertiary border border-border">
            <div className="text-text-muted text-[10px] uppercase tracking-wide">Longest</div>
            <div className="font-mono text-text-primary">{fmt(dims.longest_m)} m</div>
          </div>
          <div className="p-2 rounded bg-bg-tertiary border border-border">
            <div className="text-text-muted text-[10px] uppercase tracking-wide">Middle</div>
            <div className="font-mono text-text-primary">{fmt(dims.middle_m)} m</div>
          </div>
          <div className="p-2 rounded bg-bg-tertiary border border-border">
            <div className="text-text-muted text-[10px] uppercase tracking-wide">Shortest</div>
            <div className="font-mono text-text-primary">{fmt(dims.shortest_m)} m</div>
          </div>
        </div>
      )}

      {!failed && autoScale.view_alignment && !manual && (
        <p className="text-[10px] text-text-muted">
          View match IoU {autoScale.view_alignment.iou.toFixed(2)} · {autoScale.view_alignment.method}
          {autoScale.object_distance_m != null && ` · ${autoScale.object_distance_m.toFixed(1)}m from camera`}
        </p>
      )}

      <div className="space-y-1">
        <label className="text-[10px] text-text-muted uppercase tracking-wide">Set longest dim (m)</label>
        <div className="flex gap-1.5">
          <input
            type="number"
            step="0.01"
            min="0.001"
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            disabled={!taskId || applying}
            className="flex-1 px-2 py-1.5 text-xs font-mono bg-bg-tertiary border border-border rounded focus:border-accent focus:outline-none disabled:opacity-50"
          />
          <button
            onClick={handleApply}
            disabled={!taskId || applying}
            className="px-3 py-1.5 text-xs font-medium rounded border border-accent/30 bg-accent/10 text-accent hover:bg-accent/20 transition-colors disabled:opacity-50 flex items-center gap-1"
            title={taskId ? 'Re-bake GLB at this size' : 'Select a model first'}
          >
            {applying ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Check className="w-3.5 h-3.5" />}
            Apply
          </button>
        </div>
        {error && <p className="text-[10px] text-red-400">{error}</p>}
        {!taskId && <p className="text-[10px] text-text-muted">Override available after a fresh generation.</p>}
      </div>
    </div>
  );
}
