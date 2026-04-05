import { useState } from 'react';
import { Wand2, RotateCcw, Check, X, MousePointer2 } from 'lucide-react';
import type { MaskResult } from '../types/segmentation';
import {
  startSegmentation,
  segmentText,
  resetSegmentation,
  confirmSegmentation,
} from '../api/segmentation';

interface Props {
  imageFile: File;
  onConfirm: (segmentedImagePath: string) => void;
  onCancel: () => void;
}

export default function SegmentationWorkspace({ imageFile, onConfirm, onCancel }: Props) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [prompt, setPrompt] = useState('');
  const [masks, setMasks] = useState<MaskResult[]>([]);
  const [selectedMask, setSelectedMask] = useState(0);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageUrl] = useState(() => URL.createObjectURL(imageFile));

  const handleStartAndSegment = async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);

    try {
      let sid = sessionId;
      if (!sid) {
        const res = await startSegmentation(imageFile);
        sid = res.session_id;
        setSessionId(sid);
      }

      const segResult = await segmentText(sid, prompt);
      setMasks(segResult.masks);
      setOverlayUrl(segResult.overlay_url || null);
      setSelectedMask(0);

      if (segResult.masks.length === 0) {
        setError('No objects found. Try a different prompt.');
      }
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      await resetSegmentation(sessionId);
      setMasks([]);
      setOverlayUrl(null);
      setPrompt('');
      setError(null);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleConfirm = async () => {
    if (!sessionId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await confirmSegmentation(sessionId, selectedMask);
      onConfirm(res.segmented_image_path);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-medium text-text-secondary flex items-center gap-1.5">
          <Wand2 className="w-3.5 h-3.5 text-accent" />
          Segment Object
        </h3>
        <button onClick={onCancel} className="text-text-muted hover:text-text-secondary">
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Image preview with overlay */}
      <div className="relative rounded-lg overflow-hidden border border-border bg-bg-tertiary">
        <img
          src={overlayUrl || imageUrl}
          alt="Segmentation"
          className="w-full h-auto max-h-48 object-contain"
        />
        {loading && (
          <div className="absolute inset-0 bg-bg-primary/60 flex items-center justify-center">
            <svg className="w-6 h-6 animate-spin text-accent" viewBox="0 0 24 24" fill="none">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
              <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
            </svg>
          </div>
        )}
      </div>

      {/* Text prompt */}
      <div className="flex gap-2">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleStartAndSegment()}
          placeholder="Describe object to segment..."
          className="flex-1 bg-bg-tertiary border border-border rounded-lg px-3 py-1.5 text-sm placeholder:text-text-muted/50"
          disabled={loading}
        />
        <button
          onClick={handleStartAndSegment}
          disabled={loading || !prompt.trim()}
          className="px-3 py-1.5 rounded-lg bg-accent text-white text-xs font-medium disabled:opacity-50 hover:bg-accent-hover transition-colors flex items-center gap-1"
        >
          <MousePointer2 className="w-3.5 h-3.5" />
          Segment
        </button>
      </div>

      {/* Mask results */}
      {masks.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-text-muted">{masks.length} mask(s) found</p>
          <div className="flex flex-wrap gap-1.5">
            {masks.map((mask) => (
              <button
                key={mask.index}
                onClick={() => setSelectedMask(mask.index)}
                className={`text-xs px-2.5 py-1 rounded-lg border transition-colors
                  ${selectedMask === mask.index
                    ? 'border-accent bg-accent/10 text-accent'
                    : 'border-border bg-bg-tertiary text-text-muted hover:border-border-hover'}`}
              >
                #{mask.index + 1} ({(mask.score * 100).toFixed(0)}%)
              </button>
            ))}
          </div>

          {/* Action buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleReset}
              disabled={loading}
              className="flex-1 py-1.5 rounded-lg border border-border text-xs text-text-muted hover:bg-bg-tertiary transition-colors flex items-center justify-center gap-1"
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </button>
            <button
              onClick={handleConfirm}
              disabled={loading}
              className="flex-1 py-1.5 rounded-lg bg-success/20 text-success text-xs font-medium hover:bg-success/30 transition-colors flex items-center justify-center gap-1"
            >
              <Check className="w-3 h-3" />
              Use Mask #{selectedMask + 1}
            </button>
          </div>
        </div>
      )}

      {error && (
        <p className="text-xs text-error">{error}</p>
      )}
    </div>
  );
}
