import { useState, useRef, useCallback, useEffect } from 'react';
import { Wand2, RotateCcw, Check, X, Plus, Minus, Type, Undo2 } from 'lucide-react';
import type { MaskResult } from '../types/segmentation';
import {
  startSegmentation,
  segmentText,
  segmentPoints,
  resetSegmentation,
  confirmSegmentation,
} from '../api/segmentation';

interface PointPrompt {
  x: number;
  y: number;
  label: number; // 1 = add, 0 = remove
}

interface Props {
  imageFile: File;
  onConfirm: (segmentedImagePath: string) => void;
  onCancel: () => void;
}

export default function SegmentationWorkspace({ imageFile, onConfirm, onCancel }: Props) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [textPrompt, setTextPrompt] = useState('');
  const [points, setPoints] = useState<PointPrompt[]>([]);
  const [pointMode, setPointMode] = useState<'add' | 'remove'>('add');
  const [masks, setMasks] = useState<MaskResult[]>([]);
  const [selectedMask, setSelectedMask] = useState(0);
  const [overlayUrl, setOverlayUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageUrl] = useState(() => URL.createObjectURL(imageFile));
  const canvasRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  // Zoom & pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number; moved: boolean } | null>(null);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onCancel]);

  // Initialize session on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      setLoading(true);
      try {
        const res = await startSegmentation(imageFile);
        if (!cancelled) setSessionId(res.session_id);
      } catch (e: any) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [imageFile]);

  // -- Zoom with scroll wheel (centered on cursor) --
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      const step = e.deltaY > 0 ? -0.15 : 0.15;
      setZoom(prev => {
        const next = Math.max(1, Math.min(6, prev + step * prev));
        // Adjust pan so point under cursor stays fixed
        const rect = el.getBoundingClientRect();
        const cx = e.clientX - rect.left - rect.width / 2;
        const cy = e.clientY - rect.top - rect.height / 2;
        const ratio = 1 - next / prev;
        setPan(p => ({
          x: next <= 1 ? 0 : p.x + (cx - p.x) * ratio,
          y: next <= 1 ? 0 : p.y + (cy - p.y) * ratio,
        }));
        return next;
      });
    };
    el.addEventListener('wheel', handler, { passive: false });
    return () => el.removeEventListener('wheel', handler);
  }, []);

  // -- Drag to pan (mousedown/move/up) --
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    if (e.button !== 0) return; // left button only
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y, moved: false };
  }, [pan]);

  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const d = dragRef.current;
    if (!d) return;
    const dx = e.clientX - d.startX;
    const dy = e.clientY - d.startY;
    if (!d.moved && Math.abs(dx) + Math.abs(dy) < 5) return; // movement threshold
    d.moved = true;
    setPan({ x: d.panX + dx, y: d.panY + dy });
  }, []);

  const handleMouseUp = useCallback(async (e: React.MouseEvent) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (d?.moved) return; // was a drag, not a click

    // -- Point placement (same as old handleCanvasClick) --
    if (!sessionId || loading) return;
    const rect = imgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;

    const label = pointMode === 'add' ? 1 : 0;
    const newPoints = [...points, { x, y, label }];
    setPoints(newPoints);
    setLoading(true);
    setError(null);
    try {
      const result = await segmentPoints(sessionId, newPoints.map(p => [p.x, p.y]), newPoints.map(p => p.label));
      setMasks(result.masks);
      setOverlayUrl(result.overlay_url ? result.overlay_url + `?t=${Date.now()}` : null);
      setSelectedMask(0);
      if (result.masks.length === 0) setError('No objects found. Try clicking on the object.');
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  }, [sessionId, loading, pointMode, points]);

  // Reset zoom/pan helper
  const resetView = useCallback(() => { setZoom(1); setPan({ x: 0, y: 0 }); }, []);

  const handleTextSegment = async () => {
    if (!sessionId || !textPrompt.trim() || loading) return;
    setLoading(true);
    setError(null);
    try {
      // Don't reset — segment_text() handles its own state internally
      // and stores text mask logits for subsequent point refinement.
      // Clear points so clicks after text start fresh as refinement.
      setPoints([]);
      const result = await segmentText(sessionId, textPrompt);
      setMasks(result.masks);
      setOverlayUrl(result.overlay_url ? result.overlay_url + `?t=${Date.now()}` : null);
      setSelectedMask(0);
      if (result.masks.length === 0) setError('No objects found. Try a different prompt.');
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleReset = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      await resetSegmentation(sessionId);
      setMasks([]); setOverlayUrl(null); setPoints([]); setTextPrompt(''); setError(null);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleUndo = async () => {
    if (!sessionId || points.length === 0 || loading) return;
    const newPoints = points.slice(0, -1);
    setPoints(newPoints);
    setLoading(true);
    setError(null);
    try {
      if (newPoints.length === 0) {
        // No points left — if we had a text prompt, re-run it to restore text mask
        if (textPrompt.trim()) {
          const result = await segmentText(sessionId, textPrompt);
          setMasks(result.masks);
          setOverlayUrl(result.overlay_url ? result.overlay_url + `?t=${Date.now()}` : null);
          setSelectedMask(0);
        } else {
          setMasks([]); setOverlayUrl(null);
        }
        setLoading(false);
        return;
      }
      // Re-send remaining points — backend uses all points fresh each call
      const result = await segmentPoints(sessionId, newPoints.map(p => [p.x, p.y]), newPoints.map(p => p.label));
      setMasks(result.masks);
      setOverlayUrl(result.overlay_url ? result.overlay_url + `?t=${Date.now()}` : null);
      setSelectedMask(0);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleConfirm = async () => {
    if (!sessionId || masks.length === 0) return;
    setLoading(true);
    setError(null);
    try {
      const res = await confirmSegmentation(sessionId, selectedMask);
      onConfirm(res.segmented_image_path);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  // -- Render: Full-screen modal overlay --
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-bg-primary border border-border rounded-2xl shadow-2xl flex flex-col max-w-4xl w-[95vw] max-h-[92vh] overflow-hidden">

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-border">
          <h2 className="text-sm font-medium text-text-primary flex items-center gap-2">
            <Wand2 className="w-4 h-4 text-accent" />
            Segment Object
          </h2>
          <div className="flex items-center gap-3">
            {/* Point mode toggle */}
            <div className="flex gap-1 bg-bg-tertiary rounded-lg p-0.5">
              <button
                onClick={() => setPointMode('add')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors
                  ${pointMode === 'add' ? 'bg-green-500/20 text-green-400' : 'text-text-muted hover:text-text-secondary'}`}
              >
                <Plus className="w-3.5 h-3.5" />
                Add
              </button>
              <button
                onClick={() => setPointMode('remove')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors
                  ${pointMode === 'remove' ? 'bg-red-500/20 text-red-400' : 'text-text-muted hover:text-text-secondary'}`}
              >
                <Minus className="w-3.5 h-3.5" />
                Remove
              </button>
            </div>

            {/* Undo + Reset */}
            <div className="flex gap-1">
              {points.length > 0 && (
                <button
                  onClick={handleUndo}
                  disabled={loading}
                  className="p-2 rounded-lg border border-border text-text-muted hover:bg-bg-tertiary transition-colors disabled:opacity-50"
                  title="Undo last point"
                >
                  <Undo2 className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={handleReset}
                disabled={loading || (points.length === 0 && masks.length === 0)}
                className="p-2 rounded-lg border border-border text-text-muted hover:bg-bg-tertiary transition-colors disabled:opacity-50"
                title="Reset all"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>

            {/* Close */}
            <button onClick={onCancel} className="p-2 rounded-lg text-text-muted hover:bg-bg-tertiary hover:text-text-secondary transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Canvas area — large, with zoom & pan */}
        <div className="flex-1 min-h-0 p-4 flex items-center justify-center bg-bg-secondary/50 relative">
          <div
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => { dragRef.current = null; }}
            className={`relative rounded-xl overflow-hidden border border-border bg-black select-none max-h-full ${
              zoom > 1 ? 'cursor-grab active:cursor-grabbing' : 'cursor-crosshair'
            }`}
          >
            {/* Transformed content — zoom & pan applied here */}
            <div style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: 'center center', transition: dragRef.current?.moved ? 'none' : 'transform 0.1s ease-out' }}>
              <img
                ref={imgRef}
                src={overlayUrl || imageUrl}
                alt="Segmentation"
                className="block max-w-full max-h-[60vh] object-contain"
                draggable={false}
              />
            </div>

            {/* Point markers — positioned over transformed image */}
            {imgRef.current && points.map((pt, i) => {
              const rect = imgRef.current!.getBoundingClientRect();
              const containerRect = canvasRef.current?.getBoundingClientRect();
              if (!containerRect) return null;
              const px = (rect.left - containerRect.left) + pt.x * rect.width;
              const py = (rect.top - containerRect.top) + pt.y * rect.height;
              return (
                <div key={i} className="absolute pointer-events-none" style={{ left: px - 8, top: py - 8, width: 16, height: 16 }}>
                  <div className={`w-4 h-4 rounded-full border-2 border-white shadow-lg ${pt.label === 1 ? 'bg-green-500' : 'bg-red-500'}`} />
                </div>
              );
            })}

            {/* Loading */}
            {loading && (
              <div className="absolute inset-0 bg-black/30 flex items-center justify-center">
                <svg className="w-8 h-8 animate-spin text-accent" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
                  <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                </svg>
              </div>
            )}

            {/* Hint */}
            {!loading && masks.length === 0 && points.length === 0 && sessionId && (
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <span className="text-sm text-white/70 bg-black/50 px-4 py-2 rounded-lg">
                  Click on the object, or type a prompt below — scroll to zoom, drag to pan
                </span>
              </div>
            )}
          </div>

          {/* Zoom indicator */}
          {zoom > 1 && (
            <div className="absolute top-6 right-6 flex items-center gap-2 bg-bg-primary/80 backdrop-blur-sm border border-border rounded-lg px-3 py-1.5">
              <span className="text-xs text-text-secondary font-medium">{Math.round(zoom * 100)}%</span>
              <button
                onClick={resetView}
                className="text-[10px] text-accent hover:text-accent-hover"
              >
                Reset
              </button>
            </div>
          )}
        </div>

        {/* Bottom controls */}
        <div className="px-5 py-3 border-t border-border space-y-3">

          {/* Text prompt + mask badges row */}
          <div className="flex items-center gap-3">
            {/* Text prompt */}
            <div className="relative flex-1 max-w-md">
              <Type className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
              <input
                type="text"
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleTextSegment()}
                placeholder="Describe object to find..."
                className="w-full bg-bg-tertiary border border-border rounded-lg pl-9 pr-3 py-2 text-sm placeholder:text-text-muted/50"
                disabled={loading || !sessionId}
              />
            </div>
            <button
              onClick={handleTextSegment}
              disabled={loading || !textPrompt.trim() || !sessionId}
              className="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium disabled:opacity-50 hover:bg-accent-hover transition-colors"
            >
              Find
            </button>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Mask badges */}
            {masks.length > 0 && (
              <div className="flex items-center gap-1.5">
                {masks.map((mask) => (
                  <button
                    key={mask.index}
                    onClick={() => setSelectedMask(mask.index)}
                    className={`text-xs px-2.5 py-1.5 rounded-lg border transition-colors flex items-center gap-1.5
                      ${selectedMask === mask.index
                        ? 'border-accent bg-accent/10 text-accent'
                        : 'border-border bg-bg-tertiary text-text-muted hover:border-border-hover'}`}
                  >
                    <span className={`w-1.5 h-1.5 rounded-full ${
                      mask.score > 0.8 ? 'bg-green-400' : mask.score > 0.6 ? 'bg-yellow-400' : 'bg-red-400'
                    }`} />
                    {masks.length > 1 ? `Mask ${mask.index + 1}` : 'Mask'}
                    <span className="opacity-60">{(mask.score * 100).toFixed(0)}%</span>
                  </button>
                ))}
              </div>
            )}

            {/* Point count */}
            {points.length > 0 && (
              <span className="text-xs text-text-muted">{points.length} point{points.length !== 1 ? 's' : ''}</span>
            )}

            {/* Use Mask */}
            <button
              onClick={handleConfirm}
              disabled={loading || masks.length === 0}
              className="px-5 py-2 rounded-lg bg-success text-white text-sm font-medium disabled:opacity-30 hover:bg-success/90 transition-colors flex items-center gap-2"
            >
              <Check className="w-4 h-4" />
              Use Mask
            </button>
          </div>

          {/* Error */}
          {error && (
            <p className="text-xs text-error text-center">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
}
