import { useState } from 'react';
import { Sun, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react';
import type { ViewerSettings } from '../types';
import { DEFAULT_VIEWER_SETTINGS } from '../types';

interface Props {
  settings: ViewerSettings;
  onChange: (s: ViewerSettings) => void;
}

const ENV_PRESETS = [
  { value: 'none', label: 'None' },
  { value: 'studio', label: 'Studio' },
  { value: 'city', label: 'City' },
  { value: 'sunset', label: 'Sunset' },
  { value: 'dawn', label: 'Dawn' },
  { value: 'night', label: 'Night' },
  { value: 'warehouse', label: 'Warehouse' },
  { value: 'forest', label: 'Forest' },
  { value: 'apartment', label: 'Apartment' },
  { value: 'park', label: 'Park' },
  { value: 'lobby', label: 'Lobby' },
];

function LightPositionControl({
  x, y, onChange,
}: { x: number; y: number; onChange: (x: number, y: number) => void }) {
  const size = 80;
  const radius = size / 2 - 8;
  const cx = size / 2;
  const cy = size / 2;
  // Map -1..1 to pixel coords
  const dotX = cx + x * radius;
  const dotY = cy - y * radius; // invert Y so top = positive

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const nx = Math.max(-1, Math.min(1, (px - cx) / radius));
    const ny = Math.max(-1, Math.min(1, -(py - cy) / radius));
    onChange(Math.round(nx * 100) / 100, Math.round(ny * 100) / 100);
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <svg
        width={size}
        height={size}
        className="cursor-crosshair"
        onClick={handleClick}
      >
        {/* Sphere background */}
        <circle cx={cx} cy={cy} r={radius} fill="none" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
        <circle cx={cx} cy={cy} r={radius * 0.5} fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="0.5" />
        {/* Crosshair */}
        <line x1={cx - radius} y1={cy} x2={cx + radius} y2={cy} stroke="rgba(255,255,255,0.06)" strokeWidth="0.5" />
        <line x1={cx} y1={cy - radius} x2={cx} y2={cy + radius} stroke="rgba(255,255,255,0.06)" strokeWidth="0.5" />
        {/* Light position dot */}
        <circle cx={dotX} cy={dotY} r={5} fill="#fbbf24" />
        <circle cx={dotX} cy={dotY} r={8} fill="none" stroke="#fbbf2466" strokeWidth="1" />
      </svg>
      <span className="text-[10px] text-text-muted">Click to position light</span>
    </div>
  );
}

export default function SceneSettings({ settings, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const s = settings;
  const set = (patch: Partial<ViewerSettings>) => onChange({ ...s, ...patch });

  return (
    <div className="space-y-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 w-full text-sm font-medium text-text-primary"
      >
        <Sun className="w-4 h-4" />
        Scene Settings
        {open ? <ChevronUp className="w-3.5 h-3.5 ml-auto text-text-muted" /> : <ChevronDown className="w-3.5 h-3.5 ml-auto text-text-muted" />}
      </button>

      {open && (
        <div className="space-y-3 pt-1">
          {/* Light Position */}
          <div>
            <label className="text-xs text-text-secondary block mb-1">Light Position</label>
            <LightPositionControl
              x={s.lightPositionX}
              y={s.lightPositionY}
              onChange={(x, y) => set({ lightPositionX: x, lightPositionY: y })}
            />
          </div>

          {/* Light Color */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-text-secondary">Light Color</label>
              <input
                type="color"
                value={s.lightColor}
                onChange={(e) => set({ lightColor: e.target.value })}
                className="w-6 h-5 rounded border border-border cursor-pointer bg-transparent"
              />
            </div>
          </div>

          {/* Spotlight Intensity */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-text-secondary">Spotlight Intensity</label>
              <span className="text-xs font-mono text-text-muted">{s.spotlightIntensity.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={s.spotlightIntensity}
              onChange={(e) => set({ spotlightIntensity: Number(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-[10px] text-text-muted/50">
              <span>Weak</span>
              <span>Strong</span>
            </div>
          </div>

          {/* Planar Light Intensity */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-xs text-text-secondary">Ambient Intensity</label>
              <span className="text-xs font-mono text-text-muted">{s.planarLightIntensity.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={0}
              max={2}
              step={0.1}
              value={s.planarLightIntensity}
              onChange={(e) => set({ planarLightIntensity: Number(e.target.value) })}
              className="w-full"
            />
            <div className="flex justify-between text-[10px] text-text-muted/50">
              <span>Weak</span>
              <span>Strong</span>
            </div>
          </div>

          {/* Environment Preset */}
          <div>
            <label className="text-xs text-text-secondary block mb-1">Environment</label>
            <select
              value={s.environmentPreset}
              onChange={(e) => set({ environmentPreset: e.target.value })}
              className="w-full bg-bg-tertiary border border-border rounded px-3 py-1.5 text-sm"
            >
              {ENV_PRESETS.map((p) => (
                <option key={p.value} value={p.value}>{p.label}</option>
              ))}
            </select>
          </div>

          {/* Reset */}
          <button
            onClick={() => onChange({ ...DEFAULT_VIEWER_SETTINGS })}
            className="flex items-center gap-1 text-xs text-text-muted hover:text-text-secondary transition-colors"
          >
            <RotateCcw className="w-3 h-3" />
            Reset to defaults
          </button>
        </div>
      )}
    </div>
  );
}
