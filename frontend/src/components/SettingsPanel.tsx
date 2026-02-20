import { Shuffle, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { useState } from 'react';
import type { GenerationSettings, ExportSettings, ExportFormat } from '../types';
import { QUALITY_PRESETS } from '../types';

interface Props {
  generation: GenerationSettings;
  exportSettings: ExportSettings;
  onGenerationChange: (s: GenerationSettings) => void;
  onExportChange: (s: ExportSettings) => void;
}

function Tooltip({ text }: { text: string }) {
  return (
    <span className="group relative inline-block ml-1">
      <Info className="w-3.5 h-3.5 text-text-muted inline cursor-help" />
      <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs bg-bg-primary border border-border rounded shadow-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
        {text}
      </span>
    </span>
  );
}

function SliderField({
  label,
  tooltip,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  tooltip: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs text-text-secondary">
          {label}
          <Tooltip text={tooltip} />
        </label>
        <span className="text-xs font-mono text-text-muted">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full"
      />
    </div>
  );
}

const FORMAT_OPTIONS: { value: ExportFormat; label: string; desc: string }[] = [
  { value: 'glb', label: 'GLB', desc: 'Full PBR textured mesh' },
  { value: 'obj', label: 'OBJ', desc: 'Universal format with textures (zipped)' },
  { value: 'stl', label: 'STL', desc: '3D printing, geometry only' },
  { value: 'ply', label: 'PLY', desc: 'Gaussian splat point cloud' },
];

export default function SettingsPanel({ generation, exportSettings, onGenerationChange, onExportChange }: Props) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const g = generation;
  const setG = (patch: Partial<GenerationSettings>) => onGenerationChange({ ...g, ...patch });

  // Detect current quality preset
  const currentPreset = QUALITY_PRESETS.find(
    (p) => p.settings.ssSteps === g.ssSteps && p.settings.slatSteps === g.slatSteps,
  )?.id || null;

  const toggleFormat = (fmt: ExportFormat) => {
    const has = exportSettings.formats.includes(fmt);
    const next = has
      ? exportSettings.formats.filter((f) => f !== fmt)
      : [...exportSettings.formats, fmt];
    if (next.length > 0) onExportChange({ ...exportSettings, formats: next });
  };

  return (
    <div className="space-y-4">
      {/* Quality Preset */}
      <div>
        <label className="text-xs text-text-secondary block mb-2">
          Quality Preset
          <Tooltip text="Controls generation steps. Higher quality = slower but more detailed results." />
        </label>
        <div className="flex gap-1">
          {QUALITY_PRESETS.map((preset) => (
            <button
              key={preset.id}
              onClick={() => setG(preset.settings)}
              className={`flex-1 text-xs py-1.5 rounded-lg border transition-colors
                ${currentPreset === preset.id
                  ? 'border-accent bg-accent/10 text-accent'
                  : 'border-border bg-bg-tertiary text-text-muted hover:border-border-hover'}`}
              title={preset.desc}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Seed */}
      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-xs text-text-secondary">
            Seed
            <Tooltip text="Controls randomness. Same seed + same settings = same result." />
          </label>
          <button
            onClick={() => setG({ randomizeSeed: !g.randomizeSeed })}
            className={`flex items-center gap-1 text-xs px-2 py-0.5 rounded transition-colors
              ${g.randomizeSeed ? 'bg-accent/20 text-accent' : 'bg-bg-tertiary text-text-muted'}`}
          >
            <Shuffle className="w-3 h-3" />
            Random
          </button>
        </div>
        <input
          type="number"
          value={g.seed}
          onChange={(e) => setG({ seed: Math.max(0, Number(e.target.value)) })}
          disabled={g.randomizeSeed}
          className="w-full bg-bg-tertiary border border-border rounded px-3 py-1.5 text-sm disabled:opacity-40"
        />
      </div>

      {/* Export Formats */}
      <div>
        <label className="text-xs text-text-secondary block mb-2">
          Export Formats
          <Tooltip text="Select which file formats to generate after mesh extraction." />
        </label>
        <div className="flex flex-wrap gap-2">
          {FORMAT_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              onClick={() => toggleFormat(opt.value)}
              className={`group relative text-xs px-3 py-1.5 rounded-lg border transition-colors
                ${exportSettings.formats.includes(opt.value)
                  ? 'border-accent bg-accent/10 text-accent'
                  : 'border-border bg-bg-tertiary text-text-muted hover:border-border-hover'}`}
            >
              {opt.label}
              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 text-xs bg-bg-primary border border-border rounded shadow-lg whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-10">
                {opt.desc}
              </span>
            </button>
          ))}
        </div>
      </div>

      {/* Advanced Settings */}
      <button
        onClick={() => setAdvancedOpen(!advancedOpen)}
        className="flex items-center gap-1 text-xs text-text-muted hover:text-text-secondary transition-colors w-full"
      >
        {advancedOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        Advanced Settings
      </button>

      {advancedOpen && (
        <div className="space-y-3 pl-2 border-l-2 border-border">
          <p className="text-xs text-text-muted">Stage 1: Sparse Structure</p>
          <SliderField
            label="Steps"
            tooltip="Number of diffusion sampling steps. More steps = finer structure, slower generation."
            value={g.ssSteps}
            min={1}
            max={50}
            step={1}
            onChange={(v) => setG({ ssSteps: v })}
          />
          <SliderField
            label="Guidance"
            tooltip="How strongly to follow the input. Higher = more detail but may introduce artifacts."
            value={g.ssGuidance}
            min={0}
            max={10}
            step={0.5}
            onChange={(v) => setG({ ssGuidance: v })}
          />

          <p className="text-xs text-text-muted mt-3">Stage 2: Structured Latent</p>
          <SliderField
            label="Steps"
            tooltip="Sampling steps for appearance generation. Usually matches Stage 1."
            value={g.slatSteps}
            min={1}
            max={50}
            step={1}
            onChange={(v) => setG({ slatSteps: v })}
          />
          <SliderField
            label="Guidance"
            tooltip="Appearance guidance strength. Typically lower than Stage 1."
            value={g.slatGuidance}
            min={0}
            max={10}
            step={0.5}
            onChange={(v) => setG({ slatGuidance: v })}
          />

          <p className="text-xs text-text-muted mt-3">Mesh Export</p>
          <SliderField
            label="Simplification"
            tooltip="Ratio of faces to keep. Higher = more simplified mesh, smaller file."
            value={exportSettings.meshSimplify}
            min={0.8}
            max={0.99}
            step={0.01}
            onChange={(v) => onExportChange({ ...exportSettings, meshSimplify: v })}
          />
          <div>
            <label className="text-xs text-text-secondary">
              Texture Size
              <Tooltip text="Resolution of baked texture. Higher = better detail, larger file." />
            </label>
            <select
              value={exportSettings.textureSize}
              onChange={(e) => onExportChange({ ...exportSettings, textureSize: Number(e.target.value) })}
              className="w-full mt-1 bg-bg-tertiary border border-border rounded px-3 py-1.5 text-sm"
            >
              <option value={512}>512px</option>
              <option value={1024}>1024px</option>
              <option value={2048}>2048px</option>
              <option value={4096}>4096px (high VRAM)</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
}
