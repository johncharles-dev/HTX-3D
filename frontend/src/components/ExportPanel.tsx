import { Download, FileBox } from 'lucide-react';
import type { ExportFile } from '../types';

interface Props {
  exports: ExportFile[];
  generationTime: number | null;
  seed: number | null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

const FORMAT_LABELS: Record<string, { label: string; color: string }> = {
  glb: { label: 'GLB', color: 'bg-indigo-500/20 text-indigo-400' },
  obj: { label: 'OBJ (ZIP)', color: 'bg-emerald-500/20 text-emerald-400' },
  stl: { label: 'STL', color: 'bg-orange-500/20 text-orange-400' },
  ply: { label: 'PLY', color: 'bg-purple-500/20 text-purple-400' },
};

const CATEGORIES: { key: string; label: string; formats: string[] }[] = [
  { key: 'textured', label: 'Textured', formats: ['glb', 'obj'] },
  { key: 'geometry', label: 'Geometry', formats: ['stl'] },
  { key: 'pointcloud', label: 'Point Cloud', formats: ['ply'] },
];

function ExportLink({ exp }: { exp: ExportFile }) {
  const info = FORMAT_LABELS[exp.format] || { label: exp.format.toUpperCase(), color: 'bg-gray-500/20 text-gray-400' };
  return (
    <a
      href={exp.url}
      download={exp.filename}
      className="flex items-center gap-3 p-2.5 rounded-lg bg-bg-tertiary border border-border hover:border-border-hover transition-colors group"
    >
      <span className={`text-xs font-medium px-2 py-0.5 rounded ${info.color}`}>
        {info.label}
      </span>
      <span className="text-sm text-text-secondary flex-1 truncate">{exp.filename}</span>
      <span className="text-xs text-text-muted">{formatBytes(exp.size_bytes)}</span>
      <Download className="w-4 h-4 text-text-muted group-hover:text-accent transition-colors" />
    </a>
  );
}

export default function ExportPanel({ exports, generationTime, seed }: Props) {
  if (exports.length === 0) return null;

  // Group exports into categories
  const grouped = CATEGORIES
    .map((cat) => ({
      ...cat,
      files: exports.filter((exp) => cat.formats.includes(exp.format)),
    }))
    .filter((cat) => cat.files.length > 0);

  // If only one category, skip headers
  const showHeaders = grouped.length > 1;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-text-primary flex items-center gap-1.5">
          <FileBox className="w-4 h-4" />
          Downloads
        </h3>
        {generationTime && (
          <span className="text-xs text-text-muted">{generationTime}s</span>
        )}
      </div>

      {seed !== null && (
        <p className="text-xs text-text-muted">Seed: {seed}</p>
      )}

      <div className="space-y-3">
        {grouped.map((cat) => (
          <div key={cat.key} className="space-y-1.5">
            {showHeaders && (
              <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted/60 px-1">
                {cat.label}
              </p>
            )}
            <div className="space-y-1.5">
              {cat.files.map((exp) => (
                <ExportLink key={exp.format} exp={exp} />
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
