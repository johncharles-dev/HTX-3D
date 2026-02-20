import { Box, Image, Type, History, Activity } from 'lucide-react';
import type { HealthStatus } from '../types';

type Tab = 'image' | 'text' | 'gallery';

interface Props {
  activeTab: Tab;
  onTabChange: (tab: Tab) => void;
  health: HealthStatus | null;
}

const TABS: { id: Tab; label: string; icon: React.ReactNode; desc: string }[] = [
  { id: 'image', label: 'Image to 3D', icon: <Image className="w-4 h-4" />, desc: 'Generate from images' },
  { id: 'text', label: 'Text to 3D', icon: <Type className="w-4 h-4" />, desc: 'Generate from text' },
  { id: 'gallery', label: 'Gallery', icon: <History className="w-4 h-4" />, desc: 'Past generations' },
];

export default function Header({ activeTab, onTabChange, health }: Props) {
  return (
    <header className="bg-bg-secondary border-b border-border px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Logo + Title */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
            <Box className="w-5 h-5 text-accent" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary">HTX 3D Generation Tool</h1>
            <p className="text-xs text-text-muted">Powered by TRELLIS</p>
          </div>
        </div>

        {/* Tab Navigation */}
        <nav className="flex gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              title={tab.desc}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm transition-colors
                ${activeTab === tab.id
                  ? 'bg-accent/10 text-accent'
                  : 'text-text-muted hover:text-text-secondary hover:bg-bg-tertiary'}`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </nav>

        {/* GPU Status */}
        <div className="flex items-center gap-2">
          {health?.gpu.available ? (
            <div className="flex items-center gap-1.5 text-xs text-text-muted">
              <Activity className="w-3.5 h-3.5 text-success" />
              <span>{health.gpu.name}</span>
              <span className="text-text-muted/50">|</span>
              <span>{health.gpu.vram_gb} GB</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 text-xs text-error">
              <Activity className="w-3.5 h-3.5" />
              No GPU
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
