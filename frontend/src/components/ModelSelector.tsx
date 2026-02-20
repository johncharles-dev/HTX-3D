import { Check } from 'lucide-react';
import { MODELS, type ModelDef } from '../types';

type Tab = 'image' | 'text';

interface Props {
  activeTab: Tab;
  selectedModels: string[];
  onToggle: (modelId: string) => void;
}

export default function ModelSelector({ activeTab, selectedModels, onToggle }: Props) {
  // Filter models by what the current tab supports
  const available = MODELS.filter((m) =>
    activeTab === 'image' ? m.supportsImage : m.supportsText,
  );

  return (
    <div>
      <label className="text-xs text-text-secondary block mb-2">
        Select Models
        <span className="text-text-muted ml-1">(runs sequentially)</span>
      </label>
      <div className="space-y-2">
        {available.map((model) => (
          <ModelCard
            key={model.id}
            model={model}
            selected={selectedModels.includes(model.id)}
            onToggle={() => model.available && onToggle(model.id)}
          />
        ))}
      </div>
    </div>
  );
}

function ModelCard({ model, selected, onToggle }: { model: ModelDef; selected: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      disabled={!model.available}
      className={`w-full text-left p-3 rounded-lg border transition-all
        ${!model.available ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}
        ${selected ? 'bg-opacity-10' : 'bg-bg-tertiary hover:border-border-hover'}
      `}
      style={{
        borderColor: selected ? model.color : undefined,
        backgroundColor: selected ? `${model.color}11` : undefined,
      }}
    >
      <div className="flex items-center gap-2.5">
        {/* Checkbox */}
        <div
          className="w-4 h-4 rounded flex items-center justify-center shrink-0 transition-colors"
          style={{
            border: `1.5px solid ${selected ? model.color : 'rgba(255,255,255,0.15)'}`,
            backgroundColor: selected ? `${model.color}33` : 'transparent',
          }}
        >
          {selected && <Check className="w-3 h-3" style={{ color: model.color }} />}
        </div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold" style={{ color: selected ? model.color : undefined }}>
              {model.name}
            </span>
            {!model.available && (
              <span className="text-[9px] px-1.5 py-0.5 rounded bg-bg-tertiary text-text-muted border border-border">
                COMING SOON
              </span>
            )}
          </div>
          <p className="text-[10px] text-text-muted truncate">{model.desc}</p>
        </div>

        {/* Color dot */}
        <div
          className="w-2 h-2 rounded-full shrink-0"
          style={{ backgroundColor: model.color, opacity: selected ? 1 : 0.3 }}
        />
      </div>
    </button>
  );
}
