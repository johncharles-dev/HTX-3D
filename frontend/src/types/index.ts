export type ModelType = 'trellis-image-to-3d' | 'trellis-text-to-3d' | 'hunyuan-image-to-3d';
export type ExportFormat = 'glb' | 'obj' | 'stl' | 'ply';
export type TaskStatus = 'queued' | 'processing' | 'extracting' | 'completed' | 'failed';
export type InputMode = 'single' | 'multi';
export type MultiImageMode = 'stochastic' | 'multidiffusion';
export type TextMode = 'generate' | 'edit';
export type QualityPreset = 'draft' | 'standard' | 'high';
export type EngineName = 'trellis' | 'hunyuan';

// -- TRELLIS Quality Presets --------------------------------

export const QUALITY_PRESETS: { id: QualityPreset; label: string; desc: string; settings: Partial<GenerationSettings> }[] = [
  { id: 'draft', label: 'Draft', desc: 'Fast preview', settings: { ssSteps: 8, slatSteps: 8 } },
  { id: 'standard', label: 'Standard', desc: 'Balanced quality', settings: { ssSteps: 12, slatSteps: 12 } },
  { id: 'high', label: 'High', desc: 'Best quality', settings: { ssSteps: 20, slatSteps: 20 } },
];

// -- Hunyuan Quality Presets --------------------------------

export const HUNYUAN_QUALITY_PRESETS: { id: QualityPreset; label: string; desc: string; settings: Partial<GenerationSettings> }[] = [
  { id: 'draft', label: 'Draft', desc: 'Fast preview', settings: { numInferenceSteps: 5 } },
  { id: 'standard', label: 'Standard', desc: 'Balanced quality', settings: { numInferenceSteps: 30 } },
  { id: 'high', label: 'High', desc: 'Best quality', settings: { numInferenceSteps: 50 } },
];

// -- Model Registry ----------------------------------------

export interface ModelDef {
  id: string;
  name: string;
  desc: string;
  color: string;
  available: boolean;
  supportsImage: boolean;
  supportsText: boolean;
  supportsEdit: boolean;
  supportsMultiView: boolean;
}

export const MODELS: ModelDef[] = [
  {
    id: 'trellis',
    name: 'TRELLIS',
    desc: 'Microsoft Research — 3D editing & relighting',
    color: '#4FC3F7',
    available: true,
    supportsImage: true,
    supportsText: true,
    supportsEdit: true,
    supportsMultiView: false,
  },
  {
    id: 'hunyuan',
    name: 'Hunyuan3D 2.1',
    desc: 'Tencent — PBR material generation',
    color: '#CE93D8',
    available: true,
    supportsImage: true,
    supportsText: false,
    supportsEdit: false,
    supportsMultiView: false,
  },
  {
    id: 'triposg',
    name: 'TripoSG',
    desc: 'Tripo AI — Fast single-image generation',
    color: '#81C784',
    available: false,
    supportsImage: true,
    supportsText: false,
    supportsEdit: false,
    supportsMultiView: false,
  },
];

// -- Data Types --------------------------------------------

export interface GenerationSettings {
  seed: number;
  randomizeSeed: boolean;
  // TRELLIS params
  ssSteps: number;
  ssGuidance: number;
  slatSteps: number;
  slatGuidance: number;
  // Hunyuan params
  numInferenceSteps: number;
  guidanceScale: number;
  octreeResolution: number;
  texture: boolean;
}

export interface ExportSettings {
  formats: ExportFormat[];
  meshSimplify: number;
  textureSize: number;
}

export interface ExportFile {
  format: ExportFormat;
  filename: string;
  url: string;
  size_bytes: number;
}

export interface TaskResponse {
  task_id: string;
  status: TaskStatus;
  message: string;
}

export interface GenerationResult {
  task_id: string;
  status: TaskStatus;
  model: ModelType;
  seed: number;
  preview_video_url: string | null;
  thumbnail_url: string | null;
  exports: ExportFile[];
  generation_time_seconds: number | null;
  error: string | null;
  created_at: string | null;
}

export interface ProgressUpdate {
  task_id: string;
  status: TaskStatus;
  stage: string;
  progress: number;
  message: string;
}

export interface GalleryItem {
  task_id: string;
  model: ModelType;
  thumbnail_url: string | null;
  preview_video_url: string | null;
  exports: ExportFile[];
  seed: number;
  created_at: string;
}

export interface HealthStatus {
  status: string;
  gpu: {
    available: boolean;
    name: string | null;
    compute_capability: string | null;
    vram_gb: number;
    is_blackwell: boolean;
  };
  models_loaded: string[];
  engines_registered: string[];
  active_engine: string | null;
  queue_size: number;
}

// Per-model result when running multiple models
export interface ModelResult {
  modelId: string;
  taskId: string;
  status: TaskStatus;
  progress: ProgressUpdate | null;
  result: GenerationResult | null;
  error: string | null;
}

export const DEFAULT_GENERATION_SETTINGS: GenerationSettings = {
  seed: 42,
  randomizeSeed: true,
  // TRELLIS defaults
  ssSteps: 12,
  ssGuidance: 7.5,
  slatSteps: 12,
  slatGuidance: 3.0,
  // Hunyuan defaults
  numInferenceSteps: 30,
  guidanceScale: 5.5,
  octreeResolution: 256,
  texture: true,
};

export const DEFAULT_EXPORT_SETTINGS: ExportSettings = {
  formats: ['glb'],
  meshSimplify: 0.95,
  textureSize: 1024,
};
