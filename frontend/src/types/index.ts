export type ModelType = 'trellis-image-to-3d' | 'trellis-text-to-3d' | 'hunyuan-image-to-3d' | 'sam3d-image-to-3d' | 'edited';
export type ExportFormat = 'glb' | 'obj' | 'stl' | 'ply';
export type TaskStatus = 'queued' | 'processing' | 'extracting' | 'completed' | 'failed' | 'cancelled';
export type InputMode = 'single' | 'multi';
export type MultiImageMode = 'stochastic' | 'multidiffusion';
export type TextMode = 'generate' | 'edit';
export type QualityPreset = 'draft' | 'standard' | 'high';
export type EngineName = 'trellis' | 'hunyuan' | 'sam3d';

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

// -- SAM 3D Objects Quality Presets -------------------------

export const SAM3D_QUALITY_PRESETS: { id: QualityPreset; label: string; desc: string; settings: Partial<GenerationSettings> }[] = [
  { id: 'draft', label: 'Draft', desc: 'Fast preview', settings: { sam3dStage1Steps: 15, sam3dStage2Steps: 15 } },
  { id: 'standard', label: 'Standard', desc: 'Balanced quality', settings: { sam3dStage1Steps: 25, sam3dStage2Steps: 25 } },
  { id: 'high', label: 'High', desc: 'Best quality', settings: { sam3dStage1Steps: 40, sam3dStage2Steps: 40 } },
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
    id: 'sam3d',
    name: 'SAM 3D Objects',
    desc: 'Meta — Full 3D reconstruction with texture baking',
    color: '#81C784',
    available: true,
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
  targetFaceCount: number;  // 0 = no decimation, else target face count
  removeFloaters: boolean;  // remove disconnected mesh components
  roughnessOffset: number;  // -1.0 to 1.0, added to roughness texture
  metallicScale: number;    // 0.0 to 2.0, multiplied with metallic texture
  // SAM 3D Objects params
  sam3dStage1Steps: number;
  sam3dStage2Steps: number;
  sam3dTextureBaking: boolean;
  sam3dVertexColor: boolean;
}

export interface MaterialPreset {
  id: string;
  label: string;
  desc: string;
  roughnessOffset: number;
  metallicScale: number;
}

export const MATERIAL_PRESETS: MaterialPreset[] = [
  { id: 'default',  label: 'Default',     desc: 'AI-generated values',       roughnessOffset: 0.0,   metallicScale: 1.0  },
  { id: 'glossy',   label: 'Glossy',      desc: 'Shiny smooth surface',      roughnessOffset: -0.8,  metallicScale: 1.0  },
  { id: 'matte',    label: 'Matte',       desc: 'Flat diffuse finish',       roughnessOffset: 0.8,   metallicScale: 0.0  },
  { id: 'metal',    label: 'Metal',       desc: 'Polished metallic',         roughnessOffset: -0.7,  metallicScale: 2.0  },
  { id: 'plastic',  label: 'Plastic',     desc: 'Smooth non-metallic',       roughnessOffset: -0.4,  metallicScale: 0.0  },
  { id: 'clay',     label: 'Clay',        desc: 'Rough, fully matte',        roughnessOffset: 1.0,   metallicScale: 0.0  },
  { id: 'glass',    label: 'Glass',       desc: 'Mirror-smooth, subtle',     roughnessOffset: -1.0,  metallicScale: 0.4  },
  { id: 'rubber',   label: 'Rubber',      desc: 'Soft matte, no metal',      roughnessOffset: 0.6,   metallicScale: 0.0  },
];

export interface ViewerSettings {
  lightPositionX: number;  // -1 to 1 (left/right)
  lightPositionY: number;  // -1 to 1 (bottom/top)
  lightColor: string;      // hex color
  spotlightIntensity: number;  // 0 to 2
  planarLightIntensity: number;  // 0 to 2
  environmentPreset: string;
}

export const DEFAULT_VIEWER_SETTINGS: ViewerSettings = {
  lightPositionX: 0,
  lightPositionY: 1,
  lightColor: '#ffffff',
  spotlightIntensity: 0.4,
  planarLightIntensity: 0.3,
  environmentPreset: 'studio',
};

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
  generation_time_seconds: number | null;
  created_at: string;
  source_model?: string | null;
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
  targetFaceCount: 0,
  removeFloaters: true,
  roughnessOffset: 0.0,
  metallicScale: 1.0,
  // SAM 3D Objects defaults
  sam3dStage1Steps: 25,
  sam3dStage2Steps: 25,
  sam3dTextureBaking: true,
  sam3dVertexColor: false,
};

export const DEFAULT_EXPORT_SETTINGS: ExportSettings = {
  formats: ['glb'],
  meshSimplify: 0.95,
  textureSize: 1024,
};
