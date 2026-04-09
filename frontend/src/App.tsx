import { useState, useEffect } from 'react';
import Header from './components/Header';
import ImageUpload from './components/ImageUpload';
import SettingsPanel from './components/SettingsPanel';
import ModelSelector from './components/ModelSelector';
import ModelViewer from './components/ModelViewer';
import ProgressBar from './components/ProgressBar';
import ExportPanel from './components/ExportPanel';
import Gallery from './components/Gallery';
import SegmentationWorkspace from './components/SegmentationWorkspace';
import { Sparkles, Images, Type, Pencil, Upload, Wand2, Check } from 'lucide-react';
import {
  generateFromImage,
  generateFromMultiImage,
  generateFromText,
  editWithText,
  getTaskStatus,
  getHealth,
  connectProgress,
} from './api/client';
import type {
  HealthStatus,
  ExportFile,
  ProgressUpdate,
  TaskStatus,
  InputMode,
  MultiImageMode,
  TextMode,
  ModelResult,
  GenerationSettings,
  ExportSettings,
  EngineName,
} from './types';
import {
  MODELS,
  DEFAULT_GENERATION_SETTINGS,
  DEFAULT_EXPORT_SETTINGS,
} from './types';

type Tab = 'image' | 'text' | 'gallery';

export default function App() {
  // -- State -----------------------------------------------
  const [activeTab, setActiveTab] = useState<Tab>('image');
  const [health, setHealth] = useState<HealthStatus | null>(null);

  // Model selection (only for image tab)
  const [selectedModels, setSelectedModels] = useState<string[]>(['trellis']);

  // Image inputs
  const [inputMode, setInputMode] = useState<InputMode>('single');
  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [multiImageMode, setMultiImageMode] = useState<MultiImageMode>('stochastic');

  // Text input
  const [textPrompt, setTextPrompt] = useState('');
  const [textMode, setTextMode] = useState<TextMode>('generate');
  const [editMeshFile, setEditMeshFile] = useState<File | null>(null);
  const [editBaseTaskId, setEditBaseTaskId] = useState<string | null>(null);

  // Settings
  const [genSettings, setGenSettings] = useState<GenerationSettings>(DEFAULT_GENERATION_SETTINGS);
  const [exportSettings, setExportSettings] = useState<ExportSettings>(DEFAULT_EXPORT_SETTINGS);

  // Multi-model generation state
  const [isGenerating, setIsGenerating] = useState(false);
  const [modelResults, setModelResults] = useState<ModelResult[]>([]);
  const [activeResultModel, setActiveResultModel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Segmentation state
  const [showSegmentation, setShowSegmentation] = useState(false);
  const [segmentedImagePath, setSegmentedImagePath] = useState<string | null>(null);
  // Per-engine: which engines should use the segmented image (sam3d always true)
  const [useSegForEngine, setUseSegForEngine] = useState<Record<string, boolean>>({});

  // 3D viewer
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [viewerFormat, setViewerFormat] = useState<string>('glb');
  const [galleryExports, setGalleryExports] = useState<ExportFile[]>([]);

  // -- Derived ---------------------------------------------
  const activeModelResult = modelResults.find((mr) => mr.modelId === activeResultModel);
  const activeResult = activeModelResult?.result || null;
  const activeProgress = activeModelResult?.progress || null;
  const activeStatus = activeModelResult?.status || null;

  // For text tab, always use trellis
  const effectiveModels = activeTab === 'text' ? ['trellis'] : selectedModels;

  // Derive selected engines from selected models (for settings panel)
  const selectedEngines: EngineName[] = activeTab === 'text'
    ? ['trellis']
    : [...new Set(selectedModels.filter((id): id is EngineName => id === 'trellis' || id === 'hunyuan' || id === 'sam3d'))];

  // -- Health Check ----------------------------------------
  useEffect(() => {
    getHealth().then((h) => {
      setHealth(h);
      // Auto-select first registered engine if current selection is unavailable
      if (h.engines_registered && !h.engines_registered.includes(selectedModels[0])) {
        const first = h.engines_registered[0];
        if (first) setSelectedModels([first]);
      }
    }).catch(() => {});
    const interval = setInterval(() => {
      getHealth().then(setHealth).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  // -- Model Toggle ----------------------------------------
  const toggleModel = (modelId: string) => {
    setSelectedModels((prev) =>
      prev.includes(modelId)
        ? prev.filter((m) => m !== modelId)
        : [...prev, modelId],
    );
  };

  // -- Sequential Multi-Model Generation -------------------
  const runModelsSequentially = async (
    submitFn: (modelId: string) => Promise<{ task_id: string; status: TaskStatus }>,
  ) => {
    // Only run available and registered models
    const registered = health?.engines_registered ?? [];
    const toRun = effectiveModels.filter((id) => {
      const model = MODELS.find((m) => m.id === id);
      return model?.available && (registered.length === 0 || registered.includes(id));
    });
    if (toRun.length === 0) {
      throw new Error('No available models selected');
    }

    // Initialize result slots
    const initialResults: ModelResult[] = toRun.map((modelId) => ({
      modelId,
      taskId: '',
      status: 'queued' as TaskStatus,
      progress: null,
      result: null,
      error: null,
    }));
    setModelResults(initialResults);
    setActiveResultModel(toRun[0]);

    // Run each model sequentially
    for (const modelId of toRun) {
      try {
        const response = await submitFn(modelId);

        // Update task ID
        setModelResults((prev) =>
          prev.map((mr) =>
            mr.modelId === modelId ? { ...mr, taskId: response.task_id, status: response.status } : mr,
          ),
        );

        // Watch progress and wait for completion
        await new Promise<void>((resolve) => {
          const ws = connectProgress(
            response.task_id,
            (update: ProgressUpdate) => {
              setModelResults((prev) =>
                prev.map((mr) =>
                  mr.modelId === modelId ? { ...mr, status: update.status, progress: update } : mr,
                ),
              );

              if (update.status === 'completed' || update.status === 'failed') {
                getTaskStatus(response.task_id).then((res) => {
                  setModelResults((prev) =>
                    prev.map((mr) =>
                      mr.modelId === modelId
                        ? { ...mr, status: res.status, result: res, error: res.error }
                        : mr,
                    ),
                  );
                  if (res.status === 'completed') {
                    setEditBaseTaskId(res.task_id);
                    // Set viewer URL immediately when result arrives
                    const glb = res.exports.find((e) => e.format === 'glb');
                    if (glb) {
                      setViewerUrl(glb.url);
                      setViewerFormat('glb');
                    } else if (res.exports.length > 0) {
                      setViewerUrl(res.exports[0].url);
                      setViewerFormat(res.exports[0].format);
                    }
                  }
                  ws.close();
                  resolve();
                });
              }
            },
            () => resolve(),
          );
        });
      } catch (e: any) {
        setModelResults((prev) =>
          prev.map((mr) =>
            mr.modelId === modelId ? { ...mr, status: 'failed', error: e.message } : mr,
          ),
        );
      }
    }

    setIsGenerating(false);
  };

  // -- Generate Handler ------------------------------------
  const handleGenerate = async () => {
    setError(null);
    setGalleryExports([]);
    // Don't clear viewerUrl — keep previous result visible during generation
    setIsGenerating(true);

    try {
      if (activeTab === 'image') {
        if (imageFiles.length === 0) {
          throw new Error('Please upload an image');
        }

        await runModelsSequentially(async (modelId) => {
          const engine = modelId as EngineName;
          if (inputMode === 'multi' && imageFiles.length >= 2) {
            return generateFromMultiImage(imageFiles, multiImageMode, genSettings, exportSettings, engine);
          } else {
            // Per-engine: sam3d always uses segmentation, others check their toggle
            const engineUsesSeg = engine === 'sam3d' || (useSegForEngine[engine] !== false);
            const segPath = segmentedImagePath && engineUsesSeg ? segmentedImagePath : undefined;
            return generateFromImage(imageFiles[0], genSettings, exportSettings, engine, segPath);
          }
        });
      } else if (activeTab === 'text') {
        if (!textPrompt.trim()) {
          throw new Error('Please enter a text prompt');
        }

        if (textMode === 'edit') {
          if (!editMeshFile && !editBaseTaskId) {
            throw new Error('Upload a mesh file or generate a model first to edit');
          }
          await runModelsSequentially(async (_modelId) => {
            return editWithText(
              textPrompt,
              genSettings,
              exportSettings,
              editBaseTaskId || undefined,
              editMeshFile || undefined,
            );
          });
        } else {
          await runModelsSequentially(async (_modelId) => {
            return generateFromText(textPrompt, genSettings, exportSettings);
          });
        }
      }
    } catch (e: any) {
      setError(e.message);
      setIsGenerating(false);
    }
  };

  // -- Auto-select viewer when results change --------------
  useEffect(() => {
    if (!activeModelResult?.result) return;
    const res = activeModelResult.result;
    const glb = res.exports.find((e) => e.format === 'glb');
    if (glb) {
      setViewerUrl(glb.url);
      setViewerFormat('glb');
    } else if (res.exports.length > 0) {
      setViewerUrl(res.exports[0].url);
      setViewerFormat(res.exports[0].format);
    }
  }, [activeModelResult?.result]);

  // -- Switch viewer when active result model changes ------
  const handleSelectResultModel = (modelId: string) => {
    setActiveResultModel(modelId);
    const mr = modelResults.find((r) => r.modelId === modelId);
    if (mr?.result) {
      const glb = mr.result.exports.find((e) => e.format === 'glb');
      if (glb) {
        setViewerUrl(glb.url);
        setViewerFormat('glb');
      } else if (mr.result.exports.length > 0) {
        setViewerUrl(mr.result.exports[0].url);
        setViewerFormat(mr.result.exports[0].format);
      }
    }
  };

  // -- Gallery Preview -------------------------------------
  const handleGalleryPreview = (exports: ExportFile[]) => {
    setGalleryExports(exports);
    setModelResults([]);
    setActiveResultModel(null);
    const glb = exports.find((e) => e.format === 'glb');
    if (glb) {
      setViewerUrl(glb.url);
      setViewerFormat('glb');
    } else if (exports.length > 0) {
      setViewerUrl(exports[0].url);
      setViewerFormat(exports[0].format);
    }
    setActiveTab('image');
  };

  // -- Computed --------------------------------------------
  const hasResults = modelResults.length > 0;
  const displayExports = activeResult?.exports || galleryExports;

  // -- Render ----------------------------------------------
  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <Header activeTab={activeTab} onTabChange={setActiveTab} health={health} />

      {/* Segmentation modal — renders on top of everything */}
      {showSegmentation && imageFiles.length === 1 && (
        <SegmentationWorkspace
          imageFile={imageFiles[0]}
          onConfirm={(path) => {
            setSegmentedImagePath(path);
            setShowSegmentation(false);
          }}
          onCancel={() => setShowSegmentation(false)}
        />
      )}

      {activeTab === 'gallery' ? (
        <main className="flex-1 overflow-auto p-6">
          <Gallery onPreview={handleGalleryPreview} />
        </main>
      ) : (
        <main className="flex-1 flex overflow-hidden">
          {/* -- Left Panel: Input + Settings ---------------- */}
          <aside className="w-80 border-r border-border bg-bg-secondary flex flex-col overflow-y-auto">
            <div className="p-4 space-y-5 flex-1">

              {/* Model Selector — only for image tab */}
              {activeTab === 'image' && (
                <>
                  <ModelSelector
                    activeTab={activeTab}
                    selectedModels={selectedModels}
                    onToggle={toggleModel}
                    registeredEngines={health?.engines_registered}
                  />
                  <div className="border-t border-border" />
                </>
              )}

              {/* Text tab: TRELLIS-only indicator */}
              {activeTab === 'text' && (
                <div className="flex items-center gap-2 p-2.5 rounded-lg bg-[#4FC3F7]/10 border border-[#4FC3F7]/20">
                  <div className="w-2 h-2 rounded-full bg-[#4FC3F7]" />
                  <span className="text-xs font-medium text-[#4FC3F7]">TRELLIS</span>
                  <span className="text-xs text-text-muted ml-auto">Text-to-3D</span>
                </div>
              )}

              {activeTab === 'image' && (
                <>
                  {/* Input Mode Toggle */}
                  <div className="flex gap-1 bg-bg-tertiary rounded-lg p-1">
                    <button
                      onClick={() => { setInputMode('single'); setImageFiles([]); }}
                      className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs transition-colors
                        ${inputMode === 'single' ? 'bg-accent/10 text-accent' : 'text-text-muted hover:text-text-secondary'}`}
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      Single Image
                    </button>
                    <button
                      onClick={() => { setInputMode('multi'); setImageFiles([]); }}
                      className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs transition-colors
                        ${inputMode === 'multi' ? 'bg-accent/10 text-accent' : 'text-text-muted hover:text-text-secondary'}`}
                    >
                      <Images className="w-3.5 h-3.5" />
                      Multi Image
                    </button>
                  </div>

                  <ImageUpload
                    files={imageFiles}
                    onChange={(files) => {
                      setImageFiles(files);
                      setSegmentedImagePath(null);
                      setShowSegmentation(false);
                      setUseSegForEngine({});
                    }}
                    multiple={inputMode === 'multi'}
                    maxFiles={4}
                    isSegmented={!!segmentedImagePath}
                  />

                  {inputMode === 'multi' && imageFiles.length >= 2 && (
                    <div>
                      <label className="text-xs text-text-secondary block mb-1.5">
                        Fusion Mode
                      </label>
                      <select
                        value={multiImageMode}
                        onChange={(e) => setMultiImageMode(e.target.value as MultiImageMode)}
                        className="w-full bg-bg-tertiary border border-border rounded px-3 py-1.5 text-sm"
                      >
                        <option value="stochastic">Stochastic (faster, cycles views)</option>
                        <option value="multidiffusion">Multidiffusion (slower, averages all views)</option>
                      </select>
                    </div>
                  )}

                  {/* Segmentation — before segmenting */}
                  {inputMode === 'single' && imageFiles.length === 1 && !segmentedImagePath && !showSegmentation && (
                    <div className="space-y-2">
                      {selectedModels.includes('sam3d') && (
                        <div className="flex items-center gap-2 p-2 rounded-lg bg-[#81C784]/10 border border-[#81C784]/20">
                          <Wand2 className="w-3.5 h-3.5 text-[#81C784] shrink-0" />
                          <span className="text-xs text-[#81C784]">Segmentation required for SAM 3D Objects</span>
                        </div>
                      )}
                      <button
                        onClick={() => setShowSegmentation(true)}
                        className="w-full flex items-center justify-center gap-1.5 py-2 rounded-lg border border-accent/30 bg-accent/5 text-accent text-xs font-medium hover:bg-accent/10 transition-colors"
                      >
                        <Wand2 className="w-3.5 h-3.5" />
                        Segment Object
                      </button>
                      {!selectedModels.includes('sam3d') && (
                        <p className="text-[10px] text-text-muted">
                          Optional — selected engines have built-in background removal
                        </p>
                      )}
                    </div>
                  )}

                  {/* Segmentation — after segmenting: per-engine toggles */}
                  {segmentedImagePath && (
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 p-2.5 rounded-lg bg-success/10 border border-success/20">
                        <div className="w-2 h-2 rounded-full bg-success" />
                        <span className="text-xs text-success flex-1">Object segmented</span>
                        <button
                          onClick={() => { setSegmentedImagePath(null); setShowSegmentation(false); setUseSegForEngine({}); }}
                          className="text-xs text-text-muted hover:text-text-secondary"
                        >
                          Clear
                        </button>
                      </div>

                      {/* Per-engine segmentation usage toggles */}
                      {selectedModels.length > 1 && selectedModels.some(m => m === 'trellis' || m === 'hunyuan') && (
                        <div className="p-2.5 rounded-lg bg-bg-tertiary border border-border space-y-1.5">
                          <p className="text-[10px] text-text-muted font-medium">Use segmentation for:</p>
                          {selectedModels.map(modelId => {
                            const model = MODELS.find(m => m.id === modelId);
                            if (!model) return null;
                            const isSam3d = modelId === 'sam3d';
                            const isOn = isSam3d || (useSegForEngine[modelId] !== false);
                            return (
                              <button
                                key={modelId}
                                onClick={() => !isSam3d && setUseSegForEngine(prev => ({ ...prev, [modelId]: !isOn }))}
                                className={`w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-xs transition-colors ${
                                  isSam3d ? 'cursor-default' : 'cursor-pointer hover:bg-bg-primary/50'
                                }`}
                              >
                                <div
                                  className={`w-3.5 h-3.5 rounded flex items-center justify-center shrink-0 transition-colors ${
                                    isSam3d ? 'opacity-60' : ''
                                  }`}
                                  style={{
                                    border: `1.5px solid ${isOn ? model.color : 'rgba(255,255,255,0.15)'}`,
                                    backgroundColor: isOn ? `${model.color}33` : 'transparent',
                                  }}
                                >
                                  {isOn && <Check className="w-2.5 h-2.5" style={{ color: model.color }} />}
                                </div>
                                <span style={{ color: isOn ? model.color : undefined }}>{model.name}</span>
                                {isSam3d && <span className="text-[10px] text-text-muted ml-auto">required</span>}
                                {!isSam3d && !isOn && <span className="text-[10px] text-text-muted ml-auto">uses built-in rembg</span>}
                              </button>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </>
              )}

              {activeTab === 'text' && (
                <div className="space-y-4">
                  {/* Generate / Edit Mode Toggle */}
                  <div className="flex gap-1 bg-bg-tertiary rounded-lg p-1">
                    <button
                      onClick={() => setTextMode('generate')}
                      className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs transition-colors
                        ${textMode === 'generate' ? 'bg-accent/10 text-accent' : 'text-text-muted hover:text-text-secondary'}`}
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      Generate New
                    </button>
                    <button
                      onClick={() => setTextMode('edit')}
                      className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs transition-colors
                        ${textMode === 'edit' ? 'bg-accent/10 text-accent' : 'text-text-muted hover:text-text-secondary'}`}
                    >
                      <Pencil className="w-3.5 h-3.5" />
                      Edit Model
                    </button>
                  </div>

                  {/* Edit Mode: Base Model Source — BEFORE text prompt for visibility */}
                  {textMode === 'edit' && (
                    <div className="space-y-3 p-3 rounded-lg bg-bg-tertiary border border-border">
                      <label className="text-xs font-medium text-text-secondary block">
                        Base Model to Edit
                      </label>

                      {editBaseTaskId && (
                        <div className="flex items-center gap-2 p-2.5 rounded-lg bg-bg-primary border border-border">
                          <div className="w-2 h-2 rounded-full bg-success" />
                          <span className="text-xs text-text-secondary flex-1">
                            Last generation ({editBaseTaskId.slice(0, 8)}...)
                          </span>
                          <button
                            onClick={() => setEditBaseTaskId(null)}
                            className="text-xs text-text-muted hover:text-text-secondary"
                          >
                            Clear
                          </button>
                        </div>
                      )}

                      <div>
                        <p className="text-xs text-text-muted mb-1.5">
                          {editBaseTaskId ? 'Or upload a mesh file:' : 'Upload a mesh file (GLB, OBJ, PLY):'}
                        </p>
                        <label className="flex items-center gap-2 p-3 rounded-lg border border-dashed border-border hover:border-accent/50 cursor-pointer transition-colors">
                          <Upload className="w-4 h-4 text-text-muted" />
                          <span className="text-xs text-text-secondary flex-1">
                            {editMeshFile ? editMeshFile.name : 'Click to upload mesh file'}
                          </span>
                          <input
                            type="file"
                            accept=".glb,.obj,.ply,.stl"
                            className="hidden"
                            onChange={(e) => {
                              const file = e.target.files?.[0];
                              if (file) {
                                setEditMeshFile(file);
                                setEditBaseTaskId(null);
                              }
                            }}
                          />
                        </label>
                        {editMeshFile && (
                          <button
                            onClick={() => setEditMeshFile(null)}
                            className="text-xs text-text-muted hover:text-error mt-1"
                          >
                            Remove file
                          </button>
                        )}
                      </div>

                      {!editBaseTaskId && !editMeshFile && (
                        <p className="text-xs text-warning/80">
                          Generate a model first or upload a mesh file to edit
                        </p>
                      )}
                    </div>
                  )}

                  {/* Text Prompt */}
                  <div>
                    <label className="text-xs text-text-secondary block mb-1.5">
                      <Type className="w-3.5 h-3.5 inline mr-1" />
                      {textMode === 'generate' ? 'Describe the 3D object' : 'Describe the changes'}
                    </label>
                    <textarea
                      value={textPrompt}
                      onChange={(e) => setTextPrompt(e.target.value)}
                      placeholder={textMode === 'generate'
                        ? 'A detailed wooden treasure chest with gold trim and iron hinges...'
                        : 'Change the material to glossy red metal with chrome accents...'}
                      rows={4}
                      className="w-full bg-bg-tertiary border border-border rounded-lg px-3 py-2 text-sm resize-none placeholder:text-text-muted/50"
                    />
                    <p className="text-xs text-text-muted mt-1">
                      {textMode === 'generate'
                        ? 'Be specific about shape, material, and details'
                        : 'The base shape is preserved — only appearance/details change'}
                    </p>
                  </div>
                </div>
              )}

              <div className="border-t border-border pt-4">
                <SettingsPanel
                  generation={genSettings}
                  exportSettings={exportSettings}
                  onGenerationChange={setGenSettings}
                  onExportChange={setExportSettings}
                  selectedEngines={selectedEngines}
                />
              </div>
            </div>

            {/* Generate Button */}
            <div className="p-4 border-t border-border">
              <button
                onClick={handleGenerate}
                disabled={isGenerating || effectiveModels.filter(id => MODELS.find(m => m.id === id)?.available).length === 0}
                className="w-full py-3 rounded-xl font-medium text-sm transition-all
                  bg-accent hover:bg-accent-hover text-white
                  disabled:opacity-50 disabled:cursor-not-allowed
                  flex items-center justify-center gap-2"
              >
                {isGenerating ? (
                  <>
                    <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
                      <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                    </svg>
                    Generating...
                  </>
                ) : (
                  <>
                    {activeTab === 'text' && textMode === 'edit'
                      ? <><Pencil className="w-4 h-4" />Edit 3D Model</>
                      : <><Sparkles className="w-4 h-4" />Generate 3D Model</>
                    }
                    {activeTab === 'image' && selectedModels.length > 1 && (
                      <span className="text-xs opacity-60">({selectedModels.filter(id => MODELS.find(m => m.id === id)?.available).length} models)</span>
                    )}
                  </>
                )}
              </button>

              {error && (
                <p className="text-xs text-error mt-2 text-center">{error}</p>
              )}
            </div>
          </aside>

          {/* -- Center: 3D Viewer -------------------------- */}
          <section className="flex-1 flex flex-col min-w-0">
            {/* Result Tabs — always show when we have results */}
            {hasResults && (
              <div className="px-4 pt-3 pb-0">
                <div className="flex items-center gap-2">
                  <div className="flex gap-1 bg-bg-secondary rounded-lg p-1 border border-border">
                    {modelResults.map((mr) => {
                      const model = MODELS.find((m) => m.id === mr.modelId);
                      if (!model) return null;
                      const isActive = activeResultModel === mr.modelId;
                      const isLoading = mr.status === 'processing' || mr.status === 'extracting';
                      const isComplete = mr.status === 'completed';
                      const isFailed = mr.status === 'failed';

                      return (
                        <button
                          key={mr.modelId}
                          onClick={() => handleSelectResultModel(mr.modelId)}
                          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-all ${
                            isActive ? 'shadow-sm' : 'hover:bg-bg-tertiary'
                          }`}
                          style={{
                            backgroundColor: isActive ? `${model.color}20` : undefined,
                            color: isActive ? model.color : undefined,
                          }}
                        >
                          {/* Status indicator */}
                          <div
                            className={`w-2 h-2 rounded-full ${isLoading ? 'animate-pulse' : ''}`}
                            style={{
                              backgroundColor: isComplete ? '#22c55e' : isFailed ? '#ef4444' : isLoading ? model.color : 'rgba(255,255,255,0.15)',
                              boxShadow: isLoading ? `0 0 6px ${model.color}66` : undefined,
                            }}
                          />
                          <span className="font-medium">{model.name}</span>
                          {isLoading && mr.progress && (
                            <span className="opacity-60">{Math.round(mr.progress.progress * 100)}%</span>
                          )}
                          {isComplete && (
                            <span className="opacity-50 text-[10px]">Done</span>
                          )}
                        </button>
                      );
                    })}
                  </div>

                  {/* Generation stats */}
                  {activeResult?.generation_time_seconds && (
                    <span className="text-xs text-text-muted ml-auto">
                      {activeResult.generation_time_seconds}s
                    </span>
                  )}
                </div>
              </div>
            )}

            {/* 3D Viewer Area */}
            <div className="flex-1 min-h-0 p-4 relative">
              <ModelViewer url={viewerUrl} format={viewerFormat} exports={displayExports} />

              {/* Progress Overlay — floats on top of viewer */}
              {isGenerating && (
                <div className="absolute bottom-6 left-6 right-6 z-10">
                  <ProgressBar
                    progress={activeProgress || { task_id: '', stage: 'processing', progress: 0, message: 'Starting generation...', status: 'processing' as const }}
                    status={activeStatus || 'processing'}
                  />
                </div>
              )}

              {/* Stats overlay */}
              {activeResult && (
                <div className="absolute bottom-6 right-6 bg-bg-primary/80 backdrop-blur-sm border border-border rounded-lg px-3 py-2 text-[10px] leading-relaxed text-text-muted z-10">
                  <div>Seed: <span className="text-text-secondary">{activeResult.seed}</span></div>
                  {activeResult.generation_time_seconds && (
                    <div>Time: <span className="text-text-secondary">{activeResult.generation_time_seconds}s</span></div>
                  )}
                </div>
              )}

              {/* Drag hint */}
              {viewerUrl && (
                <div className="absolute bottom-6 left-6 text-[10px] text-text-muted/40 z-10">
                  Drag to orbit · Scroll to zoom
                </div>
              )}
            </div>

          </section>

          {/* -- Right Panel: Downloads --------------------- */}
          {displayExports.length > 0 && (
            <aside className="w-64 border-l border-border bg-bg-secondary p-4 overflow-y-auto">
              <ExportPanel
                exports={displayExports}
                generationTime={activeResult?.generation_time_seconds ?? null}
                seed={activeResult?.seed ?? null}
              />
            </aside>
          )}
        </main>
      )}
    </div>
  );
}
