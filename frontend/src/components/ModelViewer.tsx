import { Suspense, useRef, useMemo, useState, useCallback, useEffect, Component, type ReactNode } from 'react';
import { Canvas, useLoader, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Center, Environment } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { AlertTriangle, Box, Circle, Grid3x3, Dot, Eraser, Undo2, Redo2, Check, X, Minus, Plus, Triangle } from 'lucide-react';
import * as THREE from 'three';
import type { ExportFile, ViewerSettings } from '../types';
import { DEFAULT_VIEWER_SETTINGS } from '../types';
import { EditableModel } from './MeshEraser';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

type ViewMode = 'textured' | 'mesh' | 'solid' | 'wireframe' | 'pointcloud';

const BG_COLOR = '#1a1d27';

class ViewerErrorBoundary extends Component<
  { children: ReactNode; onReset?: () => void },
  { hasError: boolean; error: string }
> {
  constructor(props: { children: ReactNode; onReset?: () => void }) {
    super(props);
    this.state = { hasError: false, error: '' };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error: error.message };
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="absolute inset-0 flex items-center justify-center rounded-xl border border-border" style={{ background: BG_COLOR }}>
          <div className="text-center p-6">
            <AlertTriangle className="w-10 h-10 text-warning mx-auto mb-3" />
            <p className="text-sm text-text-secondary mb-1">3D Viewer Error</p>
            <p className="text-xs text-text-muted mb-3 max-w-xs">{this.state.error || 'Failed to load 3D model'}</p>
            <button
              onClick={() => { this.setState({ hasError: false, error: '' }); this.props.onReset?.(); }}
              className="text-xs px-3 py-1.5 rounded-lg bg-accent/10 text-accent hover:bg-accent/20 transition-colors"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function SceneInit() {
  const { scene, gl } = useThree();
  scene.background = new THREE.Color(BG_COLOR);
  gl.setClearColor(BG_COLOR, 1);
  return null;
}

function InCanvasSpinner() {
  const ref = useRef<THREE.Mesh>(null!);
  useFrame((_, delta) => { if (ref.current) ref.current.rotation.z -= delta * 2; });
  return (
    <mesh ref={ref} position={[0, 0, 0]}>
      <torusGeometry args={[0.3, 0.05, 8, 32, Math.PI * 1.5]} />
      <meshBasicMaterial color="#6366f1" />
    </mesh>
  );
}

function SafeEnvironment({ preset }: { preset: string }) {
  try {
    return <Environment preset={preset as any} />;
  } catch {
    return null;
  }
}

class EnvironmentErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() { return this.state.failed ? null : this.props.children; }
}

function GLBModel({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  return <Center><primitive object={gltf.scene.clone()} /></Center>;
}

function GLBSolid({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  const scene = useMemo(() => {
    const mat = new THREE.MeshNormalMaterial();
    const cloned = gltf.scene.clone();
    cloned.traverse((child) => { if (child instanceof THREE.Mesh) child.material = mat; });
    return cloned;
  }, [gltf]);
  return <Center><primitive object={scene} /></Center>;
}

function GLBWireframe({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  const scene = useMemo(() => {
    const cloned = gltf.scene.clone();
    cloned.traverse((child) => {
      if (child instanceof THREE.Mesh) child.material = new THREE.MeshBasicMaterial({ wireframe: true, color: new THREE.Color('#6366f1') });
    });
    return cloned;
  }, [gltf]);
  return <Center><primitive object={scene} /></Center>;
}

function GLBMesh({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  const scene = useMemo(() => {
    const mat = new THREE.MeshStandardMaterial({ color: '#b0b0b0', flatShading: true, roughness: 0.7, metalness: 0.0 });
    const cloned = gltf.scene.clone();
    cloned.traverse((child) => { if (child instanceof THREE.Mesh) child.material = mat; });
    return cloned;
  }, [gltf]);
  return <Center><primitive object={scene} /></Center>;
}

function STLModel({ url }: { url: string }) {
  const geometry = useLoader(STLLoader, url);
  return <Center><mesh geometry={geometry}><meshStandardMaterial color="#a0a0a0" /></mesh></Center>;
}

function PLYModel({ url }: { url: string }) {
  const geometry = useLoader(PLYLoader, url);
  geometry.computeVertexNormals();
  return <Center><group scale={[1, -1, 1]}><points geometry={geometry}><pointsMaterial size={0.005} vertexColors /></points></group></Center>;
}

function OBJModel({ url }: { url: string }) {
  const obj = useLoader(OBJLoader, url);
  return <Center><primitive object={obj.clone()} /></Center>;
}

interface Props {
  url: string | null;
  format?: string;
  autoRotate?: boolean;
  exports?: ExportFile[];
  viewerSettings?: ViewerSettings;
  /** Called when user finishes editing — passes a blob URL of the edited GLB */
  onEditedModel?: (blobUrl: string) => void;
}

const VIEW_MODES: { id: ViewMode; label: string; icon: typeof Box }[] = [
  { id: 'textured', label: 'Textured', icon: Box },
  { id: 'mesh', label: 'Mesh', icon: Triangle },
  { id: 'solid', label: 'Solid', icon: Circle },
  { id: 'wireframe', label: 'Wireframe', icon: Grid3x3 },
  { id: 'pointcloud', label: 'Point Cloud', icon: Dot },
];

export default function ModelViewer({ url, format = 'glb', autoRotate = true, exports = [], viewerSettings = DEFAULT_VIEWER_SETTINGS, onEditedModel }: Props) {
  const [canvasKey, setCanvasKey] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>('textured');
  const [eraserActive, setEraserActive] = useState(false);
  const [brushSize, setBrushSize] = useState(0.03);
  const [eraseCount, setEraseCount] = useState(0);
  const [eraserResetKey, setEraserResetKey] = useState(0);
  const [undoSignal, setUndoSignal] = useState(0);
  const editableGroupRef = useRef<THREE.Group | null>(null);

  const resetCanvas = useCallback(() => {
    setCanvasKey((k) => k + 1);
  }, []);

  useEffect(() => {
    setViewMode('textured');
    setEraserActive(false);
    setEraseCount(0);
    setEraserResetKey(0);
    setUndoSignal(0);
  }, [url]);

  // Enter eraser mode
  const enterEraser = useCallback(() => {
    setEraserActive(true);
  }, []);

  // Cancel eraser — discard edits, reload original
  const cancelEraser = useCallback(() => {
    setEraserActive(false);
    setEraseCount(0);
    setEraserResetKey((k) => k + 1);
    setUndoSignal(0);
  }, []);

  // Done — export edited model group as GLB blob, pass to parent, exit eraser
  const finishEraser = useCallback(() => {
    const group = editableGroupRef.current;
    if (!group) {
      console.error('No editable model group available');
      return;
    }
    const exporter = new GLTFExporter();
    exporter.parse(
      group,
      (result) => {
        const blob = result instanceof ArrayBuffer
          ? new Blob([result], { type: 'model/gltf-binary' })
          : new Blob([JSON.stringify(result)], { type: 'model/gltf+json' });
        const blobUrl = URL.createObjectURL(blob);
        setEraserActive(false);
        setEraseCount(0);
        setUndoSignal(0);
        editableGroupRef.current = null;
        onEditedModel?.(blobUrl);
      },
      (err) => console.error('GLB export failed:', err),
      { binary: true },
    );
  }, [onEditedModel]);

  // Undo last erase
  const handleUndo = useCallback(() => {
    if (eraseCount <= 0) return;
    setUndoSignal((s) => s + 1);
    setEraseCount((c) => Math.max(0, c - 1));
  }, [eraseCount]);

  // Reset all erases (back to original)
  const handleReset = useCallback(() => {
    setEraserResetKey((k) => k + 1);
    setEraseCount(0);
    setUndoSignal(0);
  }, []);

  const hasPly = exports.some((e) => e.format === 'ply');
  const hasGlb = exports.some((e) => e.format === 'glb');
  const showViewModes = exports.length > 0 && url;

  const getModelProps = (): { component: typeof GLBModel; modelUrl: string } | null => {
    if (!url) return null;
    if (viewMode === 'textured') {
      const ModelComponent = { glb: GLBModel, stl: STLModel, ply: PLYModel, obj: OBJModel }[format] || GLBModel;
      return { component: ModelComponent, modelUrl: url };
    }
    if (viewMode === 'mesh') {
      const glbExport = exports.find((e) => e.format === 'glb');
      return { component: GLBMesh, modelUrl: glbExport?.url || url };
    }
    if (viewMode === 'solid') {
      const glbExport = exports.find((e) => e.format === 'glb');
      return { component: GLBSolid, modelUrl: glbExport?.url || url };
    }
    if (viewMode === 'wireframe') {
      const glbExport = exports.find((e) => e.format === 'glb');
      return { component: GLBWireframe, modelUrl: glbExport?.url || url };
    }
    if (viewMode === 'pointcloud') {
      const plyExport = exports.find((e) => e.format === 'ply');
      if (plyExport) return { component: PLYModel, modelUrl: plyExport.url };
      return null;
    }
    return null;
  };

  if (!url) {
    return (
      <div className="w-full h-full flex items-center justify-center rounded-xl border border-border" style={{ background: BG_COLOR }}>
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-3 rounded-2xl bg-bg-tertiary flex items-center justify-center">
            <svg className="w-8 h-8 text-text-muted" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 3L2 9l10 6 10-6-10-6z" />
              <path d="M2 17l10 6 10-6" />
              <path d="M2 13l10 6 10-6" />
            </svg>
          </div>
          <p className="text-sm text-text-muted">3D preview will appear here</p>
          <p className="text-xs text-text-muted/60 mt-1">Generate a model to see the result</p>
        </div>
      </div>
    );
  }

  const modelProps = getModelProps();
  const ModelComponent = modelProps?.component || GLBModel;
  const modelUrl = modelProps?.modelUrl || url;

  // Canvas key: stable during eraser use, changes on view mode or URL
  const cKey = eraserActive
    ? `${canvasKey}-eraser-${modelUrl}`
    : `${canvasKey}-${viewMode}-${modelUrl}`;

  return (
    <div className="w-full h-full relative">
      <div className={`absolute inset-0 rounded-xl border overflow-hidden ${eraserActive ? 'border-red-500/40 cursor-crosshair' : 'border-border'}`} style={{ background: BG_COLOR }}>
        <ViewerErrorBoundary onReset={resetCanvas}>
          <Canvas
            key={cKey}
            camera={{ position: [2, 1.5, 2], fov: 40 }}
            gl={{ antialias: true, alpha: false, failIfMajorPerformanceCaveat: false }}
            onCreated={({ scene, gl }) => {
              scene.background = new THREE.Color(BG_COLOR);
              gl.setClearColor(BG_COLOR, 1);
              gl.clear();
            }}
          >
            <SceneInit />
            <ambientLight intensity={viewerSettings.planarLightIntensity} color={viewerSettings.lightColor} />
            <directionalLight
              position={[viewerSettings.lightPositionX * 5, viewerSettings.lightPositionY * 5, 5]}
              intensity={viewerSettings.spotlightIntensity * 2.5}
              color={viewerSettings.lightColor}
            />
            <directionalLight position={[-3, 3, -3]} intensity={viewerSettings.planarLightIntensity * 0.7} />
            <hemisphereLight args={['#b1e1ff', '#b97a20', viewerSettings.planarLightIntensity * 0.8]} />
            <OrbitControls autoRotate={!eraserActive && autoRotate} autoRotateSpeed={2} />

            {/* Environment loaded separately so HDR fetch failures don't block the model */}
            {viewerSettings.environmentPreset !== 'none' && (
              <EnvironmentErrorBoundary>
                <Suspense fallback={null}>
                  <SafeEnvironment preset={viewerSettings.environmentPreset} />
                </Suspense>
              </EnvironmentErrorBoundary>
            )}

            <Suspense fallback={<InCanvasSpinner />}>
              {eraserActive ? (
                <EditableModel
                  url={modelUrl}
                  eraserActive={eraserActive}
                  brushSize={brushSize}
                  onErase={() => setEraseCount((c) => c + 1)}
                  resetKey={eraserResetKey}
                  undoSignal={undoSignal}
                  onGroupReady={(g) => { editableGroupRef.current = g; }}
                />
              ) : (
                <ModelComponent url={modelUrl} />
              )}
            </Suspense>
          </Canvas>
        </ViewerErrorBoundary>

        {/* Eraser mode indicator */}
        {eraserActive && (
          <div className="absolute top-3 right-3 z-10 bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-1.5">
            <span className="text-xs text-red-400 font-medium">Eraser Mode</span>
            <span className="text-[10px] text-red-400/60 ml-2">Click to erase, drag to rotate</span>
          </div>
        )}
      </div>

      {/* View Mode Toggle + Eraser — floating at top of viewer */}
      {showViewModes && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10">
          <div className="flex gap-0.5 bg-bg-primary/80 backdrop-blur-sm rounded-lg p-0.5 border border-border shadow-lg">
            {!eraserActive && VIEW_MODES.map(({ id, label, icon: Icon }) => {
              const disabled =
                (id === 'pointcloud' && !hasPly) ||
                (id === 'mesh' && !hasGlb && format !== 'glb') ||
                (id === 'solid' && !hasGlb && format !== 'glb') ||
                (id === 'wireframe' && !hasGlb && format !== 'glb');
              return (
                <button
                  key={id}
                  onClick={() => !disabled && setViewMode(id)}
                  disabled={disabled}
                  className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors
                    ${viewMode === id
                      ? 'bg-accent/15 text-accent'
                      : disabled
                        ? 'text-text-muted/30 cursor-not-allowed'
                        : 'text-text-muted hover:text-text-secondary hover:bg-bg-tertiary/50'}`}
                  title={disabled ? `No ${id === 'pointcloud' ? 'PLY' : 'GLB'} export available` : label}
                >
                  <Icon className="w-3.5 h-3.5" />
                  {label}
                </button>
              );
            })}

            {!eraserActive && hasGlb && <div className="w-px bg-border mx-0.5" />}

            {/* Eraser toggle (only show when not in eraser mode) */}
            {!eraserActive && hasGlb && (
              <button
                onClick={enterEraser}
                className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-md transition-colors text-text-muted hover:text-text-secondary hover:bg-bg-tertiary/50"
                title="Enter eraser mode to remove unwanted mesh parts"
              >
                <Eraser className="w-3.5 h-3.5" />
                Eraser
              </button>
            )}

            {/* Eraser toolbar (replaces view modes when in eraser mode) */}
            {eraserActive && (
              <>
                {/* Brush size */}
                <span className="text-[10px] text-text-muted self-center px-1">Brush</span>
                <button onClick={() => setBrushSize((s) => Math.max(0.01, s - 0.01))} className="p-1 rounded hover:bg-bg-tertiary text-text-muted">
                  <Minus className="w-3 h-3" />
                </button>
                <span className="text-xs font-mono text-text-secondary w-8 text-center self-center">{brushSize.toFixed(2)}</span>
                <button onClick={() => setBrushSize((s) => Math.min(0.2, s + 0.01))} className="p-1 rounded hover:bg-bg-tertiary text-text-muted">
                  <Plus className="w-3 h-3" />
                </button>

                <div className="w-px bg-border mx-0.5" />

                {/* Undo */}
                <button
                  onClick={handleUndo}
                  disabled={eraseCount <= 0}
                  className={`flex items-center gap-1 text-xs px-2 py-1.5 rounded-md transition-colors
                    ${eraseCount > 0 ? 'text-text-muted hover:text-text-secondary hover:bg-bg-tertiary' : 'text-text-muted/30 cursor-not-allowed'}`}
                  title="Undo last erase"
                >
                  <Undo2 className="w-3.5 h-3.5" />
                </button>

                {/* Reset all */}
                <button
                  onClick={handleReset}
                  disabled={eraseCount <= 0}
                  className={`flex items-center gap-1 text-xs px-2 py-1.5 rounded-md transition-colors
                    ${eraseCount > 0 ? 'text-text-muted hover:text-text-secondary hover:bg-bg-tertiary' : 'text-text-muted/30 cursor-not-allowed'}`}
                  title="Reset all erases"
                >
                  <Redo2 className="w-3.5 h-3.5" />
                  Reset
                </button>

                <div className="w-px bg-border mx-0.5" />

                {/* Erase count */}
                <span className="text-[10px] text-text-muted self-center">
                  {eraseCount > 0 ? `${eraseCount} erased` : 'Click to erase'}
                </span>

                <div className="w-px bg-border mx-0.5" />

                {/* Cancel — discard edits */}
                <button
                  onClick={cancelEraser}
                  className="flex items-center gap-1 text-xs px-2 py-1.5 rounded-md text-text-muted hover:text-red-400 hover:bg-red-500/10 transition-colors"
                  title="Cancel and discard edits"
                >
                  <X className="w-3.5 h-3.5" />
                  Cancel
                </button>

                {/* Done — apply edits and return to normal view */}
                <button
                  onClick={finishEraser}
                  disabled={eraseCount <= 0}
                  className={`flex items-center gap-1 text-xs px-3 py-1.5 rounded-md transition-colors
                    ${eraseCount > 0
                      ? 'bg-green-500/15 text-green-400 hover:bg-green-500/25'
                      : 'text-text-muted/30 cursor-not-allowed'}`}
                  title="Apply edits and return to normal view"
                >
                  <Check className="w-3.5 h-3.5" />
                  Done
                </button>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
