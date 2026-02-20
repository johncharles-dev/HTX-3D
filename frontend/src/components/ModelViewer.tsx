import { Suspense, useRef, useMemo, useState, useCallback, useEffect, Component, type ReactNode } from 'react';
import { Canvas, useLoader, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Center, Environment } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { AlertTriangle, Box, Circle, Grid3x3, Dot } from 'lucide-react';
import * as THREE from 'three';
import type { ExportFile } from '../types';

type ViewMode = 'textured' | 'solid' | 'wireframe' | 'pointcloud';

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

// Immediately sets dark background — renders on first frame before model loads
function SceneInit() {
  const { scene, gl } = useThree();
  scene.background = new THREE.Color(BG_COLOR);
  gl.setClearColor(BG_COLOR, 1);
  return null;
}

// Loading indicator inside the 3D scene (renders while model suspends)
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

function GLBModel({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  return (
    <Center>
      <primitive object={gltf.scene.clone()} />
    </Center>
  );
}

function GLBSolid({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  const scene = useMemo(() => {
    const mat = new THREE.MeshNormalMaterial();
    const cloned = gltf.scene.clone();
    cloned.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = mat;
      }
    });
    return cloned;
  }, [gltf]);

  return (
    <Center>
      <primitive object={scene} />
    </Center>
  );
}

function GLBWireframe({ url }: { url: string }) {
  const gltf = useLoader(GLTFLoader, url);
  const scene = useMemo(() => {
    const cloned = gltf.scene.clone();
    cloned.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.material = new THREE.MeshBasicMaterial({
          wireframe: true,
          color: new THREE.Color('#6366f1'),
        });
      }
    });
    return cloned;
  }, [gltf]);

  return (
    <Center>
      <primitive object={scene} />
    </Center>
  );
}

function STLModel({ url }: { url: string }) {
  const geometry = useLoader(STLLoader, url);
  return (
    <Center>
      <mesh geometry={geometry}>
        <meshStandardMaterial color="#a0a0a0" />
      </mesh>
    </Center>
  );
}

function PLYModel({ url }: { url: string }) {
  const geometry = useLoader(PLYLoader, url);
  geometry.computeVertexNormals();
  return (
    <Center>
      <group scale={[1, -1, 1]}>
        <points geometry={geometry}>
          <pointsMaterial size={0.005} vertexColors />
        </points>
      </group>
    </Center>
  );
}

function OBJModel({ url }: { url: string }) {
  const obj = useLoader(OBJLoader, url);
  return (
    <Center>
      <primitive object={obj.clone()} />
    </Center>
  );
}

function AutoRotate({ enabled }: { enabled: boolean }) {
  const controlsRef = useRef<any>(null);
  useFrame(() => {
    if (enabled && controlsRef.current) {
      controlsRef.current.autoRotate = true;
      controlsRef.current.autoRotateSpeed = 2;
      controlsRef.current.update();
    }
  });
  return <OrbitControls ref={controlsRef} autoRotate={enabled} autoRotateSpeed={2} />;
}

interface Props {
  url: string | null;
  format?: string;
  autoRotate?: boolean;
  exports?: ExportFile[];
}

const VIEW_MODES: { id: ViewMode; label: string; icon: typeof Box }[] = [
  { id: 'textured', label: 'Textured', icon: Box },
  { id: 'solid', label: 'Solid', icon: Circle },
  { id: 'wireframe', label: 'Wireframe', icon: Grid3x3 },
  { id: 'pointcloud', label: 'Point Cloud', icon: Dot },
];

export default function ModelViewer({ url, format = 'glb', autoRotate = true, exports = [] }: Props) {
  const [canvasKey, setCanvasKey] = useState(0);
  const [viewMode, setViewMode] = useState<ViewMode>('textured');

  const resetCanvas = useCallback(() => {
    setCanvasKey((k) => k + 1);
  }, []);

  // Reset to textured when URL changes
  useEffect(() => {
    setViewMode('textured');
  }, [url]);

  const hasPly = exports.some((e) => e.format === 'ply');
  const hasGlb = exports.some((e) => e.format === 'glb');
  const showViewModes = exports.length > 0 && url;

  // Resolve what to render based on view mode
  const getModelProps = (): { component: typeof GLBModel; modelUrl: string } | null => {
    if (!url) return null;

    if (viewMode === 'textured') {
      // Use the format-specific component lookup for textured mode
      const ModelComponent = {
        glb: GLBModel,
        stl: STLModel,
        ply: PLYModel,
        obj: OBJModel,
      }[format] || GLBModel;
      return { component: ModelComponent, modelUrl: url };
    }

    if (viewMode === 'solid') {
      const glbExport = exports.find((e) => e.format === 'glb');
      const solidUrl = glbExport?.url || url;
      return { component: GLBSolid, modelUrl: solidUrl };
    }

    if (viewMode === 'wireframe') {
      const glbExport = exports.find((e) => e.format === 'glb');
      const wireUrl = glbExport?.url || url;
      return { component: GLBWireframe, modelUrl: wireUrl };
    }

    if (viewMode === 'pointcloud') {
      const plyExport = exports.find((e) => e.format === 'ply');
      if (plyExport) {
        return { component: PLYModel, modelUrl: plyExport.url };
      }
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

  // Fallback when switching to a mode that can't render (e.g. point cloud with no PLY)
  const ModelComponent = modelProps?.component || GLBModel;
  const modelUrl = modelProps?.modelUrl || url;

  return (
    <div className="w-full h-full relative">
      <div className="absolute inset-0 rounded-xl border border-border overflow-hidden" style={{ background: BG_COLOR }}>
        <ViewerErrorBoundary onReset={resetCanvas}>
          <Canvas
            key={`${canvasKey}-${viewMode}-${modelUrl}`}
            camera={{ position: [2, 1.5, 2], fov: 40 }}
            gl={{ antialias: true, alpha: false, failIfMajorPerformanceCaveat: false }}
            onCreated={({ scene, gl }) => {
              scene.background = new THREE.Color(BG_COLOR);
              gl.setClearColor(BG_COLOR, 1);
              gl.clear();
            }}
          >
            {/* These render immediately — no suspension */}
            <SceneInit />
            <ambientLight intensity={0.6} />
            <directionalLight position={[5, 5, 5]} intensity={1} />
            <directionalLight position={[-3, 3, -3]} intensity={0.4} />
            <hemisphereLight args={['#b1e1ff', '#b97a20', 0.5]} />
            <AutoRotate enabled={autoRotate} />

            {/* Model + Environment load async — inner Suspense prevents blocking the scene */}
            <Suspense fallback={<InCanvasSpinner />}>
              {viewMode === 'textured' && <Environment preset="studio" />}
              <ModelComponent url={modelUrl} />
            </Suspense>
          </Canvas>
        </ViewerErrorBoundary>
      </div>

      {/* View Mode Toggle — floating at top of viewer */}
      {showViewModes && (
        <div className="absolute top-3 left-1/2 -translate-x-1/2 z-10">
          <div className="flex gap-0.5 bg-bg-primary/80 backdrop-blur-sm rounded-lg p-0.5 border border-border shadow-lg">
            {VIEW_MODES.map(({ id, label, icon: Icon }) => {
              const disabled =
                (id === 'pointcloud' && !hasPly) ||
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
          </div>
        </div>
      )}
    </div>
  );
}
