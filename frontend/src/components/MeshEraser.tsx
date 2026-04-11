import { useRef, useCallback, useEffect, useState } from 'react';
import { useThree } from '@react-three/fiber';
import { Center } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import * as THREE from 'three';

// ── Geometry snapshot for undo ──────────────────────────

interface GeometrySnapshot {
  attrs: Record<string, { array: Float32Array; itemSize: number }>;
}

function takeSnapshot(geom: THREE.BufferGeometry): GeometrySnapshot {
  const attrs: GeometrySnapshot['attrs'] = {};
  for (const name of Object.keys(geom.attributes)) {
    const attr = geom.getAttribute(name) as THREE.BufferAttribute;
    attrs[name] = {
      array: new Float32Array(attr.array as Float32Array),
      itemSize: attr.itemSize,
    };
  }
  return { attrs };
}

function restoreSnapshot(geom: THREE.BufferGeometry, snap: GeometrySnapshot) {
  for (const name of Object.keys(snap.attrs)) {
    const { array, itemSize } = snap.attrs[name];
    geom.setAttribute(name, new THREE.BufferAttribute(new Float32Array(array), itemSize));
  }
  geom.setIndex(null);
  geom.computeBoundingSphere();
  geom.computeBoundingBox();
  geom.computeVertexNormals();
}

// ── EditableModel component ─────────────────────────────

interface EditableModelProps {
  url: string;
  eraserActive: boolean;
  brushSize: number;
  onErase: () => void;
  resetKey: number;
  undoSignal: number;
  /** Called with a ref to the loaded scene group — parent uses this for GLB export */
  onGroupReady?: (group: THREE.Group | null) => void;
}

export function EditableModel({ url, eraserActive, brushSize, onErase, resetKey, undoSignal, onGroupReady }: EditableModelProps) {
  const { camera, raycaster, gl } = useThree();
  const groupRef = useRef<THREE.Group>(null);
  const [loadedScene, setLoadedScene] = useState<THREE.Group | null>(null);
  const pointerDownPos = useRef<{ x: number; y: number } | null>(null);
  const undoStack = useRef<{ mesh: THREE.Mesh; snapshot: GeometrySnapshot }[]>([]);

  // Load/reload GLTF
  useEffect(() => {
    undoStack.current = [];
    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {
      const cloned = gltf.scene.clone(true);
      cloned.traverse((child) => {
        if (child instanceof THREE.Mesh && child.geometry) {
          if (child.geometry.index !== null) {
            child.geometry = child.geometry.toNonIndexed();
          }
          child.geometry.computeBoundingSphere();
          child.geometry.computeBoundingBox();
        }
      });
      setLoadedScene(cloned);
      onGroupReady?.(cloned);
    });
    return () => { onGroupReady?.(null); };
  }, [url, resetKey]);

  // Handle undo signal from parent
  useEffect(() => {
    if (undoSignal <= 0) return;
    const entry = undoStack.current.pop();
    if (entry) {
      restoreSnapshot(entry.mesh.geometry, entry.snapshot);
    }
  }, [undoSignal]);

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    pointerDownPos.current = { x: e.clientX, y: e.clientY };
  }, []);

  const handlePointerUp = useCallback(
    (e: React.PointerEvent) => {
      if (!eraserActive || !groupRef.current || !pointerDownPos.current) return;

      const dx = e.clientX - pointerDownPos.current.x;
      const dy = e.clientY - pointerDownPos.current.y;
      pointerDownPos.current = null;
      if (Math.abs(dx) + Math.abs(dy) > 5) return; // was a drag

      const rect = gl.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        -((e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(mouse, camera);

      const meshes: THREE.Mesh[] = [];
      groupRef.current.traverse((child) => {
        if (child instanceof THREE.Mesh) meshes.push(child);
      });

      const intersects = raycaster.intersectObjects(meshes, false);
      if (intersects.length === 0) return;

      const hit = intersects[0];
      const mesh = hit.object as THREE.Mesh;
      const hitPoint = hit.point;
      const geom = mesh.geometry;
      const posAttr = geom.getAttribute('position');
      if (!posAttr) return;

      // Save snapshot for undo BEFORE modifying
      undoStack.current.push({ mesh, snapshot: takeSnapshot(geom) });

      const worldMatrix = mesh.matrixWorld;
      const facesToRemove = new Set<number>();
      const faceCount = posAttr.count / 3;
      const v = new THREE.Vector3();

      // Remove face if ANY vertex is within the brush radius.
      // Using per-vertex (not centroid) ensures clean cuts with no
      // surviving boundary faces that could bridge disconnected parts.
      const brushSq = brushSize * brushSize;
      for (let f = 0; f < faceCount; f++) {
        for (let j = 0; j < 3; j++) {
          v.fromBufferAttribute(posAttr, f * 3 + j);
          v.applyMatrix4(worldMatrix);
          const dx = v.x - hitPoint.x, dy = v.y - hitPoint.y, dz = v.z - hitPoint.z;
          if (dx * dx + dy * dy + dz * dz <= brushSq) {
            facesToRemove.add(f);
            break;
          }
        }
      }

      if (facesToRemove.size === 0) {
        undoStack.current.pop(); // nothing removed, discard snapshot
        return;
      }

      const keepFaces: number[] = [];
      for (let f = 0; f < faceCount; f++) {
        if (!facesToRemove.has(f)) keepFaces.push(f);
      }

      const attrNames = Object.keys(geom.attributes);
      const attrMeta: Record<string, number> = {};
      for (const name of attrNames) {
        attrMeta[name] = (geom.getAttribute(name) as THREE.BufferAttribute).itemSize;
      }

      const newArrays: Record<string, Float32Array> = {};
      for (const name of attrNames) {
        newArrays[name] = new Float32Array(keepFaces.length * 3 * attrMeta[name]);
      }

      for (let i = 0; i < keepFaces.length; i++) {
        const f = keepFaces[i];
        for (const name of attrNames) {
          const itemSize = attrMeta[name];
          const oldArr = (geom.getAttribute(name) as THREE.BufferAttribute).array as Float32Array;
          for (let j = 0; j < 3; j++) {
            const srcOff = (f * 3 + j) * itemSize;
            const dstOff = (i * 3 + j) * itemSize;
            for (let k = 0; k < itemSize; k++) {
              newArrays[name][dstOff + k] = oldArr[srcOff + k];
            }
          }
        }
      }

      for (const name of attrNames) {
        geom.setAttribute(name, new THREE.BufferAttribute(newArrays[name], attrMeta[name]));
      }
      geom.setIndex(null);
      geom.computeBoundingSphere();
      geom.computeBoundingBox();
      geom.computeVertexNormals();

      onErase();
    },
    [eraserActive, brushSize, camera, gl, raycaster, onErase],
  );

  if (!loadedScene) return null;

  return (
    <Center>
      <group ref={groupRef} onPointerDown={handlePointerDown} onPointerUp={handlePointerUp}>
        <primitive object={loadedScene} />
      </group>
    </Center>
  );
}

// ── Client-side floater removal ─────────────────────────

/** Find connected face components in a non-indexed triangle geometry. */
function findComponents(posAttr: THREE.BufferAttribute | THREE.InterleavedBufferAttribute, faceCount: number): number[][] {
  // Build adjacency: faces sharing a vertex position are connected
  const vertexToFaces = new Map<string, number[]>();
  const v = new THREE.Vector3();
  for (let f = 0; f < faceCount; f++) {
    for (let j = 0; j < 3; j++) {
      v.fromBufferAttribute(posAttr, f * 3 + j);
      const key = `${v.x.toFixed(4)},${v.y.toFixed(4)},${v.z.toFixed(4)}`;
      let list = vertexToFaces.get(key);
      if (!list) { list = []; vertexToFaces.set(key, list); }
      list.push(f);
    }
  }

  // BFS / flood-fill
  const visited = new Uint8Array(faceCount);
  const components: number[][] = [];
  for (let f = 0; f < faceCount; f++) {
    if (visited[f]) continue;
    const component: number[] = [];
    const stack = [f];
    visited[f] = 1;
    while (stack.length > 0) {
      const cur = stack.pop()!;
      component.push(cur);
      for (let j = 0; j < 3; j++) {
        v.fromBufferAttribute(posAttr, cur * 3 + j);
        const key = `${v.x.toFixed(4)},${v.y.toFixed(4)},${v.z.toFixed(4)}`;
        const neighbors = vertexToFaces.get(key);
        if (!neighbors) continue;
        for (const nf of neighbors) {
          if (!visited[nf]) { visited[nf] = 1; stack.push(nf); }
        }
      }
    }
    components.push(component);
  }
  return components;
}

/** Rebuild a non-indexed geometry keeping only faces in keepFaces. */
function rebuildGeometry(g: THREE.BufferGeometry, keepFaces: number[]) {
  const attrNames = Object.keys(g.attributes);
  const attrMeta: Record<string, number> = {};
  for (const name of attrNames) {
    attrMeta[name] = (g.getAttribute(name) as THREE.BufferAttribute).itemSize;
  }
  const newArrays: Record<string, Float32Array> = {};
  for (const name of attrNames) {
    newArrays[name] = new Float32Array(keepFaces.length * 3 * attrMeta[name]);
  }
  for (let i = 0; i < keepFaces.length; i++) {
    const f = keepFaces[i];
    for (const name of attrNames) {
      const itemSize = attrMeta[name];
      const oldArr = (g.getAttribute(name) as THREE.BufferAttribute).array as Float32Array;
      for (let j = 0; j < 3; j++) {
        const srcOff = (f * 3 + j) * itemSize;
        const dstOff = (i * 3 + j) * itemSize;
        for (let k = 0; k < itemSize; k++) {
          newArrays[name][dstOff + k] = oldArr[srcOff + k];
        }
      }
    }
  }
  for (const name of attrNames) {
    g.setAttribute(name, new THREE.BufferAttribute(newArrays[name], attrMeta[name]));
  }
  g.setIndex(null);
  g.computeBoundingSphere();
  g.computeBoundingBox();
  g.computeVertexNormals();
}

/**
 * Remove disconnected components from a GLB blob.
 * When keepLargestOnly is true (default), removes everything except the
 * single largest connected component — ideal for post-eraser cleanup.
 * When false, uses minRatio to filter small fragments only.
 */
export async function removeFloatersFromBlob(
  blobUrl: string,
  { keepLargestOnly = true, minRatio = 0.05 }: { keepLargestOnly?: boolean; minRatio?: number } = {},
): Promise<{ url: string; removed: number }> {
  return new Promise((resolve, reject) => {
    const loader = new GLTFLoader();
    loader.load(blobUrl, (gltf) => {
      let totalRemoved = 0;

      // Collect all meshes for cross-mesh removal
      const allMeshes: { mesh: THREE.Mesh; faceCount: number }[] = [];

      gltf.scene.traverse((child) => {
        if (!(child instanceof THREE.Mesh)) return;
        const geom = child.geometry;
        if (!geom) return;

        // Ensure non-indexed for face-level processing
        if (geom.index !== null) {
          child.geometry = geom.toNonIndexed();
        }
        const g = child.geometry;
        const posAttr = g.getAttribute('position');
        if (!posAttr) return;

        const faceCount = Math.floor(posAttr.count / 3);
        if (faceCount < 1) return;

        allMeshes.push({ mesh: child, faceCount });

        // Within-mesh component detection
        if (faceCount < 2) return;
        const components = findComponents(posAttr, faceCount);
        // components found: largest first after sort;

        if (components.length <= 1) return;

        // Sort by size descending
        components.sort((a, b) => b.length - a.length);
        const largestSize = components[0].length;
        const keepFaceSet = new Set<number>();
        let keptComponents = 0;
        for (const comp of components) {
          const keep = keepLargestOnly
            ? keptComponents === 0  // only keep the very first (largest)
            : comp.length >= largestSize * minRatio;
          if (keep) {
            for (const f of comp) keepFaceSet.add(f);
            keptComponents++;
          }
        }
        const removed = components.length - keptComponents;
        if (removed === 0) return;
        totalRemoved += removed;

        const keepFaces = Array.from(keepFaceSet).sort((a, b) => a - b);
        rebuildGeometry(g, keepFaces);
      });

      // Cross-mesh removal: if multiple meshes, remove small ones
      if (allMeshes.length > 1) {
        allMeshes.sort((a, b) => b.faceCount - a.faceCount);
        const largestMeshFaces = allMeshes[0].faceCount;
        for (let i = 0; i < allMeshes.length; i++) {
          const { mesh, faceCount } = allMeshes[i];
          const shouldRemove = keepLargestOnly
            ? i > 0  // remove all except the first (largest)
            : faceCount < largestMeshFaces * minRatio;
          if (shouldRemove) {
            mesh.removeFromParent();
            mesh.geometry.dispose();
            if (mesh.material) {
              const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
              mats.forEach(m => m.dispose());
            }
            totalRemoved++;
          }
        }
      }


      // Export cleaned scene as GLB
      import('three/examples/jsm/exporters/GLTFExporter.js').then(({ GLTFExporter }) => {
        const exporter = new GLTFExporter();
        exporter.parse(
          gltf.scene,
          (result) => {
            const blob = result instanceof ArrayBuffer
              ? new Blob([result], { type: 'model/gltf-binary' })
              : new Blob([JSON.stringify(result)], { type: 'model/gltf+json' });
            const newUrl = URL.createObjectURL(blob);
            resolve({ url: newUrl, removed: totalRemoved });
          },
          (err: unknown) => reject(err),
          { binary: true },
        );
      }).catch(reject);
    }, undefined, reject);
  });
}
