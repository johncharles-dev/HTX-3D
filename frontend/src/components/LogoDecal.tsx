import { useRef, useCallback, useEffect, useState } from 'react';
import { useThree } from '@react-three/fiber';
import { Center } from '@react-three/drei';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DecalGeometry } from 'three/examples/jsm/geometries/DecalGeometry.js';
import * as THREE from 'three';

// ── Interactive logo decal component ────────────────────
//
// Loads the GLB, lets the user stamp a PNG logo onto the surface, then keep
// editing it: click a logo to select, drag it across the surface to reposition
// (re-orienting to the new surface normal), and the parent's Size/Rotate
// controls act live on the selected logo. Each decal's geometry is regenerated
// from its live placement params and baked into the target mesh's LOCAL space,
// so it stays glued to the surface and exports correctly through GLTFExporter.

const ACCENT = 0x6366f1;

interface Placement {
  id: number;
  targetMesh: THREE.Mesh;
  position: THREE.Vector3; // world-space hit point
  normal: THREE.Vector3;   // world-space surface normal
  size: number;
  rotation: number;        // in-plane rotation (radians)
  texture: THREE.Texture;
  mesh: THREE.Mesh;        // rendered decal mesh (child of targetMesh)
}

export interface LogoSelection {
  size: number;
  rotation: number;
}

interface LogoDecalModelProps {
  url: string;
  /** Uploaded logo as a texture; null until the user picks a PNG. */
  logoTexture: THREE.Texture | null;
  /** Active size — default for new stamps, and live size of the selected logo. */
  logoSize: number;
  /** Active in-plane rotation (radians) — same dual role as logoSize. */
  logoRotation: number;
  /** Called when the number of placed logos changes. */
  onCountChange: (n: number) => void;
  /** Called when selection changes — null when nothing is selected. */
  onSelectionChange?: (sel: LogoSelection | null) => void;
  /** Bumped by parent to remove all logos. */
  resetKey: number;
  /** Bumped by parent to remove the most recent logo. */
  undoSignal: number;
  /** Passes the loaded scene group to the parent for GLB export. */
  onGroupReady?: (group: THREE.Group | null) => void;
}

function orientationFromNormal(position: THREE.Vector3, normal: THREE.Vector3, rotation: number): THREE.Euler {
  const dummy = new THREE.Object3D();
  dummy.position.copy(position);
  dummy.lookAt(position.clone().add(normal));
  dummy.rotateZ(rotation);
  return dummy.rotation.clone();
}

/** Build decal geometry for a placement, baked into the target mesh's local space. */
function buildDecalGeometry(p: Placement): THREE.BufferGeometry {
  p.targetMesh.updateWorldMatrix(true, false);
  const euler = orientationFromNormal(p.position, p.normal, p.rotation);
  const sizeVec = new THREE.Vector3(p.size, p.size, Math.max(p.size, 0.5));
  const geom = new DecalGeometry(p.targetMesh, p.position, euler, sizeVec);
  geom.applyMatrix4(p.targetMesh.matrixWorld.clone().invert());
  return geom;
}

export function LogoDecalModel({
  url,
  logoTexture,
  logoSize,
  logoRotation,
  onCountChange,
  onSelectionChange,
  resetKey,
  undoSignal,
  onGroupReady,
}: LogoDecalModelProps) {
  const { camera, raycaster, gl, controls } = useThree() as any;
  const groupRef = useRef<THREE.Group>(null);
  const [loadedScene, setLoadedScene] = useState<THREE.Group | null>(null);

  const placements = useRef<Placement[]>([]);
  const selectedId = useRef<number | null>(null);
  const idCounter = useRef(0);
  const outline = useRef<THREE.BoxHelper | null>(null);

  const pointerDown = useRef<{ x: number; y: number; onDecal: boolean; dragging: boolean } | null>(null);

  // Latest props reachable from imperative handlers without rebinding them.
  const logoTextureRef = useRef(logoTexture);
  const logoSizeRef = useRef(logoSize);
  const logoRotationRef = useRef(logoRotation);
  logoTextureRef.current = logoTexture;
  logoSizeRef.current = logoSize;
  logoRotationRef.current = logoRotation;

  // ── Load / reload GLB ─────────────────────────────────
  useEffect(() => {
    placements.current = [];
    selectedId.current = null;
    idCounter.current = 0;
    const loader = new GLTFLoader();
    loader.load(url, (gltf) => {
      const cloned = gltf.scene.clone(true);
      cloned.traverse((child) => {
        if (child instanceof THREE.Mesh && child.geometry) {
          if (child.geometry.index !== null) child.geometry = child.geometry.toNonIndexed();
          child.geometry.computeBoundingSphere();
          child.geometry.computeBoundingBox();
        }
      });
      setLoadedScene(cloned);
      onGroupReady?.(cloned);
    });
    return () => { onGroupReady?.(null); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, resetKey]);

  // ── Selection outline helpers ─────────────────────────
  const clearOutline = useCallback(() => {
    if (outline.current) {
      outline.current.removeFromParent();
      outline.current.geometry.dispose();
      (outline.current.material as THREE.Material).dispose();
      outline.current = null;
    }
  }, []);

  const refreshOutline = useCallback(() => {
    const sel = placements.current.find((p) => p.id === selectedId.current);
    if (!sel || !groupRef.current) { clearOutline(); return; }
    if (!outline.current) {
      // Added to the group (a sibling of the exported scene), so it never exports.
      outline.current = new THREE.BoxHelper(sel.mesh, ACCENT);
      (outline.current.material as THREE.LineBasicMaterial).depthTest = false;
      outline.current.renderOrder = 999;
      groupRef.current.add(outline.current);
    } else {
      outline.current.setFromObject(sel.mesh);
    }
    outline.current.update();
  }, [clearOutline]);

  const selectPlacement = useCallback((id: number | null) => {
    selectedId.current = id;
    refreshOutline();
    const sel = placements.current.find((p) => p.id === id);
    onSelectionChange?.(sel ? { size: sel.size, rotation: sel.rotation } : null);
  }, [refreshOutline, onSelectionChange]);

  // ── Undo / reset ──────────────────────────────────────
  useEffect(() => {
    if (undoSignal <= 0) return;
    const p = placements.current.pop();
    if (p) {
      if (selectedId.current === p.id) { selectedId.current = null; clearOutline(); onSelectionChange?.(null); }
      p.mesh.removeFromParent();
      p.mesh.geometry.dispose();
      (p.mesh.material as THREE.Material).dispose();
      onCountChange(placements.current.length);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [undoSignal]);

  // ── Live size/rotation from parent → selected logo ────
  useEffect(() => {
    const sel = placements.current.find((p) => p.id === selectedId.current);
    if (!sel) return;
    if (sel.size === logoSize && sel.rotation === logoRotation) return;
    sel.size = logoSize;
    sel.rotation = logoRotation;
    sel.mesh.geometry.dispose();
    sel.mesh.geometry = buildDecalGeometry(sel);
    refreshOutline();
  }, [logoSize, logoRotation, refreshOutline]);

  // ── Raycast helpers ───────────────────────────────────
  const setRay = useCallback((clientX: number, clientY: number) => {
    const rect = gl.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(mouse, camera);
  }, [camera, gl, raycaster]);

  const collectMeshes = useCallback(() => {
    const base: THREE.Mesh[] = [];
    const decals: THREE.Mesh[] = [];
    groupRef.current?.traverse((c) => {
      if (!(c instanceof THREE.Mesh)) return;
      if (c.userData.isLogoDecal) decals.push(c); else base.push(c);
    });
    return { base, decals };
  }, []);

  // ── Pointer interaction ───────────────────────────────
  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    setRay(e.clientX, e.clientY);
    const { decals } = collectMeshes();
    const hitDecal = raycaster.intersectObjects(decals, false)[0];
    const onDecal = !!hitDecal;
    pointerDown.current = { x: e.clientX, y: e.clientY, onDecal, dragging: false };

    if (onDecal) {
      // Select immediately and prepare to drag; suspend camera orbit.
      const id = (hitDecal.object as THREE.Mesh).userData.placementId as number;
      selectPlacement(id);
      if (controls) controls.enabled = false;
      try { gl.domElement.setPointerCapture(e.pointerId); } catch { /* noop */ }
    }
  }, [setRay, collectMeshes, raycaster, selectPlacement, controls, gl]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    const pd = pointerDown.current;
    if (!pd || !pd.onDecal) return;
    const moved = Math.abs(e.clientX - pd.x) + Math.abs(e.clientY - pd.y);
    if (moved <= 4) return;
    pd.dragging = true;

    const sel = placements.current.find((p) => p.id === selectedId.current);
    if (!sel) return;

    setRay(e.clientX, e.clientY);
    const { base } = collectMeshes();
    const hit = raycaster.intersectObjects(base, false)[0];
    if (!hit || !hit.face) return;

    const target = hit.object as THREE.Mesh;
    target.updateWorldMatrix(true, false);
    const normalMatrix = new THREE.Matrix3().getNormalMatrix(target.matrixWorld);
    sel.targetMesh = target;
    sel.position.copy(hit.point);
    sel.normal.copy(hit.face.normal).applyMatrix3(normalMatrix).normalize();

    // Re-parent if the logo moved onto a different mesh.
    if (sel.mesh.parent !== target) target.add(sel.mesh);
    sel.mesh.geometry.dispose();
    sel.mesh.geometry = buildDecalGeometry(sel);
    refreshOutline();
  }, [setRay, collectMeshes, raycaster, refreshOutline]);

  const placeNew = useCallback((hit: THREE.Intersection) => {
    const texture = logoTextureRef.current;
    if (!texture || !hit.face) return;
    const target = hit.object as THREE.Mesh;
    target.updateWorldMatrix(true, false);
    const normalMatrix = new THREE.Matrix3().getNormalMatrix(target.matrixWorld);

    const material = new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      alphaTest: 0.05,
      depthTest: true,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -4,
      toneMapped: false,
    });
    const mesh = new THREE.Mesh(undefined, material);
    const id = ++idCounter.current;
    mesh.userData.isLogoDecal = true;
    mesh.userData.placementId = id;
    mesh.renderOrder = placements.current.length + 1;
    target.add(mesh);

    const p: Placement = {
      id, targetMesh: target,
      position: hit.point.clone(),
      normal: hit.face.normal.clone().applyMatrix3(normalMatrix).normalize(),
      size: logoSizeRef.current,
      rotation: logoRotationRef.current,
      texture, mesh,
    };
    mesh.geometry = buildDecalGeometry(p);
    placements.current.push(p);
    onCountChange(placements.current.length);
    selectPlacement(id);
  }, [onCountChange, selectPlacement]);

  const handlePointerUp = useCallback((e: React.PointerEvent) => {
    const pd = pointerDown.current;
    pointerDown.current = null;
    if (controls) controls.enabled = true;
    try { gl.domElement.releasePointerCapture(e.pointerId); } catch { /* noop */ }
    if (!pd) return;

    const moved = Math.abs(e.clientX - pd.x) + Math.abs(e.clientY - pd.y);
    if (pd.dragging || moved > 4) return; // was a drag (move logo or orbit)

    // A click: select a decal, or stamp a new logo on the bare surface.
    setRay(e.clientX, e.clientY);
    const { base, decals } = collectMeshes();
    const hitDecal = raycaster.intersectObjects(decals, false)[0];
    if (hitDecal) {
      selectPlacement((hitDecal.object as THREE.Mesh).userData.placementId as number);
      return;
    }
    const hitBase = raycaster.intersectObjects(base, false)[0];
    if (hitBase) placeNew(hitBase);
  }, [controls, gl, setRay, collectMeshes, raycaster, selectPlacement, placeNew]);

  if (!loadedScene) return null;

  return (
    <Center>
      <group
        ref={groupRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        <primitive object={loadedScene} />
      </group>
    </Center>
  );
}

/** Load a PNG/image File into an sRGB THREE.Texture suitable for a decal map. */
export function loadLogoTexture(file: File): Promise<THREE.Texture> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    new THREE.TextureLoader().load(
      url,
      (tex) => {
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.anisotropy = 4;
        URL.revokeObjectURL(url);
        resolve(tex);
      },
      undefined,
      (err) => { URL.revokeObjectURL(url); reject(err); },
    );
  });
}
