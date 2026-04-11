import type { TaskResponse, GenerationResult, GalleryItem, HealthStatus, GenerationSettings, ExportSettings, MultiImageMode } from '../types';

const API_BASE = '/api';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// -- Generation --------------------------------------------

export async function generateFromImage(
  image: File,
  settings: GenerationSettings,
  exportSettings: ExportSettings,
  engine: string = 'trellis',
  segmentedImagePath?: string,
): Promise<TaskResponse> {
  const form = new FormData();
  form.append('image', image);
  form.append('engine', engine);
  form.append('seed', String(settings.seed));
  form.append('randomize_seed', String(settings.randomizeSeed));
  // TRELLIS params
  form.append('ss_steps', String(settings.ssSteps));
  form.append('ss_guidance', String(settings.ssGuidance));
  form.append('slat_steps', String(settings.slatSteps));
  form.append('slat_guidance', String(settings.slatGuidance));
  // Hunyuan params
  if (engine === 'hunyuan') {
    form.append('num_inference_steps', String(settings.numInferenceSteps));
    form.append('guidance_scale', String(settings.guidanceScale));
    form.append('octree_resolution', String(settings.octreeResolution));
    form.append('texture', String(settings.texture));
    form.append('remove_floaters', String(settings.removeFloaters));
    if (settings.targetFaceCount > 0) {
      form.append('target_face_count', String(settings.targetFaceCount));
    }
    if (settings.roughnessOffset !== 0.0) {
      form.append('roughness_offset', String(settings.roughnessOffset));
    }
    if (settings.metallicScale !== 1.0) {
      form.append('metallic_scale', String(settings.metallicScale));
    }
  }
  // SAM 3D Objects params
  if (engine === 'sam3d') {
    form.append('sam3d_stage1_steps', String(settings.sam3dStage1Steps));
    form.append('sam3d_stage2_steps', String(settings.sam3dStage2Steps));
    form.append('sam3d_texture_baking', String(settings.sam3dTextureBaking));
    form.append('sam3d_vertex_color', String(settings.sam3dVertexColor));
  }
  // Segmented image path (from SAM3 segmentation)
  if (segmentedImagePath) {
    form.append('segmented_image_path', segmentedImagePath);
  }
  form.append('formats', exportSettings.formats.join(','));
  form.append('mesh_simplify', String(exportSettings.meshSimplify));
  form.append('texture_size', String(exportSettings.textureSize));

  return request<TaskResponse>('/generate/image', { method: 'POST', body: form });
}

export async function generateFromMultiImage(
  images: File[],
  mode: MultiImageMode,
  settings: GenerationSettings,
  exportSettings: ExportSettings,
  engine: string = 'trellis',
): Promise<TaskResponse> {
  const form = new FormData();
  images.forEach((img) => form.append('images', img));
  form.append('engine', engine);
  form.append('mode', mode);
  form.append('seed', String(settings.seed));
  form.append('randomize_seed', String(settings.randomizeSeed));
  // TRELLIS params
  form.append('ss_steps', String(settings.ssSteps));
  form.append('ss_guidance', String(settings.ssGuidance));
  form.append('slat_steps', String(settings.slatSteps));
  form.append('slat_guidance', String(settings.slatGuidance));
  // Hunyuan params
  if (engine === 'hunyuan') {
    form.append('num_inference_steps', String(settings.numInferenceSteps));
    form.append('guidance_scale', String(settings.guidanceScale));
    form.append('octree_resolution', String(settings.octreeResolution));
    form.append('texture', String(settings.texture));
    form.append('remove_floaters', String(settings.removeFloaters));
    if (settings.targetFaceCount > 0) {
      form.append('target_face_count', String(settings.targetFaceCount));
    }
    if (settings.roughnessOffset !== 0.0) {
      form.append('roughness_offset', String(settings.roughnessOffset));
    }
    if (settings.metallicScale !== 1.0) {
      form.append('metallic_scale', String(settings.metallicScale));
    }
  }
  form.append('formats', exportSettings.formats.join(','));
  form.append('mesh_simplify', String(exportSettings.meshSimplify));
  form.append('texture_size', String(exportSettings.textureSize));

  return request<TaskResponse>('/generate/multi-image', { method: 'POST', body: form });
}

export async function generateFromText(
  prompt: string,
  settings: GenerationSettings,
  exportSettings: ExportSettings,
): Promise<TaskResponse> {
  const form = new FormData();
  form.append('prompt', prompt);
  form.append('seed', String(settings.seed));
  form.append('randomize_seed', String(settings.randomizeSeed));
  form.append('ss_steps', String(settings.ssSteps));
  form.append('ss_guidance', String(settings.ssGuidance));
  form.append('slat_steps', String(settings.slatSteps));
  form.append('slat_guidance', String(settings.slatGuidance));
  form.append('formats', exportSettings.formats.join(','));
  form.append('mesh_simplify', String(exportSettings.meshSimplify));
  form.append('texture_size', String(exportSettings.textureSize));

  return request<TaskResponse>('/generate/text', { method: 'POST', body: form });
}

// -- Text-Guided Edit --------------------------------------

export async function editWithText(
  prompt: string,
  settings: GenerationSettings,
  exportSettings: ExportSettings,
  baseTaskId?: string,
  meshFile?: File,
): Promise<TaskResponse> {
  const form = new FormData();
  form.append('prompt', prompt);
  form.append('seed', String(settings.seed));
  form.append('randomize_seed', String(settings.randomizeSeed));
  form.append('slat_steps', String(settings.slatSteps));
  form.append('slat_guidance', String(settings.slatGuidance));
  form.append('formats', exportSettings.formats.join(','));
  form.append('mesh_simplify', String(exportSettings.meshSimplify));
  form.append('texture_size', String(exportSettings.textureSize));
  if (baseTaskId) form.append('base_task_id', baseTaskId);
  if (meshFile) form.append('mesh_file', meshFile);

  return request<TaskResponse>('/generate/edit', { method: 'POST', body: form });
}

// -- Task --------------------------------------------------

export async function getTaskStatus(taskId: string): Promise<GenerationResult> {
  return request<GenerationResult>(`/task/${taskId}`);
}

export async function cancelTask(taskId: string): Promise<void> {
  await request(`/task/${taskId}/cancel`, { method: 'POST' });
}

// -- Gallery -----------------------------------------------

export async function getGallery(page = 1, perPage = 20): Promise<{ items: GalleryItem[]; total: number }> {
  return request(`/gallery?page=${page}&per_page=${perPage}`);
}

export async function deleteGalleryItem(taskId: string): Promise<void> {
  await request(`/gallery/${taskId}`, { method: 'DELETE' });
}

export async function saveEditedToGallery(
  blobUrl: string,
  label = 'Edited',
  sourceModel?: string | null,
  sourceSeed?: number | null,
): Promise<GalleryItem> {
  const res = await fetch(blobUrl);
  const blob = await res.blob();
  const form = new FormData();
  form.append('file', blob, 'edited_model.glb');
  form.append('label', label);
  if (sourceModel) form.append('source_model', sourceModel);
  if (sourceSeed != null) form.append('seed', String(sourceSeed));
  const resp = await fetch(`${API_BASE}/gallery/edited`, { method: 'POST', body: form });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(err.detail || `Upload failed: ${resp.status}`);
  }
  const data = await resp.json();
  return data.item as GalleryItem;
}

// -- Health ------------------------------------------------

export async function getHealth(): Promise<HealthStatus> {
  return request('/health');
}

// -- WebSocket ---------------------------------------------

export function connectProgress(taskId: string, onMessage: (data: any) => void, onClose?: () => void): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${protocol}//${window.location.host}/ws/progress/${taskId}`);
  ws.onmessage = (e) => onMessage(JSON.parse(e.data));
  ws.onclose = () => onClose?.();
  ws.onerror = () => onClose?.();
  return ws;
}
