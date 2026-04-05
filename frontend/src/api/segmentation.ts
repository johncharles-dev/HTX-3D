import type { SegmentStartResponse, SegmentResponse, ConfirmSegmentResponse } from '../types/segmentation';

const API_BASE = '/api/segment';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${url}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function startSegmentation(image: File): Promise<SegmentStartResponse> {
  const form = new FormData();
  form.append('image', image);
  return request<SegmentStartResponse>('/start', { method: 'POST', body: form });
}

export async function segmentText(sessionId: string, prompt: string): Promise<SegmentResponse> {
  return request<SegmentResponse>('/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, prompt }),
  });
}

export async function segmentBox(
  sessionId: string,
  box: number[],
  label: boolean = true,
): Promise<SegmentResponse> {
  return request<SegmentResponse>('/box', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, box, label }),
  });
}

export async function segmentPoint(
  sessionId: string,
  x: number,
  y: number,
  label: boolean = true,
): Promise<SegmentResponse> {
  return request<SegmentResponse>('/point', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, x, y, label }),
  });
}

export async function segmentPoints(
  sessionId: string,
  points: number[][],
  labels: number[],
): Promise<SegmentResponse> {
  return request<SegmentResponse>('/points', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, points, labels }),
  });
}

export async function resetSegmentation(sessionId: string): Promise<void> {
  await request(`/reset?session_id=${sessionId}`, {
    method: 'POST',
  });
}

export async function confirmSegmentation(
  sessionId: string,
  maskIndex: number = 0,
): Promise<ConfirmSegmentResponse> {
  return request<ConfirmSegmentResponse>('/confirm', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, mask_index: maskIndex }),
  });
}
