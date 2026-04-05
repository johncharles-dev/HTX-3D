export interface SegmentStartResponse {
  session_id: string;
  width: number;
  height: number;
  message: string;
}

export interface MaskResult {
  index: number;
  score: number;
  bbox: number[];
  area_pixels: number;
}

export interface SegmentResponse {
  session_id: string;
  masks: MaskResult[];
  overlay_url: string;
  message: string;
}

export interface ConfirmSegmentResponse {
  session_id: string;
  segmented_image_path: string;
  message: string;
}
