export interface SurrogateModelReport {
  model_name: string;
  architecture: string;
  target_class: string;
  patch_cosine_sim: number;
  salient_patch_cos: number;
  global_cos: number;
  concealed_patches_pct: number;
  reid_evasion_pct: number;
  raw_score: string;
  post_concealed_score: string;
  resistance_delta_pct: number;
  evasion_status: string;
}

export interface StealthMetrics {
  psnr_db: number;
  ssim: number;
  linf_255: number;
  rmse_255: number;
  chroma_rms_255: number;
  processing_time_ms: number;
  original_resolution: [number, number];
}

export interface BenchmarkAssetData {
  sourceType: 'image' | 'video';
  fileName: string;
  fileSize?: string;
  cleanImageUrl: string;
  obfuscatedImageUrl: string;
  diffHeatmapUrl: string;
  stealthMetrics: StealthMetrics;
  models: SurrogateModelReport[];
  timestamp: number;
  originalFile?: File | Blob;
}
