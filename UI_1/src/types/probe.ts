export interface ModelProbeSpec {
  id: string;
  name: string;
  architecture: string;
  description: string;
  family: 'CLIP' | 'SigLIP' | 'DINO' | 'VLM' | 'ViT' | string;
  target_layer: string;
  badge: string;
  default_selected: boolean;
}

export interface FeatureConfidence {
  feature: string;
  clean_confidence_pct: number;
  concealed_confidence_pct: number;
  confidence_drop_pct: number;
  status: 'EVADED' | 'ATTENUATED' | 'DETECTED' | string;
  plain_english_insight: string;
}

export interface ModelProbeResult {
  model_id: string;
  model_name: string;
  architecture: string;
  family: string;
  prompt: string;
  clean_output: string;
  concealed_output: string;
  evasion_status: 'EVADED' | 'DISRUPTED' | 'DEGRADED' | string;
  evasion_score_pct: number;
  cosine_similarity_clean: number;
  cosine_similarity_concealed: number;
  similarity_drop_pct: number;
  feature_dispersion_pct: number;
  latency_ms: number;
  plain_english_verdict?: string;
  feature_confidences?: FeatureConfidence[];
  details?: Record<string, any>;
}

export interface ProbeResponse {
  success: boolean;
  prompt: string;
  total_models_probed: number;
  overall_evasion_pct: number;
  results: ModelProbeResult[];
  clean_image_url?: string | null;
  concealed_image_url?: string | null;
  processing_time_ms: number;
  conceal_engine_used: string;
  paligemma_plain_english_summary?: string;
  paligemma_feature_audit?: FeatureConfidence[];
  obfuscation_epsilon?: number | null;
  psnr_db?: number | null;
  ssim?: number | null;
  linf_255?: number | null;
  rmse_255?: number | null;
}

export interface ModelSpec {
  id: string;
  name: string;
  type: 'onnx' | 'pt' | 'algorithmic' | string;
  filename: string;
  path: string;
  available: boolean;
  description: string;
  speed_tier: string;
  default_for_video: boolean;
  default_for_image: boolean;
}

export interface ModelsCatalogResponse {
  models: ModelSpec[];
  image_default: string;
  video_default: string;
  current_image_model: string;
}

export interface SiglipOptionScore {
  option: string;
  clean_confidence_pct: number;
  concealed_confidence_pct: number;
  confidence_drop_pct: number;
  status: 'Hidden' | 'Weakened' | 'Visible' | string;
}

export interface SiglipProbeResponse {
  success: boolean;
  clean_image_url?: string | null;
  concealed_image_url?: string | null;
  overall_protection_pct: number;
  options_hidden_count: number;
  total_options: number;
  avg_confidence_drop_pct: number;
  results: SiglipOptionScore[];
  conceal_engine_used?: string | null;
  obfuscation_epsilon?: number | null;
  psnr_db?: number | null;
  ssim?: number | null;
  linf_255?: number | null;
  rmse_255?: number | null;
}

