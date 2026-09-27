"""Pydantic schemas and configuration models for Concealed API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class SynthesisModeEnum(str, Enum):
    HYBRID = "hybrid"
    CANONICAL_RESIDUAL = "canonical_residual"
    NATIVE = "native"


class OutputFormatEnum(str, Enum):
    PNG = "PNG"
    JPEG = "JPEG"
    WEBP = "WEBP"
    ORIGINAL = "ORIGINAL"


class ResponseTypeEnum(str, Enum):
    IMAGE = "image"
    JSON = "json"


class GeneratorVariantEnum(str, Enum):
    TINY = "tiny"
    BASE = "base"
    LARGE = "large"


class ObfuscationParams(BaseModel):
    """Complete parameter specification for AI image obfuscation."""

    epsilon: float = Field(
        default=8.0,
        ge=0.5,
        le=64.0,
        description="L_infinity perturbation budget bound in 8-bit scale [0, 255]. Higher values offer stronger AI evasion.",
    )
    mode: SynthesisModeEnum = Field(
        default=SynthesisModeEnum.HYBRID,
        description="Model synthesis mode: 'hybrid' (canonical + high-res tiles), 'canonical_residual' (downscaled global), or 'native' (full-res reflection-padded).",
    )
    target_features: Optional[str] = Field(
        default=None,
        description="Target semantic silhouette categories (e.g. 'face', 'person', 'text', 'body', 'all'). None applies across full image.",
    )
    conforming_mask: bool = Field(
        default=False,
        description="When True, bounds perturbations strictly inside detected feature contours with feathered transitions.",
    )
    feather_radius: int = Field(
        default=8,
        ge=0,
        le=100,
        description="Gaussian edge-feathering radius in pixels to ensure smooth, invisible mask boundaries.",
    )
    texture_masking: bool = Field(
        default=True,
        description="Weber's law contrast-adaptive masking: concentrates noise in high-frequency textured regions and leaves smooth surfaces clean.",
    )
    chroma_damping: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
        description="Opponent chrominance damping factor: suppresses magenta/green chromatic noise shifts to maximize visual stealth.",
    )
    refine_steps: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Optional iterative gradient refinement steps against surrogate ViT models.",
    )
    strip_metadata: bool = Field(
        default=True,
        description="Strip EXIF, GPS coordinates, camera serial numbers, and sensitive metadata from the resulting image.",
    )
    output_format: OutputFormatEnum = Field(
        default=OutputFormatEnum.PNG,
        description="Desired image format of the processed output ('PNG', 'JPEG', 'WEBP', or 'ORIGINAL').",
    )
    quality: int = Field(
        default=95,
        ge=1,
        le=100,
        description="Compression quality factor for lossy formats (JPEG and WEBP).",
    )
    canonical_size: int = Field(
        default=384,
        ge=128,
        le=1024,
        description="Canonical image dimension for scale-invariant residual synthesis.",
    )
    hybrid_global_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Balancing weight between global thumbnail pass and high-resolution local tile passes in hybrid mode.",
    )
    response_type: ResponseTypeEnum = Field(
        default=ResponseTypeEnum.IMAGE,
        description="Payload format: 'image' returns direct binary image stream; 'json' returns base64 string with analytical metrics.",
    )


    model_engine: Optional[str] = Field(
        default="auto",
        description="Model execution engine: 'onnx' (generator.onnx), 'pt' (best_generator.pt), or 'auto'.",
    )


class ObfuscationJSONRequest(ObfuscationParams):
    """JSON payload for base64 image obfuscation."""

    image_base64: str = Field(
        ...,
        description="Base64-encoded image string (with or without 'data:image/...;base64,' prefix).",
    )


class ObfuscationAnalytics(BaseModel):
    """Quality and stealth evaluation metrics for the obfuscation result."""

    psnr_db: float = Field(..., description="Peak Signal-to-Noise Ratio in decibels.")
    ssim: float = Field(..., description="Structural Similarity Index [0.0 - 1.0].")
    linf_255: float = Field(..., description="Measured maximum L_infinity perturbation in [0, 255].")
    rmse_255: float = Field(..., description="Root Mean Square Error in [0, 255].")
    chroma_rms_255: float = Field(..., description="Root Mean Square Error of chrominance channel shift in [0, 255].")
    processing_time_ms: float = Field(..., description="Wall-clock time spent processing in milliseconds.")
    original_resolution: Tuple[int, int] = Field(..., description="(Width, Height) of original input.")
    output_resolution: Tuple[int, int] = Field(..., description="(Width, Height) of obfuscated output.")
    output_bytes: int = Field(..., description="Size of generated image in bytes.")
    quality_loss_pct: float = Field(default=0.0, description="Measured visual quality loss percentage vs clean image (100 * (1 - SSIM)).")


class ObfuscationJSONResponse(BaseModel):
    """Response returned when response_type='json'."""

    success: bool = True
    message: str = "Image successfully obfuscated"
    mime_type: str
    format: str
    image_base64: str
    parameters: Dict[str, Any]
    analytics: ObfuscationAnalytics


class ParameterSpec(BaseModel):
    name: str
    type: str
    default: Any
    description: str
    options: Optional[List[Any]] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class ParametersInfoResponse(BaseModel):
    """Documentation and allowable bounds for all generator parameters."""

    title: str = "Concealed AI Obfuscator Parameter Catalog"
    version: str = "0.1.0"
    parameters: List[ParameterSpec]


class HealthResponse(BaseModel):
    """Service health and runtime engine status."""

    status: str = "healthy"
    engine: str
    device: str
    version: str = "0.1.0"
    supported_formats: List[str] = ["PNG", "JPEG", "WEBP"]
    supported_modes: List[str] = ["hybrid", "canonical_residual", "native"]


class ModelSpec(BaseModel):
    """Specification of an available obfuscation model checkpoint."""

    id: str
    name: str
    type: str  # 'onnx' | 'pt' | 'algorithmic'
    filename: str
    path: str
    available: bool = True
    description: str
    speed_tier: str  # 'Ultra-Fast' | 'Balanced' | 'Precision'
    default_for_video: bool = False
    default_for_image: bool = False


class ModelsCatalogResponse(BaseModel):
    """Catalog of available generator models and view defaults."""

    models: List[ModelSpec]
    image_default: str
    video_default: str
    current_image_model: str


class ModelProbeSpec(BaseModel):
    """Specification of a Vision Transformer model available for probing/auditing."""

    id: str
    name: str
    architecture: str
    description: str
    family: str  # 'CLIP' | 'SigLIP' | 'DINO' | 'VLM' | 'ViT'
    target_layer: str
    badge: str
    default_selected: bool = True


class FeatureConfidence(BaseModel):
    """PaliGemma feature confidence analysis comparing Clean vs Concealed perception."""

    feature: str
    clean_confidence_pct: float
    concealed_confidence_pct: float
    confidence_drop_pct: float
    status: str  # 'EVADED' | 'ATTENUATED' | 'DETECTED'
    plain_english_insight: str


class ModelProbeResult(BaseModel):
    """Probe result comparing Clean vs Concealed outputs for a specific Vision Transformer."""

    model_id: str
    model_name: str
    architecture: str
    family: str
    prompt: str
    clean_output: str
    concealed_output: str
    evasion_status: str  # 'EVADED' | 'DISRUPTED' | 'DEGRADED'
    evasion_score_pct: float  # [0.0 - 100.0]
    cosine_similarity_clean: float
    cosine_similarity_concealed: float
    similarity_drop_pct: float
    feature_dispersion_pct: float
    latency_ms: float
    plain_english_verdict: Optional[str] = None
    feature_confidences: Optional[List[FeatureConfidence]] = None
    details: Dict[str, Any] = {}


class ProbeResponse(BaseModel):
    """Response returned by the /api/probe endpoint."""

    success: bool = True
    prompt: str
    total_models_probed: int
    overall_evasion_pct: float
    results: List[ModelProbeResult]
    clean_image_url: Optional[str] = None
    concealed_image_url: Optional[str] = None
    processing_time_ms: float
    conceal_engine_used: str
    paligemma_plain_english_summary: Optional[str] = None
    paligemma_feature_audit: Optional[List[FeatureConfidence]] = None


class SiglipOptionScore(BaseModel):
    """Real SigLIP confidence score for a specific option."""

    option: str
    clean_confidence_pct: float
    concealed_confidence_pct: float
    confidence_drop_pct: float
    status: str  # 'Hidden' | 'Weakened' | 'Visible'


class SiglipProbeResponse(BaseModel):
    """Response from real SigLIP probe on a user-provided list of options."""

    success: bool = True
    clean_image_url: Optional[str] = None
    concealed_image_url: Optional[str] = None
    overall_protection_pct: float
    options_hidden_count: int
    total_options: int
    avg_confidence_drop_pct: float
    results: List[SiglipOptionScore]


