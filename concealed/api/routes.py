"""FastAPI route definitions for Concealed Image Obfuscation API."""

from __future__ import annotations

import base64
import json
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status

from concealed.api.schemas import (
    HealthResponse,
    ObfuscationAnalytics,
    ObfuscationJSONRequest,
    ObfuscationJSONResponse,
    ObfuscationParams,
    OutputFormatEnum,
    ParameterSpec,
    ParametersInfoResponse,
    ResponseTypeEnum,
    SynthesisModeEnum,
)
from concealed.api.service import ObfuscationService

router = APIRouter(prefix="/api", tags=["Obfuscation"])
service = ObfuscationService()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health, device backend, and supported synthesis modes."""
    status_info = service.get_status()
    return HealthResponse(
        status="healthy",
        engine=status_info.get("backend", "Algorithmic-DCT-Engine"),
        device=status_info.get("device", "cpu"),
        version="0.1.0",
        supported_formats=["PNG", "JPEG", "WEBP"],
        supported_modes=["hybrid", "canonical_residual", "native"],
    )


@router.get("/parameters", response_model=ParametersInfoResponse)
async def get_parameters_catalog() -> ParametersInfoResponse:
    """Return exhaustive documentation, ranges, and defaults for all obfuscation parameters."""
    specs = [
        ParameterSpec(
            name="epsilon",
            type="float",
            default=8.0,
            description="L_infinity perturbation budget bound in 8-bit scale [0, 255]. Higher values offer stronger AI evasion.",
            min_value=0.5,
            max_value=64.0,
        ),
        ParameterSpec(
            name="mode",
            type="string",
            default="hybrid",
            description="Synthesis mode: 'hybrid' (canonical + high-res tiles), 'canonical_residual' (downscaled global), or 'native' (full-res reflection-padded).",
            options=["hybrid", "canonical_residual", "native"],
        ),
        ParameterSpec(
            name="target_features",
            type="string",
            default=None,
            description="Comma-separated target silhouette categories (e.g. 'face', 'person', 'text', 'body'). If omitted, applies across entire image.",
            options=["face", "person", "body", "text", "all"],
        ),
        ParameterSpec(
            name="conforming_mask",
            type="boolean",
            default=False,
            description="When True, bounds perturbations strictly inside detected feature contours with feathered transitions.",
            options=[True, False],
        ),
        ParameterSpec(
            name="feather_radius",
            type="integer",
            default=8,
            description="Gaussian edge-feathering radius in pixels to ensure smooth, invisible mask boundaries.",
            min_value=0,
            max_value=100,
        ),
        ParameterSpec(
            name="texture_masking",
            type="boolean",
            default=True,
            description="Weber's law contrast-adaptive masking: concentrates noise in high-frequency textured regions and leaves smooth surfaces clean.",
            options=[True, False],
        ),
        ParameterSpec(
            name="chroma_damping",
            type="float",
            default=0.70,
            description="Opponent chrominance damping factor: suppresses magenta/green chromatic noise shifts to maximize visual stealth.",
            min_value=0.0,
            max_value=1.0,
        ),
        ParameterSpec(
            name="refine_steps",
            type="integer",
            default=0,
            description="Optional iterative gradient refinement steps against surrogate ViT models.",
            min_value=0,
            max_value=50,
        ),
        ParameterSpec(
            name="strip_metadata",
            type="boolean",
            default=True,
            description="Strip EXIF, GPS coordinates, camera serial numbers, and sensitive metadata from the resulting image.",
            options=[True, False],
        ),
        ParameterSpec(
            name="output_format",
            type="string",
            default="PNG",
            description="Target image format: 'PNG', 'JPEG', 'WEBP', or 'ORIGINAL'.",
            options=["PNG", "JPEG", "WEBP", "ORIGINAL"],
        ),
        ParameterSpec(
            name="quality",
            type="integer",
            default=95,
            description="Compression quality factor for lossy formats (JPEG and WEBP).",
            min_value=1,
            max_value=100,
        ),
        ParameterSpec(
            name="canonical_size",
            type="integer",
            default=384,
            description="Canonical image dimension for scale-invariant residual synthesis.",
            min_value=128,
            max_value=1024,
        ),
        ParameterSpec(
            name="hybrid_global_weight",
            type="float",
            default=0.60,
            description="Balancing weight between global thumbnail pass and high-resolution local tile passes in hybrid mode.",
            min_value=0.0,
            max_value=1.0,
        ),
        ParameterSpec(
            name="response_type",
            type="string",
            default="image",
            description="Response payload format: 'image' (binary stream) or 'json' (base64 string + analytics).",
            options=["image", "json"],
        ),
    ]
    return ParametersInfoResponse(parameters=specs)


@router.post("/obfuscate")
async def obfuscate_image_upload(
    file: UploadFile = File(..., description="Input image file to obfuscate"),
    epsilon: float = Form(8.0, ge=0.5, le=64.0, description="L_infinity perturbation budget in [0, 255]"),
    mode: SynthesisModeEnum = Form(SynthesisModeEnum.HYBRID, description="Synthesis mode"),
    target_features: Optional[str] = Form(None, description="Comma-separated target silhouette categories"),
    conforming_mask: bool = Form(False, description="Restrict perturbation strictly within contours"),
    feather_radius: int = Form(8, ge=0, le=100, description="Mask edge feather radius in pixels"),
    texture_masking: bool = Form(True, description="Weber's law contrast-adaptive masking"),
    chroma_damping: float = Form(0.70, ge=0.0, le=1.0, description="Opponent chrominance damping factor"),
    refine_steps: int = Form(0, ge=0, le=50, description="Test-time ViT refinement steps"),
    strip_metadata: bool = Form(True, description="Remove EXIF, GPS, camera metadata"),
    output_format: OutputFormatEnum = Form(OutputFormatEnum.PNG, description="Output image format"),
    quality: int = Form(95, ge=1, le=100, description="Quality for lossy formats"),
    canonical_size: int = Form(384, ge=128, le=1024, description="Canonical residual size"),
    hybrid_global_weight: float = Form(0.60, ge=0.0, le=1.0, description="Global vs tile weight"),
    response_type: ResponseTypeEnum = Form(ResponseTypeEnum.IMAGE, description="Return binary image or JSON"),
) -> Response:
    """Upload an image, configure all obfuscation parameters, and receive the processed image back."""
    # Read uploaded file
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading uploaded file: {e}",
        ) from e

    # Build params object
    params = ObfuscationParams(
        epsilon=epsilon,
        mode=mode,
        target_features=target_features,
        conforming_mask=conforming_mask,
        feather_radius=feather_radius,
        texture_masking=texture_masking,
        chroma_damping=chroma_damping,
        refine_steps=refine_steps,
        strip_metadata=strip_metadata,
        output_format=output_format,
        quality=quality,
        canonical_size=canonical_size,
        hybrid_global_weight=hybrid_global_weight,
        response_type=response_type,
    )

    # Execute obfuscation
    try:
        out_bytes, mime_type, analytics = service.process_image(image_bytes, params)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image obfuscation failed: {e}",
        ) from e

    # Determine file extension for Content-Disposition
    ext = mime_type.split("/")[-1]
    if ext == "jpeg":
        ext = "jpg"

    # Handle binary image stream response
    if response_type == ResponseTypeEnum.IMAGE:
        headers = {
            "Content-Disposition": f'inline; filename="concealed_obfuscated.{ext}"',
            "X-Processing-Time-Ms": str(analytics.processing_time_ms),
            "X-Epsilon": str(params.epsilon),
            "X-Mode": params.mode.value,
            "X-PSNR-dB": str(analytics.psnr_db),
            "X-SSIM": str(analytics.ssim),
            "X-Linf-255": str(analytics.linf_255),
            "X-Output-Format": ext.upper(),
            "X-Model": getattr(service, "model_filename", "best_generator.pt"),
            "X-Quality-Loss-Pct": str(analytics.quality_loss_pct),
        }
        return Response(content=out_bytes, media_type=mime_type, headers=headers)

    # Handle JSON response with base64 encoded image
    b64_str = base64.b64encode(out_bytes).decode("ascii")
    resp_obj = ObfuscationJSONResponse(
        success=True,
        message="Image successfully obfuscated",
        mime_type=mime_type,
        format=ext.upper(),
        image_base64=f"data:{mime_type};base64,{b64_str}",
        parameters=params.model_dump(),
        analytics=analytics,
    )
    return Response(
        content=resp_obj.model_dump_json(),
        media_type="application/json",
    )


@router.post("/obfuscate/json", response_model=ObfuscationJSONResponse)
async def obfuscate_image_json(request: ObfuscationJSONRequest) -> ObfuscationJSONResponse:
    """Submit a base64-encoded image with complete configuration parameters and receive base64 output with metrics."""
    try:
        b64_out, mime_type, analytics = service.process_base64(request.image_base64, request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Base64 image obfuscation failed: {e}",
        ) from e

    ext = mime_type.split("/")[-1].upper()
    return ObfuscationJSONResponse(
        success=True,
        message="Image successfully obfuscated",
        mime_type=mime_type,
        format=ext,
        image_base64=f"data:{mime_type};base64,{b64_out}",
        parameters=request.model_dump(exclude={"image_base64"}),
        analytics=analytics,
    )


@router.post("/obscure-feature")
@router.post("/feature-obscure")
async def obscure_feature(
    file: UploadFile = File(..., description="Input image file to obscure specific features in"),
    feature: str = Form(..., description="Target feature or comma-separated features (e.g. face, text, person, car, waterbottle, etc.)"),
    conf: float = Form(0.25, ge=0.01, le=1.0, description="Detection confidence threshold"),
    show_boxes: bool = Form(False, description="Draw bounding boxes around detected features"),
    show_contours: bool = Form(False, description="Draw conforming silhouette contours"),
    output_format: str = Form("PNG", description="Output format (PNG, JPEG, WEBP)"),
    response_type: ResponseTypeEnum = Form(ResponseTypeEnum.IMAGE, description="Response format ('image' or 'json')"),
) -> Response:
    """Detect specified features conforming to object silhouettes and obscure them using Imageprocessor & detector."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading uploaded file: {e}",
        ) from e

    try:
        out_bytes, mime_type, regions, elapsed_ms = service.obscure_feature(
            image_bytes=image_bytes,
            feature=feature,
            conf=conf,
            show_boxes=show_boxes,
            show_contours=show_contours,
            output_format=output_format,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Feature obscuring failed: {e}",
        ) from e

    ext = mime_type.split("/")[-1]
    if ext == "jpeg":
        ext = "jpg"

    if response_type == ResponseTypeEnum.IMAGE:
        headers = {
            "Content-Disposition": f'inline; filename="obscured_{feature}.{ext}"',
            "X-Feature": str(feature),
            "X-Regions-Found": str(len(regions)),
            "X-Processing-Time-Ms": str(elapsed_ms),
            "Access-Control-Expose-Headers": "Content-Disposition, X-Feature, X-Regions-Found, X-Processing-Time-Ms",
        }
        return Response(content=out_bytes, media_type=mime_type, headers=headers)

    b64_out = base64.b64encode(out_bytes).decode("ascii")
    return Response(
        content=json.dumps({
            "success": True,
            "feature": feature,
            "regions_count": len(regions),
            "regions": regions,
            "processing_time_ms": elapsed_ms,
            "mime_type": mime_type,
            "image_base64": f"data:{mime_type};base64,{b64_out}",
        }),
        media_type="application/json",
    )


@router.post("/obfuscate/video")
async def obfuscate_video(
    file: UploadFile = File(..., description="Video file to obfuscate"),
    epsilon: float = Form(8.0, ge=0.5, le=64.0, description="L_infinity perturbation budget"),
    mode: SynthesisModeEnum = Form(SynthesisModeEnum.HYBRID, description="Synthesis mode"),
    max_frames: Optional[int] = Form(None, description="Optional cap on frames to process"),
) -> Response:
    """Process video file through adversarial obfuscation engine, shielding frames against AI vision models."""
    try:
        video_bytes = await file.read()
        if not video_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded video file is empty.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error reading uploaded video file: {e}",
        ) from e

    params = ObfuscationParams(
        epsilon=epsilon,
        mode=mode,
    )

    try:
        out_bytes, mime_type, meta = service.process_video(video_bytes, params, max_frames=max_frames)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Video obfuscation failed: {e}",
        ) from e

    headers = {
        "Content-Disposition": 'inline; filename="concealed_video.mp4"',
        "X-Frames-Processed": str(meta.get("frames_processed", 0)),
        "X-FPS": str(meta.get("fps", 24)),
        "X-Processing-Time-Ms": str(meta.get("processing_time_ms", 0)),
        "Access-Control-Expose-Headers": "Content-Disposition, X-Frames-Processed, X-FPS, X-Processing-Time-Ms",
    }
    return Response(content=out_bytes, media_type=mime_type, headers=headers)


