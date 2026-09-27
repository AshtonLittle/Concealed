"""
ConcealedPipeline: Master pipeline coordinating Client-Side Pre-Formatting,
High-Fidelity Adaptive Compression, and Platform Counter-Prevention Verification.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from PIL import Image

from .preformatting.formatter import ClientSideFormatter, PLATFORM_PROFILES
from .compression.compressor import AdaptiveCompressor, CompressionResult
from .counter_prevention.verifier import CounterPreventionVerifier, IngestionSimulationResult


class ConcealedPipeline:
    """
    High-level pipeline for client-side pre-formatting, compression,
    and counter-prevention verification.
    """

    def __init__(
        self,
        platform: str = "instagram_feed",
        fit_mode: str = "fit_width",
        format_type: str = "jpeg",
        default_quality: int = 90,
        chroma_subsampling: str = "444",
        progressive: bool = True,
        optimize: bool = True,
    ):
        self.platform = platform
        self.formatter = ClientSideFormatter(platform=platform, fit_mode=fit_mode)
        self.compressor = AdaptiveCompressor(
            format_type=format_type,
            default_quality=default_quality,
            chroma_subsampling=chroma_subsampling,
            progressive=progressive,
            optimize=optimize,
        )
        self.verifier = CounterPreventionVerifier()

    def process_image(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        target_size_kb: Optional[float] = None,
        quality: Optional[int] = None,
        chroma_subsampling: Optional[str] = None,
        platform: Optional[str] = None,
        verify_counter_prevention: bool = True,
    ) -> Tuple[Path, Dict[str, Any]]:
        """
        Processes an image end-to-end:
        1. Client-side pre-formatting (resolution, sRGB profile, EXIF sanitization)
        2. Adaptive high-fidelity compression
        3. Platform counter-prevention verification

        Returns:
            Tuple of (output_file_path, comprehensive_metrics_dict).
        """
        src = Path(input_path)
        if not src.is_file():
            raise FileNotFoundError(f"Input image not found: {src.resolve()}")

        # Determine output file path
        if output_path is None:
            suffix = f".{self.compressor.format_type.lower()}"
            if suffix == ".jpg":
                suffix = ".jpeg"
            output_path = src.parent / f"{src.stem}_compressed{suffix}"
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        orig_file_size = src.stat().st_size

        # 1. Load image
        with Image.open(src) as img:
            # 2. Client-side pre-formatting
            active_platform = platform or self.platform
            if active_platform != self.formatter.profile.name:
                active_formatter = ClientSideFormatter(platform=active_platform)
            else:
                active_formatter = self.formatter

            preformatted_img, format_metrics = active_formatter.format_image(img)

        # 3. High-fidelity adaptive compression
        compression_result: CompressionResult = self.compressor.compress_image(
            image=preformatted_img,
            output_path=dest,
            quality=quality,
            chroma_subsampling=chroma_subsampling,
            target_size_kb=target_size_kb,
            original_size_bytes=orig_file_size,
        )

        metrics: Dict[str, Any] = {
            "source_path": str(src.resolve()),
            "output_path": str(dest.resolve()),
            "initial_resolution": format_metrics["initial_size"],
            "formatted_resolution": format_metrics["final_size"],
            "platform_target": active_platform,
            "color_converted_to_srgb": format_metrics["color_converted_to_srgb"],
            "exif_stripped": format_metrics["exif_stripped"],
            "original_file_size_bytes": orig_file_size,
            "original_file_size_kb": round(orig_file_size / 1024.0, 2),
            "compressed_file_size_bytes": compression_result.compressed_size_bytes,
            "compressed_file_size_kb": compression_result.compressed_size_kb,
            "compression_ratio_percent": compression_result.compression_ratio_percent,
            "quality_factor": compression_result.quality,
            "chroma_subsampling": compression_result.chroma_subsampling,
            "bits_per_pixel": compression_result.bits_per_pixel,
        }

        # 4. Counter-prevention verification
        if verify_counter_prevention:
            # Reload saved image to verify actual on-disk bitstream
            with Image.open(dest) as saved_img:
                survival: IngestionSimulationResult = self.verifier.evaluate_survival(
                    candidate_image=saved_img,
                    platform=active_platform,
                    file_size_bytes=compression_result.compressed_size_bytes,
                )
                metrics["counter_prevention"] = {
                    "survival_psnr_db": survival.survival_psnr,
                    "survival_ssim": survival.survival_ssim,
                    "survival_mae": survival.survival_mae,
                    "simulated_platform_size_kb": survival.simulated_size_kb,
                    "risk_level": survival.risk_level,
                    "risk_factors": survival.risk_factors,
                    "recommendations": survival.recommendations,
                }

        return dest.resolve(), metrics


# Convenience alias
CompressionPipeline = ConcealedPipeline
