"""
AdaptiveCompressor: High-fidelity image compression engine.
Features:
- Adaptive quality tuning (Q=1 to 100)
- Chroma subsampling management (4:4:4 pristine color vs 4:2:0 bandwidth efficiency)
- Target file size rate-control (binary search optimization for strict size budgets)
- Progressive encoding and Huffman table optimization
- Support for JPEG, WebP, and PNG
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from PIL import Image, ImageFile

# Ensure large block sizes for libjpeg encoder buffer
ImageFile.MAXBLOCK = 64 * 1024 * 1024


@dataclass
class CompressionResult:
    output_path: Optional[Path]
    format: str
    dimensions: Tuple[int, int]
    original_size_bytes: int
    compressed_size_bytes: int
    compressed_size_kb: float
    compression_ratio_percent: float
    quality: int
    chroma_subsampling: str
    bits_per_pixel: float
    data_buffer: Optional[bytes] = None


class AdaptiveCompressor:
    """
    High-fidelity multi-format compression engine with target size budgeting.
    """

    def __init__(
        self,
        format_type: str = "jpeg",
        default_quality: int = 90,
        chroma_subsampling: str = "444",
        progressive: bool = True,
        optimize: bool = True,
    ):
        """
        Args:
            format_type: 'jpeg', 'webp', or 'png'.
            default_quality: Default quality factor Q (1-100). Default 90.
            chroma_subsampling: '444' (0), '422' (1), or '420' (2). Default '444'.
            progressive: Whether to enable progressive JPEG rendering.
            optimize: Whether to optimize Huffman/quantization tables.
        """
        self.format_type = format_type.lower()
        self.default_quality = default_quality
        self.chroma_subsampling = str(chroma_subsampling)
        self.progressive = progressive
        self.optimize = optimize

    def _resolve_subsampling_flag(self, subsampling_str: str) -> Union[int, str]:
        """Maps '444', '422', '420' to Pillow subsampling flags."""
        s = str(subsampling_str).strip().lower()
        if s in ("444", "0", "keep"):
            return 0  # 4:4:4 (no downsampling)
        elif s in ("422", "1"):
            return 1  # 4:2:2
        elif s in ("420", "2"):
            return 2  # 4:2:0
        return 0

    def compress_to_bytes(
        self,
        image: Image.Image,
        quality: Optional[int] = None,
        chroma_subsampling: Optional[str] = None,
        target_size_kb: Optional[float] = None,
    ) -> Tuple[bytes, int]:
        """
        Compresses PIL image into raw bytes in memory.
        If target_size_kb is specified, automatically searches for optimal quality Q.

        Returns:
            Tuple of (compressed_bytes, quality_used).
        """
        fmt = self.format_type.upper()
        if fmt == "JPG":
            fmt = "JPEG"

        q = quality if quality is not None else self.default_quality
        subsampling_flag = self._resolve_subsampling_flag(
            chroma_subsampling or self.chroma_subsampling
        )
        icc_profile = image.info.get("icc_profile")

        def _encode_at_quality(curr_q: int) -> bytes:
            save_kwargs: Dict[str, Any] = {"format": fmt}
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile

            if fmt == "JPEG":
                save_kwargs.update({
                    "quality": curr_q,
                    "optimize": self.optimize,
                    "subsampling": subsampling_flag,
                })
                # Attempt progressive encoding; fallback to baseline if libjpeg buffer complains
                if self.progressive:
                    try:
                        buf = io.BytesIO()
                        image.save(buf, progressive=True, **save_kwargs)
                        return buf.getvalue()
                    except Exception:
                        pass
                buf = io.BytesIO()
                image.save(buf, progressive=False, **save_kwargs)
                return buf.getvalue()

            elif fmt == "WEBP":
                save_kwargs.update({
                    "quality": curr_q,
                    "method": 6,
                    "lossless": False,
                })
                buf = io.BytesIO()
                image.save(buf, **save_kwargs)
                return buf.getvalue()

            elif fmt == "PNG":
                save_kwargs.update({
                    "optimize": self.optimize,
                })
                buf = io.BytesIO()
                image.save(buf, **save_kwargs)
                return buf.getvalue()

            buf = io.BytesIO()
            image.save(buf, **save_kwargs)
            return buf.getvalue()

        # If PNG or no target size specified, encode directly at requested quality
        if fmt == "PNG" or target_size_kb is None:
            data = _encode_at_quality(q)
            return data, q

        # Target size budget specified -> Binary search for optimal Q
        target_bytes = int(target_size_kb * 1024)
        low_q = 5
        high_q = 98
        best_data: Optional[bytes] = None
        best_q = low_q

        while low_q <= high_q:
            mid_q = (low_q + high_q) // 2
            encoded = _encode_at_quality(mid_q)
            encoded_len = len(encoded)

            if encoded_len <= target_bytes:
                # Valid candidate, try for higher quality
                best_data = encoded
                best_q = mid_q
                low_q = mid_q + 1
            else:
                # Exceeded target size, reduce quality
                high_q = mid_q - 1

        if best_data is None:
            # Even minimum quality was above target bytes, return lowest quality data
            best_data = _encode_at_quality(5)
            best_q = 5

        return best_data, best_q

    def compress_image(
        self,
        image: Image.Image,
        output_path: Optional[Union[str, Path]] = None,
        quality: Optional[int] = None,
        chroma_subsampling: Optional[str] = None,
        target_size_kb: Optional[float] = None,
        original_size_bytes: int = 0,
    ) -> CompressionResult:
        """
        Compresses an image, writes to file (if path provided), and returns detailed metrics.
        """
        raw_bytes, used_q = self.compress_to_bytes(
            image=image,
            quality=quality,
            chroma_subsampling=chroma_subsampling,
            target_size_kb=target_size_kb,
        )

        compressed_len = len(raw_bytes)
        compressed_kb = round(compressed_len / 1024.0, 2)

        orig_len = original_size_bytes if original_size_bytes > 0 else (image.width * image.height * 3)
        saved_pct = round(max(0.0, (1.0 - (compressed_len / float(orig_len))) * 100.0), 2)
        total_pixels = image.width * image.height
        bpp = round((compressed_len * 8.0) / float(total_pixels), 4)

        dest_path = None
        if output_path is not None:
            dest_path = Path(output_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as f:
                f.write(raw_bytes)

        return CompressionResult(
            output_path=dest_path,
            format=self.format_type.upper(),
            dimensions=image.size,
            original_size_bytes=orig_len,
            compressed_size_bytes=compressed_len,
            compressed_size_kb=compressed_kb,
            compression_ratio_percent=saved_pct,
            quality=used_q,
            chroma_subsampling=chroma_subsampling or self.chroma_subsampling,
            bits_per_pixel=bpp,
            data_buffer=raw_bytes if dest_path is None else None,
        )
