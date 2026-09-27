"""Python client SDK for the Concealed Obfuscation API.

Allows calling the Concealed backend API in Python with full control over all
obfuscation parameters and receiving the processed image back as a PIL Image,
file, or byte buffer.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import Image

try:
    import httpx
    _HTTP_BACKEND = "httpx"
except ImportError:
    try:
        import requests
        _HTTP_BACKEND = "requests"
    except ImportError:
        _HTTP_BACKEND = "urllib"


class ConcealedClient:
    """Python client for interacting with the Concealed Image Obfuscation API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001", timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Check API service health, active backend engine, and supported formats."""
        url = f"{self.base_url}/api/health"
        if _HTTP_BACKEND == "httpx":
            import httpx
            resp = httpx.get(url, timeout=self.timeout)
            return resp.json()
        elif _HTTP_BACKEND == "requests":
            import requests
            resp = requests.get(url, timeout=self.timeout)
            return resp.json()
        else:
            import json
            import urllib.request
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                return json.loads(r.read())

    def get_parameters(self) -> Dict[str, Any]:
        """Retrieve full documentation and allowable bounds for all model parameters."""
        url = f"{self.base_url}/api/parameters"
        if _HTTP_BACKEND == "httpx":
            import httpx
            resp = httpx.get(url, timeout=self.timeout)
            return resp.json()
        elif _HTTP_BACKEND == "requests":
            import requests
            resp = requests.get(url, timeout=self.timeout)
            return resp.json()
        else:
            import json
            import urllib.request
            with urllib.request.urlopen(url, timeout=self.timeout) as r:
                return json.loads(r.read())

    def obfuscate(
        self,
        image: Union[str, Path, bytes, Image.Image],
        epsilon: float = 8.0,
        mode: str = "hybrid",
        target_features: Optional[str] = None,
        conforming_mask: bool = False,
        feather_radius: int = 8,
        texture_masking: bool = True,
        chroma_damping: float = 0.70,
        refine_steps: int = 0,
        strip_metadata: bool = True,
        output_format: str = "PNG",
        quality: int = 95,
        canonical_size: int = 384,
        hybrid_global_weight: float = 0.60,
    ) -> Image.Image:
        """Upload an image with custom obfuscation parameters and receive the obfuscated PIL Image back.

        Args:
            image: Path to image file, raw image bytes, or an existing PIL Image.
            epsilon: L_infinity perturbation budget bound in [0.5, 64.0] (default: 8.0).
            mode: Synthesis mode ('hybrid', 'canonical_residual', 'native').
            target_features: Optional target classes (e.g. 'face', 'person', 'text').
            conforming_mask: Restrict perturbations strictly inside detected contours.
            feather_radius: Blur radius in pixels for smooth contour edges.
            texture_masking: Weber's law contrast-adaptive masking.
            chroma_damping: Factor [0.0 - 1.0] suppressing magenta/green color tinting.
            refine_steps: Gradient ViT repulsion refinement steps.
            strip_metadata: Strip EXIF, GPS coordinates, and camera metadata.
            output_format: Target format ('PNG', 'JPEG', 'WEBP', or 'ORIGINAL').
            quality: Compression quality factor for JPEG/WEBP.
            canonical_size: Dimension for scale-invariant residual synthesis.
            hybrid_global_weight: Global vs high-res tile blending ratio.

        Returns:
            PIL.Image.Image: The processed obfuscated image at native resolution.
        """
        out_image, _ = self.obfuscate_with_analytics(
            image=image,
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
        )
        return out_image

    def obfuscate_with_analytics(
        self,
        image: Union[str, Path, bytes, Image.Image],
        epsilon: float = 8.0,
        mode: str = "hybrid",
        target_features: Optional[str] = None,
        conforming_mask: bool = False,
        feather_radius: int = 8,
        texture_masking: bool = True,
        chroma_damping: float = 0.70,
        refine_steps: int = 0,
        strip_metadata: bool = True,
        output_format: str = "PNG",
        quality: int = 95,
        canonical_size: int = 384,
        hybrid_global_weight: float = 0.60,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """Upload an image, receive the obfuscated PIL Image back along with stealth telemetry analytics."""
        # Convert input to raw bytes
        if isinstance(image, (str, Path)):
            with open(image, "rb") as f:
                img_bytes = f.read()
            filename = Path(image).name
        elif isinstance(image, Image.Image):
            buf = io.BytesIO()
            fmt = image.format or "PNG"
            image.save(buf, format=fmt)
            img_bytes = buf.getvalue()
            filename = f"image.{fmt.lower()}"
        elif isinstance(image, bytes):
            img_bytes = image
            filename = "image.png"
        else:
            raise TypeError(f"Unsupported image input type: {type(image)}")

        # Prepare form fields
        data = {
            "epsilon": str(epsilon),
            "mode": mode,
            "conforming_mask": "true" if conforming_mask else "false",
            "feather_radius": str(feather_radius),
            "texture_masking": "true" if texture_masking else "false",
            "chroma_damping": str(chroma_damping),
            "refine_steps": str(refine_steps),
            "strip_metadata": "true" if strip_metadata else "false",
            "output_format": output_format,
            "quality": str(quality),
            "canonical_size": str(canonical_size),
            "hybrid_global_weight": str(hybrid_global_weight),
            "response_type": "image",
        }
        if target_features:
            data["target_features"] = target_features

        files = {"file": (filename, img_bytes, "application/octet-stream")}
        url = f"{self.base_url}/api/obfuscate"

        # Make HTTP request
        if _HTTP_BACKEND == "httpx":
            import httpx
            resp = httpx.post(url, files=files, data=data, timeout=self.timeout)
            resp.raise_for_status()
            out_bytes = resp.content
            headers = dict(resp.headers)
        elif _HTTP_BACKEND == "requests":
            import requests
            resp = requests.post(url, files=files, data=data, timeout=self.timeout)
            resp.raise_for_status()
            out_bytes = resp.content
            headers = dict(resp.headers)
        else:
            raise RuntimeError("httpx or requests is required for multipart uploads.")

        out_img = Image.open(io.BytesIO(out_bytes))

        # Extract telemetry headers
        analytics = {
            "processing_time_ms": float(headers.get("x-processing-time-ms", 0.0)),
            "epsilon": float(headers.get("x-epsilon", epsilon)),
            "mode": headers.get("x-mode", mode),
            "psnr_db": float(headers.get("x-psnr-db", 0.0)),
            "ssim": float(headers.get("x-ssim", 0.0)),
            "linf_255": float(headers.get("x-linf-255", 0.0)),
            "output_format": headers.get("x-output-format", output_format),
        }

        return out_img, analytics
