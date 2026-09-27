"""
CounterPreventionVerifier: Simulates platform ingestion pipelines
(Instagram, WhatsApp, Facebook) and verifies survival metrics
(PSNR, SSIM, MAE, and Re-compression risk).

Helps ensure client-side pre-formatting avoids triggering aggressive
server-side platform downscaling, re-compression, and color corruption.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
from PIL import Image, ImageFile
import numpy as np

# Prevent buffer overflow during high-entropy JPEG encoding
ImageFile.MAXBLOCK = 64 * 1024 * 1024


@dataclass
class IngestionSimulationResult:
    platform: str
    survival_psnr: float
    survival_ssim: float
    survival_mae: float
    simulated_size_kb: float
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    risk_factors: List[str]
    recommendations: List[str]


class CounterPreventionVerifier:
    """
    Simulates platform ingestion degradation and evaluates survival metrics.
    """

    @staticmethod
    def simulate_platform_ingestion(
        image: Image.Image,
        platform: str = "instagram_feed",
    ) -> Image.Image:
        """
        Simulates what social media platforms do when an image is uploaded:
        - Downscaling if exceeding platform thresholds
        - Forcing 4:2:0 subsampling
        - Re-compressing with quality factors between 70 and 80
        - Stripping color management
        """
        sim_img = image.copy()
        w, h = sim_img.size

        if platform == "instagram_feed":
            # Instagram downscales any image with width > 1080px
            if w > 1080:
                new_w = 1080
                new_h = int(round(1080 * (h / float(w))))
                sim_img = sim_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            sim_img.save(buf, format="JPEG", quality=75, subsampling=2, optimize=True)
            buf.seek(0)
            return Image.open(buf).copy()

        elif platform == "whatsapp":
            # WhatsApp caps max dimension at 1600px
            if max(w, h) > 1600:
                if w >= h:
                    new_w = 1600
                    new_h = int(round(1600 * (h / float(w))))
                else:
                    new_h = 1600
                    new_w = int(round(1600 * (w / float(h))))
                sim_img = sim_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            sim_img.save(buf, format="JPEG", quality=70, subsampling=2, optimize=True)
            buf.seek(0)
            return Image.open(buf).copy()

        elif platform == "facebook":
            # Facebook caps high-res at 2048px
            if max(w, h) > 2048:
                if w >= h:
                    new_w = 2048
                    new_h = int(round(2048 * (h / float(w))))
                else:
                    new_h = 2048
                    new_w = int(round(2048 * (w / float(h))))
                sim_img = sim_img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

            buf = io.BytesIO()
            sim_img.save(buf, format="JPEG", quality=76, subsampling=2, optimize=True)
            buf.seek(0)
            return Image.open(buf).copy()

        # Universal fallback
        buf = io.BytesIO()
        sim_img.save(buf, format="JPEG", quality=75, subsampling=2, optimize=True)
        buf.seek(0)
        return Image.open(buf).copy()

    @staticmethod
    def compute_psnr(img1: Image.Image, img2: Image.Image) -> float:
        """Computes Peak Signal-to-Noise Ratio (dB) between two PIL images."""
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, resample=Image.Resampling.LANCZOS)

        arr1 = np.array(img1.convert("RGB"), dtype=np.float32)
        arr2 = np.array(img2.convert("RGB"), dtype=np.float32)

        mse = float(np.mean((arr1 - arr2) ** 2))
        if mse == 0:
            return 100.0
        return float(10.0 * np.log10((255.0 ** 2) / mse))

    @staticmethod
    def compute_mae(img1: Image.Image, img2: Image.Image) -> float:
        """Computes Mean Absolute Error between two PIL images."""
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, resample=Image.Resampling.LANCZOS)

        arr1 = np.array(img1.convert("RGB"), dtype=np.float32)
        arr2 = np.array(img2.convert("RGB"), dtype=np.float32)
        return float(np.mean(np.abs(arr1 - arr2)))

    @staticmethod
    def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
        """
        Computes Structural Similarity Index (SSIM) on luminance (Y) channel.
        Uses vectorized 8x8 block formulation (Wang et al.).
        """
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, resample=Image.Resampling.LANCZOS)

        y1 = np.array(img1.convert("L"), dtype=np.float64)
        y2 = np.array(img2.convert("L"), dtype=np.float64)

        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2

        h, w = y1.shape
        bh, bw = h // 8, w // 8
        if bh == 0 or bw == 0:
            return 1.0

        # Vectorize into (bh*bw, 64) blocks
        b1 = y1[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
        b2 = y2[:bh * 8, :bw * 8].reshape(bh, 8, bw, 8).transpose(0, 2, 1, 3).reshape(-1, 64)

        mu1 = np.mean(b1, axis=1)
        mu2 = np.mean(b2, axis=1)
        sigma1_sq = np.var(b1, axis=1)
        sigma2_sq = np.var(b2, axis=1)
        sigma12 = np.mean((b1 - mu1[:, None]) * (b2 - mu2[:, None]), axis=1)

        num = (2.0 * mu1 * mu2 + c1) * (2.0 * sigma12 + c2)
        den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
        ssim_map = num / (den + 1e-10)

        return float(np.clip(np.mean(ssim_map), -1.0, 1.0))

    def evaluate_survival(
        self,
        candidate_image: Image.Image,
        platform: str = "instagram_feed",
        file_size_bytes: Optional[int] = None,
    ) -> IngestionSimulationResult:
        """
        Evaluates how well candidate_image survives platform counter-measures and ingestion.
        """
        simulated = self.simulate_platform_ingestion(candidate_image, platform=platform)

        psnr = self.compute_psnr(candidate_image, simulated)
        ssim = self.compute_ssim(candidate_image, simulated)
        mae = self.compute_mae(candidate_image, simulated)

        # In-memory byte size of simulated output
        sim_buf = io.BytesIO()
        simulated.save(sim_buf, format="JPEG", quality=75)
        sim_size_kb = round(len(sim_buf.getvalue()) / 1024.0, 2)

        w, h = candidate_image.size
        ar = w / float(h)
        risk_factors: List[str] = []
        recommendations: List[str] = []

        # Risk Factor 1: Resolution & Aspect Ratio
        if platform == "instagram_feed":
            if w > 1080:
                risk_factors.append(f"Width ({w}px) exceeds Instagram 1080px threshold -> server downscale will occur.")
                recommendations.append("Apply client-side pre-formatting to standard 1080px Lanczos width.")
            if ar < (4.0 / 5.0) or ar > 1.91:
                risk_factors.append(f"Aspect ratio ({ar:.2f}) outside [0.80, 1.91] -> platform will auto-crop edges.")
                recommendations.append("Use 'contain' or 'crop' pre-formatting to prevent unaligned platform cropping.")
        elif platform == "whatsapp":
            if max(w, h) > 1600:
                risk_factors.append(f"Dimensions ({w}x{h}) exceed WhatsApp 1600px ceiling -> downscaling will occur.")
                recommendations.append("Pre-format with WhatsApp profile to 1600px max dimension.")
        elif platform == "facebook":
            if max(w, h) > 2048:
                risk_factors.append(f"Dimensions ({w}x{h}) exceed Facebook 2048px ceiling -> server re-scaling will occur.")
                recommendations.append("Pre-format with Facebook profile to 2048px max dimension.")

        # Risk Factor 2: File Size
        if file_size_bytes is not None:
            size_mb = file_size_bytes / (1024 * 1024)
            if size_mb > 2.0:
                risk_factors.append(f"File size ({size_mb:.2f}MB) exceeds safe 1.5MB ceiling -> triggers heavy server re-compression.")
                recommendations.append("Use adaptive compression with --target-size-kb 1200.")

        # Risk Factor 3: Color Profile
        icc = candidate_image.info.get("icc_profile")
        if not icc:
            recommendations.append("Ensure sRGB IEC61966-2.1 ICC profile is embedded to prevent color shifting.")

        # Risk categorization based primarily on ingestion risk factors, supported by fidelity
        if len(risk_factors) == 0:
            risk_level = "LOW"
        elif len(risk_factors) == 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return IngestionSimulationResult(
            platform=platform,
            survival_psnr=round(psnr, 2),
            survival_ssim=round(ssim, 4),
            survival_mae=round(mae, 2),
            simulated_size_kb=sim_size_kb,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )
