"""Differentiable Expectation Over Transformations (EOT).

Ensures perturbations synthesized by the Amortized Generator survive real-world
preprocessing pipelines:
  1. Differentiable JPEG compression (8x8 DCT + ITU-T.81 quantization + STE rounding)
  2. Input Diversity (DI-FGSM) random multi-scale bilinear resize & padding
  3. High-resolution VLM tile crop simulation (mimicking GPT-4o / Qwen2-VL / InternVL tiling)
  4. Differentiable exposure / contrast jitter
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffJPEG(nn.Module):
    """Differentiable 8x8 Block-DCT JPEG Compression Simulator with Straight-Through Estimator."""

    def __init__(self) -> None:
        super().__init__()
        # Standard ITU-T.81 Luminance (Y) and Chrominance (CbCr) 8x8 quantization tables
        luma_table = torch.tensor(
            [
                [16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99],
            ],
            dtype=torch.float32,
        )
        chroma_table = torch.tensor(
            [
                [17, 18, 24, 47, 99, 99, 99, 99],
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
            ],
            dtype=torch.float32,
        )
        q_tables = torch.stack([luma_table, chroma_table, chroma_table], dim=0)  # [3, 8, 8]
        self.register_buffer("q_tables", q_tables.view(1, 3, 1, 1, 8, 8))

        # Orthonormal 8x8 1D DCT-II matrix
        dct_mat = torch.zeros(8, 8, dtype=torch.float32)
        for k in range(8):
            alpha = math.sqrt(1.0 / 8.0) if k == 0 else math.sqrt(2.0 / 8.0)
            for n in range(8):
                dct_mat[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / 16.0)
        self.register_buffer("dct_mat", dct_mat)
        self.register_buffer("idct_mat", dct_mat.t().contiguous())

    @staticmethod
    def _quality_to_scale(quality: float) -> float:
        q = max(1.0, min(100.0, float(quality)))
        scale = 5000.0 / q if q < 50.0 else 200.0 - 2.0 * q
        return max(0.01, scale / 100.0)

    def forward(self, x: torch.Tensor, quality: float = 75.0) -> torch.Tensor:
        """Apply differentiable JPEG simulation to RGB tensor x in [0, 1]."""
        b, c, h, w = x.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_pad = x
        _, _, hp, wp = x_pad.shape

        # RGB [0, 1] -> YCbCr [0, 255] shifted by -128
        rgb = x_pad * 255.0
        r, g, bl = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * bl - 128.0
        cb = -0.168736 * r - 0.331264 * g + 0.5 * bl
        cr = 0.5 * r - 0.418688 * g - 0.081312 * bl
        ycbcr = torch.cat([y, cb, cr], dim=1)

        # Partition into [B, 3, hp//8, wp//8, 8, 8] blocks
        bh, bw = hp // 8, wp // 8
        blocks = ycbcr.view(b, c, bh, 8, bw, 8).permute(0, 1, 2, 4, 3, 5).contiguous()

        # Forward 2D DCT: D @ block @ D^T
        dct_blocks = torch.matmul(torch.matmul(self.dct_mat, blocks), self.idct_mat)

        # Quantize with STE rounding
        scale = self._quality_to_scale(quality)
        q_step = torch.clamp(self.q_tables * scale, min=1.0)
        q_coeffs = dct_blocks / q_step
        q_rounded = q_coeffs + (torch.round(q_coeffs) - q_coeffs).detach()
        dequant_blocks = q_rounded * q_step

        # Inverse 2D DCT: D^T @ coeff @ D
        idct_blocks = torch.matmul(torch.matmul(self.idct_mat, dequant_blocks), self.dct_mat)
        ycbcr_rec = idct_blocks.permute(0, 1, 2, 4, 3, 5).contiguous().view(b, c, hp, wp)

        # YCbCr -> RGB [0, 1]
        y_r = ycbcr_rec[:, 0:1] + 128.0
        cb_r = ycbcr_rec[:, 1:2]
        cr_r = ycbcr_rec[:, 2:3]
        r_out = y_r + 1.402 * cr_r
        g_out = y_r - 0.344136 * cb_r - 0.714136 * cr_r
        b_out = y_r + 1.772 * cb_r
        rgb_out = torch.cat([r_out, g_out, b_out], dim=1) / 255.0

        rgb_out = rgb_out[:, :, :h, :w]
        return torch.clamp(rgb_out, 0.0, 1.0)


class DifferentiableEOT(nn.Module):
    """Differentiable Expectation Over Transformations (EOT) module for training.

    Applies paired spatial transforms (multi-scale DI-FGSM resize or VLM tile cropping)
    to both ``x_clean`` and ``x_obf`` so their patch grids remain spatially aligned,
    while applying compression (`DiffJPEG`) and photometric jitter to ``x_obf``.
    """

    def __init__(
        self,
        enabled: bool = True,
        jpeg_prob: float = 0.8,
        jpeg_quality_min: float = 60.0,
        jpeg_quality_max: float = 95.0,
        random_scale_min: float = 0.75,
        random_scale_max: float = 1.25,
        tile_crop_prob: float = 0.5,
        color_jitter_strength: float = 0.03,
    ) -> None:
        super().__init__()
        self.enabled = bool(enabled)
        self.jpeg_prob = float(jpeg_prob)
        self.jpeg_quality_min = float(jpeg_quality_min)
        self.jpeg_quality_max = float(jpeg_quality_max)
        self.random_scale_min = float(random_scale_min)
        self.random_scale_max = float(random_scale_max)
        self.tile_crop_prob = float(tile_crop_prob)
        self.color_jitter_strength = float(color_jitter_strength)
        self.diff_jpeg = DiffJPEG()

    def forward(
        self, x_clean: torch.Tensor, x_obf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return transformed ``(x_clean_aug, x_obf_aug)`` pair for surrogate feature extraction."""
        if not self.enabled:
            return x_clean, x_obf

        b, _, h, w = x_obf.shape
        device = x_obf.device

        # 1. Differentiable JPEG compression on obfuscated image
        if torch.rand(1, device=device).item() < self.jpeg_prob:
            q = torch.empty(1, device=device).uniform_(self.jpeg_quality_min, self.jpeg_quality_max).item()
            x_obf = self.diff_jpeg(x_obf, quality=q)

        # 2. Mild photometric jitter on obfuscated image
        if self.color_jitter_strength > 0.0:
            brightness = (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * self.color_jitter_strength
            contrast = 1.0 + (torch.rand(b, 1, 1, 1, device=device) * 2.0 - 1.0) * self.color_jitter_strength
            x_obf = torch.clamp((x_obf - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

        # 3. Paired spatial transform: either VLM high-res tile crop OR multi-scale DI-FGSM resize
        if h >= 256 and w >= 256 and torch.rand(1, device=device).item() < self.tile_crop_prob:
            # Simulate a high-res VLM local tile crop (50%-75% of image dimensions)
            crop_ratio = torch.empty(1, device=device).uniform_(0.5, 0.75).item()
            ch = max(64, int(h * crop_ratio))
            cw = max(64, int(w * crop_ratio))
            top = int(torch.randint(0, max(1, h - ch + 1), (1,), device=device).item())
            left = int(torch.randint(0, max(1, w - cw + 1), (1,), device=device).item())
            x_clean_aug = x_clean[:, :, top : top + ch, left : left + cw]
            x_obf_aug = x_obf[:, :, top : top + ch, left : left + cw]
        else:
            # Multi-scale DI-FGSM random resize
            scale = torch.empty(1, device=device).uniform_(self.random_scale_min, self.random_scale_max).item()
            nh = max(64, int(round(h * scale)))
            nw = max(64, int(round(w * scale)))
            x_clean_aug = F.interpolate(x_clean, size=(nh, nw), mode="bilinear", align_corners=False)
            x_obf_aug = F.interpolate(x_obf, size=(nh, nw), mode="bilinear", align_corners=False)

        return x_clean_aug, x_obf_aug


def build_eot(config: dict) -> DifferentiableEOT:
    """Factory helper to construct DifferentiableEOT from a config dictionary."""
    eot_cfg = config.get("eot", config)
    return DifferentiableEOT(
        enabled=eot_cfg.get("enabled", True),
        jpeg_prob=eot_cfg.get("jpeg_prob", 0.8),
        jpeg_quality_min=eot_cfg.get("jpeg_quality_min", 60.0),
        jpeg_quality_max=eot_cfg.get("jpeg_quality_max", 95.0),
        random_scale_min=eot_cfg.get("random_scale_min", 0.75),
        random_scale_max=eot_cfg.get("random_scale_max", 1.25),
        tile_crop_prob=eot_cfg.get("tile_crop_prob", 0.5),
        color_jitter_strength=eot_cfg.get("color_jitter_strength", 0.03),
    )
