"""Amortized Adversarial Generator Network for Vision Transformer Obfuscation.

Designed for 100% ONNX / TensorRT / WebGPU compatibility while maintaining
explicit spatial-frequency control (via a 2D DCT Conv2d basis stem), global
context (via bottleneck self-attention), and scale-invariant residual synthesis
(canonical_residual, native, and hybrid global+tile modes).
"""

from __future__ import annotations

import math
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SynthesisMode = Literal["canonical_residual", "native", "hybrid"]
GeneratorVariant = Literal["tiny", "base", "large"]

VARIANT_CONFIGS: Dict[GeneratorVariant, Dict[str, int]] = {
    "tiny": {"base_channels": 24, "blocks_per_stage": 1, "attn_heads": 4},
    "base": {"base_channels": 40, "blocks_per_stage": 2, "attn_heads": 4},
    "large": {"base_channels": 64, "blocks_per_stage": 3, "attn_heads": 8},
}


def _create_2d_dct_kernels(block_size: int = 8, in_channels: int = 3) -> torch.Tensor:
    """Create an orthonormal 2D DCT-II basis filter bank as a Conv2d weight tensor.

    Returns a tensor of shape [in_channels * block_size * block_size, 1, block_size, block_size]
    suitable for a depthwise Conv2d, or [out_channels, in_channels, block_size, block_size].
    Here we select the 16 most informative mid/high-frequency 2D DCT basis functions per channel
    (total 48 filters for RGB) that overlap with 14x14 and 16x16 ViT patch harmonics.
    """
    basis_1d = torch.zeros(block_size, block_size)
    for k in range(block_size):
        alpha = math.sqrt(1.0 / block_size) if k == 0 else math.sqrt(2.0 / block_size)
        for n in range(block_size):
            basis_1d[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2.0 * block_size))

    # Construct all 64 2D basis filters [64, block_size, block_size]
    basis_2d = []
    for u in range(block_size):
        for v in range(block_size):
            filt = torch.outer(basis_1d[u], basis_1d[v])
            basis_2d.append(filt)
    basis_2d_tensor = torch.stack(basis_2d, dim=0)  # [64, 8, 8]

    # Select 16 representative frequencies across DC, low, mid (ViT patch harmonics), and high
    selected_indices = [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 24, 25, 32, 36, 45]
    selected = basis_2d_tensor[selected_indices]  # [16, 8, 8]

    # Repeat for each RGB input channel in depthwise format: [3 * 16, 1, 8, 8]
    weights = selected.unsqueeze(1).repeat(in_channels, 1, 1, 1)
    return weights


class DCTFilterBankStem(nn.Module):
    """ONNX-native 2D Discrete Cosine Transform (DCT) frequency stem.

    Projects RGB inputs into 48 spatial-frequency bands using an 8x8 depthwise Conv2d
    initialized with orthonormal 2D DCT-II basis filters, fused with a 3x3 spatial stem.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 40) -> None:
        super().__init__()
        self.dct_channels = in_channels * 16
        self.pad = nn.ReflectionPad2d((3, 4, 3, 4))
        self.dct_conv = nn.Conv2d(
            in_channels,
            self.dct_channels,
            kernel_size=8,
            stride=1,
            padding=0,
            groups=in_channels,
            bias=False,
        )
        # Initialize with orthonormal 2D DCT basis
        with torch.no_grad():
            self.dct_conv.weight.copy_(_create_2d_dct_kernels(8, in_channels))

        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(self.dct_channels + out_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dct_feat = self.dct_conv(self.pad(x))
        spa_feat = self.spatial_conv(x)
        return self.fuse(torch.cat([spa_feat, dct_feat], dim=1))


class SqueezeExcitation(nn.Module):
    """Lightweight Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(8, channels // reduction)
        self.fc1 = nn.Conv2d(channels, mid, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(-2, -1), keepdim=True)
        scale = torch.sigmoid(self.fc2(self.act(self.fc1(scale))))
        return x * scale


class ConvNeXtDepthwiseBlock(nn.Module):
    """7x7 Depthwise-Separable Inverted Residual Block with SE attention.

    The 7x7 depthwise receptive field matches half the 14x14 ViT patch stride,
    enabling efficient synthesis of patch-boundary-disrupting patterns.
    """

    def __init__(self, channels: int, expansion: int = 2) -> None:
        super().__init__()
        hidden = channels * expansion
        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.pwconv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)
        self.se = SqueezeExcitation(hidden, reduction=4)
        self.pwconv2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.25))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.dwconv(x)
        out = self.norm(out)
        out = self.act(self.pwconv1(out))
        out = self.se(out)
        out = self.pwconv2(out)
        return residual + self.gamma * out


class BottleneckSelfAttention(nn.Module):
    """ONNX-native multi-head self-attention at the 1/16 spatial bottleneck.

    Provides global receptive field across the entire image so the generator can
    coordinate long-range self-attention disruption across distant ViT patches.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        num_groups = min(8, channels)
        while channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kv_proj = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        normed = self.norm(x)
        q = self.q_proj(normed).reshape(b, self.num_heads, self.head_dim, h * w).transpose(-2, -1)  # [B, heads, HW, D]

        # Spatial-Reduction Attention: pool K, V to a fixed 16x16 (256-token) context grid
        # so attention complexity is O(HW * 256) instead of O((HW)^2), preventing VRAM spikes at high res.
        kv_spatial = F.interpolate(normed, size=(16, 16), mode="bilinear", align_corners=False)
        kv = self.kv_proj(kv_spatial).reshape(b, 2, self.num_heads, self.head_dim, 256)
        k = kv[:, 0]                    # [B, heads, D, 256]
        v = kv[:, 1].transpose(-2, -1)  # [B, heads, 256, D]

        attn = torch.matmul(q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, HW, D]
        out = out.transpose(-2, -1).reshape(b, c, h, w)
        return x + self.gamma * self.proj(out)


class WeberTextureMask(nn.Module):
    """Perceptual luminance/texture mask based on Weber's Law.

    Computes local gradient magnitude using fixed Sobel filters and modulates
    the perturbation strength so flat, smooth regions receive slightly calmer
    perturbations while textured regions use the full epsilon budget.
    """

    def __init__(self, min_mask_scale: float = 0.14) -> None:
        super().__init__()
        self.min_mask_scale = min_mask_scale
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        kernels = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)  # [2, 1, 3, 3]
        self.register_buffer("sobel_kernels", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert RGB [B, 3, H, W] to luminance [B, 1, H, W]
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        grads = F.conv2d(F.pad(lum, (1, 1, 1, 1), mode="reflect"), self.sobel_kernels)
        mag = torch.sqrt(grads[:, 0:1] ** 2 + grads[:, 1:2] ** 2 + 1e-6)
        # Smooth local texture energy over a 5x5 neighborhood
        energy = F.avg_pool2d(mag, kernel_size=5, stride=1, padding=2)

        # Combine with canonical 224x224 Sobel gradient energy so high-resolution (720p/1080p/4K)
        # images have the exact same structural mask amplitude as 224x224 training images
        lum_canon = F.interpolate(lum, size=(224, 224), mode="bilinear", align_corners=False)
        grads_c = F.conv2d(F.pad(lum_canon, (1, 1, 1, 1), mode="reflect"), self.sobel_kernels)
        mag_c = torch.sqrt(grads_c[:, 0:1] ** 2 + grads_c[:, 1:2] ** 2 + 1e-6)
        energy_c = F.avg_pool2d(mag_c, kernel_size=5, stride=1, padding=2)
        energy_c = F.interpolate(energy_c, size=(lum.shape[-2], lum.shape[-1]), mode="bilinear", align_corners=False)
        energy = torch.maximum(energy, energy_c)

        normalized = torch.tanh(energy * 12.0)
        mask = self.min_mask_scale + (1.0 - self.min_mask_scale) * normalized
        return mask


class AmortizedObfuscationGenerator(nn.Module):
    """Amortized Generator Network that synthesizes ViT-disrupting perturbations.

    Supports three perturbation synthesis modes:
    - ``"canonical_residual"``: Predicts delta at ``canonical_size`` (e.g., 384x384)
      and bilinearly upsamples only the residual delta to the native image size.
      Prevents anti-aliasing cancellation when downstream AI models downsample 4K/1080p images.
    - ``"native"``: Predicts delta directly at the native input resolution.
    - ``"hybrid"``: Synthesizes both a global canonical delta (defeating global thumbnail ViTs)
      and a higher-resolution local tile-scale delta (defeating high-res tile-slicing VLMs
      like GPT-4o high-detail, Qwen2-VL, and InternVL).
    """

    def __init__(
        self,
        variant: GeneratorVariant = "base",
        epsilon_255: float = 8.0,
        mode: SynthesisMode = "hybrid",
        canonical_size: int = 384,
        tile_size: int = 384,
        hybrid_global_weight: float = 0.25,
        luminance_texture_masking: bool = True,
        use_dct_stem: bool = True,
    ) -> None:
        super().__init__()
        if variant not in VARIANT_CONFIGS:
            raise ValueError(f"Unknown generator variant '{variant}'. Choose from {list(VARIANT_CONFIGS)}")

        cfg = VARIANT_CONFIGS[variant]
        c1 = cfg["base_channels"]
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 4
        blocks = cfg["blocks_per_stage"]
        heads = cfg["attn_heads"]

        self.variant = variant
        self.epsilon_255 = float(epsilon_255)
        self.register_buffer("epsilon", torch.tensor(float(epsilon_255) / 255.0))
        self.mode: SynthesisMode = mode
        self.canonical_size = int(canonical_size)
        self.tile_size = int(tile_size)
        self.hybrid_global_weight = float(hybrid_global_weight)
        self.luminance_texture_masking = bool(luminance_texture_masking)
        self.use_dct_stem = bool(use_dct_stem)

        # Stem
        if use_dct_stem:
            self.stem = DCTFilterBankStem(in_channels=3, out_channels=c1)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, c1, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(8, c1), num_channels=c1),
                nn.SiLU(inplace=True),
            )

        # Encoder stages
        self.enc1 = nn.Sequential(*[ConvNeXtDepthwiseBlock(c1) for _ in range(blocks)])
        self.down1 = nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc2 = nn.Sequential(*[ConvNeXtDepthwiseBlock(c2) for _ in range(blocks)])
        self.down2 = nn.Conv2d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False)

        self.enc3 = nn.Sequential(*[ConvNeXtDepthwiseBlock(c3) for _ in range(blocks)])
        self.down3 = nn.Conv2d(c3, c4, kernel_size=3, stride=2, padding=1, bias=False)

        # Bottleneck at 1/8 scale with Global Self-Attention
        self.bottleneck = nn.Sequential(
            ConvNeXtDepthwiseBlock(c4),
            BottleneckSelfAttention(c4, num_heads=heads),
            ConvNeXtDepthwiseBlock(c4),
        )

        # Decoder stages with skip connections
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c4, c3, kernel_size=3, padding=1, bias=False),
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(c3 * 2, c3, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, c3), num_channels=c3),
            nn.SiLU(inplace=True),
            *[ConvNeXtDepthwiseBlock(c3) for _ in range(blocks)],
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c3, c2, kernel_size=3, padding=1, bias=False),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(c2 * 2, c2, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, c2), num_channels=c2),
            nn.SiLU(inplace=True),
            *[ConvNeXtDepthwiseBlock(c2) for _ in range(blocks)],
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c2, c1, kernel_size=3, padding=1, bias=False),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(8, c1), num_channels=c1),
            nn.SiLU(inplace=True),
            *[ConvNeXtDepthwiseBlock(c1) for _ in range(blocks)],
        )

        # Perturbation head (zero-bias to prevent static image-agnostic UAP mode collapse)
        self.head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1, 3, kernel_size=3, padding=1, bias=False),
        )
        # Initialize final head with small weights for stable start
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.02)

        self.texture_mask = WeberTextureMask(min_mask_scale=0.06)

    def set_epsilon_255(self, epsilon_255: float) -> None:
        """Dynamically adjust the L_infinity perturbation bound at inference or training time."""
        self.epsilon_255 = float(epsilon_255)
        self.epsilon.fill_(float(epsilon_255) / 255.0)

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run the U-Net backbone on a tensor whose H, W are multiples of 8."""
        s0 = self.stem(x)
        e1 = self.enc1(s0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.bottleneck(self.down3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        raw = self.head(d1)

        # 1. Zero-DC Spatial High-Pass Filter:
        # Strip low-frequency (>9x9 px) wavy blobs so the generator cannot emit visible background blotches.
        raw = raw - F.avg_pool2d(raw, kernel_size=9, stride=1, padding=4)

        # 2. Content-adaptive structural gate: tie perturbation phase/amplitude to input structure
        # so the generator cannot collapse to a static image-independent background grating.
        x_hp = (x - F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)).abs().mean(dim=1, keepdim=True)
        content_gate = 0.65 + 0.70 * torch.tanh(x_hp * 16.0)
        return raw * content_gate

    def _forward_padded(self, x: torch.Tensor) -> torch.Tensor:
        """Pad input to a multiple of 8, run backbone, and crop back to exact (H, W)."""
        _, _, h, w = x.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            raw = self._forward_backbone(x_padded)
            return raw[:, :, :h, :w]
        return self._forward_backbone(x)

    def compute_delta(self, x: torch.Tensor, mode: Optional[SynthesisMode] = None) -> torch.Tensor:
        """Synthesize bounded perturbation delta in [-epsilon, +epsilon] for input x in [0, 1]."""
        active_mode: SynthesisMode = mode if mode is not None else self.mode
        h, w = x.shape[-2], x.shape[-1]
        s = (self.canonical_size // 8) * 8

        if active_mode == "canonical_residual":
            x_canon = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            raw_canon = self._forward_backbone(x_canon)
            delta = self.epsilon * torch.tanh(raw_canon)
            delta = F.interpolate(delta, size=(h, w), mode="bilinear", align_corners=False)

        elif active_mode == "native":
            raw_native = self._forward_padded(x)
            delta = self.epsilon * torch.tanh(raw_native)

        elif active_mode == "hybrid":
            # 1. Global canonical pass (survives downsampling to thumbnail ViTs)
            x_canon = F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False)
            raw_global = self._forward_backbone(x_canon)
            raw_global = F.interpolate(raw_global, size=(h, w), mode="bilinear", align_corners=False)

            # 2. High-res tile grid pass (defeats high-res tile-slicing VLMs like GPT-4o / Gemini / Qwen2-VL)
            local_s = max(16, (self.tile_size // 8) * 8)
            x_local = F.interpolate(x, size=(local_s, local_s), mode="bilinear", align_corners=False)
            raw_local = self._forward_backbone(x_local)
            raw_local = F.interpolate(raw_local, size=(h, w), mode="bilinear", align_corners=False)

            alpha = self.hybrid_global_weight
            raw_blended = alpha * raw_global + (1.0 - alpha) * raw_local
            delta = self.epsilon * torch.tanh(raw_blended)

        else:
            raise ValueError(f"Unsupported synthesis mode: {active_mode}")

        # YCbCr Opponent Chrominance Damping:
        # Suppress magenta/green (+G vs -R/-B) chromatic waves by 70% while keeping full luminance/edge budget.
        delta_y = 0.299 * delta[:, 0:1] + 0.587 * delta[:, 1:2] + 0.114 * delta[:, 2:3]
        delta_chroma = delta - delta_y
        delta = delta_y + 0.30 * delta_chroma

        if self.luminance_texture_masking:
            mask = self.texture_mask(x)
            delta = delta * mask

        # Guarantee strict L_infinity bound
        delta = torch.max(torch.min(delta, self.epsilon), -self.epsilon)
        return delta

    def forward(
        self,
        x: torch.Tensor,
        return_delta: bool = False,
        mode: Optional[SynthesisMode] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Generate obfuscated image x_tilde in [0, 1].

        When exported to ONNX with default ``return_delta=False``, returns a single
        tensor ``obfuscated`` of the exact same shape ``[B, 3, H, W]`` as ``x``.
        """
        delta = self.compute_delta(x, mode=mode)
        obfuscated = torch.clamp(x + delta, 0.0, 1.0)
        if return_delta:
            actual_delta = obfuscated - x
            return obfuscated, actual_delta
        return obfuscated


def build_generator(config: dict) -> AmortizedObfuscationGenerator:
    """Factory helper to build an AmortizedObfuscationGenerator from a config dictionary."""
    gen_cfg = config.get("generator", config)
    return AmortizedObfuscationGenerator(
        variant=gen_cfg.get("variant", "base"),
        epsilon_255=gen_cfg.get("epsilon_255", 8.0),
        mode=gen_cfg.get("mode", "hybrid"),
        canonical_size=gen_cfg.get("canonical_size", 384),
        tile_size=gen_cfg.get("tile_size", 384),
        hybrid_global_weight=gen_cfg.get("hybrid_global_weight", 0.6),
        luminance_texture_masking=gen_cfg.get("luminance_texture_masking", True),
        use_dct_stem=gen_cfg.get("use_dct_stem", True),
    )
