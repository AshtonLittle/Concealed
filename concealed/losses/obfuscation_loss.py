"""Composite Multi-Layer ViT Disruption & Perceptual Stealth Loss.

Combines:
  1. Multi-layer spatial patch token cosine repulsion (targets intermediate ViT blocks
     used by multimodal LLM projectors)
  2. Global [CLS] / pooled semantic token cosine repulsion
  3. Spatial patch feature variance / attention dispersion collapse loss
  4. Perceptual stealth regularization (SSIM, Low-Frequency Total Variation, L2, and optional LPIPS)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from concealed.models.surrogates import SurrogateOutput


def compute_ssim_loss(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute differentiable 1 - SSIM perceptual loss between RGB images x and y in [0, 1]."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2

    mu_x = F.avg_pool2d(x, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(y, window_size, stride=1, padding=pad)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(x * x, window_size, stride=1, padding=pad) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y * y, window_size, stride=1, padding=pad) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, window_size, stride=1, padding=pad) - mu_xy

    ssim_num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    ssim_den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = ssim_num / (ssim_den + 1e-8)
    return torch.clamp(1.0 - ssim_map.mean(), min=0.0)


def compute_low_freq_tv_loss(delta: torch.Tensor) -> torch.Tensor:
    """Penalize low-frequency blotches in the perturbation delta while allowing fine patch-grid frequencies."""
    # Downsample delta with a 4x4 average pool to isolate low-frequency color shifts
    delta_low = F.avg_pool2d(delta, kernel_size=4, stride=2, padding=1)
    tv_h = torch.mean(torch.abs(delta_low[:, :, 1:, :] - delta_low[:, :, :-1, :]))
    tv_w = torch.mean(torch.abs(delta_low[:, :, :, 1:] - delta_low[:, :, :, :-1]))
    dc_energy = torch.mean(delta_low.pow(2))
    return tv_h + tv_w + 2.0 * dc_energy


class CompositeObfuscationLoss(nn.Module):
    """Computes the joint adversarial ViT disruption and human perceptual stealth objective."""

    def __init__(
        self,
        patch_cosine_weight: float = 2.5,
        global_cosine_weight: float = 1.5,
        patch_dispersion_weight: float = 0.5,
        cosine_margin: float = -0.15,
        ssim_weight: float = 1.2,
        low_freq_tv_weight: float = 2.0,
        l2_reg_weight: float = 0.2,
        use_lpips: bool = False,
        lpips_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.patch_cosine_weight = float(patch_cosine_weight)
        self.global_cosine_weight = float(global_cosine_weight)
        self.patch_dispersion_weight = float(patch_dispersion_weight)
        self.cosine_margin = float(cosine_margin)
        self.ssim_weight = float(ssim_weight)
        self.low_freq_tv_weight = float(low_freq_tv_weight)
        self.l2_reg_weight = float(l2_reg_weight)
        self.use_lpips = bool(use_lpips)
        self.lpips_weight = float(lpips_weight)

        self.lpips_fn: Optional[nn.Module] = None
        if self.use_lpips:
            try:
                import lpips

                self.lpips_fn = lpips.LPIPS(net="alex").eval()
                for p in self.lpips_fn.parameters():
                    p.requires_grad_(False)
            except Exception:
                self.lpips_fn = None

    def compute_surrogate_disruption(
        self,
        clean_outputs: List[SurrogateOutput],
        obf_outputs: List[SurrogateOutput],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute weighted multi-surrogate ViT feature disruption loss."""
        total_patch_cos = torch.tensor(0.0, device=obf_outputs[0].global_embedding.device)
        total_global_cos = torch.tensor(0.0, device=obf_outputs[0].global_embedding.device)
        total_dispersion = torch.tensor(0.0, device=obf_outputs[0].global_embedding.device)
        total_weight = 0.0

        raw_patch_cos_sum = 0.0
        raw_global_cos_sum = 0.0

        for clean_out, obf_out in zip(clean_outputs, obf_outputs):
            w = obf_out.weight
            total_weight += w

            # 1. Global [CLS] / pooled cosine similarity repulsion
            # Both embeddings are already L2-normalized [B, D]
            global_cos = (clean_out.global_embedding.detach() * obf_out.global_embedding).sum(dim=-1).mean()
            raw_global_cos_sum += float(global_cos.detach().item()) * w
            global_loss = F.relu(global_cos - self.cosine_margin)
            total_global_cos = total_global_cos + w * global_loss

            # 2. Multi-layer spatial patch token cosine repulsion & spatial dispersion collapse
            if obf_out.patch_tokens and clean_out.patch_tokens:
                layer_cos_losses = []
                layer_disp_losses = []
                layer_raw_cos = []
                for clean_patches, obf_patches in zip(clean_out.patch_tokens, obf_out.patch_tokens):
                    # [B, N, D] dot product per spatial patch
                    patch_cos = (clean_patches.detach() * obf_patches).sum(dim=-1).mean()
                    layer_raw_cos.append(float(patch_cos.detach().item()))
                    layer_cos_losses.append(F.relu(patch_cos - self.cosine_margin))

                    # Spatial dispersion loss: push spatial patches toward their mean (uniformity collapse)
                    # or scramble spatial covariance so self-attention loses object boundaries
                    spatial_mean = F.normalize(obf_patches.mean(dim=1, keepdim=True), p=2, dim=-1)
                    # Penalize patches for retaining distinct structured variance aligned with clean image
                    clean_centered = clean_patches.detach() - clean_patches.detach().mean(dim=1, keepdim=True)
                    obf_centered = obf_patches - obf_patches.mean(dim=1, keepdim=True)
                    cov_align = F.cosine_similarity(clean_centered, obf_centered, dim=-1).mean()
                    uniform_sim = (obf_patches * spatial_mean).sum(dim=-1).mean()
                    layer_disp_losses.append(F.relu(cov_align) + (1.0 - uniform_sim) * 0.25)

                total_patch_cos = total_patch_cos + w * torch.stack(layer_cos_losses).mean()
                total_dispersion = total_dispersion + w * torch.stack(layer_disp_losses).mean()
                raw_patch_cos_sum += (sum(layer_raw_cos) / len(layer_raw_cos)) * w

        norm = max(total_weight, 1e-6)
        patch_loss = total_patch_cos / norm
        global_loss = total_global_cos / norm
        disp_loss = total_dispersion / norm

        surrogate_loss = (
            self.patch_cosine_weight * patch_loss
            + self.global_cosine_weight * global_loss
            + self.patch_dispersion_weight * disp_loss
        )
        metrics = {
            "patch_cos_sim": raw_patch_cos_sum / norm,
            "global_cos_sim": raw_global_cos_sum / norm,
            "surrogate_loss": float(surrogate_loss.detach().item()),
        }
        return surrogate_loss, metrics

    def compute_perceptual_stealth(
        self, x_clean: torch.Tensor, x_obf: torch.Tensor, delta: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute perceptual stealth regularization to keep perturbations invisible to humans."""
        ssim_loss = compute_ssim_loss(x_clean, x_obf)
        tv_loss = compute_low_freq_tv_loss(delta)
        l2_loss = torch.mean(delta.pow(2))

        stealth_loss = (
            self.ssim_weight * ssim_loss
            + self.low_freq_tv_weight * tv_loss
            + self.l2_reg_weight * l2_loss
        )

        lpips_val = 0.0
        if self.lpips_fn is not None:
            self.lpips_fn.to(x_clean.device)
            lp = self.lpips_fn(x_clean * 2.0 - 1.0, x_obf * 2.0 - 1.0).mean()
            lpips_val = float(lp.detach().item())
            stealth_loss = stealth_loss + self.lpips_weight * lp

        metrics = {
            "ssim_loss": float(ssim_loss.detach().item()),
            "low_freq_tv_loss": float(tv_loss.detach().item()),
            "lpips_loss": lpips_val,
            "stealth_loss": float(stealth_loss.detach().item()),
        }
        return stealth_loss, metrics

    def forward(
        self,
        x_clean: torch.Tensor,
        x_obf: torch.Tensor,
        delta: torch.Tensor,
        clean_outputs: List[SurrogateOutput],
        obf_outputs: List[SurrogateOutput],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute total composite loss and diagnostic metrics dictionary."""
        sur_loss, sur_metrics = self.compute_surrogate_disruption(clean_outputs, obf_outputs)
        stealth_loss, stealth_metrics = self.compute_perceptual_stealth(x_clean, x_obf, delta)
        total_loss = sur_loss + stealth_loss

        metrics = {**sur_metrics, **stealth_metrics, "total_loss": float(total_loss.detach().item())}
        return total_loss, metrics


def build_loss(config: dict) -> CompositeObfuscationLoss:
    """Factory helper to construct CompositeObfuscationLoss from a config dictionary."""
    loss_cfg = config.get("loss", config)
    return CompositeObfuscationLoss(
        patch_cosine_weight=loss_cfg.get("patch_cosine_weight", 2.5),
        global_cosine_weight=loss_cfg.get("global_cosine_weight", 1.5),
        patch_dispersion_weight=loss_cfg.get("patch_dispersion_weight", 0.5),
        cosine_margin=loss_cfg.get("cosine_margin", -0.15),
        ssim_weight=loss_cfg.get("ssim_weight", 1.2),
        low_freq_tv_weight=loss_cfg.get("low_freq_tv_weight", 2.0),
        l2_reg_weight=loss_cfg.get("l2_reg_weight", 0.2),
        use_lpips=loss_cfg.get("use_lpips", False),
        lpips_weight=loss_cfg.get("lpips_weight", 0.5),
    )
