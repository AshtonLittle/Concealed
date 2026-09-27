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


def compute_low_freq_tv_loss(delta: torch.Tensor, x_clean: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Suppress multi-scale low-frequency blotches, opponent chroma waves, and smooth-region edge lines.

    Unlike standard pixel TV (which blurs fine textures into wide waves), this penalizes:
      1. 8x8 and 16x16 pooled low-frequency energy (blocking wide stripes/bands)
      2. YCbCr opponent chrominance energy (blocking magenta/green +G/-R-B color casts)
      3. Batch-mean UAP collapse (forcing perturbations to be content-adaptive per image)
      4. Smooth-region & dark-shadow gradient steps (teaching the generator to avoid visible edge lines)
    """
    # 1. Multi-scale low-frequency energy suppression (blocks >=8px and >=16px waves)
    low_8 = F.avg_pool2d(delta, kernel_size=8, stride=4, padding=2)
    low_16 = F.avg_pool2d(delta, kernel_size=16, stride=8, padding=4)
    low_freq_energy = torch.mean(low_8.pow(2)) + 2.5 * torch.mean(low_16.pow(2))

    # 2. Opponent chrominance suppression in YCbCr space
    delta_y = 0.299 * delta[:, 0:1] + 0.587 * delta[:, 1:2] + 0.114 * delta[:, 2:3]
    delta_chroma = delta - delta_y
    chroma_energy = torch.mean(delta_chroma.pow(2))

    # 3. Anti-UAP mode-collapse penalty across batch dimension (penalizes batch-constant gratings)
    uap_penalty = torch.tensor(0.0, device=delta.device, dtype=delta.dtype)
    if delta.shape[0] > 1:
        batch_mean = delta.mean(dim=0, keepdim=True)
        uap_penalty = torch.mean(batch_mean.pow(2))

    # 4. Edge-line & dark-shadow step suppression when clean reference image is provided
    edge_step_penalty = torch.tensor(0.0, device=delta.device, dtype=delta.dtype)
    if x_clean is not None and x_clean.shape == delta.shape:
        lum = 0.299 * x_clean[:, 0:1] + 0.587 * x_clean[:, 1:2] + 0.114 * x_clean[:, 2:3]
        # Measure local isotropic texture vs flat/shadow regions
        lum_mean = F.avg_pool2d(lum, kernel_size=7, stride=1, padding=3)
        lum_var = torch.clamp(F.avg_pool2d(lum * lum, kernel_size=7, stride=1, padding=3) - lum_mean.pow(2), min=0.0)
        flat_or_dark_weight = torch.exp(-lum_var * 120.0) * (1.0 + 0.8 * torch.exp(-lum_mean * 12.0))
        dy = (delta_y[:, :, 1:, :] - delta_y[:, :, :-1, :]).pow(2)
        dx = (delta_y[:, :, :, 1:] - delta_y[:, :, :, :-1]).pow(2)
        edge_step_penalty = (dy * flat_or_dark_weight[:, :, 1:, :]).mean() + (dx * flat_or_dark_weight[:, :, :, 1:]).mean()

    return 2.0 * low_freq_energy + 1.5 * chroma_energy + 1.0 * uap_penalty + 0.75 * edge_step_penalty


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
        """Compute salience-weighted multi-surrogate ViT feature disruption loss and identification metrics."""
        device = obf_outputs[0].global_embedding.device
        total_patch_cos = torch.tensor(0.0, device=device)
        total_global_cos = torch.tensor(0.0, device=device)
        total_dispersion = torch.tensor(0.0, device=device)
        total_loss_weight = 0.0
        total_metric_weight = 0.0

        raw_patch_cos_sum = 0.0
        raw_salient_cos_sum = 0.0
        raw_global_cos_sum = 0.0
        concealed_patches_50_sum = 0.0
        concealed_patches_70_sum = 0.0
        per_model_metrics: Dict[str, float] = {}

        for clean_out, obf_out in zip(clean_outputs, obf_outputs):
            w = obf_out.weight
            total_metric_weight += w
            short_name = obf_out.name.split("/")[-1]

            # 1. Global [CLS] / pooled cosine similarity repulsion + anti-anchor misdirection
            clean_glob = clean_out.global_embedding.detach()
            obf_glob = obf_out.global_embedding
            global_cos = (clean_glob * obf_glob).sum(dim=-1).mean()
            g_cos_val = float(global_cos.detach().item())
            raw_global_cos_sum += g_cos_val * w
            per_model_metrics[f"global_cos/{short_name}"] = g_cos_val

            global_loss = F.relu(global_cos - self.cosine_margin)

            # 2. Multi-layer spatial patch token cosine repulsion & self-attention Gram scrambling
            m_p_cos = g_cos_val
            if obf_out.patch_tokens and clean_out.patch_tokens:
                layer_cos_losses = []
                layer_disp_losses = []
                layer_raw_cos = []
                layer_salient_cos = []
                layer_conc_50 = []
                layer_conc_70 = []

                for clean_patches, obf_patches in zip(clean_out.patch_tokens, obf_out.patch_tokens):
                    cp = clean_patches.detach()
                    # Per-patch cosine similarity [B, N]
                    per_patch_cos = (cp * obf_patches).sum(dim=-1)
                    mean_patch_cos = per_patch_cos.mean()
                    layer_raw_cos.append(float(mean_patch_cos.detach().item()))

                    # Estimate foreground patch salience from alignment with clean global semantics + patch variance
                    cp_centered = cp - cp.mean(dim=1, keepdim=True)
                    patch_distinctiveness = cp_centered.norm(dim=-1)  # [B, N]
                    salience_weights = torch.softmax(patch_distinctiveness * 3.0, dim=-1) * cp.shape[1]
                    salient_weighted_cos = (per_patch_cos * salience_weights.detach()).mean()

                    # Top-25% most salient foreground patches (faces, objects, text)
                    k_top = max(1, cp.shape[1] // 4)
                    top_idx = torch.topk(patch_distinctiveness, k=k_top, dim=-1).indices
                    top_patch_cos = torch.gather(per_patch_cos.detach(), 1, top_idx).mean()
                    layer_salient_cos.append(float(top_patch_cos.item()))

                    # Percentage of patches disrupted below recognition thresholds (0.50 and 0.70)
                    layer_conc_50.append(float((per_patch_cos.detach() < 0.50).float().mean().item() * 100.0))
                    layer_conc_70.append(float((per_patch_cos.detach() < 0.70).float().mean().item() * 100.0))

                    # Combine uniform + salience-focused patch repulsion
                    combined_patch_cos = 0.35 * mean_patch_cos + 0.65 * salient_weighted_cos
                    layer_cos_losses.append(F.relu(combined_patch_cos - self.cosine_margin))

                    # Spatial relational / covariance + Cross-Patch Self-Attention Gram Matrix scrambling:
                    # Breaks how the ViT routes attention between foreground tokens (e.g., eyes, mouth, clothing)
                    obf_centered = obf_patches - obf_patches.mean(dim=1, keepdim=True)
                    cov_align = F.cosine_similarity(cp_centered, obf_centered, dim=-1).mean()
                    spatial_mean = F.normalize(obf_patches.mean(dim=1, keepdim=True), p=2, dim=-1)
                    uniform_sim = (obf_patches * spatial_mean).sum(dim=-1).mean()

                    # Cross-patch Gram matrix & Impostor Patch Contrastive Margin [B, N, N]
                    if cp.shape[1] <= 256:
                        cp_sub = cp
                        op_sub = obf_patches
                        cp_norm = F.normalize(cp_centered, p=2, dim=-1)
                        op_norm = F.normalize(obf_centered, p=2, dim=-1)
                    else:
                        stride = max(1, cp.shape[1] // 196)
                        cp_sub = cp[:, ::stride]
                        op_sub = obf_patches[:, ::stride]
                        cp_norm = F.normalize(cp_centered[:, ::stride], p=2, dim=-1)
                        op_norm = F.normalize(obf_centered[:, ::stride], p=2, dim=-1)
                    gram_clean = torch.bmm(cp_norm, cp_norm.transpose(1, 2))
                    gram_obf = torch.bmm(op_norm, op_norm.transpose(1, 2))
                    gram_align = (gram_clean * gram_obf).mean()

                    # Pull each obfuscated patch toward its strongest off-diagonal clean impostor patch j != i
                    sim_grid = torch.bmm(op_sub, cp_sub.transpose(1, 2))
                    eye_mask = torch.eye(cp_sub.shape[1], device=cp.device, dtype=torch.bool).unsqueeze(0)
                    off_diag = sim_grid.masked_fill(eye_mask, -1e4)
                    impostor_sim = (torch.logsumexp(off_diag * 10.0, dim=-1) / 10.0).mean()

                    layer_disp_losses.append(
                        F.relu(cov_align - self.cosine_margin)
                        + 0.35 * F.relu(gram_align)
                        + 0.45 * (1.0 - impostor_sim)
                        + (1.0 - uniform_sim) * 0.25
                    )

                m_p_cos = sum(layer_raw_cos) / len(layer_raw_cos)
                m_s_cos = sum(layer_salient_cos) / len(layer_salient_cos)
                m_c50 = sum(layer_conc_50) / len(layer_conc_50)
                m_c70 = sum(layer_conc_70) / len(layer_conc_70)

                raw_patch_cos_sum += m_p_cos * w
                raw_salient_cos_sum += m_s_cos * w
                concealed_patches_50_sum += m_c50 * w
                concealed_patches_70_sum += m_c70 * w
                per_model_metrics[f"patch_cos/{short_name}"] = m_p_cos
                per_model_metrics[f"salient_cos/{short_name}"] = m_s_cos
                per_model_metrics[f"conc70/{short_name}"] = m_c70
                per_model_metrics[f"conc50/{short_name}"] = m_c50

                # Hard-Surrogate Focal Weighting: models with higher remaining similarity (e.g. CLIP)
                # automatically receive up to 1.85x stronger gradient focus so all models reach [EVADED / SCRAMBLED]
                hardness_score = max(0.0, min(1.0, 0.6 * m_p_cos + 0.4 * g_cos_val))
                w_loss = w * (1.0 + 0.85 * (hardness_score ** 1.5))
                total_loss_weight += w_loss

                total_patch_cos = total_patch_cos + w_loss * torch.stack(layer_cos_losses).mean()
                total_dispersion = total_dispersion + w_loss * torch.stack(layer_disp_losses).mean()
                total_global_cos = total_global_cos + w_loss * global_loss
            else:
                hardness_score = max(0.0, min(1.0, g_cos_val))
                w_loss = w * (1.0 + 0.85 * (hardness_score ** 1.5))
                total_loss_weight += w_loss
                total_global_cos = total_global_cos + w_loss * global_loss

        loss_norm = max(total_loss_weight, 1e-6)
        metric_norm = max(total_metric_weight, 1e-6)
        patch_loss = total_patch_cos / loss_norm
        global_loss = total_global_cos / loss_norm
        disp_loss = total_dispersion / loss_norm

        surrogate_loss = (
            self.patch_cosine_weight * patch_loss
            + self.global_cosine_weight * global_loss
            + self.patch_dispersion_weight * disp_loss
        )
        metrics = {
            "patch_cos_sim": raw_patch_cos_sum / metric_norm,
            "salient_patch_cos": raw_salient_cos_sum / metric_norm,
            "global_cos_sim": raw_global_cos_sum / metric_norm,
            "concealed_patches_50_pct": concealed_patches_50_sum / metric_norm,
            "concealed_patches_70_pct": concealed_patches_70_sum / metric_norm,
            "surrogate_loss": float(surrogate_loss.detach().item()),
            **per_model_metrics,
        }
        return surrogate_loss, metrics

    def compute_perceptual_stealth(
        self, x_clean: torch.Tensor, x_obf: torch.Tensor, delta: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute perceptual stealth regularization to keep perturbations invisible to humans."""
        ssim_loss = compute_ssim_loss(x_clean, x_obf)
        tv_loss = compute_low_freq_tv_loss(delta, x_clean=x_clean)
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

        # Measure opponent chrominance shift in [0, 255] units and UAP collapse ratio
        delta_y = 0.299 * delta[:, 0:1] + 0.587 * delta[:, 1:2] + 0.114 * delta[:, 2:3]
        chroma_rms_255 = float((delta - delta_y).pow(2).mean().sqrt().detach().item() * 255.0)
        uap_ratio = 0.0
        if delta.shape[0] > 1:
            uap_ratio = float(
                (delta.mean(dim=0).pow(2).mean() / (delta.pow(2).mean() + 1e-6)).detach().item()
            )

        metrics = {
            "ssim_loss": float(ssim_loss.detach().item()),
            "low_freq_tv_loss": float(tv_loss.detach().item()),
            "lpips_loss": lpips_val,
            "stealth_loss": float(stealth_loss.detach().item()),
            "chroma_rms_255": chroma_rms_255,
            "uap_collapse_ratio": uap_ratio,
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
