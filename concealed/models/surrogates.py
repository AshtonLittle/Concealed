"""Pluggable Open-Source Vision Transformer Surrogate Ensemble.

Wraps HuggingFace `transformers` vision towers (CLIP, SigLIP, DINOv2, EVA-CLIP, InternViT, etc.)
and `timm` Vision Transformers into a unified differentiable interface that extracts both:
  1. Global semantic embeddings ([CLS] / pooled output)
  2. Multi-layer intermediate spatial patch token grids (crucial for transferring to
     frontier VLMs that project spatial patch tokens directly into LLM decoders).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SurrogateOutput:
    """Standardized multi-layer representation extracted from a Vision Transformer."""

    name: str
    weight: float
    global_embedding: torch.Tensor  # [B, D] L2-normalized
    patch_tokens: List[torch.Tensor]  # List of [B, N_patches, D] L2-normalized tensors


class TinyViTBackbone(nn.Module):
    """Lightweight in-memory Vision Transformer for fast offline testing and dry runs."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=embed_dim * 2,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        b = pixel_values.shape[0]
        x = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)  # [B, N, D]
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed

        hidden_states: List[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states.append(x)

        x = self.norm(x)
        global_emb = x[:, 0]
        return global_emb, hidden_states


class VisionTransformerSurrogate(nn.Module):
    """Unified differentiable wrapper around a single frozen Vision Transformer model."""

    def __init__(
        self,
        model_name: str,
        weight: float = 1.0,
        tap_layers: Sequence[int] = (-4, -2, -1),
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.weight = float(weight)
        self.tap_layers = list(tap_layers)
        self.backend_type: str = "hf"
        self._timm_features: List[torch.Tensor] = []
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        if model_name.startswith("mock/"):
            self.backend_type = "mock"
            patch_size = 14 if "14" in model_name else 16
            img_size = 224
            self.model = TinyViTBackbone(image_size=img_size, patch_size=patch_size, embed_dim=128, depth=4)
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
            self.input_size = (img_size, img_size)
            self.has_cls_token = True

        elif model_name.startswith("timm/"):
            import timm

            self.backend_type = "timm"
            timm_name = model_name.split("timm/", 1)[1]
            self.model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
            data_cfg = timm.data.resolve_model_data_config(self.model)
            in_size = data_cfg.get("input_size", (3, 224, 224))
            self.input_size = (int(in_size[-2]), int(in_size[-1]))
            mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
            std = data_cfg.get("std", (0.229, 0.224, 0.225))
            self.has_cls_token = getattr(self.model, "num_prefix_tokens", 1) > 0
            self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 1)
            self._register_timm_hooks()

        else:
            self.backend_type = "hf"
            self.model, self.input_size, mean, std, self.has_cls_token = self._load_hf_vision_model(
                model_name, pretrained=pretrained
            )

        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))

        # Freeze all surrogate parameters while preserving input gradient flow
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    @staticmethod
    def _load_hf_vision_model(
        model_name: str, pretrained: bool = True
    ) -> Tuple[nn.Module, Tuple[int, int], Tuple[float, ...], Tuple[float, ...], bool]:
        from transformers import (
            AutoConfig,
            AutoImageProcessor,
            AutoModel,
            CLIPVisionConfig,
            CLIPVisionModel,
            Dinov2Config,
            Dinov2Model,
            SiglipVisionConfig,
            SiglipVisionModel,
        )

        lower_name = model_name.lower()
        # Default preprocessing constants by family
        if "siglip" in lower_name:
            default_mean = (0.5, 0.5, 0.5)
            default_std = (0.5, 0.5, 0.5)
            default_size = 384 if "384" in lower_name else 224
            has_cls = False
        elif "clip" in lower_name:
            default_mean = (0.48145466, 0.4578275, 0.40821073)
            default_std = (0.26862954, 0.26130258, 0.27577711)
            default_size = 336 if "336" in lower_name else 224
            has_cls = True
        else:
            default_mean = (0.485, 0.456, 0.406)
            default_std = (0.229, 0.224, 0.225)
            default_size = 224
            has_cls = True

        if pretrained:
            try:
                processor = AutoImageProcessor.from_pretrained(model_name)
                mean = tuple(getattr(processor, "image_mean", default_mean))
                std = tuple(getattr(processor, "image_std", default_std))
                size_dict = getattr(processor, "size", {})
                if isinstance(size_dict, int):
                    img_h = img_w = size_dict
                elif isinstance(size_dict, dict):
                    img_h = size_dict.get("height", size_dict.get("shortest_edge", default_size))
                    img_w = size_dict.get("width", size_dict.get("shortest_edge", default_size))
                else:
                    img_h = img_w = default_size
            except Exception:
                mean, std, img_h, img_w = default_mean, default_std, default_size, default_size

            if "siglip" in lower_name:
                model = SiglipVisionModel.from_pretrained(model_name)
            elif "clip" in lower_name:
                model = CLIPVisionModel.from_pretrained(model_name)
            elif "dinov2" in lower_name:
                model = Dinov2Model.from_pretrained(model_name)
            else:
                full_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                model = getattr(full_model, "vision_model", full_model)
        else:
            # Uninitialized config mode (useful for offline architecture tests)
            mean, std, img_h, img_w = default_mean, default_std, default_size, default_size
            if "siglip" in lower_name:
                model = SiglipVisionModel(SiglipVisionConfig(image_size=img_h, num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4))
            elif "clip" in lower_name:
                model = CLIPVisionModel(CLIPVisionConfig(image_size=img_h, num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4))
            elif "dinov2" in lower_name:
                model = Dinov2Model(Dinov2Config(image_size=img_h, num_hidden_layers=4, hidden_size=128, intermediate_size=256, num_attention_heads=4))
            else:
                cfg = AutoConfig.from_pretrained(model_name)
                model = AutoModel.from_config(cfg)

        return model, (int(img_h), int(img_w)), mean, std, has_cls

    def _register_timm_hooks(self) -> None:
        """Attach forward hooks to intermediate transformer blocks in a timm model."""
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            blocks = getattr(self.model, "layers", None)
        if blocks is None:
            return

        num_blocks = len(blocks)
        resolved_indices = set()
        for idx in self.tap_layers:
            pos_idx = idx if idx >= 0 else max(0, num_blocks + idx)
            if 0 <= pos_idx < num_blocks:
                resolved_indices.add(pos_idx)

        def _make_hook():
            def _hook(_module, _inp, output):
                tensor = output[0] if isinstance(output, tuple) else output
                self._timm_features.append(tensor)

            return _hook

        for i in sorted(resolved_indices):
            self._hooks.append(blocks[i].register_forward_hook(_make_hook()))

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiably resize and normalize an RGB image batch in [0, 1]."""
        if (x.shape[-2], x.shape[-1]) != self.input_size:
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> SurrogateOutput:
        """Extract normalized global embedding and multi-layer spatial patch tokens."""
        self.model.eval()
        pixel_values = self.preprocess(x)

        if self.backend_type == "mock":
            global_emb, all_hidden = self.model(pixel_values)
            patch_list = []
            num_layers = len(all_hidden)
            for idx in self.tap_layers:
                pos_idx = idx if idx >= 0 else max(0, num_layers + idx)
                if 0 <= pos_idx < num_layers:
                    tokens = all_hidden[pos_idx][:, 1:, :]  # Strip CLS token
                    patch_list.append(F.normalize(tokens, p=2, dim=-1))
            global_emb = F.normalize(global_emb, p=2, dim=-1)
            return SurrogateOutput(
                name=self.model_name,
                weight=self.weight,
                global_embedding=global_emb,
                patch_tokens=patch_list,
            )

        if self.backend_type == "timm":
            self._timm_features.clear()
            out = self.model.forward_features(pixel_values)
            if hasattr(self.model, "forward_head"):
                global_emb = self.model.forward_head(out, pre_logits=True)
            elif out.ndim == 3:
                global_emb = out[:, 0] if self.has_cls_token else out.mean(dim=1)
            else:
                global_emb = out.flatten(1)

            patch_list = []
            prefix = getattr(self, "num_prefix_tokens", 1 if self.has_cls_token else 0)
            for feat in self._timm_features:
                if feat.ndim == 4:
                    # [B, C, H, W] or [B, H, W, C] -> [B, N, D]
                    if feat.shape[1] > feat.shape[-1]:
                        tokens = feat.flatten(2).transpose(1, 2)
                    else:
                        tokens = feat.reshape(feat.shape[0], -1, feat.shape[-1])
                elif feat.ndim == 3:
                    tokens = feat[:, prefix:, :] if prefix > 0 and feat.shape[1] > prefix else feat
                else:
                    continue
                patch_list.append(F.normalize(tokens, p=2, dim=-1))

            if not patch_list and out.ndim == 3:
                tokens = out[:, prefix:, :] if prefix > 0 else out
                patch_list.append(F.normalize(tokens, p=2, dim=-1))

            global_emb = F.normalize(global_emb, p=2, dim=-1)
            return SurrogateOutput(
                name=self.model_name,
                weight=self.weight,
                global_embedding=global_emb,
                patch_tokens=patch_list,
            )

        # HuggingFace transformers vision model
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states  # Tuple of [B, N (+1), D]
        last_hidden = outputs.last_hidden_state

        if getattr(outputs, "pooler_output", None) is not None:
            global_emb = outputs.pooler_output
        elif self.has_cls_token:
            global_emb = last_hidden[:, 0]
        else:
            global_emb = last_hidden.mean(dim=1)

        patch_list = []
        if hidden_states is not None:
            num_layers = len(hidden_states)
            seen = set()
            for idx in self.tap_layers:
                pos_idx = idx if idx >= 0 else max(0, num_layers + idx)
                if 0 <= pos_idx < num_layers and pos_idx not in seen:
                    seen.add(pos_idx)
                    layer_tokens = hidden_states[pos_idx]
                    if self.has_cls_token and layer_tokens.shape[1] > 1:
                        # Handle DINOv2 register tokens if present, or standard [CLS] at index 0
                        num_reg = getattr(self.model.config, "num_register_tokens", 0)
                        layer_tokens = layer_tokens[:, 1 + num_reg :, :]
                    patch_list.append(F.normalize(layer_tokens, p=2, dim=-1))
        else:
            tokens = last_hidden[:, 1:, :] if self.has_cls_token else last_hidden
            patch_list.append(F.normalize(tokens, p=2, dim=-1))

        global_emb = F.normalize(global_emb, p=2, dim=-1)
        return SurrogateOutput(
            name=self.model_name,
            weight=self.weight,
            global_embedding=global_emb,
            patch_tokens=patch_list,
        )


class SurrogateEnsemble(nn.Module):
    """Manages an ensemble of frozen open-source Vision Transformer surrogates."""

    def __init__(
        self,
        model_specs: Sequence[Dict[str, object]],
        tap_layers: Sequence[int] = (-4, -2, -1),
        pretrained: bool = True,
        sequential_offload: bool = False,
    ) -> None:
        super().__init__()
        self.sequential_offload = bool(sequential_offload)
        self.surrogates = nn.ModuleList()

        for spec in model_specs:
            name = str(spec["name"])
            weight = float(spec.get("weight", 1.0))
            layers = spec.get("tap_layers", tap_layers)
            surrogate = VisionTransformerSurrogate(
                model_name=name,
                weight=weight,
                tap_layers=layers,  # type: ignore[arg-type]
                pretrained=pretrained,
            )
            self.surrogates.append(surrogate)

    def forward(self, x: torch.Tensor) -> List[SurrogateOutput]:
        """Extract features across all surrogates in the ensemble."""
        outputs: List[SurrogateOutput] = []
        device = x.device

        for surrogate in self.surrogates:
            if self.sequential_offload and device.type == "cuda":
                surrogate.to(device)
                out = surrogate(x)
                surrogate.to("cpu")
            else:
                out = surrogate(x)
            outputs.append(out)
        return outputs


def build_surrogate_ensemble(
    config: dict,
    key: str = "train_models",
    pretrained: bool = True,
) -> SurrogateEnsemble:
    """Factory helper to construct a SurrogateEnsemble from a configuration dictionary."""
    sur_cfg = config.get("surrogates", config)
    model_specs = sur_cfg.get(key, [])
    tap_layers = sur_cfg.get("tap_layers", [-4, -2, -1])
    sequential_offload = sur_cfg.get("sequential_offload", False)
    return SurrogateEnsemble(
        model_specs=model_specs,
        tap_layers=tap_layers,
        pretrained=pretrained,
        sequential_offload=sequential_offload,
    )
