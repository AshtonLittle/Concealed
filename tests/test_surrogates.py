"""Unit tests for VisionTransformerSurrogate and SurrogateEnsemble across model families."""

from __future__ import annotations

import pytest
import torch

from concealed.models.surrogates import SurrogateEnsemble, VisionTransformerSurrogate


@pytest.mark.parametrize(
    "model_name",
    [
        "mock/tiny-vit-16",
        "mock/tiny-vit-14",
        "openai/clip-vit-base-patch16",
        "google/siglip-base-patch16-224",
        "facebook/dinov2-base",
    ],
)
def test_surrogate_families_offline(model_name: str) -> None:
    """Verify CLIP, SigLIP, DINOv2, and Mock ViT wrappers extract global & multi-layer patch tokens."""
    surrogate = VisionTransformerSurrogate(
        model_name=model_name,
        weight=1.0,
        tap_layers=[-3, -2, -1],
        pretrained=False,  # Use fast local config initialization for unit tests
    )
    x = torch.rand(2, 3, 200, 200, requires_grad=True)
    out = surrogate(x)

    assert out.global_embedding.shape[0] == 2
    assert out.global_embedding.ndim == 2
    assert len(out.patch_tokens) >= 2
    for pt in out.patch_tokens:
        assert pt.shape[0] == 2
        assert pt.ndim == 3  # [B, N_patches, D]

    # Verify input gradient flows through frozen surrogate back to input image x
    loss = out.global_embedding.sum() + out.patch_tokens[-1].sum()
    loss.backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum().item()) > 0.0


def test_surrogate_ensemble_multi_model() -> None:
    ensemble = SurrogateEnsemble(
        model_specs=[
            {"name": "openai/clip-vit-base-patch16", "weight": 1.0},
            {"name": "google/siglip-base-patch16-224", "weight": 1.0},
            {"name": "facebook/dinov2-base", "weight": 1.0},
        ],
        tap_layers=[-2, -1],
        pretrained=False,
    )
    x = torch.rand(1, 3, 224, 224, requires_grad=True)
    outputs = ensemble(x)
    assert len(outputs) == 3
    names = [o.name for o in outputs]
    assert "openai/clip-vit-base-patch16" in names
    assert "google/siglip-base-patch16-224" in names
    assert "facebook/dinov2-base" in names


def test_model_aliases_and_convnext_4d_stage_hooks() -> None:
    from concealed.models.surrogates import resolve_model_name

    assert resolve_model_name("dfn5b") == "timm/hf-hub:apple/DFN5B-CLIP-ViT-H-14-378"
    assert (
        resolve_model_name("openclip-convnext-large")
        == "timm/hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft"
    )
    assert resolve_model_name("glm-ocr") == "zai-org/GLM-OCR"
    assert resolve_model_name("qwen-image-2.1") == "Qwen/Qwen-Image-2.1"

    # Test ConvNeXt-style 4D stage feature extraction via timm offline backbone
    convnext_surrogate = VisionTransformerSurrogate(
        model_name="timm/convnext_atto",
        weight=1.0,
        tap_layers=[-2, -1],
        pretrained=False,
    )
    x = torch.rand(1, 3, 160, 160, requires_grad=True)
    out = convnext_surrogate(x)
    assert out.global_embedding.shape[0] == 1
    assert len(out.patch_tokens) >= 2
    for pt in out.patch_tokens:
        assert pt.ndim == 3  # 4D [B, C, H, W] stages converted to [B, H*W, C] tokens
    (out.global_embedding.sum() + out.patch_tokens[-1].sum()).backward()
    assert x.grad is not None

