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
