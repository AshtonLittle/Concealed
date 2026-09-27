"""Unit tests for DiffJPEG, DifferentiableEOT, and CompositeObfuscationLoss."""

from __future__ import annotations

import torch

from concealed.losses.eot import DifferentiableEOT, DiffJPEG
from concealed.losses.obfuscation_loss import CompositeObfuscationLoss
from concealed.models.generator import AmortizedObfuscationGenerator
from concealed.models.surrogates import SurrogateEnsemble


def test_diff_jpeg_forward_and_ste_backward() -> None:
    jpeg = DiffJPEG()
    x = torch.rand(2, 3, 130, 142, requires_grad=True)
    y = jpeg(x, quality=70.0)

    assert y.shape == x.shape
    assert float(y.min().item()) >= 0.0
    assert float(y.max().item()) <= 1.0

    loss = (y ** 2).mean()
    loss.backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum().item()) > 0.0


def test_differentiable_eot_spatial_alignment() -> None:
    eot = DifferentiableEOT(
        enabled=True,
        jpeg_prob=1.0,
        tile_crop_prob=0.5,
        color_jitter_strength=0.03,
    )
    x_clean = torch.rand(2, 3, 280, 320)
    x_obf = (x_clean + 0.02).clamp(0.0, 1.0).requires_grad_(True)

    c_aug, o_aug = eot(x_clean, x_obf)
    assert c_aug.shape == o_aug.shape
    loss = (o_aug - c_aug).pow(2).mean()
    loss.backward()
    assert x_obf.grad is not None


def test_composite_obfuscation_loss_decreases_similarity() -> None:
    gen = AmortizedObfuscationGenerator(
        variant="tiny",
        epsilon_255=12.0,
        mode="canonical_residual",
        canonical_size=128,
    )
    ensemble = SurrogateEnsemble(
        model_specs=[{"name": "mock/tiny-vit-16", "weight": 1.0}],
        tap_layers=[-2, -1],
        pretrained=False,
    )
    loss_fn = CompositeObfuscationLoss()

    x = torch.rand(2, 3, 128, 128)
    optimizer = torch.optim.AdamW(gen.parameters(), lr=3e-3)

    with torch.no_grad():
        clean_outs = ensemble(x)
        x_obf_init, delta_init = gen(x, return_delta=True)
        obf_outs_init = ensemble(x_obf_init)
        _, init_metrics = loss_fn(x, x_obf_init, delta_init, clean_outs, obf_outs_init)

    # Run 4 optimization steps and verify patch cosine similarity drops
    for _ in range(4):
        optimizer.zero_grad()
        x_obf, delta = gen(x, return_delta=True)
        obf_outs = ensemble(x_obf)
        loss, step_metrics = loss_fn(x, x_obf, delta, clean_outs, obf_outs)
        loss.backward()
        optimizer.step()

    assert step_metrics["patch_cos_sim"] < init_metrics["patch_cos_sim"]
