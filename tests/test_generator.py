"""Unit tests for AmortizedObfuscationGenerator across variants, modes, and resolutions."""

from __future__ import annotations

import pytest
import torch

from concealed.models.generator import AmortizedObfuscationGenerator


@pytest.mark.parametrize("variant", ["tiny", "base", "large"])
def test_generator_variants_and_bounds(variant: str) -> None:
    eps_255 = 8.0
    eps = eps_255 / 255.0
    gen = AmortizedObfuscationGenerator(
        variant=variant,  # type: ignore[arg-type]
        epsilon_255=eps_255,
        mode="canonical_residual",
        canonical_size=128,
        tile_size=128,
    ).eval()

    x = torch.rand(2, 3, 160, 200)
    with torch.no_grad():
        x_obf, delta = gen(x, return_delta=True)

    assert x_obf.shape == x.shape
    assert delta.shape == x.shape
    assert float(x_obf.min().item()) >= 0.0
    assert float(x_obf.max().item()) <= 1.0
    assert float(torch.max(torch.abs(delta)).item()) <= eps + 1e-5


@pytest.mark.parametrize("mode", ["canonical_residual", "native", "hybrid"])
@pytest.mark.parametrize("hw", [(128, 128), (155, 213), (300, 420)])
def test_generator_modes_and_arbitrary_resolutions(mode: str, hw: tuple[int, int]) -> None:
    gen = AmortizedObfuscationGenerator(
        variant="tiny",
        epsilon_255=10.0,
        mode=mode,  # type: ignore[arg-type]
        canonical_size=128,
        tile_size=128,
        luminance_texture_masking=True,
        use_dct_stem=True,
    )
    x = torch.rand(1, 3, hw[0], hw[1], requires_grad=True)
    x_obf, delta = gen(x, return_delta=True)

    assert x_obf.shape == (1, 3, hw[0], hw[1])
    assert delta.shape == (1, 3, hw[0], hw[1])

    loss = x_obf.mean() + delta.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    # Verify generator parameters receive non-zero gradients
    grad_norms = [p.grad.norm().item() for p in gen.parameters() if p.grad is not None]
    assert len(grad_norms) > 0 and sum(grad_norms) > 0.0
