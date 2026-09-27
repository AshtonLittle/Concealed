"""Unit tests verifying ModelProbeService SigLIP-SO400M training architecture surrogate & evaluations."""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from concealed.api.probe_service import ModelProbeService
from concealed.models.surrogates import VisionTransformerSurrogate


def test_siglip_so400m_surrogate_initialization() -> None:
    """Verify VisionTransformerSurrogate initializes with google/siglip-so400m-patch14-384."""
    surrogate = VisionTransformerSurrogate(
        model_name="google/siglip-so400m-patch14-384",
        weight=1.0,
        tap_layers=[-3, -2, -1],
        pretrained=False,
    )
    x = torch.rand(1, 3, 384, 384, requires_grad=True)
    out = surrogate(x)

    assert out.name == "google/siglip-so400m-patch14-384"
    assert out.global_embedding.shape[0] == 1
    assert len(out.patch_tokens) == 3
    for pt in out.patch_tokens:
        assert pt.shape[0] == 1
        assert pt.ndim == 3


def test_model_probe_service_training_surrogate_eval() -> None:
    """Verify _run_training_surrogate_eval computes the full suite of training ViT metrics."""
    service = ModelProbeService()
    surrogate = VisionTransformerSurrogate(
        model_name="google/siglip-so400m-patch14-384",
        weight=1.0,
        tap_layers=[-3, -2, -1],
        pretrained=False,
    )

    clean_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    obf_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    eval_res = service._run_training_surrogate_eval(surrogate, clean_img, obf_img)

    expected_keys = [
        "patch_cos_sim",
        "salient_patch_cos",
        "global_cos_sim",
        "patch_reid_evasion_pct",
        "salient_displacement_pct",
        "concealed_patches_70_pct",
        "concealed_patches_50_pct",
        "evasion_score_pct",
        "sim_drop_pct",
        "identification_evaded",
        "evasion_status",
    ]
    for key in expected_keys:
        assert key in eval_res, f"Missing metric key: {key}"

    assert isinstance(eval_res["patch_cos_sim"], float)
    assert isinstance(eval_res["salient_patch_cos"], float)
    assert isinstance(eval_res["global_cos_sim"], float)
    assert 0.0 <= eval_res["patch_reid_evasion_pct"] <= 100.0
    assert 0.0 <= eval_res["salient_displacement_pct"] <= 100.0
    assert isinstance(eval_res["identification_evaded"], bool)
    assert eval_res["evasion_status"] in ["EVADED", "WEAKENED", "VISIBLE"]


def test_probe_options_siglip_end_to_end() -> None:
    """Verify probe_options_siglip outputs SiglipProbeResponse with SiglipTrainingEvaluations."""
    service = ModelProbeService()

    # Pre-inject offline surrogate to prevent network calls during CI
    service._siglip_surrogate = VisionTransformerSurrogate(
        model_name="google/siglip-so400m-patch14-384",
        weight=1.0,
        tap_layers=[-3, -2, -1],
        pretrained=False,
    )

    import io

    clean_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))
    obf_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

    clean_buf = io.BytesIO()
    clean_img.save(clean_buf, format="JPEG")
    obf_buf = io.BytesIO()
    obf_img.save(obf_buf, format="JPEG")

    resp = service.probe_options_siglip(
        clean_image_bytes=clean_buf.getvalue(),
        obfuscated_image_bytes=obf_buf.getvalue(),
        options=["person", "car", "face"],
    )

    assert resp.success is True
    assert resp.model_name == "google/siglip-so400m-patch14-384"
    assert resp.total_options == 3
    assert len(resp.results) == 3
    assert resp.training_evaluations is not None
    assert resp.training_evaluations.model_name == "google/siglip-so400m-patch14-384"
    assert resp.training_evaluations.patch_cos_sim is not None
    assert resp.training_evaluations.salient_patch_cos is not None
    assert resp.training_evaluations.global_cos_sim is not None
    assert resp.training_evaluations.patch_reid_evasion_pct is not None
    assert resp.training_evaluations.salient_displacement_pct is not None
    assert resp.training_evaluations.evasion_status in ["EVADED", "WEAKENED", "VISIBLE"]
