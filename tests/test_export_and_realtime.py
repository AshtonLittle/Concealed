"""End-to-end integration tests for training, ONNX export, evaluation, and RealtimeObfuscator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from concealed.evaluate import evaluate_generator
from concealed.pipeline.export import export_to_onnx, export_to_torchscript
from concealed.pipeline.realtime import RealtimeObfuscator
from concealed.train import train


def _create_dummy_dataset(folder: Path, num_images: int = 6) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(num_images):
        arr = np.random.randint(0, 256, size=(160, 160, 3), dtype=np.uint8)
        p = folder / f"sample_{i:02d}.png"
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def test_end_to_end_train_export_and_realtime_pipeline(tmp_path: Path) -> None:
    data_dir = tmp_path / "images"
    out_dir = tmp_path / "run"
    image_paths = _create_dummy_dataset(data_dir, num_images=6)

    config = {
        "generator": {
            "variant": "tiny",
            "epsilon_255": 8.0,
            "mode": "hybrid",
            "canonical_size": 128,
            "tile_size": 64,
            "hybrid_global_weight": 0.6,
            "luminance_texture_masking": True,
            "use_dct_stem": True,
        },
        "surrogates": {
            "sequential_offload": False,
            "tap_layers": [-2, -1],
            "train_models": [
                {"name": "mock/tiny-vit-16", "weight": 1.0},
                {"name": "openai/clip-vit-base-patch16", "weight": 1.0},
            ],
        },
        "eot": {
            "enabled": True,
            "jpeg_prob": 0.5,
            "random_scale_min": 0.85,
            "random_scale_max": 1.15,
            "tile_crop_prob": 0.5,
        },
        "loss": {
            "patch_cosine_weight": 2.5,
            "global_cosine_weight": 1.5,
            "patch_dispersion_weight": 0.5,
            "ssim_weight": 1.0,
            "low_freq_tv_weight": 1.0,
        },
        "training": {
            "train_resolution": 128,
            "batch_size": 2,
            "grad_accumulation_steps": 1,
            "epochs": 2,
            "lr": 5e-4,
            "warmup_epochs": 1,
            "val_split": 0.33,
            "num_workers": 0,
            "seed": 42,
        },
    }

    # 1. Run end-to-end training
    trained_gen, val_metrics = train(
        config=config,
        data_dir=data_dir,
        output_dir=out_dir,
        device_str="cpu",
        pretrained_surrogates=False,
    )
    best_ckpt = out_dir / "best_generator.pt"
    assert best_ckpt.exists()
    assert "patch_cos_sim" in val_metrics

    # 2. Export to ONNX & verify dynamic spatial resolution parity
    onnx_path = out_dir / "generator.onnx"
    export_info = export_to_onnx(
        generator=trained_gen,
        output_path=onnx_path,
        verify=True,
        quantize_int8=False,
    )
    assert onnx_path.exists()
    assert export_info["verified"] is True

    # 3. Export to TorchScript
    ts_path = export_to_torchscript(trained_gen, out_dir / "generator.torchscript.pt")
    assert ts_path.exists()

    # 4. Run RealtimeObfuscator with both PyTorch (.pt) and ONNX (.onnx) backends
    obf_pt = RealtimeObfuscator(best_ckpt, device="cpu")
    obf_onnx = RealtimeObfuscator(onnx_path, device="cpu")

    # Test on an arbitrary non-square resolution (190x250)
    pil_in = Image.fromarray(np.random.randint(0, 256, size=(190, 250, 3), dtype=np.uint8))
    pil_out_pt = obf_pt.obfuscate_pil(pil_in)
    pil_out_onnx = obf_onnx.obfuscate_pil(pil_in)
    assert pil_out_pt.size == pil_in.size
    assert pil_out_onnx.size == pil_in.size

    # Verify PT and ONNX uint8 outputs match within +/-1 rounding step
    diff_uint8 = np.abs(np.asarray(pil_out_pt, dtype=np.int16) - np.asarray(pil_out_onnx, dtype=np.int16))
    assert int(diff_uint8.max()) <= 1

    # 5. Evaluate against seen and held-out ViT models
    eval_report = evaluate_generator(
        generator=trained_gen,
        image_paths=image_paths[:4],
        surrogate_names=["mock/tiny-vit-16", "mock/tiny-vit-14"],
        resolution=128,
        batch_size=2,
        device="cpu",
        pretrained_surrogates=False,
    )
    assert "models" in eval_report
    assert "mock/tiny-vit-14" in eval_report["models"]
    assert eval_report["stealth"]["max_linf_255"] <= 8.5
