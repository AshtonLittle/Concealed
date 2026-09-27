"""Evaluation and Black-Box Transferability Benchmark Suite.

Evaluates a trained AmortizedObfuscationGenerator on both:
  1. Training surrogates (white-box / seen Vision Transformers)
  2. Held-out open-source Vision Transformers (black-box transferability test,
     mimicking unseen frontier models)

Reports per-model and aggregate metrics:
  - Spatial Patch Token Cosine Similarity (1.0 = identical, <= 0.2 = severe disruption)
  - Global Embedding Cosine Similarity
  - Top-1 Image Feature Self-Retrieval Collapse Rate (%)
  - Visual Stealth Metrics: PSNR (dB), SSIM, and L_infinity (out of 255)
  - Real-time Inference Latency (ms/image and FPS)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from concealed.data.dataset import ImageObfuscationDataset, discover_images
from concealed.losses.obfuscation_loss import compute_ssim_loss
from concealed.models.generator import AmortizedObfuscationGenerator
from concealed.models.surrogates import VisionTransformerSurrogate
from concealed.train import load_generator_checkpoint


@torch.no_grad()
def evaluate_generator(
    generator: AmortizedObfuscationGenerator,
    image_paths: Sequence[Path],
    surrogate_names: Sequence[str],
    resolution: int = 512,
    batch_size: int = 8,
    device: str | torch.device = "cpu",
    pretrained_surrogates: bool = True,
) -> Dict[str, object]:
    """Benchmark generator obfuscation efficacy and stealth across multiple ViT models."""
    dev = torch.device(device)
    generator.to(dev).eval()

    dataset = ImageObfuscationDataset(image_paths, resolution=resolution, is_train=False)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=False)

    # First pass: generate obfuscated images, measure latency & visual quality
    clean_batches: List[torch.Tensor] = []
    obf_batches: List[torch.Tensor] = []

    total_psnr = 0.0
    total_ssim = 0.0
    max_linf_255 = 0.0
    total_images = 0
    total_gen_time = 0.0

    for x_clean in loader:
        x_clean = x_clean.to(dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_obf, delta = generator(x_clean, return_delta=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        total_gen_time += time.perf_counter() - t0

        bsz = x_clean.shape[0]
        total_images += bsz

        mse = torch.mean((x_clean - x_obf) ** 2, dim=(1, 2, 3))
        batch_psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-10))
        total_psnr += float(batch_psnr.sum().item())

        ssim_val = 1.0 - float(compute_ssim_loss(x_clean, x_obf).item())
        total_ssim += ssim_val * bsz

        linf = float(torch.max(torch.abs(delta)).item() * 255.0)
        max_linf_255 = max(max_linf_255, linf)

        clean_batches.append(x_clean.cpu())
        obf_batches.append(x_obf.cpu())

    avg_latency_ms = (total_gen_time / max(1, total_images)) * 1000.0
    fps = 1000.0 / max(avg_latency_ms, 1e-6)

    # Second pass: evaluate each surrogate model sequentially to conserve VRAM
    per_model_results: Dict[str, Dict[str, float]] = {}
    for model_name in surrogate_names:
        surrogate = VisionTransformerSurrogate(
            model_name=model_name,
            weight=1.0,
            pretrained=pretrained_surrogates,
        ).to(dev)

        clean_global_list = []
        obf_global_list = []
        patch_cos_sum = 0.0
        global_cos_sum = 0.0
        n_samples = 0

        for x_c_cpu, x_o_cpu in zip(clean_batches, obf_batches):
            x_c = x_c_cpu.to(dev)
            x_o = x_o_cpu.to(dev)
            bsz = x_c.shape[0]

            out_c = surrogate(x_c)
            out_o = surrogate(x_o)

            g_cos = (out_c.global_embedding * out_o.global_embedding).sum(dim=-1)
            global_cos_sum += float(g_cos.sum().item())

            if out_c.patch_tokens and out_o.patch_tokens:
                layer_cos = []
                for pc, po in zip(out_c.patch_tokens, out_o.patch_tokens):
                    layer_cos.append((pc * po).sum(dim=-1).mean(dim=-1))
                p_cos = torch.stack(layer_cos, dim=0).mean(dim=0)
                patch_cos_sum += float(p_cos.sum().item())

            clean_global_list.append(out_c.global_embedding.cpu())
            obf_global_list.append(out_o.global_embedding.cpu())
            n_samples += bsz

        # Compute Top-1 Image Feature Self-Retrieval Collapse Rate
        # (Given obfuscated query embedding, does it still match its own clean image in the gallery?)
        all_clean_g = torch.cat(clean_global_list, dim=0)  # [N, D]
        all_obf_g = torch.cat(obf_global_list, dim=0)    # [N, D]
        sim_matrix = torch.matmul(all_obf_g, all_clean_g.t())  # [N, N]
        top1_Pred = sim_matrix.argmax(dim=-1)
        ground_truth = torch.arange(all_clean_g.shape[0])
        fooling_rate = float((top1_Pred != ground_truth).float().mean().item() * 100.0)

        per_model_results[model_name] = {
            "patch_cosine_similarity": round(patch_cos_sum / max(1, n_samples), 4),
            "global_cosine_similarity": round(global_cos_sum / max(1, n_samples), 4),
            "retrieval_fooling_rate_pct": round(fooling_rate, 2),
        }
        del surrogate

    mean_patch_cos = sum(m["patch_cosine_similarity"] for m in per_model_results.values()) / max(
        1, len(per_model_results)
    )
    mean_global_cos = sum(m["global_cosine_similarity"] for m in per_model_results.values()) / max(
        1, len(per_model_results)
    )

    report = {
        "num_images": total_images,
        "stealth": {
            "psnr_db": round(total_psnr / max(1, total_images), 2),
            "ssim": round(total_ssim / max(1, total_images), 4),
            "max_linf_255": round(max_linf_255, 2),
        },
        "performance": {
            "avg_latency_ms": round(avg_latency_ms, 2),
            "throughput_fps": round(fps, 2),
        },
        "aggregate_obfuscation": {
            "mean_patch_cosine_similarity": round(mean_patch_cos, 4),
            "mean_global_cosine_similarity": round(mean_global_cos, 4),
        },
        "models": per_model_results,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Concealed Generator against seen & unseen ViTs")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained generator .pt checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory of evaluation images")
    parser.add_argument(
        "--extra-surrogates",
        type=str,
        nargs="*",
        default=[],
        help="Additional unseen open-source ViT models to test black-box transferability",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--resolution", type=int, default=512, help="Evaluation image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save evaluation JSON report")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator, config = load_generator_checkpoint(args.checkpoint, device=device)

    sur_cfg = config.get("surrogates", {})
    seen_models = [m["name"] for m in sur_cfg.get("train_models", [])]
    holdout_models = [m["name"] for m in sur_cfg.get("eval_holdout_models", [])]
    all_models = list(dict.fromkeys(seen_models + holdout_models + list(args.extra_surrogates)))
    if not all_models:
        all_models = ["openai/clip-vit-base-patch16"]

    image_paths = discover_images(args.data_dir)
    report = evaluate_generator(
        generator=generator,
        image_paths=image_paths,
        surrogate_names=all_models,
        resolution=args.resolution,
        batch_size=args.batch_size,
        device=device,
    )

    print(json.dumps(report, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
