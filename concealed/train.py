"""Training engine for the Concealed Amortized Obfuscation Generator."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from tqdm import tqdm

from concealed.data.dataset import create_train_val_dataloaders
from concealed.losses.eot import DifferentiableEOT, build_eot
from concealed.losses.obfuscation_loss import CompositeObfuscationLoss, build_loss
from concealed.models.generator import AmortizedObfuscationGenerator, build_generator
from concealed.models.surrogates import SurrogateEnsemble, build_surrogate_ensemble


class ModelEMA:
    """Exponential Moving Average (EMA) of generator weights for stable inference."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_visual_comparison(
    x_clean: torch.Tensor,
    x_obf: torch.Tensor,
    delta: torch.Tensor,
    out_path: Path,
    max_samples: int = 4,
) -> None:
    """Save a side-by-side image grid: [Clean | 10x Magnified Delta | Obfuscated]."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = min(max_samples, x_clean.shape[0])
    rows = []
    for i in range(n):
        clean_np = (x_clean[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        obf_np = (x_obf[i].detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        # Visualize delta centered at 0.5 gray with 10x magnification
        delta_vis = ((delta[i].detach().cpu().permute(1, 2, 0).numpy() * 10.0 + 0.5) * 255.0).clip(0, 255).astype(
            np.uint8
        )
        row = np.concatenate([clean_np, delta_vis, obf_np], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(out_path)


def save_checkpoint(
    path: Path,
    generator: AmortizedObfuscationGenerator,
    ema: Optional[ModelEMA],
    config: dict,
    epoch: int,
    metrics: Dict[str, float],
) -> None:
    """Save generator checkpoint with full architecture config metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    active_model = ema.ema_model if ema is not None else generator
    payload = {
        "generator_state_dict": active_model.state_dict(),
        "raw_generator_state_dict": generator.state_dict(),
        "config": config,
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_generator_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    override_mode: Optional[str] = None,
    override_epsilon_255: Optional[float] = None,
) -> Tuple[AmortizedObfuscationGenerator, dict]:
    """Load a trained AmortizedObfuscationGenerator from a `.pt` checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    generator = build_generator(config)
    raw_sd = ckpt.get("generator_state_dict", ckpt)
    target_sd = generator.state_dict()
    filtered_sd = {k: v for k, v in raw_sd.items() if k in target_sd and target_sd[k].shape == v.shape}
    generator.load_state_dict(filtered_sd, strict=False)
    if override_mode is not None:
        generator.mode = override_mode  # type: ignore[assignment]
    if override_epsilon_255 is not None:
        generator.set_epsilon_255(override_epsilon_255)
    generator.to(device).eval()
    return generator, config


@torch.no_grad()
def validate(
    generator: AmortizedObfuscationGenerator,
    surrogates: SurrogateEnsemble,
    loss_fn: CompositeObfuscationLoss,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    sample_out_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Run validation over val_loader and compute mean obfuscation, stealth, and feature identification metrics."""
    generator.eval()
    accum: Dict[str, float] = {}
    count = 0
    use_amp = device.type == "cuda"

    # Collect embeddings across the validation gallery to measure real Feature Identification & Semantic Flip rates
    clean_glob_gallery: Dict[str, list] = {}
    obf_glob_gallery: Dict[str, list] = {}
    clean_patch_gallery: Dict[str, list] = {}
    obf_patch_gallery: Dict[str, list] = {}

    for batch_idx, x_clean in enumerate(val_loader):
        x_clean = x_clean.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            x_obf, delta = generator(x_clean, return_delta=True)
            clean_outputs = surrogates(x_clean)
            obf_outputs = surrogates(x_obf)
            _, metrics = loss_fn(x_clean, x_obf, delta, clean_outputs, obf_outputs)

        for c_out, o_out in zip(clean_outputs, obf_outputs):
            short_name = o_out.model_name.split("/")[-1]
            clean_glob_gallery.setdefault(short_name, []).append(c_out.global_embedding.detach().float().cpu())
            obf_glob_gallery.setdefault(short_name, []).append(o_out.global_embedding.detach().float().cpu())
            if c_out.patch_tokens and o_out.patch_tokens:
                # Pool final tapped layer's patch tokens into a spatial-structural descriptor
                cp_desc = torch.nn.functional.normalize(c_out.patch_tokens[-1].detach().float().mean(dim=1), p=2, dim=-1)
                op_desc = torch.nn.functional.normalize(o_out.patch_tokens[-1].detach().float().mean(dim=1), p=2, dim=-1)
                clean_patch_gallery.setdefault(short_name, []).append(cp_desc.cpu())
                obf_patch_gallery.setdefault(short_name, []).append(op_desc.cpu())

        # Compute PSNR in dB
        mse = torch.mean((x_clean.float() - x_obf.float()) ** 2).item()
        psnr = 10.0 * math.log10(1.0 / max(mse, 1e-10))
        metrics["psnr_db"] = psnr
        metrics["linf_255"] = float(torch.max(torch.abs(delta)).item() * 255.0)

        for k, v in metrics.items():
            accum[k] = accum.get(k, 0.0) + v
        count += 1

        if batch_idx == 0 and sample_out_path is not None:
            save_visual_comparison(x_clean.float(), x_obf.float(), delta.float(), sample_out_path)

    summary = {k: v / max(count, 1) for k, v in accum.items()}

    # Compute Gallery Feature Identification Evasion (%) and Semantic Neighbor Flip Rate (%) per Vision Transformer
    reid_evasion_list = []
    sem_flip_list = []
    for short_name, c_list in clean_glob_gallery.items():
        c_mat = torch.cat(c_list, dim=0)  # [N, D]
        o_mat = torch.cat(obf_glob_gallery[short_name], dim=0)  # [N, D]
        if short_name in clean_patch_gallery and clean_patch_gallery[short_name]:
            cp_mat = torch.cat(clean_patch_gallery[short_name], dim=0)
            op_mat = torch.cat(obf_patch_gallery[short_name], dim=0)
            c_joint = torch.nn.functional.normalize(0.5 * c_mat + 0.5 * cp_mat, p=2, dim=-1)
            o_joint = torch.nn.functional.normalize(0.5 * o_mat + 0.5 * op_mat, p=2, dim=-1)
        else:
            c_joint, o_joint = c_mat, o_mat

        n_gal = c_joint.shape[0]
        sim_obf_to_clean = torch.matmul(o_joint, c_joint.T)  # [N, N]
        diag_sim = torch.diag(sim_obf_to_clean)  # Self-similarity [N]
        top1_match = torch.argmax(sim_obf_to_clean, dim=-1)
        arange_idx = torch.arange(n_gal)

        # Feature Identification Evaded if Top-1 match flips to a different image OR self-similarity drops < 0.65
        evaded = (top1_match != arange_idx) | (diag_sim < 0.65)
        evasion_pct = float(evaded.float().mean().item() * 100.0)
        summary[f"reid_evasion_pct/{short_name}"] = evasion_pct
        reid_evasion_list.append(evasion_pct)

        # Semantic Neighbor Flip Rate: does the nearest non-self neighbor in the gallery change?
        if n_gal > 2:
            sim_clean_to_clean = torch.matmul(c_joint, c_joint.T)
            eye_mask = torch.eye(n_gal, dtype=torch.bool)
            sim_clean_no_self = sim_clean_to_clean.masked_fill(eye_mask, -2.0)
            sim_obf_no_self = sim_obf_to_clean.masked_fill(eye_mask, -2.0)
            nn_clean = torch.argmax(sim_clean_no_self, dim=-1)
            nn_obf = torch.argmax(sim_obf_no_self, dim=-1)
            flip_pct = float((nn_clean != nn_obf).float().mean().item() * 100.0)
        else:
            flip_pct = evasion_pct
        summary[f"semantic_flip_pct/{short_name}"] = flip_pct
        sem_flip_list.append(flip_pct)

    summary["feature_reid_evasion_pct"] = sum(reid_evasion_list) / max(len(reid_evasion_list), 1)
    summary["semantic_neighbor_flip_pct"] = sum(sem_flip_list) / max(len(sem_flip_list), 1)
    return summary


def train(
    config: dict,
    data_dir: str | Path,
    output_dir: str | Path,
    device_str: Optional[str] = None,
    pretrained_surrogates: bool = True,
    on_epoch_end: Optional[object] = None,
) -> Tuple[AmortizedObfuscationGenerator, Dict[str, float]]:
    """Execute end-to-end training of the Amortized Obfuscation Generator."""
    train_cfg = config.get("training", {})
    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    if device_str:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved configuration
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    resolution = int(train_cfg.get("train_resolution", 384))
    batch_size = int(train_cfg.get("batch_size", 4))
    val_split = float(train_cfg.get("val_split", 0.1))
    num_workers = int(train_cfg.get("num_workers", 0 if device.type == "cpu" else 4))
    max_images = train_cfg.get("max_images", None)
    if max_images is not None:
        max_images = int(max_images)

    train_loader, val_loader = create_train_val_dataloaders(
        data_dir=data_dir,
        resolution=resolution,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=num_workers,
        seed=seed,
        max_images=max_images,
    )

    generator = build_generator(config).to(device)
    ema = ModelEMA(generator, decay=float(train_cfg.get("ema_decay", 0.995)))

    surrogates = build_surrogate_ensemble(config, key="train_models", pretrained=pretrained_surrogates)
    cpu_offload = bool(config.get("surrogates", {}).get("cpu_offload", False))
    seq_grad_accum = bool(config.get("surrogates", {}).get("sequential_grad_accum", True))
    if not cpu_offload:
        surrogates.to(device)

    eot = build_eot(config).to(device)
    loss_fn = build_loss(config).to(device)

    epochs = int(train_cfg.get("epochs", 40))
    lr = float(train_cfg.get("lr", 2e-4))
    min_lr = float(train_cfg.get("min_lr", 1e-5))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    warmup_epochs = int(train_cfg.get("warmup_epochs", 2))
    grad_accum = max(1, int(train_cfg.get("grad_accumulation_steps", 1)))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    save_every = max(1, int(train_cfg.get("save_every_epochs", 5)))

    optimizer = torch.optim.AdamW(generator.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    total_steps = max(1, epochs * len(train_loader) // grad_accum)
    warmup_steps = max(1, warmup_epochs * len(train_loader) // grad_accum)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return (min_lr / lr) + (1.0 - min_lr / lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    mp_mode = str(train_cfg.get("mixed_precision", "fp16")).lower()
    use_amp = device.type == "cuda" and mp_mode in ("fp16", "bf16")
    amp_dtype = torch.bfloat16 if mp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and mp_mode == "fp16"))

    best_patch_cos = float("inf")
    last_val_metrics: Dict[str, float] = {}
    history = []
    train_start_time = time.perf_counter()
    model_short_names = [s.model_name.split("/")[-1] for s in surrogates.surrogates]

    print(
        f"\n=== Concealed Training Started ===\n"
        f"  Device     : {device} (AMP: {use_amp})\n"
        f"  Dataset    : {len(train_loader.dataset)} train / {len(val_loader.dataset)} val images @ {resolution}x{resolution}\n"
        f"  Generator  : variant={generator.variant}, mode={generator.mode}, eps={generator.epsilon_255:.1f}/255\n"
        f"  Surrogates : {', '.join(model_short_names)}\n"
        f"  Schedule   : {epochs} epochs x {len(train_loader)} steps/epoch (batch_size={batch_size}, lr={lr})\n"
        f"==================================\n",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        generator.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_metrics: Dict[str, float] = {}
        step_count = 0
        t0 = time.perf_counter()
        log_every = max(1, len(train_loader) // 4)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for idx, x_clean in enumerate(pbar):
            x_clean = x_clean.to(device, non_blocking=True)

            if (seq_grad_accum or cpu_offload) and len(surrogates.surrogates) > 1:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    x_obf, delta = generator(x_clean, return_delta=True)
                    x_clean_aug, x_obf_aug = eot(x_clean, x_obf)

                x_obf_aug_leaf = x_obf_aug.detach().requires_grad_(True)

                total_w = sum(s.weight for s in surrogates.surrogates) or 1.0
                agg_sur_metrics: Dict[str, float] = {}

                for s_mod in surrogates.surrogates:
                    if cpu_offload and device.type == "cuda":
                        s_mod.to(device)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        with torch.no_grad():
                            c_out = s_mod(x_clean_aug)
                        o_out = s_mod(x_obf_aug_leaf)
                        s_loss, s_metrics = loss_fn.compute_surrogate_disruption([c_out], [o_out])
                        weighted_s_loss = s_loss * (s_mod.weight / total_w) / grad_accum
                    scaler.scale(weighted_s_loss).backward()
                    if cpu_offload and device.type == "cuda":
                        s_mod.to("cpu")

                    w_ratio = s_mod.weight / total_w
                    for mk, mv in s_metrics.items():
                        if "/" in mk:
                            agg_sur_metrics[mk] = mv
                        else:
                            agg_sur_metrics[mk] = agg_sur_metrics.get(mk, 0.0) + mv * w_ratio

                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    stealth_loss, stealth_metrics = loss_fn.compute_perceptual_stealth(x_clean, x_obf, delta)
                    scaled_stealth = stealth_loss / grad_accum

                assert x_obf_aug_leaf.grad is not None
                surrogate_grad_bridge = (x_obf_aug.float() * x_obf_aug_leaf.grad.detach().float()).sum()
                (surrogate_grad_bridge + scaler.scale(scaled_stealth).float()).backward()

                metrics = {
                    **agg_sur_metrics,
                    **stealth_metrics,
                    "total_loss": agg_sur_metrics.get("surrogate_loss", 0.0) + stealth_metrics["stealth_loss"],
                }
            else:
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    x_obf, delta = generator(x_clean, return_delta=True)
                    x_clean_aug, x_obf_aug = eot(x_clean, x_obf)

                    with torch.no_grad():
                        clean_outputs = surrogates(x_clean_aug)
                    obf_outputs = surrogates(x_obf_aug)

                    loss, metrics = loss_fn(x_clean, x_obf, delta, clean_outputs, obf_outputs)
                    scaled_loss = loss / grad_accum

                scaler.scale(scaled_loss).backward()

            if (idx + 1) % grad_accum == 0 or (idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(generator)

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
            step_count += 1
            pbar.set_postfix(
                patch_cos=f"{metrics['patch_cos_sim']:.3f}",
                salient=f"{metrics.get('salient_patch_cos', metrics['patch_cos_sim']):.3f}",
                glob_cos=f"{metrics['global_cos_sim']:.3f}",
                conc70=f"{metrics.get('concealed_patches_70_pct', 0.0):.0f}%",
            )

            if (idx + 1) % log_every == 0 or (idx + 1) == len(train_loader):
                done_total = (epoch - 1) * len(train_loader) + (idx + 1)
                all_total = epochs * len(train_loader)
                elapsed_all = time.perf_counter() - train_start_time
                eta_sec = (elapsed_all / max(done_total, 1)) * max(0, all_total - done_total)
                eta_m, eta_s = int(eta_sec // 60), int(eta_sec % 60)
                print(
                    f"  [Live Step {idx + 1:03d}/{len(train_loader):03d} | Ep {epoch}/{epochs} | ETA {eta_m}m{eta_s:02d}s] "
                    f"PatchCos={metrics['patch_cos_sim']:.3f} | "
                    f"SalientCos={metrics.get('salient_patch_cos', 0.0):.3f} | "
                    f"GlobalCos={metrics['global_cos_sim']:.3f} | "
                    f"ConcealedPatches(<0.7)={metrics.get('concealed_patches_70_pct', 0.0):.1f}% | "
                    f"ChromaRMS={metrics.get('chroma_rms_255', 0.0):.2f}/255",
                    flush=True,
                )

        train_summary = {k: v / max(1, step_count) for k, v in epoch_metrics.items()}
        val_sample_path = out_dir / "samples" / f"epoch_{epoch:03d}.png"
        val_summary = validate(
            ema.ema_model,
            surrogates,
            loss_fn,
            val_loader,
            device,
            sample_out_path=val_sample_path,
        )
        last_val_metrics = val_summary
        elapsed = time.perf_counter() - t0
        total_elapsed = time.perf_counter() - train_start_time
        rem_sec = (total_elapsed / epoch) * (epochs - epoch)
        rem_m, rem_s = int(rem_sec // 60), int(rem_sec % 60)

        record = {
            "epoch": epoch,
            "elapsed_sec": round(elapsed, 2),
            "train": train_summary,
            "val": val_summary,
        }
        history.append(record)
        with open(out_dir / "training_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        print(
            f"\n+---------------------------------------------------------------------------------------+\n"
            f"| EPOCH {epoch:02d}/{epochs:02d} ANALYTICS REPORT ({elapsed:.1f}s | Remaining ETA: {rem_m}m{rem_s:02d}s)\n"
            f"+---------------------------------------------------------------------------------------+\n"
            f"| Feature Disruption : Patch Cos = {val_summary['patch_cos_sim']:.4f} | Salient Foreground Cos = {val_summary.get('salient_patch_cos', 0.0):.4f} | Global Cos = {val_summary['global_cos_sim']:.4f}\n"
            f"| Identification Stat: Feature ID Evasion = {val_summary.get('feature_reid_evasion_pct', 0.0):5.1f}% | Semantic Category Flip = {val_summary.get('semantic_neighbor_flip_pct', 0.0):5.1f}%\n"
            f"| Spatial Masking    : Patches <0.70 Sim  = {val_summary.get('concealed_patches_70_pct', 0.0):5.1f}% | Patches <0.50 Sim      = {val_summary.get('concealed_patches_50_pct', 0.0):5.1f}%\n"
            f"| Visual Stealth     : PSNR = {val_summary['psnr_db']:.2f} dB | Chroma Shift = {val_summary.get('chroma_rms_255', 0.0):.2f}/255 | L_inf = {val_summary['linf_255']:.2f}/255 | UAP Ratio = {val_summary.get('uap_collapse_ratio', 0.0):.3f}\n"
            f"| Per-Transformer Breakdown:",
            flush=True,
        )
        for s_name in model_short_names:
            p_c = val_summary.get(f"patch_cos/{s_name}", 0.0)
            s_c = val_summary.get(f"salient_cos/{s_name}", 0.0)
            g_c = val_summary.get(f"global_cos/{s_name}", 0.0)
            ev_p = val_summary.get(f"reid_evasion_pct/{s_name}", 0.0)
            fl_p = val_summary.get(f"semantic_flip_pct/{s_name}", 0.0)
            print(
                f"|   * {s_name:28s} -> PatchCos: {p_c:.4f} | SalientCos: {s_c:.4f} | GlobalCos: {g_c:.4f} | ID Evasion: {ev_p:5.1f}% | SemFlip: {fl_p:5.1f}%",
                flush=True,
            )
        print("+---------------------------------------------------------------------------------------+\n", flush=True)

        # Save latest & best checkpoints
        save_checkpoint(out_dir / "latest_generator.pt", generator, ema, config, epoch, val_summary)
        if val_summary["patch_cos_sim"] < best_patch_cos:
            best_patch_cos = val_summary["patch_cos_sim"]
            save_checkpoint(out_dir / "best_generator.pt", generator, ema, config, epoch, val_summary)

        if epoch % save_every == 0:
            save_checkpoint(out_dir / f"generator_epoch_{epoch:03d}.pt", generator, ema, config, epoch, val_summary)

        if callable(on_epoch_end):
            on_epoch_end(epoch, out_dir, val_summary)

    # Automatically export the best generator to ONNX for immediate real-time pipeline use
    if train_cfg.get("auto_export_onnx", True):
        from concealed.pipeline.export import export_to_onnx

        onnx_path = out_dir / "generator.onnx"
        export_info = export_to_onnx(
            generator=ema.ema_model,
            output_path=onnx_path,
            verify=True,
            quantize_int8=False,
        )
        print(f"Auto-exported real-time ONNX model -> {export_info['onnx_path']} ({export_info['size_mb']} MB)")

    return ema.ema_model, last_val_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Concealed Amortized Obfuscation Generator")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config file")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing training images")
    parser.add_argument("--output-dir", type=str, default="runs/default", help="Directory to save checkpoints & logs")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon_255 perturbation budget")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["canonical_residual", "native", "hybrid"],
        default=None,
        help="Override generator synthesis mode",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["tiny", "base", "large"],
        default=None,
        help="Override generator capacity preset",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["default", "fast", "ocr", "full"],
        default=None,
        help="Select built-in surrogate model profile ('default'=SigLIP+DFN5B+ConvNeXt+DINOv2, 'ocr'=GLM-OCR priority, 'full'=all 6 models)",
    )
    parser.add_argument(
        "--surrogates",
        type=str,
        nargs="+",
        default=None,
        help="Override training surrogate model names or aliases (e.g., siglip-so400m dfn5b openclip-convnext-large dinov2-large glm-ocr)",
    )
    parser.add_argument("--device", type=str, default=None, help="Override compute device (cuda or cpu)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.profile and args.profile != "default":
        prof = config.get("surrogates", {}).get("profiles", {}).get(args.profile)
        if prof:
            config["surrogates"]["train_models"] = prof["train_models"]
            if "sequential_offload" in prof:
                config["surrogates"]["sequential_offload"] = prof["sequential_offload"]
        if args.profile == "ocr":
            # For OCR obfuscation, prioritize high-res local tile branch
            config.setdefault("generator", {})["hybrid_global_weight"] = 0.35

    if args.epochs is not None:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        config.setdefault("training", {})["lr"] = args.lr
    if args.epsilon is not None:
        config.setdefault("generator", {})["epsilon_255"] = args.epsilon
    if args.mode is not None:
        config.setdefault("generator", {})["mode"] = args.mode
    if args.variant is not None:
        config.setdefault("generator", {})["variant"] = args.variant
    if args.surrogates is not None:
        config.setdefault("surrogates", {})["train_models"] = [{"name": s, "weight": 1.0} for s in args.surrogates]

    train(config=config, data_dir=args.data_dir, output_dir=args.output_dir, device_str=args.device)


if __name__ == "__main__":
    main()
