"""Real-Time Image & Video Stream Obfuscation Pipeline.

Provides the `RealtimeObfuscator` class supporting both PyTorch (`.pt`) and
ONNX Runtime (`.onnx`) backends for low-latency integration into:
  - Live video / webcam / RTSP streaming pipelines
  - Web API image upload middleware (PIL / NumPy / bytes)
  - High-throughput batch folder obfuscation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from concealed.data.dataset import discover_images
from concealed.models.generator import AmortizedObfuscationGenerator, SynthesisMode
from concealed.train import load_generator_checkpoint


class RealtimeObfuscator:
    """High-throughput real-time image obfuscation engine (PyTorch or ONNX Runtime)."""

    def __init__(
        self,
        model_path: str | Path | AmortizedObfuscationGenerator,
        device: Optional[str] = None,
        mode: Optional[SynthesisMode] = None,
        epsilon_255: Optional[float] = None,
        fp16: bool = True,
    ) -> None:
        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(self.device_str)
        self.mode = mode
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.backend = "torch"
        self.ort_session = None
        self.generator: Optional[AmortizedObfuscationGenerator] = None

        if isinstance(model_path, AmortizedObfuscationGenerator):
            self.generator = model_path.to(self.device).eval()
            if mode is not None:
                self.generator.mode = mode
            if epsilon_255 is not None:
                self.generator.set_epsilon_255(epsilon_255)
        else:
            path = Path(model_path)
            if path.suffix.lower() == ".onnx":
                import onnxruntime as ort

                self.backend = "onnx"
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device.type == "cuda"
                    else ["CPUExecutionProvider"]
                )
                available = ort.get_available_providers()
                active_providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
                self.ort_session = ort.InferenceSession(str(path), providers=active_providers)
            else:
                self.generator, _ = load_generator_checkpoint(
                    path,
                    device=self.device,
                    override_mode=mode,
                    override_epsilon_255=epsilon_255,
                )

    @torch.no_grad()
    def obfuscate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Obfuscate a batch of RGB tensors ``[B, 3, H, W]`` or ``[3, H, W]`` in ``[0, 1]``."""
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze = True

        if self.backend == "onnx":
            assert self.ort_session is not None
            inp_np = x.detach().cpu().numpy().astype(np.float32)
            out_np = self.ort_session.run(["obfuscated_image"], {"input_image": inp_np})[0]
            out = torch.from_numpy(out_np).to(x.device)
        else:
            assert self.generator is not None
            x_dev = x.to(self.device, non_blocking=True)
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                out = self.generator(x_dev, return_delta=False, mode=self.mode)
            assert isinstance(out, torch.Tensor)
            out = out.float()

        return out.squeeze(0) if squeeze else out

    def obfuscate_numpy(self, rgb_uint8: np.ndarray) -> np.ndarray:
        """Obfuscate an HxWx3 uint8 RGB numpy array and return an HxWx3 uint8 RGB array."""
        tensor = torch.from_numpy(rgb_uint8).permute(2, 0, 1).float().div(255.0)
        obf_tensor = self.obfuscate_tensor(tensor)
        obf_np = (obf_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        return obf_np

    def obfuscate_bgr_frame(self, bgr_uint8: np.ndarray) -> np.ndarray:
        """Obfuscate an OpenCV HxWx3 uint8 BGR video frame in real time."""
        rgb = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
        obf_rgb = self.obfuscate_numpy(rgb)
        return cv2.cvtColor(obf_rgb, cv2.COLOR_RGB2BGR)

    def obfuscate_pil(self, image: Image.Image) -> Image.Image:
        """Obfuscate a PIL Image and return a new PIL Image at the exact native resolution."""
        rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
        obf_rgb = self.obfuscate_numpy(rgb)
        return Image.fromarray(obf_rgb)

    def obfuscate_file(self, input_path: str | Path, output_path: str | Path) -> None:
        """Obfuscate a single image file on disk."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(input_path) as img:
            obf_img = self.obfuscate_pil(img)
            obf_img.save(out, quality=95)

    def obfuscate_directory(self, input_dir: str | Path, output_dir: str | Path) -> int:
        """Obfuscate all supported images in ``input_dir`` and write them to ``output_dir``."""
        in_root = Path(input_dir)
        out_root = Path(output_dir)
        files = discover_images(in_root)
        for fpath in files:
            rel = fpath.relative_to(in_root) if fpath != in_root else Path(fpath.name)
            self.obfuscate_file(fpath, out_root / rel)
        return len(files)

    def obfuscate_video(
        self,
        input_video: str | Path,
        output_video: str | Path,
        max_frames: Optional[int] = None,
    ) -> Dict[str, float]:
        """Run the generator frame-by-frame over a video file and write the obfuscated video."""
        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(input_video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_video}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        frame_count = 0
        t0 = time.perf_counter()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                obf_frame = self.obfuscate_bgr_frame(frame)
                writer.write(obf_frame)
                frame_count += 1
                if max_frames is not None and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()

        elapsed = max(1e-6, time.perf_counter() - t0)
        return {
            "frames_processed": frame_count,
            "elapsed_sec": round(elapsed, 3),
            "avg_fps": round(frame_count / elapsed, 2),
            "avg_latency_ms": round((elapsed / max(1, frame_count)) * 1000.0, 2),
        }

    def benchmark(
        self,
        resolution: Tuple[int, int] = (1080, 1920),
        warmup: int = 3,
        iterations: int = 15,
    ) -> Dict[str, float]:
        """Benchmark real-time inference latency and FPS at a target (H, W) resolution."""
        h, w = resolution
        dummy = torch.rand(1, 3, h, w, dtype=torch.float32)

        for _ in range(warmup):
            _ = self.obfuscate_tensor(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = self.obfuscate_tensor(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        avg_ms = (elapsed / iterations) * 1000.0
        return {
            "height": h,
            "width": w,
            "backend": self.backend,
            "device": self.device_str,
            "avg_latency_ms": round(avg_ms, 2),
            "throughput_fps": round(1000.0 / max(avg_ms, 1e-6), 2),
        }


    def refine_tensor(
        self,
        x_clean: torch.Tensor,
        x_init: torch.Tensor,
        steps: int = 15,
        epsilon_255: Optional[float] = None,
        surrogate_names: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Hybrid test-time refinement: warm-start from generator output and run high-pass edge-masked ViT repulsion."""
        import torch.nn.functional as F
        from concealed.models.generator import WeberTextureMask
        from concealed.models.surrogates import VisionTransformerSurrogate

        squeeze = False
        if x_clean.ndim == 3:
            x_clean = x_clean.unsqueeze(0)
            x_init = x_init.unsqueeze(0)
            squeeze = True

        dev = self.device
        x_c = x_clean.to(dev).float()
        x_0 = x_init.to(dev).float()
        eps = float(epsilon_255 if epsilon_255 is not None else (self.generator.epsilon_255 if self.generator else 10.0)) / 255.0

        if Path(".snowflake_hf_cache").exists():
            os.environ.setdefault("HF_HOME", str(Path(".snowflake_hf_cache").resolve()))
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        names = surrogate_names or [
            "google/siglip-base-patch16-224",
            "openai/clip-vit-base-patch16",
            "facebook/dinov2-base",
            "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        ]
        models = [VisionTransformerSurrogate(n, tap_layers=(-3, -2, -1)).to(dev).eval() for n in names]
        weber = WeberTextureMask(min_mask_scale=0.28).to(dev)

        h, w = x_c.shape[-2], x_c.shape[-1]
        with torch.no_grad():
            # Evaluate texture + shadow mask at canonical 384x384 so it is 100% scale-invariant across 720p and 20MP+
            x_384_clean = F.interpolate(x_c, size=(384, 384), mode="bilinear", align_corners=False)
            lum_384 = 0.299 * x_384_clean[:, 0:1] + 0.587 * x_384_clean[:, 1:2] + 0.114 * x_384_clean[:, 2:3]
            shadow_gate_384 = (lum_384 / 0.08).clamp(0.35, 1.0)
            tex_mask_384 = weber(x_384_clean) * shadow_gate_384

            # Exact 224x224 clean image as seen by VisionTransformerSurrogate.preprocess(x_c)
            x_224_clean = F.interpolate(x_c, size=(224, 224), mode="bilinear", align_corners=False)
            clean_targets = [m(x_224_clean) for m in models]

            if self.generator is not None:
                raw_init_224 = (0.25 * torch.tanh(self.generator._forward_padded(x_224_clean))).detach().clone()
                raw_init_384 = (0.25 * torch.tanh(self.generator._forward_padded(x_384_clean))).detach().clone()
            else:
                init_delta = (x_0 - x_c).clamp(-eps * 0.99, eps * 0.99)
                init_224 = F.interpolate(init_delta, size=(224, 224), mode="bilinear", align_corners=False)
                init_384 = F.interpolate(init_delta, size=(384, 384), mode="bilinear", align_corners=False)
                raw_init_224 = torch.atanh((init_224 / max(eps, 1e-6)).clamp(-0.95, 0.95)).detach().clone()
                raw_init_384 = torch.atanh((init_384 / max(eps, 1e-6)).clamp(-0.95, 0.95)).detach().clone()

        param_224 = raw_init_224.requires_grad_(True)
        param_384 = raw_init_384.requires_grad_(True)
        opt = torch.optim.Adam([param_224, param_384], lr=0.22)

        def _smooth_hp(z: torch.Tensor) -> torch.Tensor:
            lp = z
            for _ in range(3):
                lp = F.avg_pool2d(lp, kernel_size=5, stride=1, padding=2)
            return z - lp

        def _synthesize_delta_384() -> torch.Tensor:
            hp_224 = _smooth_hp(param_224)
            hp_384 = _smooth_hp(param_384)
            # Local RMS wave normalization at 224/384 prevents tanh saturation plateaus & contour lines!
            rms_224 = F.avg_pool2d(hp_224.square(), kernel_size=9, stride=1, padding=4).mean(dim=1, keepdim=True).sqrt() + 1e-4
            rms_384 = F.avg_pool2d(hp_384.square(), kernel_size=9, stride=1, padding=4).mean(dim=1, keepdim=True).sqrt() + 1e-4
            norm_224 = hp_224 / rms_224
            norm_384 = hp_384 / rms_384

            up_224 = F.interpolate(norm_224, size=(384, 384), mode="bicubic", align_corners=False)
            raw = 0.65 * up_224 + 0.35 * norm_384

            # Subtle cap (L_inf <= 5.5/255 on textures, ~1.5/255 on smooth skin/walls -> PSNR ~ 44-45 dB)
            soft_cap = min(eps, 5.5 / 255.0)
            d = soft_cap * torch.tanh(raw * 0.68)
            # YCbCr chrominance damping (85% pure luminance so there is zero color tint)
            d_y = 0.299 * d[:, 0:1] + 0.587 * d[:, 1:2] + 0.114 * d[:, 2:3]
            d = (d_y + 0.15 * (d - d_y)) * tex_mask_384
            return torch.clamp(d, -soft_cap, soft_cap)

        print(f"\n--- Running Hybrid Smooth ViT Refinement ({steps} steps, eps={eps * 255:.1f}/255) ---", flush=True)
        for step in range(1, steps + 1):
            opt.zero_grad()
            delta_384 = _synthesize_delta_384()
            delta_224 = F.interpolate(delta_384, size=(224, 224), mode="bilinear", align_corners=False)
            x_adv = torch.clamp(x_224_clean + delta_224, 0.0, 1.0)

            total_loss = torch.tensor(0.0, device=dev)
            p_cos_list = []
            s_cos_list = []
            g_cos_list = []
            c50_list = []

            for m, c_out in zip(models, clean_targets):
                o_out = m(x_adv)
                g_cos = (c_out.global_embedding.detach() * o_out.global_embedding).sum(dim=-1).mean()
                g_cos_list.append(float(g_cos.detach().item()))
                total_loss = total_loss + 2.5 * g_cos

                for cp, op in zip(c_out.patch_tokens, o_out.patch_tokens):
                    cp_d = cp.detach()
                    per_p = (cp_d * op).sum(dim=-1)
                    p_cos_list.append(float(per_p.mean().detach().item()))
                    c50_list.append(float((per_p.detach() < 0.50).float().mean().item() * 100.0))

                    # Salience-weighted foreground attack
                    cp_cent = cp_d - cp_d.mean(dim=1, keepdim=True)
                    sal = cp_cent.norm(dim=-1)
                    sal_w = torch.softmax(sal * 3.0, dim=-1) * cp_d.shape[1]
                    total_loss = total_loss + 3.5 * (per_p * sal_w).mean()

                    k_top = max(1, cp_d.shape[1] // 4)
                    top_idx = torch.topk(sal, k=k_top, dim=-1).indices
                    s_cos_list.append(float(torch.gather(per_p.detach(), 1, top_idx).mean().item()))

            total_loss.backward()
            opt.step()

            if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
                print(
                    f"  [Step {step:02d}/{steps:02d}] "
                    f"PatchCos={np.mean(p_cos_list):.4f} | "
                    f"SalientForegroundCos={np.mean(s_cos_list):.4f} | "
                    f"GlobalCos={np.mean(g_cos_list):.4f} | "
                    f"ConcealedPatches(<0.5)={np.mean(c50_list):.1f}%",
                    flush=True,
                )

        with torch.no_grad():
            delta_384 = _synthesize_delta_384()
            delta = F.interpolate(delta_384, size=(h, w), mode="bicubic", align_corners=False)
            soft_cap = min(eps, 5.5 / 255.0)
            delta = torch.clamp(delta, -soft_cap, soft_cap)
            x_final = torch.clamp(x_c + delta, 0.0, 1.0).to(x_clean.device)

        return x_final.squeeze(0) if squeeze else x_final

    @torch.no_grad()
    def analyze_image_pair(
        self,
        clean_pil: Image.Image,
        obf_pil: Image.Image,
        eval_models: Optional[list[str]] = None,
    ) -> Dict[str, object]:
        """Compute and print a comprehensive ViT Feature Identification & Visual Stealth Analytics report."""
        import math
        import torch.nn.functional as F
        from concealed.losses.obfuscation_loss import compute_ssim_loss
        from concealed.models.surrogates import VisionTransformerSurrogate

        clean_np = np.asarray(clean_pil.convert("RGB"), dtype=np.float32) / 255.0
        obf_np = np.asarray(obf_pil.convert("RGB"), dtype=np.float32) / 255.0
        x_c = torch.from_numpy(clean_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x_o = torch.from_numpy(obf_np).permute(2, 0, 1).unsqueeze(0).to(self.device)
        delta = x_o - x_c

        mse = float(delta.pow(2).mean().item())
        psnr_db = 10.0 * math.log10(1.0 / max(mse, 1e-10))
        if max(x_c.shape[-2], x_c.shape[-1]) > 1280:
            s_scale = 1280.0 / float(max(x_c.shape[-2], x_c.shape[-1]))
            sh, sw = int(round(x_c.shape[-2] * s_scale)), int(round(x_c.shape[-1] * s_scale))
            ssim_val = 1.0 - float(
                compute_ssim_loss(
                    F.interpolate(x_c, size=(sh, sw), mode="bilinear", align_corners=False),
                    F.interpolate(x_o, size=(sh, sw), mode="bilinear", align_corners=False),
                ).item()
            )
        else:
            ssim_val = 1.0 - float(compute_ssim_loss(x_c, x_o).item())
        linf_255 = float(delta.abs().max().item() * 255.0)
        rmse_255 = float(math.sqrt(mse) * 255.0)
        delta_y = 0.299 * delta[:, 0:1] + 0.587 * delta[:, 1:2] + 0.114 * delta[:, 2:3]
        chroma_rms_255 = float((delta - delta_y).pow(2).mean().sqrt().item() * 255.0)

        if Path(".snowflake_hf_cache").exists():
            os.environ.setdefault("HF_HOME", str(Path(".snowflake_hf_cache").resolve()))
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error()
        except Exception:
            pass

        model_list = eval_models or [
            "google/siglip-base-patch16-224",
            "openai/clip-vit-base-patch16",
            "facebook/dinov2-base",
            "timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k",
        ]
        model_reports = []

        for mname in model_list:
            sur = VisionTransformerSurrogate(mname, tap_layers=(-3, -2, -1)).to(self.device).eval()
            c_out = sur(x_c)
            o_out = sur(x_o)

            g_cos = float((c_out.global_embedding * o_out.global_embedding).sum(dim=-1).mean().item())
            p_cos_layers = []
            s_cos_layers = []
            c70_layers = []
            c50_layers = []
            patch_reid_evasion_layers = []
            salient_displacement_layers = []

            for cp, op in zip(c_out.patch_tokens, o_out.patch_tokens):
                per_p = (cp * op).sum(dim=-1)  # [1, N]
                p_cos_layers.append(float(per_p.mean().item()))
                c70_layers.append(float((per_p < 0.70).float().mean().item() * 100.0))
                c50_layers.append(float((per_p < 0.50).float().mean().item() * 100.0))

                sal_c = (cp - cp.mean(dim=1, keepdim=True)).norm(dim=-1)
                sal_o = (op - op.mean(dim=1, keepdim=True)).norm(dim=-1)
                k_top = max(1, cp.shape[1] // 4)
                top_c_idx = torch.topk(sal_c, k=k_top, dim=-1).indices
                top_o_idx = torch.topk(sal_o, k=k_top, dim=-1).indices
                s_cos_layers.append(float(torch.gather(per_p, 1, top_c_idx).mean().item()))

                # 1. Patch-to-Patch Re-ID Evasion (% of spatial patches that no longer self-match in the 196-patch grid)
                sim_grid = torch.matmul(op[0], cp[0].T)  # [N, N]
                matched_idx = torch.argmax(sim_grid, dim=-1)
                true_idx = torch.arange(cp.shape[1], device=cp.device)
                evaded_patches = (matched_idx != true_idx) | (per_p[0] < 0.50)
                patch_reid_evasion_layers.append(float(evaded_patches.float().mean().item() * 100.0))

                # 2. Salient Foreground Attention Displacement (% of top-25% foreground patches displaced)
                c_set = set(top_c_idx[0].cpu().tolist())
                o_set = set(top_o_idx[0].cpu().tolist())
                displaced_pct = (1.0 - len(c_set.intersection(o_set)) / max(1, len(c_set))) * 100.0
                salient_displacement_layers.append(float(displaced_pct))

            mean_s_cos = float(np.mean(s_cos_layers))
            mean_reid_ev = float(np.mean(patch_reid_evasion_layers))
            mean_disp = float(np.mean(salient_displacement_layers))

            model_reports.append(
                {
                    "model": mname.split("/")[-1],
                    "patch_cos": round(float(np.mean(p_cos_layers)), 4),
                    "salient_patch_cos": round(mean_s_cos, 4),
                    "global_cos": round(g_cos, 4),
                    "concealed_patches_70_pct": round(float(np.mean(c70_layers)), 1),
                    "concealed_patches_50_pct": round(float(np.mean(c50_layers)), 1),
                    "patch_reid_evasion_pct": round(mean_reid_ev, 1),
                    "salient_displacement_pct": round(mean_disp, 1),
                    "identification_evaded": bool(mean_reid_ev >= 50.0 or mean_s_cos < 0.50 or g_cos < 0.45),
                }
            )

        print(
            f"\n+=======================================================================================+\n"
            f"| CONCEALED IMAGE OBFUSCATION & IDENTIFICATION ANALYTICS REPORT\n"
            f"+=======================================================================================+\n"
            f"| Resolution     : {clean_pil.size[0]}x{clean_pil.size[1]}\n"
            f"| Visual Stealth : PSNR = {psnr_db:.2f} dB | SSIM = {ssim_val:.4f} | L_inf = {linf_255:.2f}/255 | RMSE = {rmse_255:.2f}/255\n"
            f"| Color Purity   : Opponent Chroma Shift = {chroma_rms_255:.2f}/255 (Low = no purple/green tint)\n"
            f"+---------------------------------------------------------------------------------------+\n"
            f"| Vision Transformer Feature & Identification Breakdown:",
            flush=True,
        )
        for rep in model_reports:
            status_str = "EVADED / SCRAMBLED" if rep["identification_evaded"] else "PARTIALLY DISRUPTED"
            print(
                f"|  * {rep['model']}\n"
                f"|      Patch Cosine Sim        : {rep['patch_cos']:.4f}  (Salient Foreground Faces/Objects: {rep['salient_patch_cos']:.4f})\n"
                f"|      Global [CLS] Cosine Sim : {rep['global_cos']:.4f}\n"
                f"|      Concealed Spatial Grid  : {rep['concealed_patches_70_pct']:5.1f}% (<0.70 sim) | {rep['concealed_patches_50_pct']:5.1f}% (<0.50 sim)\n"
                f"|      Feature Re-ID Evasion   : {rep['patch_reid_evasion_pct']:5.1f}% of patch features misidentified | {rep['salient_displacement_pct']:5.1f}% salient focus displaced\n"
                f"|      Identification Status   : [{status_str}]",
                flush=True,
            )
        print("+=======================================================================================+\n", flush=True)

        return {
            "psnr_db": round(psnr_db, 2),
            "ssim": round(ssim_val, 4),
            "linf_255": round(linf_255, 2),
            "chroma_rms_255": round(chroma_rms_255, 2),
            "models": model_reports,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time Image / Video Obfuscation Pipeline")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt checkpoint or .onnx model")
    parser.add_argument("--input", type=str, default=None, help="Input image file, directory, or .mp4 video")
    parser.add_argument("--output", type=str, default=None, help="Output image file, directory, or .mp4 video")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["canonical_residual", "native", "hybrid"],
        default=None,
        help="Override synthesis mode",
    )
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon_255 perturbation bound")
    parser.add_argument(
        "--refine-steps",
        type=int,
        default=0,
        help="Optional hybrid test-time ViT feature refinement steps on top of generator pass (e.g. 15)",
    )
    parser.add_argument(
        "--no-analytics",
        action="store_true",
        help="Skip printing the ViT feature identification analytics report after single-image obfuscation",
    )
    parser.add_argument(
        "--save-comparison",
        action="store_true",
        help="Save a side-by-side [Original | Obfuscated | 10x Perturbation Map] image next to --output",
    )
    parser.add_argument("--device", type=str, default=None, help="Compute device (cuda or cpu)")
    parser.add_argument("--benchmark", action="store_true", help="Run real-time latency & FPS benchmark")
    parser.add_argument(
        "--bench-res",
        type=int,
        nargs=2,
        default=[1080, 1920],
        metavar=("HEIGHT", "WIDTH"),
        help="Resolution (H W) for --benchmark (default: 1080 1920)",
    )
    args = parser.parse_args()

    obfuscator = RealtimeObfuscator(
        model_path=args.model,
        device=args.device,
        mode=args.mode,  # type: ignore[arg-type]
        epsilon_255=args.epsilon,
    )

    if args.input and args.output:
        in_path = Path(args.input)
        out_path = Path(args.output)
        if in_path.is_dir():
            count = obfuscator.obfuscate_directory(in_path, out_path)
            print(f"Obfuscated {count} images -> {out_path}")
        elif in_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            stats = obfuscator.obfuscate_video(in_path, out_path)
            print(f"Obfuscated video -> {out_path}: {json.dumps(stats)}")
        else:
            clean_pil = Image.open(in_path).convert("RGB")
            rgb = np.array(clean_pil, dtype=np.uint8, copy=True)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
            obf_tensor = obfuscator.obfuscate_tensor(tensor)

            if args.refine_steps > 0:
                obf_tensor = obfuscator.refine_tensor(
                    tensor, obf_tensor, steps=args.refine_steps, epsilon_255=args.epsilon
                )

            obf_np = (obf_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
            obf_pil = Image.fromarray(obf_np)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            obf_pil.save(out_path, quality=95)
            print(f"Obfuscated image -> {out_path}")

            if args.save_comparison:
                c_vis, o_vis = clean_pil, obf_pil
                if max(c_vis.size) > 1400:
                    vis_scale = 1400.0 / float(max(c_vis.size))
                    vw, vh = int(round(c_vis.width * vis_scale)), int(round(c_vis.height * vis_scale))
                    c_vis = c_vis.resize((vw, vh), Image.Resampling.BICUBIC)
                    o_vis = o_vis.resize((vw, vh), Image.Resampling.BICUBIC)
                clean_f = np.asarray(c_vis, dtype=np.float32) / 255.0
                obf_f = np.asarray(o_vis, dtype=np.float32) / 255.0
                diff_10x = np.clip(np.abs(obf_f - clean_f) * 10.0, 0.0, 1.0)
                comp = np.concatenate([clean_f, obf_f, diff_10x], axis=1)
                comp_path = out_path.with_name(f"{out_path.stem}_comparison.png")
                Image.fromarray((comp * 255.0).astype(np.uint8)).save(comp_path)
                print(f"Saved visual comparison -> {comp_path}")

            if not args.no_analytics:
                obfuscator.analyze_image_pair(clean_pil, obf_pil)

    if args.benchmark:
        stats = obfuscator.benchmark(resolution=(args.bench_res[0], args.bench_res[1]))
        print(f"Benchmark Results: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
