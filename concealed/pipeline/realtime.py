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
        if in_path.is_dir():
            count = obfuscator.obfuscate_directory(in_path, args.output)
            print(f"Obfuscated {count} images -> {args.output}")
        elif in_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            stats = obfuscator.obfuscate_video(in_path, args.output)
            print(f"Obfuscated video -> {args.output}: {json.dumps(stats)}")
        else:
            obfuscator.obfuscate_file(in_path, args.output)
            print(f"Obfuscated image -> {args.output}")

    if args.benchmark:
        stats = obfuscator.benchmark(resolution=(args.bench_res[0], args.bench_res[1]))
        print(f"Benchmark Results: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
