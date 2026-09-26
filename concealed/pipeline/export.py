"""ONNX and TorchScript Export Pipeline for Real-Time Deployment.

Exports a trained AmortizedObfuscationGenerator checkpoint to:
  1. Dynamic-Axis ONNX (`.onnx`) supporting arbitrary batch sizes and (H, W) resolutions
  2. Quantized INT8 ONNX (`.int8.onnx`) for CPU / edge / browser WebGPU deployment
  3. TorchScript (`.torchscript.pt`) for C++ / LibTorch pipelines
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from concealed.models.generator import AmortizedObfuscationGenerator, SynthesisMode
from concealed.train import load_generator_checkpoint


class _ONNXExportWrapper(torch.nn.Module):
    """Thin wrapper that exposes a single tensor input/output for clean ONNX graph tracing."""

    def __init__(self, generator: AmortizedObfuscationGenerator, mode: Optional[SynthesisMode] = None) -> None:
        super().__init__()
        self.generator = generator
        self.mode = mode

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        out = self.generator(input_image, return_delta=False, mode=self.mode)
        assert isinstance(out, torch.Tensor)
        return out


def export_to_onnx(
    generator: AmortizedObfuscationGenerator,
    output_path: str | Path,
    mode: Optional[SynthesisMode] = None,
    opset_version: int = 17,
    verify: bool = True,
    quantize_int8: bool = False,
) -> Dict[str, object]:
    """Export an AmortizedObfuscationGenerator to ONNX with dynamic batch, height, and width axes."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    generator_cpu = generator.to("cpu").eval()
    wrapper = _ONNXExportWrapper(generator_cpu, mode=mode).eval()

    # Use a non-canonical dummy resolution (e.g., 400x480) during tracing so dynamic H/W
    # interpolation nodes are preserved in the exported ONNX graph.
    dummy_input = torch.rand(1, 3, 400, 480, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(out_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["obfuscated_image"],
        dynamic_axes={
            "input_image": {0: "batch_size", 2: "height", 3: "width"},
            "obfuscated_image": {0: "batch_size", 2: "height", 3: "width"},
        },
        dynamo=False,
    )

    result: Dict[str, object] = {
        "onnx_path": str(out_path),
        "size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
    }

    if quantize_int8:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quant_path = out_path.with_suffix(".int8.onnx")
        quantize_dynamic(
            model_input=str(out_path),
            model_output=str(quant_path),
            weight_type=QuantType.QUInt8,
        )
        result["int8_onnx_path"] = str(quant_path)
        result["int8_size_mb"] = round(quant_path.stat().st_size / (1024 * 1024), 2)

    if verify:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        # Verify on a different resolution (e.g., 320x512) to confirm dynamic spatial axes
        test_x = torch.rand(1, 3, 320, 512, dtype=torch.float32)
        with torch.no_grad():
            torch_out = wrapper(test_x).numpy()
        ort_out = sess.run(["obfuscated_image"], {"input_image": test_x.numpy()})[0]
        max_diff = float(np.max(np.abs(torch_out - ort_out)))
        result["verified"] = max_diff < 1e-3
        result["max_abs_diff"] = max_diff

    return result


def export_to_torchscript(
    generator: AmortizedObfuscationGenerator,
    output_path: str | Path,
    mode: Optional[SynthesisMode] = None,
) -> Path:
    """Export generator to TorchScript via trace for LibTorch / C++ pipelines."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generator_cpu = generator.to("cpu").eval()
    wrapper = _ONNXExportWrapper(generator_cpu, mode=mode).eval()
    dummy_input = torch.rand(1, 3, 384, 384, dtype=torch.float32)
    traced = torch.jit.trace(wrapper, dummy_input)
    traced.save(str(out_path))
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Concealed Generator to ONNX / TorchScript")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output .onnx file path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["canonical_residual", "native", "hybrid"],
        default=None,
        help="Override synthesis mode baked into exported graph",
    )
    parser.add_argument("--epsilon", type=float, default=None, help="Override epsilon_255 bound")
    parser.add_argument("--quantize-int8", action="store_true", help="Also generate INT8 dynamically quantized ONNX")
    parser.add_argument("--torchscript", type=str, default=None, help="Optional path to also save TorchScript model")
    args = parser.parse_args()

    generator, _ = load_generator_checkpoint(
        args.checkpoint,
        device="cpu",
        override_mode=args.mode,
        override_epsilon_255=args.epsilon,
    )
    info = export_to_onnx(
        generator=generator,
        output_path=args.output,
        mode=args.mode,  # type: ignore[arg-type]
        verify=True,
        quantize_int8=args.quantize_int8,
    )
    print(f"Exported ONNX model: {info}")

    if args.torchscript:
        ts_path = export_to_torchscript(generator, args.torchscript, mode=args.mode)  # type: ignore[arg-type]
        print(f"Exported TorchScript model: {ts_path}")


if __name__ == "__main__":
    main()
