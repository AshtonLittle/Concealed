"""Concealed: Amortized Generator Network for Real-Time Vision Transformer Obfuscation,
Adaptive High-Fidelity Image Compression, Client-Side Pre-Formatting & Counter-Prevention.
"""

from concealed.pipeline import (
    ConcealedPipeline,
    CompressionPipeline,
    RealtimeObfuscator,
    export_to_onnx,
    export_to_torchscript,
)
from concealed.preformatting.formatter import ClientSideFormatter, PlatformProfile, PLATFORM_PROFILES
from concealed.compression.compressor import AdaptiveCompressor, CompressionResult
from concealed.counter_prevention.verifier import CounterPreventionVerifier, IngestionSimulationResult
from concealed.models.generator import AmortizedObfuscationGenerator, SynthesisMode

__version__ = "2.1.0"

__all__ = [
    "ConcealedPipeline",
    "CompressionPipeline",
    "RealtimeObfuscator",
    "AmortizedObfuscationGenerator",
    "SynthesisMode",
    "ClientSideFormatter",
    "PlatformProfile",
    "PLATFORM_PROFILES",
    "AdaptiveCompressor",
    "CompressionResult",
    "CounterPreventionVerifier",
    "IngestionSimulationResult",
    "export_to_onnx",
    "export_to_torchscript",
]
