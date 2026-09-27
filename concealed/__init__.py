"""
Concealed AI: Image Compression, Client-Side Pre-Formatting, and Counter-Prevention Engine.
"""

from .pipeline import ConcealedPipeline, CompressionPipeline
from .preformatting.formatter import ClientSideFormatter, PlatformProfile, PLATFORM_PROFILES
from .compression.compressor import AdaptiveCompressor, CompressionResult
from .counter_prevention.verifier import CounterPreventionVerifier, IngestionSimulationResult

__version__ = "2.1.0"

__all__ = [
    "ConcealedPipeline",
    "CompressionPipeline",
    "ClientSideFormatter",
    "PlatformProfile",
    "PLATFORM_PROFILES",
    "AdaptiveCompressor",
    "CompressionResult",
    "CounterPreventionVerifier",
    "IngestionSimulationResult",
]
