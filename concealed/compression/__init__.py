"""
Compression module for Concealed AI.
Provides adaptive loss-controlled compression, chroma subsampling management,
and target file size rate-control.
"""

from .compressor import AdaptiveCompressor, CompressionResult

__all__ = ["AdaptiveCompressor", "CompressionResult"]
