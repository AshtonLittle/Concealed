"""
Preformatting module for Concealed AI.
Handles client-side standardization: resolution, aspect ratio, sRGB color spaces, and metadata stripping.
"""

from .formatter import ClientSideFormatter, PlatformProfile

__all__ = ["ClientSideFormatter", "PlatformProfile"]
