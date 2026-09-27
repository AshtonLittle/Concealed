"""Concealed Image Processor Module.

Alias module providing PascalCase `Imageprocessor` compatibility for `image_processor.py`.
"""

from __future__ import annotations

import image_processor as _ip
from image_processor import (
    ConformingRegion,
    apply_algorithm,
    build_parser,
    detect_feature_regions,
    draw_conforming_contours,
    load_image,
    main,
    process_image,
    process_image_bytes,
    save_image,
)

__all__ = [
    "ConformingRegion",
    "apply_algorithm",
    "build_parser",
    "detect_feature_regions",
    "draw_conforming_contours",
    "load_image",
    "main",
    "process_image",
    "process_image_bytes",
    "save_image",
]

if __name__ == "__main__":
    main()
