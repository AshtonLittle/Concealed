#!/usr/bin/env python3
"""
Concealed AI - Image Processor CLI
Performs client-side pre-formatting, adaptive high-fidelity image compression,
and platform counter-prevention verification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PIL import Image

from concealed.pipeline import ConcealedPipeline
from concealed.preformatting.formatter import PLATFORM_PROFILES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concealed AI: Image Compression, Client-Side Pre-Formatting & Counter-Prevention Engine."
    )
    parser.add_argument(
        "image_file",
        type=str,
        help="Path to the input photo to process."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional destination path for the compressed image. (Default: <name>_compressed.<ext>)"
    )
    parser.add_argument(
        "--platform",
        type=str,
        choices=list(PLATFORM_PROFILES.keys()),
        default="instagram_feed",
        help="Target platform profile. Options: instagram_feed (1080px), instagram_story (1080x1920), whatsapp (1600px), facebook (2048px), universal. Default: instagram_feed"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="Export compression quality factor Q (1-100). Default: 90"
    )
    parser.add_argument(
        "--target-size-kb",
        type=float,
        default=None,
        help="Target file size budget in KB (e.g. 1000, 1500). If specified, uses binary search rate-control to achieve maximum quality within budget."
    )
    parser.add_argument(
        "--chroma",
        type=str,
        choices=["444", "422", "420"],
        default="444",
        help="Chroma subsampling mode. '444': pristine edge & color fidelity (subsampling=0). '420': standard web compression. Default: 444"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jpeg", "webp", "png"],
        default="jpeg",
        help="Output compression format. Default: jpeg"
    )
    parser.add_argument(
        "--fit-mode",
        type=str,
        choices=["fit_width", "contain", "crop"],
        default="fit_width",
        help="Aspect ratio adjustment strategy. Default: fit_width"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Disable platform counter-prevention simulation and survival metrics."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.image_file)
    if not input_path.is_file():
        print(f"[Error] Input image file not found: {input_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("\n=======================================================")
    print("      CONCEALED AI: COMPRESSION & PRE-FORMATTING       ")
    print("=======================================================")
    print(f"[+] Input Photo: {input_path}")

    with Image.open(input_path) as im:
        orig_kb = round(input_path.stat().st_size / 1024.0, 2)
        print(f"    Source Resolution: {im.width}x{im.height} px | Mode: {im.mode} | Size: {orig_kb} KB")

    print(f"[+] Client-Side Pre-Formatting Target: Platform '{args.platform}'")
    print(f"    - Color standard: sRGB IEC61966-2.1 with embedded ICC profile")
    print(f"    - Resampling: High-order Lanczos interpolation")
    print(f"    - Privacy: Stripping all EXIF / GPS / device metadata")
    print(f"    - Fit mode: {args.fit_mode}")

    print(f"[+] Compression Settings:")
    print(f"    - Format: {args.format.upper()}")
    print(f"    - Quality Factor: {args.quality if args.target_size_kb is None else f'Adaptive (target <= {args.target_size_kb} KB)'}")
    print(f"    - Chroma Subsampling: {args.chroma} ({'Pristine Color' if args.chroma == '444' else 'Bandwidth Saving'})")

    pipeline = ConcealedPipeline(
        platform=args.platform,
        fit_mode=args.fit_mode,
        format_type=args.format,
        default_quality=args.quality,
        chroma_subsampling=args.chroma,
        progressive=True,
        optimize=True,
    )

    print("\n[*] Executing client-side pre-formatting and compression...")
    saved_path, metrics = pipeline.process_image(
        input_path=input_path,
        output_path=args.output,
        target_size_kb=args.target_size_kb,
        quality=args.quality,
        chroma_subsampling=args.chroma,
        platform=args.platform,
        verify_counter_prevention=not args.no_verify,
    )

    print(f"\n[OK] Compression Completed Successfully!")
    print(f"    Output File: {saved_path}")
    print(f"    Output Resolution: {metrics['formatted_resolution'][0]}x{metrics['formatted_resolution'][1]} px")
    print(f"    Compressed Size: {metrics['compressed_file_size_kb']} KB ({metrics['compression_ratio_percent']}% reduction from raw buffer)")
    print(f"    Quality Factor Used: Q={metrics['quality_factor']}")
    print(f"    Bits Per Pixel (bpp): {metrics['bits_per_pixel']}")

    if "counter_prevention" in metrics:
        cp = metrics["counter_prevention"]
        print(f"\n[+] Platform Counter-Prevention Verification Report ({args.platform}):")
        print(f"    Ingestion Survival PSNR: {cp['survival_psnr_db']} dB")
        print(f"    Ingestion Survival SSIM: {cp['survival_ssim']}")
        print(f"    Re-compression Risk Level: [{cp['risk_level']}]")
        if cp["risk_factors"]:
            print("    Potential Risk Factors Detected:")
            for rf in cp["risk_factors"]:
                print(f"      - {rf}")
        else:
            print("    [Passed] Zero platform degradation risk. Output conforms directly to ingestion standards.")
        if cp["recommendations"]:
            print("    Recommendations:")
            for rec in cp["recommendations"]:
                print(f"      * {rec}")

    print("=======================================================\n")


if __name__ == "__main__":
    main()
