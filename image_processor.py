#!/usr/bin/env python3
"""
Concealed AI - Image Processor CLI
Performs client-side pre-formatting, adaptive high-fidelity image compression,
platform counter-prevention verification, and optional neural representation obfuscation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from PIL import Image

from concealed.pipeline import ConcealedPipeline, RealtimeObfuscator
from concealed.preformatting.formatter import PLATFORM_PROFILES
from concealed.models.generator import AmortizedObfuscationGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concealed AI: Image Obfuscation, Compression, Client-Side Pre-Formatting & Counter-Prevention Engine."
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
        help="Optional destination path for the output image. (Default: <name>_processed.<ext>)"
    )
    parser.add_argument(
        "--obfuscate",
        action="store_true",
        help="Apply real-time amortized neural representation obfuscation against Vision Transformers before compression."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to generator checkpoint (.pt or .onnx). If omitted and --obfuscate is set, an initialized generator is used."
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8.0,
        help="Maximum L_inf perturbation bound in 0-255 scale (default: 8.0)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device for neural obfuscation ('cpu' or 'cuda')."
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
    print("      CONCEALED AI: OBFUSCATION & COMPRESSION ENGINE   ")
    print("=======================================================")
    print(f"[+] Input Photo: {input_path}")

    with Image.open(input_path) as im:
        orig_kb = round(input_path.stat().st_size / 1024.0, 2)
        print(f"    Source Resolution: {im.width}x{im.height} px | Mode: {im.mode} | Size: {orig_kb} KB")
        working_img = im.convert("RGB")

    # Step 1: Optional Neural Representation Obfuscation
    if args.obfuscate:
        print("\n[*] Applying Amortized Generator Neural Obfuscation...")
        if args.model_path and Path(args.model_path).is_file():
            obfuscator = RealtimeObfuscator(
                model_path=args.model_path,
                device=args.device,
                epsilon_255=args.epsilon,
            )
        else:
            gen = AmortizedObfuscationGenerator(epsilon=args.epsilon / 255.0)
            obfuscator = RealtimeObfuscator(
                model_path=gen,
                device=args.device or "cpu",
                epsilon_255=args.epsilon,
            )
        working_img = obfuscator.obfuscate_pil(working_img)
        print(f"[OK] Neural Obfuscation Synthesized (L_inf <= {args.epsilon}/255)")

    print(f"\n[+] Client-Side Pre-Formatting Target: Platform '{args.platform}'")
    print(f"    - Color standard: sRGB IEC61966-2.1 with embedded ICC profile")
    print(f"    - Resampling: High-order Lanczos interpolation")
    print(f"    - Privacy: Stripping all EXIF / GPS / device metadata")
    print(f"    - Fit mode: {args.fit_mode}")

    print(f"\n[+] Compression Settings:")
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

    # Save intermediate working image to temporary location if obfuscated, or process directly
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_input = Path(tmpdir) / f"input_{input_path.name}"
        working_img.save(temp_input, format="PNG")

        dest_output = args.output
        if dest_output is None:
            prefix = "concealed" if args.obfuscate else "compressed"
            suffix = f".{args.format.lower()}"
            if suffix == ".jpg":
                suffix = ".jpeg"
            dest_output = input_path.parent / f"{input_path.stem}_{prefix}{suffix}"

        print("\n[*] Executing client-side pre-formatting and compression...")
        saved_path, metrics = pipeline.process_image(
            input_path=temp_input,
            output_path=dest_output,
            target_size_kb=args.target_size_kb,
            quality=args.quality,
            chroma_subsampling=args.chroma,
            platform=args.platform,
            verify_counter_prevention=not args.no_verify,
        )

    print(f"\n[OK] Processing Completed Successfully!")
    print(f"    Output File: {saved_path}")
    print(f"    Output Resolution: {metrics['formatted_resolution'][0]}x{metrics['formatted_resolution'][1]} px")
    print(f"    Compressed Size: {metrics['compressed_file_size_kb']} KB")
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
