#!/usr/bin/env python3
"""
Feature-Targeted Image Processing Pipeline (Conforming Masks)
Accepts an image file and an optional target feature name from the command line,
detects the specified feature (e.g. face, arms, table, waterbottle, text, people,
or any arbitrary object), generates a pixel-precise mask conforming to the object's
contours, applies the modification algorithm ONLY to the conforming pixels,
and returns the modified image.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from detector import detect_feature_regions, ConformingRegion


def load_image(filepath: str | Path) -> Image.Image:
    """
    Loads an image file and returns a PIL Image object.
    
    Args:
        filepath: Path to the input image file.
        
    Returns:
        PIL.Image.Image: The loaded image object.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Input image file not found: {path.resolve()}")
    
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception as exc:
        raise ValueError(f"Failed to open '{filepath}' as a valid image: {exc}") from exc


def apply_algorithm(section: Image.Image, mask: Image.Image | None = None) -> Image.Image:
    """
    Placeholder for your custom modification algorithm.
    
    Args:
        section: A PIL Image object of the target region.
        mask: Optional PIL Image (mode 'L') of the conforming silhouette 
              (255 = inside object, 0 = outside object).
              Even if your algorithm modifies the entire rectangle, the pipeline
              automatically uses this mask to ensure only the object's conforming
              pixels are applied to the final image!
        
    Returns:
        PIL.Image.Image: The modified section image.
    """
    # ------------------------------------------------------------------
    # TODO: INSERT YOUR CUSTOM ALGORITHM HERE.
    #
    # You receive `section`, which is cropped to the object bounds.
    # The pipeline will automatically mask the output to the object's
    # exact silhouette when compositing.
    #
    # You can:
    #   1. Manipulate pixels directly:
    #      pixels = section.load()
    #
    #   2. Convert to a NumPy array for OpenCV / matrix operations:
    #      import numpy as np
    #      arr = np.array(section)
    #      # ... your array operations ...
    #      return Image.fromarray(arr)
    #
    #   3. Apply custom neural networks, filters, or transformations.
    # ------------------------------------------------------------------
    
    # Demonstration effect (Pixelation / Conceal):
    # This provides immediate visual feedback conforming to the object silhouette.
    factor = 25
    w_small = max(1, section.width // factor)
    h_small = max(1, section.height // factor)
    small = section.resize((w_small, h_small), resample=Image.Resampling.BILINEAR)
    demonstration_modified = small.resize(section.size, Image.Resampling.NEAREST)
    
    return demonstration_modified


def save_image(image: Image.Image, output_path: str | Path) -> Path:
    """Saves the image object back to disk."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path.resolve()


def draw_conforming_contours(canvas: Image.Image, region: ConformingRegion, color=(0, 255, 0)) -> None:
    """Draws the conforming contour outline onto the canvas image."""
    mask_np = region.mask
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Convert canvas to numpy array for drawing anti-aliased polyline
    canvas_np = np.array(canvas)
    for c in contours:
        # Shift contour to global image coordinates
        shifted = c + np.array([region.x1, region.y1])
        cv2.polylines(canvas_np, [shifted], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
        
    # Put text label near top-left of contour
    label_pos = (region.x1 + 4, max(15, region.y1 - 6))
    cv2.putText(canvas_np, region.label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    
    canvas.paste(Image.fromarray(canvas_np))


def process_image(
    input_path: str | Path,
    output_path: str | Path | None = None,
    features: list[str] | None = None,
    conf: float = 0.25,
    show_boxes: bool = False,
    show_contours: bool = False
) -> Path:
    """
    High-level processing pipeline with conforming masks:
    1. Loads input image -> PIL Image object.
    2. If features are specified, detects all conforming regions matching the feature(s).
    3. Crops each section, runs `apply_algorithm(section, mask=mask)`, and pastes back
       STRICTLY conforming to the object's silhouette mask.
    4. Saves the resulting image to disk.
    
    Args:
        input_path: Path to the original image file.
        output_path: Destination path for the new image.
        features: Optional list of target feature strings (e.g. ['face'], ['table'], etc.)
        conf: Confidence threshold for detection (0.0 to 1.0).
        show_boxes: If True, draws rectangular bounding box outlines.
        show_contours: If True, draws contours conforming to the object silhouette.
        
    Returns:
        Path: The absolute path of the generated image.
    """
    src = Path(input_path)
    if output_path is None:
        feat_tag = f"_{'_'.join(features)}" if features else ""
        output_path = src.parent / f"modified{feat_tag}_{src.stem}{src.suffix or '.png'}"
        
    print(f"[+] Loading image: {src}")
    image_obj = load_image(src)
    print(f"    Dimensions: {image_obj.width}x{image_obj.height} | Mode: {image_obj.mode} | Format: {image_obj.format}")
    
    if image_obj.mode not in ("RGB", "RGBA"):
        image_obj = image_obj.convert("RGB")
        
    canvas = image_obj.copy()
    
    if not features:
        # No feature specified: apply algorithm across the entire image
        print("[*] No specific feature requested. Applying algorithm to full image...")
        canvas = apply_algorithm(canvas)
    else:
        # Detect conforming regions for each requested feature
        all_regions: list[ConformingRegion] = []
        for feat in features:
            feat_name = feat.strip()
            print(f"[*] Detecting conforming feature '{feat_name}' (confidence >= {conf})...")
            regions = detect_feature_regions(src, feat_name, conf=conf)
            print(f"    Found {len(regions)} conforming region(s) for '{feat_name}'")
            all_regions.extend(regions)
                
        if not all_regions:
            print(f"[!] Warning: No matching features recognized for {features}.")
            print("    The image will be saved without section modifications.")
        else:
            print(f"[+] Applying algorithm conforming strictly to {len(all_regions)} object silhouette(s)...")
            for idx, region in enumerate(all_regions, 1):
                # Clamp coordinates to image dimensions
                x1 = max(0, min(image_obj.width - 1, region.x1))
                y1 = max(0, min(image_obj.height - 1, region.y1))
                x2 = max(x1 + 1, min(image_obj.width, region.x2))
                y2 = max(y1 + 1, min(image_obj.height, region.y2))
                
                # Crop section and get its conforming mask
                section = canvas.crop((x1, y1, x2, y2))
                mask_pil = region.pil_mask
                
                if mask_pil.size != section.size:
                    mask_pil = mask_pil.resize(section.size, Image.Resampling.NEAREST)
                    
                # Smooth mask edges slightly (1px) for seamless, natural edge blending
                smoothed_mask = mask_pil.filter(ImageFilter.GaussianBlur(radius=1.0))
                
                # Run the modification algorithm on the section
                try:
                    modified_section = apply_algorithm(section, mask=smoothed_mask)
                except TypeError:
                    # Backward compatibility if user's algorithm takes only 1 argument
                    modified_section = apply_algorithm(section)
                    
                if modified_section.size != section.size:
                    modified_section = modified_section.resize(section.size)
                    
                # Paste modified pixels ONLY where the conforming mask is active
                canvas.paste(modified_section, (x1, y1), mask=smoothed_mask)
                
                # Calculate pixel coverage for informative logging
                fg_count = np.count_nonzero(region.mask)
                total_count = max(1, region.mask.size)
                coverage_pct = (fg_count / total_count) * 100
                print(f"    - Object {idx} [{region.label}]: bounds ({x1}, {y1})->({x2}, {y2}), conforming mask coverage: {coverage_pct:.1f}%")
                
            # Optional visualization overlays
            if show_contours:
                for region in all_regions:
                    draw_conforming_contours(canvas, region, color=(0, 255, 128))
            elif show_boxes:
                draw = ImageDraw.Draw(canvas)
                for region in all_regions:
                    draw.rectangle([region.x1, region.y1, region.x2, region.y2], outline="red", width=2)
                    draw.text((region.x1 + 4, max(0, region.y1 - 15)), region.label, fill="red")
                    
    print(f"[+] Saving output image to: {output_path}")
    saved_path = save_image(canvas, output_path)
    print(f"[OK] Successfully created: {saved_path}")
    return saved_path


def process_image_bytes(
    image_bytes: bytes,
    features: list[str] | str | None = None,
    conf: float = 0.25,
    show_boxes: bool = False,
    show_contours: bool = False,
    output_format: str = "PNG",
) -> tuple[bytes, str, list[dict]]:
    """
    In-memory image processing pipeline:
    1. Writes input image bytes to a temporary file.
    2. Runs detection and applies algorithm conforming strictly to object silhouettes.
    3. Encodes modified image to bytes in the requested format.
    4. Returns (output_bytes, mime_type, detected_regions_info).
    """
    import io
    import tempfile
    import os

    if isinstance(features, str):
        features = [f.strip() for f in features.split(",") if f.strip()]

    # Write to a temp file for detector functions
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_in:
        tmp_in.write(image_bytes)
        tmp_in_path = Path(tmp_in.name)

    try:
        image_obj = load_image(tmp_in_path)
        if image_obj.mode not in ("RGB", "RGBA"):
            image_obj = image_obj.convert("RGB")

        canvas = image_obj.copy()
        regions_info: list[dict] = []

        if not features:
            canvas = apply_algorithm(canvas)
        else:
            all_regions: list[ConformingRegion] = []
            for feat in features:
                feat_name = feat.strip()
                regions = detect_feature_regions(tmp_in_path, feat_name, conf=conf)
                all_regions.extend(regions)

            for idx, region in enumerate(all_regions, 1):
                x1 = max(0, min(image_obj.width - 1, region.x1))
                y1 = max(0, min(image_obj.height - 1, region.y1))
                x2 = max(x1 + 1, min(image_obj.width, region.x2))
                y2 = max(y1 + 1, min(image_obj.height, region.y2))

                section = canvas.crop((x1, y1, x2, y2))
                mask_pil = region.pil_mask
                if mask_pil.size != section.size:
                    mask_pil = mask_pil.resize(section.size, Image.Resampling.NEAREST)
                smoothed_mask = mask_pil.filter(ImageFilter.GaussianBlur(radius=1.0))

                try:
                    modified_section = apply_algorithm(section, mask=smoothed_mask)
                except TypeError:
                    modified_section = apply_algorithm(section)

                if modified_section.size != section.size:
                    modified_section = modified_section.resize(section.size)

                canvas.paste(modified_section, (x1, y1), mask=smoothed_mask)

                fg_count = int(np.count_nonzero(region.mask))
                total_count = max(1, region.mask.size)
                coverage = (fg_count / total_count) * 100
                regions_info.append({
                    "id": idx,
                    "label": region.label,
                    "confidence": round(float(region.confidence), 4),
                    "bbox": [x1, y1, x2, y2],
                    "mask_coverage_pct": round(coverage, 2),
                })

            if show_contours:
                for region in all_regions:
                    draw_conforming_contours(canvas, region, color=(0, 255, 128))
            elif show_boxes:
                draw = ImageDraw.Draw(canvas)
                for region in all_regions:
                    draw.rectangle([region.x1, region.y1, region.x2, region.y2], outline="red", width=2)
                    draw.text((region.x1 + 4, max(0, region.y1 - 15)), region.label, fill="red")

        # Encode canvas to bytes
        fmt = output_format.upper()
        if fmt not in ["PNG", "JPEG", "JPG", "WEBP"]:
            fmt = "PNG"
        save_fmt = "JPEG" if fmt in ["JPEG", "JPG"] else fmt
        mime = f"image/{save_fmt.lower()}"

        buf = io.BytesIO()
        if save_fmt == "JPEG" and canvas.mode == "RGBA":
            canvas = canvas.convert("RGB")
        canvas.save(buf, format=save_fmt)
        out_bytes = buf.getvalue()

        return out_bytes, mime, regions_info
    finally:
        try:
            if tmp_in_path.exists():
                os.remove(tmp_in_path)
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply an image-modification algorithm conforming strictly to the silhouette "
            "of recognized features (face, arms, table, waterbottle, text, people, or any custom object)."
        )
    )
    parser.add_argument(
        "image_file",
        type=str,
        nargs="?",
        default=None,
        help="Path or filename of the input image."
    )
    parser.add_argument(
        "-f", "--feature",
        type=str,
        default=None,
        help=(
            "Feature to recognize and modify (e.g. 'face', 'arms', 'table', "
            "'waterbottle', 'text', 'people', 'chair', 'dog', etc.). "
            "Supports comma-separated lists for multiple features, e.g. 'face, arms'."
        )
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional destination path for the modified image."
    )
    parser.add_argument(
        "-c", "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold between 0.0 and 1.0 (default: 0.25)."
    )
    parser.add_argument(
        "--show-contours",
        action="store_true",
        help="Draw the conforming silhouette contour outlines on the output image."
    )
    parser.add_argument(
        "--show-boxes",
        action="store_true",
        help="Draw rectangular bounding box outlines on the output image."
    )
    parser.add_argument(
        "--list-features",
        action="store_true",
        help="Display examples of recognized features and exit."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    if args.list_features:
        print("\n=== Recognized Feature Categories ===")
        print("1. Body Parts (Conforming to Human Pose & Segmentation):")
        print("   - face, head, arms, hands, legs")
        print("2. Text Regions (Conforming to Word & Letter Strokes):")
        print("   - text, words, writing, ocr")
        print("3. Common Objects (Conforming Instance Segmentation - COCO 80 classes):")
        print("   - person/people, table/desk, bottle/waterbottle, chair, couch, cup,")
        print("     laptop, cell phone, tv, book, car, bus, bicycle, dog, cat, etc.")
        print("4. Any Arbitrary Object (Conforming via YOLO-World + GrabCut Silhouette):")
        print("   - Any custom object query entered (e.g. 'backpack', 'guitar', 'hat', etc.)\n")
        return

    if not args.image_file:
        parser.print_help()
        sys.exit(1)

    features = None
    if args.feature:
        features = [f.strip() for f in args.feature.split(",") if f.strip()]
        
    try:
        process_image(
            input_path=args.image_file,
            output_path=args.output,
            features=features,
            conf=args.conf,
            show_boxes=args.show_boxes,
            show_contours=args.show_contours
        )
    except Exception as err:
        print(f"[Error] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
