#!/usr/bin/env python3
"""
Image Processing Pipeline
Accepts an image file from the command line, converts it into a workable object,
passes it through a customizable algorithm placeholder, and saves/returns the result.
"""

import argparse
import sys
from pathlib import Path
from PIL import Image


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
        # Load image and ensure image data is read into memory
        img = Image.open(path)
        img.load()
        return img
    except Exception as exc:
        raise ValueError(f"Failed to open '{filepath}' as a valid image: {exc}") from exc


def apply_algorithm(image: Image.Image) -> Image.Image:
    """
    Placeholder for your custom modification algorithm.
    
    Args:
        image: A PIL Image object representing the loaded image.
        
    Returns:
        PIL.Image.Image: The modified image object.
    """
    # -------------------------------------------------------------
    # TODO: Insert your modification algorithm below.
    #
    # You have full access to the PIL Image object:
    #   - Inspect attributes: image.size, image.mode, image.format
    #   - Pixel manipulation: image.load(), image.getpixel(), image.putpixel()
    #   - Transformations:    image.filter(), image.resize(), etc.
    #   - NumPy conversion:   import numpy as np; arr = np.array(image)
    #                         ... modify arr ...
    #                         image = Image.fromarray(arr)
    # -------------------------------------------------------------
    
    print("[*] Running algorithm placeholder (pass-through)...")
    
    # Example placeholder: currently leaves the image intact.
    # Replace or add your code here:
    modified_image = image.copy()
    
    return modified_image


def save_image(image: Image.Image, output_path: str | Path) -> Path:
    """
    Saves the image object back to disk.
    
    Args:
        image: The PIL Image object to save.
        output_path: Destination path for the new image file.
        
    Returns:
        Path: The absolute path of the saved file.
    """
    out_path = Path(output_path)
    # Ensure destination directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    image.save(out_path)
    return out_path.resolve()


def process_image(input_path: str | Path, output_path: str | Path | None = None) -> Path:
    """
    High-level pipeline:
    1. Reads input file -> image object
    2. Runs modification algorithm on image object
    3. Saves image object -> new output image file
    
    Args:
        input_path: Path to the original image.
        output_path: Destination path. Defaults to 'output_<original_name>'.
        
    Returns:
        Path: The path to the newly generated image.
    """
    src = Path(input_path)
    
    if output_path is None:
        output_path = src.parent / f"modified_{src.stem}{src.suffix or '.png'}"
    
    print(f"[+] Loading image: {src}")
    image_obj = load_image(src)
    print(f"    Dimensions: {image_obj.width}x{image_obj.height} | Mode: {image_obj.mode} | Format: {image_obj.format}")
    
    print("[+] Applying algorithm...")
    modified_obj = apply_algorithm(image_obj)
    
    print(f"[+] Saving modified image to: {output_path}")
    saved_path = save_image(modified_obj, output_path)
    print(f"[OK] Successfully created: {saved_path}")
    
    return saved_path


def build_parser() -> argparse.ArgumentParser:
    """Constructs the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Load an image, modify it via an algorithm, and export the result."
    )
    parser.add_argument(
        "image_file",
        type=str,
        help="Path or filename of the input image to process."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path/filename for the output image. (Default: modified_<filename>)"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    
    try:
        process_image(args.image_file, args.output)
    except Exception as err:
        print(f"[Error] {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
