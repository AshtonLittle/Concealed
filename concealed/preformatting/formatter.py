"""
ClientSideFormatter: Pre-formats and standardizes photos on the client side
prior to platform upload and compression.

Solves:
1. Destructive platform re-sampling: Enforces exact platform width (e.g. 1080px for Instagram)
   using high-quality Lanczos interpolation on the client side.
2. Color shift / washing out: Converts wide-gamut (Display P3, Adobe RGB, CMYK) to standard
   sRGB (IEC61966-2.1) and embeds the standard sRGB ICC profile.
3. Aspect ratio truncation: Validates or adapts aspect ratio to avoid abrupt platform cropping.
4. Privacy leakage: Strips all sensitive EXIF/GPS/device metadata.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union
from PIL import Image, ImageCms, ImageOps


@dataclass
class PlatformProfile:
    name: str
    target_width: Optional[int] = None
    max_dimension: Optional[int] = None
    min_aspect_ratio: Optional[float] = None  # width / height (e.g. 4/5 = 0.8)
    max_aspect_ratio: Optional[float] = None  # width / height (e.g. 1.91/1 = 1.91)
    enforce_srgb: bool = True
    strip_exif: bool = True


PLATFORM_PROFILES: Dict[str, PlatformProfile] = {
    "instagram_feed": PlatformProfile(
        name="instagram_feed",
        target_width=1080,
        min_aspect_ratio=4.0 / 5.0,     # 0.80 (Portrait 1080x1350)
        max_aspect_ratio=1.91 / 1.0,    # 1.91 (Landscape 1080x566)
        enforce_srgb=True,
        strip_exif=True,
    ),
    "instagram_story": PlatformProfile(
        name="instagram_story",
        target_width=1080,
        min_aspect_ratio=9.0 / 16.0,    # 0.5625 (1080x1920)
        max_aspect_ratio=9.0 / 16.0,
        enforce_srgb=True,
        strip_exif=True,
    ),
    "whatsapp": PlatformProfile(
        name="whatsapp",
        max_dimension=1600,
        enforce_srgb=True,
        strip_exif=True,
    ),
    "facebook": PlatformProfile(
        name="facebook",
        max_dimension=2048,
        enforce_srgb=True,
        strip_exif=True,
    ),
    "universal": PlatformProfile(
        name="universal",
        target_width=1080,
        enforce_srgb=True,
        strip_exif=True,
    ),
}


class ClientSideFormatter:
    """
    Standardizes client-side images before compression and network transmission.
    Prevents platform ingestion degradation and counter-measures.
    """

    def __init__(
        self,
        platform: str = "instagram_feed",
        fit_mode: str = "fit_width",
        bg_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        """
        Args:
            platform: Platform name ('instagram_feed', 'instagram_story', 'whatsapp', 'facebook', 'universal').
            fit_mode: Aspect fitting strategy ('fit_width', 'contain', 'crop', 'preserve').
            bg_color: RGB background padding color when using 'contain' mode.
        """
        if platform not in PLATFORM_PROFILES:
            raise ValueError(f"Unknown platform '{platform}'. Supported: {list(PLATFORM_PROFILES.keys())}")
        self.profile = PLATFORM_PROFILES[platform]
        self.fit_mode = fit_mode
        self.bg_color = bg_color

        # Cache standard sRGB profile
        self._srgb_profile = ImageCms.createProfile("sRGB")

    def format_image(
        self,
        image: Image.Image,
        target_width_override: Optional[int] = None,
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Executes complete client-side pre-formatting:
        1. Normalizes EXIF orientation
        2. Converts color space to sRGB IEC61966-2.1 with proper ICC tag
        3. Scales dimensions using high-order Lanczos interpolation
        4. Adjusts aspect ratio if specified
        5. Sanitizes metadata

        Returns:
            Tuple of (formatted_pil_image, formatting_metrics_dict).
        """
        metrics: Dict[str, Any] = {
            "initial_size": image.size,
            "initial_mode": image.mode,
            "platform": self.profile.name,
            "exif_stripped": False,
            "color_converted_to_srgb": False,
        }

        # Step 1: Normalize EXIF orientation (e.g. portrait photos taken on smartphones)
        transposed = ImageOps.exif_transpose(image)
        if transposed is not None:
            working_img = transposed
        else:
            working_img = image.copy()

        # Step 2: Harmonize color space to standard sRGB
        working_img, converted = self._standardize_color_space(working_img)
        metrics["color_converted_to_srgb"] = converted

        # Step 3: Resize and format aspect ratio according to platform requirements
        target_width = target_width_override or self.profile.target_width
        working_img, resize_info = self._apply_dimensions(
            working_img,
            target_width=target_width,
            max_dimension=self.profile.max_dimension,
            min_ar=self.profile.min_aspect_ratio,
            max_ar=self.profile.max_aspect_ratio,
        )
        metrics.update(resize_info)

        # Step 4: Metadata sanitization (strip all EXIF tags, GPS, maker notes)
        if self.profile.strip_exif:
            # Recreate an unencumbered fresh Image object devoid of EXIF dicts
            clean_img = Image.new("RGB", working_img.size)
            clean_img.paste(working_img, (0, 0))
            # Attach sRGB ICC profile to ensure platforms render colors faithfully
            srgb_bytes = ImageCms.ImageCmsProfile(self._srgb_profile).tobytes()
            clean_img.info["icc_profile"] = srgb_bytes
            working_img = clean_img
            metrics["exif_stripped"] = True

        metrics["final_size"] = working_img.size
        metrics["final_aspect_ratio"] = round(working_img.width / float(working_img.height), 4)

        return working_img, metrics

    def _standardize_color_space(self, img: Image.Image) -> Tuple[Image.Image, bool]:
        """
        Converts non-RGB or wide-gamut images (Display P3, AdobeRGB, CMYK, RGBA)
        to pristine standard sRGB IEC61966-2.1.
        """
        converted = False
        icc_profile_data = img.info.get("icc_profile")

        # Handle alpha channels (RGBA) by compositing over background color
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            bg = Image.new("RGB", img.size, self.bg_color)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[3])
            img = bg
            converted = True

        # Handle non-RGB modes (CMYK, Grayscale, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")
            converted = True

        # Handle ICC color transformations
        if icc_profile_data:
            try:
                input_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile_data))
                # Transform to sRGB if input profile is valid and different
                img = ImageCms.profileToProfile(
                    img,
                    input_profile,
                    self._srgb_profile,
                    outputMode="RGB",
                    renderingIntent=ImageCms.Intent.PERCEPTUAL,
                )
                converted = True
            except Exception:
                # If ICC parsing fails, standard RGB conversion is already applied
                pass

        return img, converted

    def _apply_dimensions(
        self,
        img: Image.Image,
        target_width: Optional[int],
        max_dimension: Optional[int],
        min_ar: Optional[float],
        max_ar: Optional[float],
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Applies dimension scaling using high-precision Lanczos interpolation.
        """
        w, h = img.size
        orig_ar = w / float(h)
        info: Dict[str, Any] = {
            "resampled": False,
            "resample_filter": "LANCZOS",
            "padding_applied": False,
            "cropping_applied": False,
        }

        # 1. Scale by target width if defined (e.g. 1080px for Instagram)
        if target_width is not None and w != target_width:
            new_w = target_width
            new_h = int(round(target_width / orig_ar))
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            w, h = img.size
            info["resampled"] = True

        # 2. Scale by max_dimension if defined (e.g. WhatsApp 1600px, Facebook 2048px)
        elif max_dimension is not None and max(w, h) > max_dimension:
            if w >= h:
                new_w = max_dimension
                new_h = int(round(max_dimension / orig_ar))
            else:
                new_h = max_dimension
                new_w = int(round(max_dimension * orig_ar))
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            w, h = img.size
            info["resampled"] = True

        # 3. Check and adjust aspect ratio limits if specified
        if min_ar is not None and max_ar is not None:
            current_ar = w / float(h)

            if current_ar < min_ar or current_ar > max_ar:
                if self.fit_mode == "contain":
                    # Pad image to fit within bounds without losing pixels
                    target_ar = min_ar if current_ar < min_ar else max_ar
                    if current_ar < target_ar:
                        # Too tall: add side padding (pillarbox)
                        canvas_w = int(round(h * target_ar))
                        canvas_h = h
                    else:
                        # Too wide: add top/bottom padding (letterbox)
                        canvas_w = w
                        canvas_h = int(round(w / target_ar))

                    canvas = Image.new("RGB", (canvas_w, canvas_h), self.bg_color)
                    paste_x = (canvas_w - w) // 2
                    paste_y = (canvas_h - h) // 2
                    canvas.paste(img, (paste_x, paste_y))
                    img = canvas
                    info["padding_applied"] = True
                    info["resampled"] = True

                elif self.fit_mode == "crop":
                    # Center crop to closest valid aspect ratio
                    target_ar = min_ar if current_ar < min_ar else max_ar
                    if current_ar < target_ar:
                        # Crop top and bottom
                        target_h = int(round(w / target_ar))
                        top = (h - target_h) // 2
                        img = img.crop((0, top, w, top + target_h))
                    else:
                        # Crop left and right
                        target_w_crop = int(round(h * target_ar))
                        left = (w - target_w_crop) // 2
                        img = img.crop((left, 0, left + target_w_crop, h))
                    info["cropping_applied"] = True
                    info["resampled"] = True

        return img, info
