"""
Unit tests for ClientSideFormatter.
Validates Lanczos resizing, aspect ratio fitting, sRGB color conversion, and EXIF sanitization.
"""

import unittest
from PIL import Image, ImageCms
import numpy as np

from concealed.preformatting.formatter import ClientSideFormatter, PLATFORM_PROFILES


class TestPreformatting(unittest.TestCase):
    def setUp(self):
        # Create synthetic test image
        self.image_landscape = Image.new("RGB", (1920, 1080), color=(120, 180, 240))
        self.image_portrait = Image.new("RGB", (1000, 2000), color=(240, 180, 120))
        self.image_rgba = Image.new("RGBA", (800, 600), color=(50, 100, 150, 200))

    def test_instagram_feed_fit_width(self):
        formatter = ClientSideFormatter(platform="instagram_feed", fit_mode="fit_width")
        formatted, metrics = formatter.format_image(self.image_landscape)

        # Width must be exactly 1080px
        self.assertEqual(formatted.width, 1080)
        # Aspect ratio should be preserved
        expected_height = int(round(1080 / (1920 / 1080)))
        self.assertEqual(formatted.height, expected_height)
        self.assertEqual(formatted.mode, "RGB")
        self.assertTrue(metrics["exif_stripped"])
        self.assertIn("icc_profile", formatted.info)

    def test_instagram_feed_contain_padding(self):
        # An overly tall image (1000x2000 -> ar = 0.5 < min_ar 0.8)
        formatter = ClientSideFormatter(platform="instagram_feed", fit_mode="contain")
        formatted, metrics = formatter.format_image(self.image_portrait)

        ar = formatted.width / float(formatted.height)
        self.assertAlmostEqual(ar, 0.8, places=2)
        self.assertTrue(metrics["padding_applied"])

    def test_instagram_feed_crop(self):
        # An overly tall image with crop mode
        formatter = ClientSideFormatter(platform="instagram_feed", fit_mode="crop")
        formatted, metrics = formatter.format_image(self.image_portrait)

        ar = formatted.width / float(formatted.height)
        self.assertAlmostEqual(ar, 0.8, places=2)
        self.assertTrue(metrics["cropping_applied"])

    def test_rgba_to_srgb_conversion(self):
        formatter = ClientSideFormatter(platform="universal")
        formatted, metrics = formatter.format_image(self.image_rgba)

        self.assertEqual(formatted.mode, "RGB")
        self.assertTrue(metrics["color_converted_to_srgb"])

    def test_whatsapp_max_dimension(self):
        large_img = Image.new("RGB", (3000, 1500), color=(100, 100, 100))
        formatter = ClientSideFormatter(platform="whatsapp")
        formatted, metrics = formatter.format_image(large_img)

        self.assertLessEqual(max(formatted.size), 1600)
        self.assertEqual(formatted.width, 1600)
        self.assertEqual(formatted.height, 800)


if __name__ == "__main__":
    unittest.main()
