"""
Unit tests for ConcealedPipeline.
Verifies end-to-end integration of Pre-Formatting, Compression, and Counter-Prevention.
"""

import unittest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from concealed.pipeline import ConcealedPipeline


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 1920x1080 gradient image
        x = np.linspace(0, 1, 1920)
        y = np.linspace(0, 1, 1080)
        xx, yy = np.meshgrid(x, y)
        arr = np.stack([xx * 255, yy * 255, (1.0 - xx) * 255], axis=-1).astype(np.uint8)
        self.img = Image.fromarray(arr)

    def test_end_to_end_instagram_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_file = Path(tmpdir) / "test_input.jpg"
            out_file = Path(tmpdir) / "test_output.jpg"
            self.img.save(in_file, quality=95)

            pipeline = ConcealedPipeline(platform="instagram_feed", default_quality=90)
            saved_path, metrics = pipeline.process_image(
                input_path=in_file,
                output_path=out_file,
                verify_counter_prevention=True,
            )

            self.assertTrue(saved_path.is_file())
            self.assertEqual(metrics["initial_resolution"], (1920, 1080))
            self.assertEqual(metrics["formatted_resolution"][0], 1080)
            self.assertGreater(metrics["compressed_file_size_kb"], 0)
            self.assertIn("counter_prevention", metrics)
            self.assertIn("survival_psnr_db", metrics["counter_prevention"])

            with Image.open(saved_path) as reloaded:
                self.assertEqual(reloaded.width, 1080)
                self.assertEqual(reloaded.mode, "RGB")
                self.assertIn("icc_profile", reloaded.info)

    def test_pipeline_with_target_size_budget(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_file = Path(tmpdir) / "test_input.jpg"
            out_file = Path(tmpdir) / "test_output.jpg"
            self.img.save(in_file, quality=95)

            target_kb = 35.0
            pipeline = ConcealedPipeline(platform="universal")
            saved_path, metrics = pipeline.process_image(
                input_path=in_file,
                output_path=out_file,
                target_size_kb=target_kb,
            )

            self.assertLessEqual(metrics["compressed_file_size_kb"], target_kb)


if __name__ == "__main__":
    unittest.main()
