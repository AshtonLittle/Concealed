"""
Unit tests for AdaptiveCompressor.
Validates multi-format compression, quality factors, chroma subsampling, and binary search rate-control.
"""

import unittest
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from concealed.compression.compressor import AdaptiveCompressor


class TestCompression(unittest.TestCase):
    def setUp(self):
        # Create a smooth continuous gradient pattern (realistic natural image)
        x = np.linspace(0, 1, 512)
        y = np.linspace(0, 1, 512)
        xx, yy = np.meshgrid(x, y)
        r = (xx * 255).astype(np.uint8)
        g = (yy * 255).astype(np.uint8)
        b = ((1.0 - xx * 0.5 - yy * 0.5) * 255).astype(np.uint8)
        arr = np.stack([r, g, b], axis=-1)
        self.test_img = Image.fromarray(arr)

    def test_jpeg_quality_scaling(self):
        compressor = AdaptiveCompressor(format_type="jpeg")
        data_q95, q_95 = compressor.compress_to_bytes(self.test_img, quality=95)
        data_q50, q_50 = compressor.compress_to_bytes(self.test_img, quality=50)

        self.assertEqual(q_95, 95)
        self.assertEqual(q_50, 50)
        self.assertGreater(len(data_q95), len(data_q50))

    def test_chroma_subsampling_modes(self):
        compressor = AdaptiveCompressor(format_type="jpeg")
        data_444, _ = compressor.compress_to_bytes(self.test_img, quality=85, chroma_subsampling="444")
        data_420, _ = compressor.compress_to_bytes(self.test_img, quality=85, chroma_subsampling="420")

        # 4:4:4 maintains full chroma detail, yielding higher fidelity and slightly larger buffer
        self.assertGreater(len(data_444), len(data_420))

    def test_target_size_budget_rate_control(self):
        compressor = AdaptiveCompressor(format_type="jpeg")
        # Target budget of 20 KB
        target_kb = 20.0
        data, used_q = compressor.compress_to_bytes(self.test_img, target_size_kb=target_kb)

        actual_kb = len(data) / 1024.0
        self.assertLessEqual(actual_kb, target_kb)
        self.assertGreater(used_q, 5)

    def test_webp_compression(self):
        compressor = AdaptiveCompressor(format_type="webp")
        data_webp, used_q = compressor.compress_to_bytes(self.test_img, quality=80)

        self.assertGreater(len(data_webp), 0)
        self.assertEqual(used_q, 80)

    def test_save_to_disk_metrics(self):
        compressor = AdaptiveCompressor(format_type="jpeg", default_quality=90)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "output.jpg"
            result = compressor.compress_image(self.test_img, output_path=out_file)

            self.assertTrue(out_file.is_file())
            self.assertEqual(result.output_path, out_file)
            self.assertGreater(result.compression_ratio_percent, 0.0)
            self.assertEqual(result.quality, 90)


if __name__ == "__main__":
    unittest.main()
