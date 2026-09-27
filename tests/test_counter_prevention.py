"""
Unit tests for CounterPreventionVerifier.
Validates platform ingestion simulation, metric calculation (PSNR/SSIM/MAE), and risk assessment.
"""

import unittest
from PIL import Image
import numpy as np

from concealed.counter_prevention.verifier import CounterPreventionVerifier


class TestCounterPrevention(unittest.TestCase):
    def setUp(self):
        # 1080x1080 continuous gradient image
        x = np.linspace(0, 1, 1080)
        y = np.linspace(0, 1, 1080)
        xx, yy = np.meshgrid(x, y)
        arr = np.stack([xx * 255, yy * 255, (1.0 - xx) * 255], axis=-1).astype(np.uint8)
        self.compliant_img = Image.fromarray(arr)
        # Add sRGB icc profile tag
        self.compliant_img.info["icc_profile"] = b"IEC61966-2.1"

        # Oversized 3000x2000 image
        x_large = np.linspace(0, 1, 3000)
        y_large = np.linspace(0, 1, 2000)
        xx_l, yy_l = np.meshgrid(x_large, y_large)
        arr_oversized = np.stack([xx_l * 255, yy_l * 255, (1.0 - yy_l) * 255], axis=-1).astype(np.uint8)
        self.oversized_img = Image.fromarray(arr_oversized)

        self.verifier = CounterPreventionVerifier()

    def test_psnr_and_ssim_identical(self):
        # Identical images should yield near-perfect metrics
        psnr = self.verifier.compute_psnr(self.compliant_img, self.compliant_img)
        ssim = self.verifier.compute_ssim(self.compliant_img, self.compliant_img)
        mae = self.verifier.compute_mae(self.compliant_img, self.compliant_img)

        self.assertEqual(psnr, 100.0)
        self.assertAlmostEqual(ssim, 1.0, places=3)
        self.assertEqual(mae, 0.0)

    def test_instagram_simulation(self):
        simulated = self.verifier.simulate_platform_ingestion(self.oversized_img, platform="instagram_feed")
        # Oversized image must be scaled down to 1080px width
        self.assertEqual(simulated.width, 1080)

    def test_risk_evaluation_compliant_image(self):
        # A 1080px image under 1MB should have LOW risk
        result = self.verifier.evaluate_survival(
            self.compliant_img,
            platform="instagram_feed",
            file_size_bytes=800 * 1024,
        )
        self.assertEqual(result.risk_level, "LOW")
        self.assertEqual(len(result.risk_factors), 0)
        self.assertGreater(result.survival_psnr, 30.0)

    def test_risk_evaluation_oversized_image(self):
        # A 3000px image with large file size should have HIGH risk
        result = self.verifier.evaluate_survival(
            self.oversized_img,
            platform="instagram_feed",
            file_size_bytes=5 * 1024 * 1024,
        )
        self.assertEqual(result.risk_level, "HIGH")
        self.assertGreater(len(result.risk_factors), 0)
        self.assertGreater(len(result.recommendations), 0)


if __name__ == "__main__":
    unittest.main()
