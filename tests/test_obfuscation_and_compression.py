"""
Integration test suite verifying Amortized Generator Obfuscation,
Surrogate Feature Representation Shift, Client-Side Pre-Formatting,
and Adaptive Compression / Counter-Prevention Pipelines.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import pytest

from concealed.models.generator import AmortizedObfuscationGenerator
from concealed.models.surrogates import VisionTransformerSurrogate
from concealed.pipeline.realtime import RealtimeObfuscator
from concealed.pipeline.compression_pipeline import ConcealedPipeline
from concealed.preformatting.formatter import ClientSideFormatter
from concealed.compression.compressor import AdaptiveCompressor
from concealed.counter_prevention.verifier import CounterPreventionVerifier


@pytest.fixture
def sample_image() -> Image.Image:
    """Generate a photographic gradient image representing natural scene dynamics."""
    x = np.linspace(0, 1, 1280)
    y = np.linspace(0, 1, 720)
    xx, yy = np.meshgrid(x, y)
    arr = np.stack([xx * 255, yy * 255, (1.0 - xx) * 255], axis=-1).astype(np.uint8)
    return Image.fromarray(arr)


def test_generator_pipeline_end_to_end(sample_image: Image.Image) -> None:
    """Test full cycle: Generator -> Pre-Formatting -> Adaptive Compression -> Counter Prevention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "source.jpg"
        output_path = Path(tmpdir) / "concealed_output.jpg"
        sample_image.save(input_path, quality=95)

        # 1. Synthesize obfuscation perturbations
        gen = AmortizedObfuscationGenerator(variant="tiny", epsilon_255=8.0)
        obfuscator = RealtimeObfuscator(model_path=gen, device="cpu", epsilon_255=8.0)
        obfuscated_pil = obfuscator.obfuscate_pil(sample_image)

        assert obfuscated_pil.size == sample_image.size
        assert obfuscated_pil.mode == "RGB"

        # Check perturbation bound
        orig_arr = np.asarray(sample_image, dtype=np.int16)
        obf_arr = np.asarray(obfuscated_pil, dtype=np.int16)
        max_diff = np.max(np.abs(orig_arr - obf_arr))
        assert max_diff <= 9  # 8.0/255 with integer rounding tolerances

        # 2. Feed through ConcealedPipeline
        obfuscated_temp = Path(tmpdir) / "obf_temp.png"
        obfuscated_pil.save(obfuscated_temp)

        pipeline = ConcealedPipeline(
            platform="instagram_feed",
            default_quality=92,
            chroma_subsampling="444",
        )
        saved_path, metrics = pipeline.process_image(
            input_path=obfuscated_temp,
            output_path=output_path,
            verify_counter_prevention=True,
        )

        assert saved_path.is_file()
        assert metrics["formatted_resolution"][0] == 1080  # Conforms to Instagram width
        assert metrics["compressed_file_size_kb"] > 0
        assert metrics["counter_prevention"]["survival_psnr_db"] > 35.0
        assert metrics["counter_prevention"]["risk_level"] in ("LOW", "NEGLIGIBLE")

        # 3. Verify ICC and EXIF stripping
        with Image.open(saved_path) as out_img:
            assert out_img.width == 1080
            assert "icc_profile" in out_img.info
            # EXIF should be stripped
            assert "exif" not in out_img.info


def test_obfuscation_with_rate_control_budget(sample_image: Image.Image) -> None:
    """Test that obfuscated images adhere to strict rate-control size budgets."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input.jpg"
        output_path = Path(tmpdir) / "budgeted_output.jpg"
        sample_image.save(input_path, quality=95)

        gen = AmortizedObfuscationGenerator(variant="tiny", epsilon_255=8.0)
        obfuscator = RealtimeObfuscator(model_path=gen, device="cpu", epsilon_255=8.0)
        obf_pil = obfuscator.obfuscate_pil(sample_image)

        temp_obf = Path(tmpdir) / "obf.png"
        obf_pil.save(temp_obf)

        target_kb = 60.0
        pipeline = ConcealedPipeline(platform="universal")
        saved_path, metrics = pipeline.process_image(
            input_path=temp_obf,
            output_path=output_path,
            target_size_kb=target_kb,
        )

        assert saved_path.is_file()
        assert metrics["compressed_file_size_kb"] <= target_kb + 1.0


def test_surrogate_representation_shift_post_compression(sample_image: Image.Image) -> None:
    """Verify that ViT surrogate feature representations shift and survive compression."""
    surrogate = VisionTransformerSurrogate(
        model_name="mock/tiny-vit-16",
        pretrained=False,
    )

    # Convert PIL to Tensor [1, 3, H, W] in [0, 1]
    def pil_to_tensor(p_img: Image.Image) -> torch.Tensor:
        arr = np.asarray(p_img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    clean_tensor = pil_to_tensor(sample_image)
    with torch.no_grad():
        clean_out = surrogate(clean_tensor)
        clean_emb = clean_out.global_embedding

    # Obfuscate and Compress
    gen = AmortizedObfuscationGenerator(variant="tiny", epsilon_255=8.0)
    obfuscator = RealtimeObfuscator(model_path=gen, device="cpu", epsilon_255=8.0)
    obf_pil = obfuscator.obfuscate_pil(sample_image)

    formatter = ClientSideFormatter(platform="instagram_feed")
    formatted_img, _ = formatter.format_image(obf_pil)

    compressor = AdaptiveCompressor(default_quality=90, chroma_subsampling="444")
    comp_bytes, q_used = compressor.compress_to_bytes(formatted_img, quality=90, chroma_subsampling="444")

    # Reload compressed bytes
    import io
    reloaded_pil = Image.open(io.BytesIO(comp_bytes))
    reloaded_tensor = pil_to_tensor(reloaded_pil)

    with torch.no_grad():
        comp_out = surrogate(reloaded_tensor)
        comp_emb = comp_out.global_embedding

    # Feature distance / cosine similarity between clean and compressed obfuscated
    cos_sim = torch.cosine_similarity(clean_emb, comp_emb).item()
    # Ensure representation is shifted (embeddings are not identical)
    assert cos_sim < 1.0
    assert not torch.allclose(clean_emb, comp_emb, atol=1e-6)
    assert comp_emb.shape == clean_emb.shape
