# Concealed

**Amortized Adversarial Generator Network for Real-Time Image Obfuscation Against Vision Transformers, with Client-Side Pre-Formatting & Platform Counter-Prevention Compression.**

`Concealed` trains a compact, single-pass neural network ($G_\theta$) that synthesizes visually imperceptible, $L_\infty$-bounded perturbations ($\delta = G_\theta(x)$, $\|\delta\|_\infty \le \epsilon$) designed to disrupt open-source and frontier Vision Transformer (ViT) representations in real-time pipelines.

Additionally, this branch includes the dedicated **Client-Side Pre-Formatting, Adaptive Compression, and Counter-Prevention Engine** to ensure protected photos survive downstream social media ingestion (Instagram, WhatsApp, Facebook) without server-side re-compression or color degradation.

---

## Key Capabilities

1. **Single-Pass Real-Time Inference**:
   - Unlike iterative per-image PGD methods (Glaze, Nightshade, PhotoGuard) that take seconds to minutes per image, `Concealed` amortizes perturbation optimization over your image dataset during training.
   - At inference time, the ViT surrogates are discarded—leaving a ~1.8M parameter generator that executes in **<10ms on GPU** and **<40ms on CPU** and exports cleanly to **ONNX**, **TensorRT**, and **TorchScript**.
2. **Hybrid Global + High-Res Tile Synthesis (`hybrid` mode)**:
   - Evaluates a full-image global context branch alongside non-overlapping $512\times 512$ high-resolution tiles, blending boundary transitions using dynamic feathering masks to prevent block edge seams.
3. **Four-Backbone ViT Ensemble Defense**:
   - Multi-target representation collapse across distinct visual architectures:
     - `google/vit-base-patch16-224` (Standard classification ViT)
     - `facebook/dino-vitb16` (Self-supervised patch representations)
     - `openai/clip-vit-base-patch32` (Multimodal vision-language space)
     - `facebook/dinov2-base` (Dense high-resolution geometric features)
4. **Client-Side Pre-Formatting (`concealed/preformatting/`)**:
   - Scales to platform standards (e.g. 1080px width) using high-precision Lanczos interpolation (`Image.Resampling.LANCZOS`).
   - Converts wide-gamut photos (Display P3, Adobe RGB, CMYK) to standard **sRGB (IEC61966-2.1)** with embedded ICC profile to prevent platform color-mangling.
   - Strips EXIF metadata (GPS, camera serials, timestamps) for user privacy.
5. **Adaptive Loss-Controlled Compression (`concealed/compression/`)**:
   - Multi-format support (JPEG, WebP, PNG).
   - Chroma subsampling management (4:4:4 pristine color preservation vs 4:2:0 bandwidth saving).
   - Rate-control binary search budgeting to hit strict file size caps (e.g. `< 1.5MB`).
6. **Platform Counter-Prevention Verification (`concealed/counter_prevention/`)**:
   - Simulates downstream platform ingestion (Instagram, WhatsApp, Facebook).
   - Computes survival PSNR, SSIM, and MAE to verify images withstand downstream compression without degradation.

---

## Quickstart CLI

### 1. Compression, Pre-Formatting & Counter-Prevention
```bash
# Standard Instagram compression (1080px Lanczos, sRGB, 4:4:4 chroma, Q=90)
python image_processor.py my_photo.jpg -o my_photo_compressed.jpg --platform instagram_feed --quality 90

# Target file size budgeting (e.g. cap under 800 KB)
python image_processor.py my_photo.jpg -o my_photo_budget.jpg --target-size-kb 800

# WhatsApp profile (max 1600px dimension)
python image_processor.py my_photo.jpg --platform whatsapp --chroma 420
```

### 2. Real-Time Obfuscation Inference
```bash
# Obfuscate a single image or folder with Amortized Generator
python -m concealed.pipeline.realtime --input my_photo.jpg --output protected_photo.png --device cuda
```

### 3. Model Training & Evaluation
```bash
# Run local generator training
python -m concealed.train --data-dir ./data --epochs 20 --batch-size 4

# Run surrogate representation collapse evaluation
python -m concealed.evaluate --checkpoint ./checkpoints/generator_latest.pt --data-dir ./test_images
```

---

## Running Tests

Run the test suite:

```bash
python -m unittest discover -s tests -v
```