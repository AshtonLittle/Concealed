# Concealed AI - Compression & Pre-Formatting Engine

Dedicated repository branch for high-fidelity image compression, client-side pre-formatting, and platform counter-prevention verification.

## Architecture

The engine is structured into three specialized modules:

1. **Client-Side Pre-Formatting (`concealed/preformatting/`)**
   - **Platform Profiles**: Native standardization for Instagram (`instagram_feed`, `instagram_story`), WhatsApp, Facebook, and Universal.
   - **Resolution & Lanczos Scaling**: Scales to platform thresholds (e.g. 1080px width) using high-precision Lanczos interpolation.
   - **sRGB Color Space Compliance**: Harmonizes wide-gamut photos (Display P3, Adobe RGB, CMYK) to standard sRGB (IEC61966-2.1) and embeds the standard ICC profile.
   - **EXIF Sanitization**: Strips GPS coordinates, camera/device serials, timestamps, and metadata to protect privacy and reduce file size.
   - **Aspect Ratio Formatting**: Supports `fit_width`, `contain` (padding), and `crop` strategies.

2. **Adaptive Compression (`concealed/compression/`)**
   - **Multi-Format Support**: High-performance encoding for JPEG, WebP, and PNG.
   - **Adaptive Quality ($Q \in [1, 100]$)**: Loss-controlled compression with progressive rendering and Huffman table optimization.
   - **Chroma Subsampling Control**:
     - `444` (`0`): Zero chroma subsampling for pristine color and edge fidelity.
     - `420` (`2`): Standard web bandwidth optimization.
     - `422` (`1`): Balanced chroma subsampling.
   - **Target-Size Budgeting (Rate Control)**: In-memory binary search optimization to hit strict file size caps (e.g. `< 1.5MB` or custom KB budget).

3. **Counter-Prevention & Ingestion Verification (`concealed/counter_prevention/`)**
   - **Ingestion Simulator**: Simulates server-side platform upload pipelines (MozJPEG/LibJPEG downsampling, forced 4:2:0 subsampling, re-compression).
   - **Quality Survival Metrics**: Computes PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index), and MAE.
   - **Re-Compression Risk Analysis**: Categorizes degradation risk (`LOW`, `MEDIUM`, `HIGH`) and generates actionable counter-prevention recommendations.

---

## Quickstart CLI

Run the image processor via `image_processor.py`:

```bash
# Standard Instagram compression (1080px Lanczos, sRGB, 4:4:4 chroma, Q=90)
python image_processor.py my_photo.jpg -o my_photo_compressed.jpg --platform instagram_feed --quality 90

# Target file size budgeting (e.g. cap at 800 KB)
python image_processor.py my_photo.jpg -o my_photo_budget.jpg --target-size-kb 800

# WhatsApp profile (max 1600px dimension)
python image_processor.py my_photo.jpg --platform whatsapp --chroma 420
```

---

## Running Tests

Execute the automated test suite:

```bash
python -m unittest discover -s tests -v
```