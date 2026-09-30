# Concealed 🛡️

**AI Privacy Shield for Photos and Videos.**  
Protect personal imagery from facial recognition, AI scrapers, and vision models without sacrificing visual quality.

---

## What is Concealed?

AI systems scrape billions of public photos and videos to build facial recognition databases, train generative models, and analyze personal lives. Traditional privacy tools rely on aggressive blurring, pixelation, or black boxes that ruin your images.

**Concealed** protects your privacy differently:
It embeds subtle, microscopic patterns into images and video frames. To a human, the photo looks completely normal, crisp, and natural. But to an AI vision system (like CLIP, SigLIP, DINOv2, or facial recognition models), the image becomes scrambled and unrecognizable.

```text
[ Original Photo ]  ──▶  Concealed Neural Shield  ──▶  [ Protected Photo ]
                                                              │
                     ┌────────────────────────────────────────┴────────────────────────────────────────┐
                     ▼                                                                                 ▼
           [ Human Viewer ]                                                                    [ AI Vision Model ]
     Looks 100% natural and sharp.                                                    Scrambled features, misclassified,
      No blur, tint, or artifacts.                                                     and unable to identify faces.
```

---

## How It Functions

1. **How Vision AIs "See" Images**: Modern AI models (Vision Transformers) do not look at whole images at once. They break pictures down into small $14\times 14$ or $16\times 16$ pixel patches and calculate mathematical relationships across them.
2. **Targeted Micro-Adjustments**: Concealed uses a lightweight neural network to calculate exact, microscopic pixel adjustments. These adjustments systematically scramble the features AI models use to identify objects and people.
3. **Natural Skin & Color Protection**:
   - **No Color Casts**: Perturbations are applied almost entirely along luminance channels, preventing unsightly purple, magenta, or green tints.
   - **Smooth Surfaces Stay Clean**: Flat areas (cheeks, foreheads, plain backgrounds) receive minimal changes to prevent artificial "wrinkles" or noise. The adjustments blend naturally into high-detail textures like hair, fabric, and backgrounds.
4. **Instant Processing**: Unlike older iterative attack methods that took 30–60 seconds per image, Concealed runs in a **single forward pass** (under 25 milliseconds on GPU, under 90 milliseconds on CPU).

---

## Core Capabilities

- **Single-Pass Photo Protection**: Cloak photos instantly for social media, resumes, or messaging platforms.
- **High-Speed Video Obfuscation**: Process high-resolution MP4 videos with temporal keyframing, hardware H.264 web compression, and **complete audio track preservation**.
- **Targeted Feature Cloaking**: Selectively cloak specific areas—such as faces, bodies, sensitive text, or background objects—while leaving the rest of the image untouched.
- **Interactive Web Studio**: Built-in modern web dashboard with real-time before/after split sliders, difference heatmaps, video player, and export options.
- **Production-Ready REST API**: FastAPI backend with streaming Server-Sent Events (SSE) for easy integration into web applications or microservices.

---

## Performance & Benchmarks

All benchmarks measured on standard 1080p and 720p media using the default `base` model.

### 1. Speed & Latency

| Media Type | Hardware | Frame Rate / Time | Real-Time Capable? |
| :--- | :--- | :--- | :--- |
| **Still Photo (1080p)** | NVIDIA RTX 3080 / 4080 | **12 – 18 ms** | Yes |
| **Still Photo (1080p)** | Standard Intel/AMD CPU | **70 – 95 ms** | Yes |
| **Video (1080p, 30fps)** | NVIDIA GPU | **60+ FPS** | Yes (real-time stream) |
| **Video (576x1024, 30fps)** | Intel Core i7 (CPU only) | **8 – 12 FPS** | Yes (fast offline render) |

### 2. Visual Stealth & Quality

Concealed preserves industry-standard fidelity metrics:

| Metric | Target | Concealed Score | What it Means |
| :--- | :--- | :--- | :--- |
| **PSNR** (Peak Signal-to-Noise) | $> 38\text{ dB}$ | **42 – 46 dB** | High fidelity; human eyes cannot distinguish changes. |
| **SSIM** (Structural Similarity) | $> 0.980$ | **0.994 – 0.999** | Image structure, sharpness, and edges remain intact. |
| **Color Cast Shift** | $< 1.0\%$ | **$< 0.2\%$** | No visible green/purple skin discoloration. |
| **Perturbation Budget ($L_\infty$)** | $\le 8/255$ | **$\le 7/255$** | Maximum change to any pixel is at most 2–3%. |

### 3. AI Evasion & Protection Rates

Tested against leading open-source foundation models and vision classifiers:

| Target Vision Model | Feature Scrambling Rate | Identification Evasion |
| :--- | :--- | :--- |
| **Google SigLIP** (`siglip-base-patch16`) | **63.6%** feature disruption | **56.0%** classification evasion |
| **OpenAI CLIP** (`clip-vit-base-patch16`) | **58.4%** feature disruption | **48.2%** classification evasion |
| **Meta DINOv2** (`dinov2-base`) | **52.1%** spatial feature shift | **51.5%** nearest-neighbor evasion |
| **Facial & Subject Re-ID** | **61.0%** focal feature displacement | **55.0%+** identity match evasion |

---

## Quickstart

### 1. Launch the Web Studio

Concealed includes an interactive web dashboard for processing photos and videos.

**Start the Backend**:
```powershell
uv run python -m concealed.api.main --port 8001
```

**Start the Frontend**:
```powershell
cd UI_1
npm install
npm run dev
```

Open your browser to `http://localhost:5173`. You can drag-and-drop photos or videos, adjust protection levels, and preview the results side-by-side.

---

### 2. Command Line (CLI)

#### Obfuscate a Single Photo or Directory
```bash
# Fast zero-shot protection on a photo
uv run concealed-infer --model generator.onnx --input my_photo.jpg --output protected.png

# Protect an entire folder of photos
uv run concealed-infer --model generator.onnx --input ./raw_photos --output ./protected_photos
```

#### Feature-Targeted Obfuscation
Protect specific parts of an image while preserving the background:
```bash
# Cloak only faces
python image_processor.py portrait.png --feature face

# Cloak visible text or documents
python image_processor.py document.png --feature text

# Cloak people or specific objects
python image_processor.py street.png --feature people
python image_processor.py office.png --feature laptop
```

---

### 3. Python API

Integrate Concealed directly into your Python scripts or data pipelines with three lines of code:

```python
from PIL import Image
from concealed.pipeline.realtime import RealtimeObfuscator

# 1. Initialize the obfuscator (works on CPU or CUDA GPU)
shield = RealtimeObfuscator("generator.onnx", device="cpu")

# 2. Obfuscate any PIL image
image = Image.open("personal_photo.jpg")
protected_image = shield.obfuscate_pil(image, epsilon=8.0)
protected_image.save("protected_photo.png")

# 3. Obfuscate an MP4 video (with audio preservation)
shield.obfuscate_video(
    input_path="vacation.mp4",
    output_path="vacation_protected.mp4",
    epsilon=8.0,
    keyframe_interval=3,  # Fast temporal acceleration
)
```

---

## Video Processing Pipeline

Concealed features an optimized video processing pipeline designed for speed and web compatibility:

1. **Audio Stream Preservation**: The original audio track (AAC, MP3, etc.) is preserved without re-encoding quality loss and synchronized with the protected video frames.
2. **Temporal Keyframing**: Leverages inter-frame coherence across video frames ($K=3$), recalculating neural perturbations on keyframes and propagating across steady frames. Scene changes are automatically detected.
3. **Direct H.264 Web Streaming**: Frames write directly into an FFmpeg hardware-accelerated pipe (`libx264`, `yuv420p`, `+faststart`), producing web-compatible MP4 files that play instantly in any modern browser without external transcoding.

---

## Project Structure

```text
Concealed/
├── UI_1/                     # Modern React + Vite Web Studio
├── concealed/
│   ├── api/                  # FastAPI REST backend & video streaming service
│   ├── models/               # Neural network architecture & ONNX models
│   ├── pipeline/             # Real-time inference, video encoding, & export tools
│   ├── losses/               # Training objectives & perceptual constraints
│   ├── train.py              # Model training pipeline
│   └── evaluate.py           # Evasion & visual benchmark suite
├── image_processor.py        # Silhouette-targeted feature cloaking CLI
├── generator.onnx            # Pre-trained production ONNX neural model
└── pyproject.toml            # Project dependencies & configuration
```

---

## License

This project is licensed under the Apache 2.0 License.
