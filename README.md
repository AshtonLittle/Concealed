# Concealed

**Amortized Adversarial Generator Network for Real-Time Image Obfuscation Against Vision Transformers.**

`Concealed` trains a compact, single-pass neural network ($G_\theta$) that synthesizes visually imperceptible, $L_\infty$-bounded perturbations ($\delta = G_\theta(x)$, $\|\delta\|_\infty \le \epsilon$) designed to disrupt open-source and frontier Vision Transformer (ViT) representations in real-time pipelines.

---

## Key Capabilities

1. **Single-Pass Real-Time Inference**:
   - Unlike iterative per-image PGD methods (Glaze, Nightshade, PhotoGuard) that take seconds to minutes per image, `Concealed` amortizes perturbation optimization over your image dataset during training.
   - At inference time, the ViT surrogates are discarded—leaving a ~1.8M parameter generator that executes in **<10ms on GPU** and **<40ms on CPU** and exports cleanly to **ONNX**, **TensorRT**, and **TorchScript**.
2. **Hybrid Global + High-Res Tile Synthesis (`hybrid` mode)**:
   - **Global Canonical Residual Branch**: Synthesizes perturbations at a ViT-aligned canonical scale ($384\times 384$) and bilinearly upsamples only the residual $\delta$ to the native image resolution—preventing anti-aliased downsampling cancellation on $1080\text{p}/4\text{K}$ images while keeping original pixels 100% sharp.
   - **High-Res $2\times 2$ Tile Grid Branch**: Simultaneously targets high-resolution local tile crops used by modern frontier VLMs (GPT-4o high-detail mode, Qwen2-VL, InternVL, LLaVA-NeXT).
3. **100% ONNX-Native Frequency Control**:
   - Embeds an orthonormal $8\times 8$ **2D Discrete Cosine Transform (DCT)** filter bank implemented as a standard `nn.Conv2d` stem alongside $7\times 7$ depthwise-separable ConvNeXt blocks, bottleneck self-attention (`MatMul` + `Softmax`), and **Weber's Law perceptual luminance/texture masking**.
4. **Pluggable Open-Source Transformer Surrogate Ensemble**:
   - Out-of-the-box support for **OpenAI CLIP** (`openai/clip-vit-*`), **Google SigLIP** (`google/siglip-*`), **Meta DINOv2** (`facebook/dinov2-*`), generic HuggingFace `AutoModel` vision towers, and any **`timm` Vision Transformer** (`timm/vit_*`, `timm/eva02_*`, `timm/swin_*`).
   - Disrupts **both global `[CLS]`/pooled embeddings and multi-layer intermediate spatial patch tokens** (`tap_layers: [-4, -2, -1]`) wrapped in **Differentiable EOT** (`DiffJPEG` + DI-FGSM multi-scale resize + VLM tile cropping).

---

## Installation

```bash
pip install -e ".[dev,perceptual]"
```

---

## Quickstart

### 1. Train on Your Image Dataset (Cloud CUDA GPU Recommended)

Place your dataset of images (`.jpg`, `.png`, `.webp`) in any folder (no labels required) and run:

```bash
concealed-train \
  --config configs/default.yaml \
  --data-dir /path/to/your/images \
  --output-dir runs/exp1
```

You can also override surrogates, perturbation budget $\epsilon$, or synthesis mode directly from the CLI:

```bash
concealed-train \
  --config configs/default.yaml \
  --data-dir /path/to/your/images \
  --output-dir runs/exp_custom \
  --mode hybrid \
  --epsilon 8.0 \
  --surrogates openai/clip-vit-base-patch16 google/siglip-base-patch16-224 facebook/dinov2-base
```

### 2. Evaluate Black-Box Transferability on Unseen Open-Source Transformers

Test how well your trained generator disrupts **held-out, unseen** Vision Transformers (simulating transfer to closed-source frontier models):

```bash
concealed-eval \
  --checkpoint runs/exp1/best_generator.pt \
  --data-dir /path/to/val_images \
  --extra-surrogates openai/clip-vit-large-patch14 timm/vit_base_patch16_224 \
  --output-json runs/exp1/eval_report.json
```

### 3. Export to Dynamic-Axis ONNX / Quantized INT8 / TorchScript

```bash
concealed-export \
  --checkpoint runs/exp1/best_generator.pt \
  --output runs/exp1/generator.onnx \
  --quantize-int8 \
  --torchscript runs/exp1/generator.torchscript.pt
```

### 4. Run Real-Time Inference (Images, Folders, or Live Video Streams)

#### CLI Usage
```bash
# Obfuscate a folder of images
concealed-infer --model runs/exp1/generator.onnx --input /path/to/raw_imgs --output /path/to/obf_imgs

# Obfuscate an MP4 video stream and run latency benchmark
concealed-infer --model runs/exp1/generator.onnx --input input.mp4 --output obfuscated.mp4 --benchmark
```

#### Python Pipeline Usage
```python
from PIL import Image
from concealed.pipeline.realtime import RealtimeObfuscator

# Load either a PyTorch (.pt) checkpoint or an exported ONNX (.onnx) model
obfuscator = RealtimeObfuscator("runs/exp1/generator.onnx", device="cuda")

# 1. PIL Image (preserves exact native resolution)
img = Image.open("photo.jpg")
protected_img = obfuscator.obfuscate_pil(img)
protected_img.save("photo_protected.jpg")

# 2. OpenCV BGR Video Frame (for 30-60+ FPS streaming pipelines)
# protected_frame = obfuscator.obfuscate_bgr_frame(frame_bgr)
```