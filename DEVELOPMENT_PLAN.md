# AI Watermark Recovery — Complete Prototype Development Plan

> Based on: *Forensic Reconstruction of Hidden AI Watermarks: Deep Learning Solutions for Degradation Recovery*

---

## Project Overview

A full-stack system implementing the 5-phase blueprint from the research paper:
- **Dual-domain watermarking** (latent + pixel)
- **Inverse Generative Restoration Module (IGRM)** for blind recovery
- **Attack Simulation Layer (ASL)** with differentiable JPEG
- **Content-dependent cryptographic binding** (MetaSeal / ViT + ECDSA)
- **BCH + Hamming error correction** for extreme attack survival

**Stack:** React + TypeScript · FastAPI · PyTorch 2.x · PostgreSQL · Redis · Celery

---

## Repository Structure

```
ai-watermark-recovery/
├── frontend/                   # React + TypeScript + Vite
│   └── src/
│       ├── components/         # ImageUploader, ResultPanel, TamperMap
│       ├── hooks/              # useWatermarkJob, useAuth
│       └── pages/              # Dashboard, EmbedPage
├── backend/
│   ├── app/                    # FastAPI routes, schemas, auth, tasks
│   ├── ml/                     # All ML modules (see below)
│   │   ├── encoder.py          # Dual-domain watermark encoder
│   │   ├── decoder.py          # Swin Transformer-based decoder + ADN
│   │   ├── igrm.py             # Inverse Generative Restoration Module
│   │   ├── asl.py              # Attack Simulation Layer
│   │   ├── jsnet.py            # Differentiable JPEG approximation
│   │   ├── iiwn.py             # Integer Invertible Watermark Network
│   │   ├── latent_embed.py     # WIND-style latent watermarking
│   │   ├── pixel_embed.py      # Pixel-domain HF watermarking
│   │   ├── perceptual_hash.py  # DINOv2 ViT perceptual hashing
│   │   ├── ecdsa_signer.py     # ECDSA signing + verification
│   │   ├── forgery_detector.py # Transplantation attack detection
│   │   ├── discrepancy.py      # Dual-layer discrepancy + attack classification
│   │   └── crypto.py           # BCH codes + Hamming parity
│   └── training/               # Training scripts + configs
├── datasets/
│   └── download.sh             # Dataset download automation
├── notebooks/                  # Jupyter experiments
├── .github/workflows/          # CI/CD pipelines
├── docker-compose.yml
├── .copilot-instructions.md    # Custom Copilot domain context
└── README.md
```

---

## Development Phases

### Phase 0 — Repository Setup & GitHub Copilot Config
**Timeline:** Week 1 | **Difficulty:** Easy

**Tasks:**
- Initialize GitHub repo with MIT license, `.gitignore` for Python/Node
- Set up monorepo: `/frontend` (Vite + React + TS), `/backend` (FastAPI), `/training` (Lightning)
- Create `.copilot-instructions.md` — teaches Copilot domain context (watermarking, diffusion, BCH)
- Configure `.vscode/settings.json` for Copilot workspace
- Docker Compose: backend + PostgreSQL + Redis + Celery worker
- Pre-commit hooks: black, ruff, mypy, eslint
- GitHub Actions CI skeleton

**GitHub Copilot Prompts:**
```
Generate a FastAPI project structure with SQLAlchemy, Celery, and Redis for an image processing service

Create a .copilot-instructions.md for a deep learning watermarking research project using PyTorch and diffusion models

Write a docker-compose.yml with services: fastapi, postgres, redis, celery-worker, and celery-beat

Create a GitHub Actions CI workflow that runs pytest, ruff, mypy, and npm test on pull_request events
```

**Key Files:** `.copilot-instructions.md`, `docker-compose.yml`, `pyproject.toml`, `.github/workflows/ci.yml`

---

### Phase 1 — Core Encoder-Decoder Watermarking Network
**Timeline:** Weeks 2–4 | **Difficulty:** High

**Tasks:**
- **Encoder network** — CNN-based autoencoder (HiDDeN-style): takes cover image (3×H×W) + N-bit payload → watermarked image with <0.5 dB PSNR loss
- **Decoder network** — Swin Transformer (RoWSFormer-inspired) with multi-head self-attention across spatial patches; outputs N-bit probability logits
- **Attention Decoding Network (ADN)** — identifies surviving ROIs without global image coordinates (enables recovery from extreme crops)
- **Dual loss function:** `L = λ₁·LPIPS + λ₂·MSE + λ₃·BCE`
- **Payload pipeline** — 48-bit base → BCH expand to 256-bit → interleaved Hamming parity blocks
- **PyTorch Lightning training loop** with W&B logging (PSNR, SSIM, BER)

**GitHub Copilot Prompts:**
```
Implement a HiDDeN-style encoder in PyTorch: takes a 3xHxW image tensor and Nbit message, outputs watermarked image with residual block architecture

Create a Swin Transformer-based watermark decoder that uses shifted window self-attention to decode a binary message from a possibly-cropped image

Write a combined loss function for watermarking: LPIPS perceptual loss + MSE + binary cross-entropy, with learnable weight balancing

Implement BCH error correction encoding in Python using the reedsolo library to expand a 48-bit payload to 256 bits with burst error correction capability

Create a PyTorch Lightning Module for training an encoder-decoder watermarking network, logging PSNR, SSIM, and BER metrics to wandb
```

**Key Files:** `backend/ml/encoder.py`, `backend/ml/decoder.py`, `backend/ml/adn.py`, `backend/ml/losses.py`, `backend/ml/crypto.py`

---

### Phase 2 — Attack Simulation Layer (ASL) with Differentiable JPEG
**Timeline:** Weeks 5–6 | **Difficulty:** Very High

**Tasks:**
- **Differentiable JPEG (JSNet)** — circular convolutions approximating DCT quantization as continuous ops; supports QF 5–100
- **Differentiable random crop** — Spatial Transformer Networks (STN) for gradient-preserving crop simulation (5%–100% canvas)
- **Geometric distortion layer** — differentiable rotation/scale/shear via `torch.nn.functional.grid_sample`
- **Diffusion regeneration attack simulator** — DDPM forward noise at varied timesteps + partial denoise (simulates D2RA/DAWN)
- **Curriculum scheduler** — start QF=70 / 80% crop, progressively reach QF=5 / 5% crop by epoch 80%

**GitHub Copilot Prompts:**
```
Implement differentiable JPEG compression in PyTorch using circular convolutions to approximate DCT block quantization, supporting quality factors from 5 to 100

Create a differentiable random crop module using Spatial Transformer Networks that preserves gradient flow and supports crop ratios from 0.05 to 1.0

Implement a curriculum learning scheduler for attack simulation that linearly increases attack severity (JPEG QF decreases, crop ratio decreases) over training epochs

Write a differentiable forward diffusion attack layer that adds Gaussian noise at varying DDPM timesteps to simulate regeneration attacks during training
```

**Key Files:** `backend/ml/asl.py`, `backend/ml/jsnet.py`, `backend/ml/stn_crop.py`, `backend/training/curriculum.py`

---

### Phase 3 — Inverse Generative Restoration Module (IGRM)
**Timeline:** Weeks 7–9 | **Difficulty:** Very High

**Tasks:**
- **Restoration-aware U-Net** — skip connections at every downsampling level; optimized for bit recovery (NOT visual quality). Custom loss: `λ_bit >> λ_perceptual`
- **Conditional diffusion restoration** — lightweight diffusion model conditioned on degraded image to restore watermark keypoints
- **Watermark keypoint detector** — attention-based detection of surviving watermark ROIs in fragmented images
- **DDIM inversion** — reconstructs initial noise space from degraded image (WIND-compatible); enables latent watermark extraction without model access
- **Blind extraction mode** — no knowledge of original embedding model required

**GitHub Copilot Prompts:**
```
Implement a U-Net in PyTorch optimized for watermark bit recovery (not perceptual quality): skip connections at every downsampling level, weighted loss prioritizing high-frequency reconstruction

Create a conditional diffusion restoration module that takes a degraded watermarked image and restores cryptographic bit patterns, using a custom loss with bit_accuracy_weight >> perceptual_weight

Implement DDIM inversion in PyTorch that reconstructs the initial noise latent from a generated image without access to the original model weights (approximation-based)

Build an attention-based watermark keypoint detector that identifies surviving ROI patches in a severely cropped image using spatial self-attention
```

**Key Files:** `backend/ml/igrm.py`, `backend/ml/unet_restore.py`, `backend/ml/ddim_inversion.py`, `backend/ml/keypoint_detector.py`

---

### Phase 4 — Dual-Domain Watermarking (Latent + Pixel)
**Timeline:** Weeks 10–12 | **Difficulty:** Very High

**Tasks:**
- **Layer 1 — Latent semantic (WIND-inspired):** `z_w = z + α·P(H(salt‖id))` — Fourier pattern injected into initial Gaussian noise; statistically indistinguishable from true Gaussian
- **Layer 2 — Pixel-level (iIWN):** Integer Invertible Watermark Network (normalizing flows / RealNVP) for perfectly reversible embedding
- **Discrepancy detector** — Layer 1 intact + Layer 2 scrubbed → D2RA/DAWN attack detected and classified
- **Dual-domain decoder** — independent extraction per layer, fusion logic reports payload + confidence + attack type

**GitHub Copilot Prompts:**
```
Implement WIND-style latent watermarking: inject a Fourier-domain pattern into Gaussian noise latent using HMAC-SHA256 hash of a secret salt, producing statistically Gaussian-distributed watermarked noise

Build an Integer Invertible Watermark Network (iIWN) using normalizing flows (RealNVP) for lossless pixel-domain watermark embedding and exact extraction

Create a dual-domain discrepancy detector that compares latent-layer and pixel-layer watermark extraction results to classify the attack type (D2RA, DAWN, JPEG, crop, or clean)
```

**Key Files:** `backend/ml/latent_embed.py`, `backend/ml/pixel_embed.py`, `backend/ml/iiwn.py`, `backend/ml/discrepancy.py`

---

### Phase 5 — Content-Dependent Cryptographic Binding (MetaSeal)
**Timeline:** Week 13 | **Difficulty:** Medium

**Tasks:**
- **ViT perceptual hasher** — frozen DINOv2 ViT-B/16 extracts 512-dim semantic embedding of the image
- **ECDSA signing** — SHA-256 hash of embedding signed with secp256k1; signature is part of watermark payload
- **Forgery detection** — recompute ViT embedding of test image; mismatch with extracted signature → transplantation attack flagged
- **Key management** — per-user ECDSA keys stored encrypted in PostgreSQL; optional on-chain registry (Ethereum EIP-712)

**GitHub Copilot Prompts:**
```
Use a frozen DINOv2 ViT-B/16 from torch.hub to extract a 512-dimensional perceptual hash from an input image tensor, normalized and L2-reduced

Implement ECDSA signing and verification using the Python cryptography library: sign a SHA-256 hash of a ViT embedding with secp256k1, serialize to compact 64-byte format for watermark payload

Create a forgery detector that compares the extracted ECDSA signature payload against a freshly computed ViT embedding of the test image to flag transplanted watermarks
```

**Key Files:** `backend/ml/perceptual_hash.py`, `backend/ml/ecdsa_signer.py`, `backend/ml/forgery_detector.py`

---

### Phase 6 — FastAPI Backend & Async Pipeline
**Timeline:** Week 14 | **Difficulty:** Medium

**Endpoints:**
- `POST /embed` — embed watermark into image
- `POST /detect` — detect + recover watermark from (possibly attacked) image
- `GET /job/{id}` — poll async job status
- `GET /audit/{image_hash}` — forensic audit log

**Response schema:**
```json
{
  "payload": "hex string",
  "confidence": 0.97,
  "attack_type": "JPEG_QF_10 | CROP_95 | D2RA | GUID_DIFFUSION | CLEAN",
  "tamper_mask": "base64 PNG",
  "latent_layer_intact": true,
  "pixel_layer_intact": false,
  "forgery_detected": false
}
```

**GitHub Copilot Prompts:**
```
Create FastAPI endpoints for a watermark detection service: POST /embed and POST /detect that enqueue Celery tasks, return job_id, and support GET /job/{id} for polling

Write a Pydantic response schema for watermark detection results: payload hex, confidence score, attack_type enum, tamper_mask as base64, dual-layer integrity flags

Implement JWT authentication with per-user rate limiting using Redis in FastAPI, including a dependency injection pattern for protected routes
```

---

### Phase 7 — React Frontend Dashboard
**Timeline:** Week 15 | **Difficulty:** Medium

**Components:**
- `ImageUploader` — drag-and-drop, file validation, upload progress
- `DetectionResultPanel` — payload display, confidence meter, attack badge
- `TamperVisualization` — canvas overlay of tamper_mask on original image
- `EmbedWorkflow` — embed custom payload, show before/after PSNR
- `useWatermarkJob` hook — polls `/job/{id}` every 2s

**GitHub Copilot Prompts:**
```
Create a React drag-and-drop image uploader component with TypeScript, preview, file size validation, and upload progress tracking using axios

Build a React hook useWatermarkJob(jobId) that polls an API endpoint every 2 seconds and returns {status, result, error} with proper cleanup on unmount

Create a canvas-based TamperVisualization component that overlays a base64-encoded binary mask on an image, highlighting tampered regions with a semi-transparent red overlay

Implement in-browser ECDSA key pair generation using the WebCrypto API (P-256), export as PEM, and POST the public key to a FastAPI backend
```

---

### Phase 8 — Model Training & Dataset Pipeline
**Timeline:** Weeks 16–20 | **Difficulty:** Expert

**Training Stages:**
| Stage | Description | Epochs | Target |
|-------|------------|--------|--------|
| 1 | Encoder+Decoder, mild attacks (QF 50–100, crop 60–100%) | 30 | PSNR >40dB, BER <0.01 |
| 2 | +IGRM, hard curriculum (QF 5–30, crop 5–20%) | 50 | BER <0.05 under extreme attacks |
| 3 | End-to-end fine-tune all modules | 20 | BER <0.02 combined attacks |

**GitHub Copilot Prompts:**
```
Create a PyTorch Dataset class WatermarkDataset that loads images from COCO, embeds a random 48-bit payload using the encoder, applies a random ASL attack, and returns the tuple (attacked, original, payload_bits)

Write a multi-stage training script using PyTorch Lightning with 3 stages: clean training, hard-attack curriculum, and end-to-end fine-tuning, with automatic checkpoint saving for best BER

Export a trained PyTorch watermark decoder to ONNX with dynamic axes for variable image sizes, then apply INT8 quantization using torch.quantization
```

---

### Phase 9 — Evaluation, Benchmarking & CI/CD
**Timeline:** Weeks 21–24 | **Difficulty:** Medium

**Tasks:**
- Reproduce WAVES benchmark; test all 5 attack vectors from the paper
- Track: TPR@1%FPR, BER, PSNR, SSIM, NCC per attack type in W&B
- pytest unit + integration tests; coverage >85%
- GitHub Actions CD: test → build Docker → push GHCR → deploy staging

**GitHub Copilot Prompts:**
```
Create a pytest benchmark that embeds a watermark, applies 5 attack types (JPEG QF=10, 95% crop, diffusion, guided diffusion, D2RA), and asserts BER stays below 0.05 for each

Write a GitHub Actions workflow that builds a Docker image, runs pytest, pushes to GitHub Container Registry, and deploys to a fly.io staging environment on merge to main
```

---

## How to Train the Backend Vision Model

> This system uses specialized **vision models** (not text LLMs). The AI backend is PyTorch-based encoder-decoder + IGRM networks.

### Step 1: Choose Base Architectures
- **Decoder backbone:** `microsoft/swin-base-patch4-window7-224` (HuggingFace)
- **Encoder backbone:** U-Net with ResNet-50
- **IGRM:** Fine-tune `stabilityai/stable-diffusion-2-1` UNet only (freeze VAE + CLIP)

### Step 2: Hardware
- Minimum: 2× RTX 3090 (48GB VRAM) or 1× A100 80GB
- Use `torch.compile()` + `bfloat16` mixed precision
- Cloud GPU: Lambda Cloud / RunPod / Vast.ai (~$2–3/hr A100)
- Full training estimate: 3–5 days

### Step 3: Configuration
```yaml
# config/stage2.yaml
model:
  encoder: resnet50_unet
  decoder: swin_base
  igrm: conditional_diffusion_unet

training:
  batch_size: 32
  grad_accumulate: 4
  effective_batch: 128
  lr: 1.0e-4
  weight_decay: 0.01
  scheduler: cosine
  epochs: 50

attacks:
  jpeg_qf_range: [5, 30]
  crop_ratio_range: [0.05, 0.20]
  diffusion_timesteps: [400, 900]
```

### Step 4 (Optional): Text LLM for Forensic Reporting
Fine-tune **Mistral-7B-Instruct** with QLoRA (4-bit) using `unsloth` on ~1,000 synthetic forensic report examples. Outputs human-readable attack analysis from detection JSON.

---

## Training Datasets

### Primary Image Datasets
| Dataset | Size | Use | URL |
|---------|------|-----|-----|
| COCO 2017 | ~18 GB | Main training (118k diverse images) | `cocodataset.org` |
| DIV2K | ~7 GB | High-res image restoration training | `data.vision.ee.ethz.ch/cvl/DIV2K` |
| RAISE-1k | ~4 GB TIFF | Pristine uncompressed images (no prior JPEG) | `loki.disi.unitn.it/RAISE` |
| BSDS500 | ~250 MB | Edge-aware fragile watermarking | HuggingFace `datasets/bsds500` |
| MIT-Adobe FiveK | ~50 GB | Tone/color editing attack testing | `data.csail.mit.edu/graphics/fivek` |

### AI-Generated Image Datasets (Critical for Latent WM Training)
| Dataset | Size | Use | URL |
|---------|------|-----|-----|
| DiffusionDB | 1.6 TB (use 2M subset) | Real SD outputs for latent WM recovery | `huggingface.co/datasets/poloclub/diffusiondb` |
| LAION-Aesthetics v2 | ~600k images | High-quality mixed AI/real images | `huggingface.co/datasets/laion/laion-high-resolution` |
| GenImage Benchmark | ~400 GB | 1.3M images from 8 generators (SD, MJ, DALL-E) | `github.com/GenImage-Dataset/GenImage` |

### Attack Evaluation Datasets
| Dataset | Size | Use | URL |
|---------|------|-----|-----|
| WAVES Benchmark | ~50 GB | Direct benchmark validation | `github.com/umd-meirl/waves` |
| KADID-10k | ~2 GB | Distortion-aware IGRM training | `database.mmsp-kn.de/kadid-10k-database.html` |
| HiDDeN Benchmark | ~12 GB | Baseline comparison (BER/PSNR) | `github.com/ando-khachatryan/HiDDeN` |

### Download All Datasets
```bash
cd datasets/
# Run automated downloader (generate with Copilot prompt below)
bash download.sh --datasets coco div2k raise1k waves kadid --output ./data/
```

**Copilot prompt to generate `download.sh`:**
```
Write a bash script that downloads COCO 2017 train, DIV2K, WAVES benchmark, and KADID-10k datasets 
with progress bars, MD5 checksum verification, and automatic extraction to ./data/ subdirectories
```

---

## GitHub Copilot Best Practices for This Project

1. **Use Copilot Chat in Agent mode** (`@workspace`) for project-wide context
2. **Write math equations as comments first** — `# z_w = z + α·P(H(salt||id))` then let Copilot implement
3. **Reference papers in docstrings** — `"""From WIND (ICLR 2025): inject Fourier group identifier into noise latent"""`
4. **Use `/tests` command** — highlight any function → Copilot Chat → `/tests` for auto-pytest generation
5. **Use `/fix` on training loops** — paste error tracebacks directly into Copilot Chat

---

## Target Performance Metrics (From Research Paper Blueprint)

| Attack Vector | Current SOTA | Target (This System) | Key Component |
|--------------|-------------|---------------------|---------------|
| Clean TPR | 90–96% | >99.9% | BCH + ECC + Crypto binding |
| Generative Regeneration | 0–10% survival | >95% survival | Dual-Stage Latent + WIND |
| Cropping (<10% retained) | Fails | >90% accuracy | ADN + IGRM |
| JPEG QF=10 | Complete signal loss | BER <0.05 | JSNet + 16-bit BCH |
| Forgery / Transplantation | Misattribution likely | 0% success rate | ViT perceptual binding |

---

*Development plan derived from: "Forensic Reconstruction of Hidden AI Watermarks: Deep Learning Solutions for Degradation Recovery" (2026)*
