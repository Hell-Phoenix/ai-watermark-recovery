# GitHub Copilot Instructions — AI Watermark Recovery System

## Project Domain
This is a **forensic AI watermark recovery system** implementing deep learning-based watermark embedding, attack simulation, and blind recovery from degraded images.

## Core Concepts (Always Relevant)
- **Watermarking**: Encoding hidden bit payloads into images imperceptibly (HiDDeN, StegaStamp, InvisMark patterns)
- **Attack types**: JPEG compression (QF 5–20), extreme cropping (<10% retained), diffusion regeneration attacks (D2RA, DAWN), guided diffusion attacks
- **Dual-domain embedding**: Latent-space (WIND/Fourier noise) + pixel-space (iIWN normalizing flows)
- **IGRM**: Inverse Generative Restoration Module — restores watermark bit structure (NOT visual quality) from degraded images
- **ASL**: Attack Simulation Layer — differentiable approximations of all attack types for end-to-end training
- **BCH codes**: Bose-Chaudhuri-Hocquenghem error correction — expands 48-bit payload to 256 bits for burst error resilience
- **Hamming parity**: Distributed parity bit matrix across interleaved image blocks

## Architecture References
- Encoder: CNN-based autoencoder (HiDDeN / InvisMark style)
- Decoder: Swin Transformer (RoWSFormer) with Attention Decoding Networks (ADN)
- IGRM: Conditional diffusion U-Net, bit-accuracy loss weighted above perceptual loss
- Latent embed: WIND framework (ICLR 2025) — Fourier patterns in initial Gaussian noise
- Pixel embed: iIWN — Integer Invertible Watermark Network (normalizing flows)
- Crypto: DINOv2 ViT-B/16 perceptual hash + ECDSA secp256k1 signing (MetaSeal)

## Key Python Libraries
- `torch`, `torchvision`, `timm` — core ML
- `lightning` (PyTorch Lightning) — training
- `reedsolo` — BCH error correction
- `cryptography` — ECDSA
- `fastapi`, `celery`, `redis` — backend
- `wandb` — experiment tracking

## Coding Conventions
- All ML modules return typed tensors with explicit shape comments: `# (B, C, H, W)`
- Loss functions always return a dict: `{"total": ..., "perceptual": ..., "ber": ...}`
- Training steps log to W&B via `self.log_dict()`
- Use `@torch.compile` decorator on all inference-critical modules
- Type hints required on all public functions
- Docstrings reference the paper/framework the technique is from

## File Structure
```
backend/ml/       → PyTorch modules
backend/app/      → FastAPI routes + Celery tasks
backend/training/ → Lightning modules + training scripts
frontend/src/     → React + TypeScript components
datasets/         → Download scripts + PyTorch Dataset classes
```
