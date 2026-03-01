<p align="center">
  <img src="frontend/public/favicon.svg" width="80" alt="WatermarkAI Logo" />
</p>

<h1 align="center">AI Watermark Recovery</h1>

<p align="center">
  <b>Forensic Reconstruction of Hidden AI Watermarks — Deep Learning Solutions for Degradation Recovery</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-18.3-61DAFB?logo=react&logoColor=black" alt="React" />
  <img src="https://img.shields.io/badge/TypeScript-5.6-3178C6?logo=typescript&logoColor=white" alt="TypeScript" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
</p>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
  - [ML Module Inventory (21 modules)](#ml-module-inventory-21-modules)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Getting Started (Step-by-Step)](#getting-started-step-by-step)
  - [Step 1 — Clone the Repository](#step-1--clone-the-repository)
  - [Step 2 — Configure Environment Variables](#step-2--configure-environment-variables)
  - [Option A — Docker (Recommended)](#option-a--docker-recommended)
  - [Option B — Manual Local Setup](#option-b--manual-local-setup)
    - [1. Install PostgreSQL \& Redis](#1-install-postgresql--redis)
    - [2. Update `.env` for local hostnames](#2-update-env-for-local-hostnames)
    - [3. Create Python virtual environment \& install](#3-create-python-virtual-environment--install)
    - [4. Run database migrations](#4-run-database-migrations)
    - [5. Start the FastAPI server](#5-start-the-fastapi-server)
    - [6. Start the Celery worker (separate terminal)](#6-start-the-celery-worker-separate-terminal)
    - [7. Verify](#7-verify)
- [Frontend Setup](#frontend-setup)
  - [Step 1 — Install Node.js dependencies](#step-1--install-nodejs-dependencies)
  - [Step 2 — Start the development server](#step-2--start-the-development-server)
  - [Step 3 — Build for production](#step-3--build-for-production)
  - [Frontend + Backend Together](#frontend--backend-together)
- [API Documentation](#api-documentation)
  - [Endpoints](#endpoints)
  - [Job Types](#job-types)
  - [Example Usage](#example-usage)
- [Environment Variables](#environment-variables)
- [Running Tests](#running-tests)
  - [Linting \& Type Checking](#linting--type-checking)
- [Target Performance](#target-performance)
- [Frontend Features](#frontend-features)
  - [Visual Effects](#visual-effects)
  - [Interactive Elements](#interactive-elements)
  - [Sound Effects (Web Audio API)](#sound-effects-web-audio-api)
- [Troubleshooting](#troubleshooting)
  - [Docker Issues](#docker-issues)
  - [Frontend Issues](#frontend-issues)
  - [Backend Issues](#backend-issues)
- [Development Roadmap](#development-roadmap)
- [License](#license)

---

## Overview

**AI Watermark Recovery** is a full-stack system for **embedding**, **detecting**, and **recovering** invisible watermarks from images — even after severe degradation such as JPEG compression (QF=5), extreme cropping (95%), and diffusion regeneration attacks (D2RA/DAWN).

The project implements a 5-layer ML pipeline based on the research paper _"Forensic Reconstruction of Hidden AI Watermarks: Deep Learning Solutions for Degradation Recovery"_ with a production-ready FastAPI backend, async Celery task queue, and a visually rich React frontend.

---

## Key Features

| Capability                           | Description                                                                                |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| **Dual-Domain Watermarking**         | Latent-space (WIND / Fourier noise) + pixel-space (iIWN normalizing flows) embedding       |
| **Blind Recovery (IGRM)**            | Inverse Generative Restoration Module recovers watermark bits from heavily degraded images |
| **Attack Simulation Layer**          | Differentiable JPEG, crop, rotation, and diffusion attacks for end-to-end training         |
| **Cryptographic Binding (MetaSeal)** | DINOv2 ViT perceptual hash + ECDSA signing prevents forgery & transplantation              |
| **Error Correction**                 | BCH + Hamming codes expand a 48-bit payload to 256 bits for burst-error resilience         |
| **Async Processing**                 | Celery workers handle heavy ML inference; poll job status via REST API                     |
| **Interactive Frontend**             | React dashboard with visual effects, sound, 3D tilt cards, particle navigation             |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│   Vite 6 · TypeScript · Framer Motion · Web Audio API           │
│   ┌──────────┐  ┌──────────────┐  ┌────────────────────┐       │
│   │   Hero   │  │ EmbedWorkflow│  │  DetectWorkflow    │       │
│   │  Landing │  │  Upload+Embed│  │  Upload+Detect     │       │
│   └──────────┘  └──────────────┘  └────────────────────┘       │
│         │               │                   │                   │
│         └───────────────┼───────────────────┘                   │
│                         │  Axios HTTP (JWT)                     │
└─────────────────────────┼───────────────────────────────────────┘
                          │  /api/v1/*
┌─────────────────────────┼───────────────────────────────────────┐
│                    FastAPI Server (:8000)                        │
│   ┌──────────┐  ┌──────────────┐  ┌────────────────────┐       │
│   │  Routes  │  │   Schemas    │  │     Security       │       │
│   │ /images  │  │  Pydantic v2 │  │  JWT + bcrypt      │       │
│   │  /jobs   │  │              │  │  Rate Limiting     │       │
│   └────┬─────┘  └──────────────┘  └────────────────────┘       │
│        │                                                         │
│   ┌────▼─────────────────────────────────────────────────┐      │
│   │              Celery Task Queue                        │      │
│   │  embed_watermark · extract_watermark · recover_wm     │      │
│   └────┬──────────────────────────────────────────────────┘      │
│        │                                                         │
│   ┌────▼─────────────────────────────────────────────────┐      │
│   │              ML Pipeline (PyTorch)                    │      │
│   │  Encoder → ASL → IGRM → Decoder → MetaSeal          │      │
│   └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
          │                    │
    ┌─────▼──────┐      ┌─────▼──────┐
    │ PostgreSQL │      │   Redis    │
    │    16      │      │     7      │
    │ (metadata) │      │  (broker)  │
    └────────────┘      └────────────┘
```

---

## ML Pipeline Architecture

The core ML system is a **5-stage pipeline** that processes images end-to-end:

```
Input Image + Payload
        │
        ▼
┌───────────────────┐
│  1. Encoder       │  HiDDeN-style CNN autoencoder
│  (encoder.py)     │  Embeds N-bit payload into cover image
│                   │  Target: <0.5 dB PSNR loss
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  2. ASL            │  Attack Simulation Layer
│  (asl.py)          │  Differentiable JPEG (JSNet), crop (STN),
│  (jsnet.py)        │  rotation, diffusion noise
│  (stn_crop.py)     │  Curriculum: QF 70→5, crop 80%→5%
└───────┬────────────┘
        │
        ▼
┌───────────────────┐
│  3. IGRM           │  Inverse Generative Restoration Module
│  (igrm.py)         │  U-Net + conditional diffusion
│  (unet_restore.py) │  Restores watermark bit structure
│  (ddim_inversion)  │  from heavily degraded images
└───────┬────────────┘
        │
        ▼
┌───────────────────┐
│  4. Decoder        │  Swin Transformer decoder + ADN
│  (decoder.py)      │  Multi-head self-attention across patches
│  (adn.py)          │  Outputs N-bit probability logits
│  (losses.py)       │  Loss: LPIPS + MSE + BCE
└───────┬────────────┘
        │
        ▼
┌───────────────────┐
│  5. MetaSeal       │  Cryptographic Binding
│  (perceptual_hash) │  DINOv2 ViT-B/16 perceptual hash
│  (ecdsa_signer)    │  ECDSA secp256k1 signing
│  (forgery_detector)│  Transplantation attack detection
│  (discrepancy.py)  │  Dual-layer integrity verification
└───────────────────┘
        │
        ▼
  Detection Result:
  {payload, confidence, attack_type,
   tamper_mask, integrity_flags}
```

### ML Module Inventory (21 modules)

| Module             | File                   | Purpose                               |
| ------------------ | ---------------------- | ------------------------------------- |
| Encoder            | `encoder.py`           | HiDDeN-style CNN watermark embedding  |
| Decoder            | `decoder.py`           | Swin Transformer bit recovery         |
| Attention Decoding | `adn.py`               | ROI detection for extreme crops       |
| Loss Functions     | `losses.py`            | LPIPS + MSE + BCE combined loss       |
| Attack Simulation  | `asl.py`               | Differentiable attack layer           |
| JPEG Simulation    | `jsnet.py`             | Differentiable JPEG compression       |
| Crop Simulation    | `stn_crop.py`          | STN-based differentiable crop         |
| IGRM               | `igrm.py`              | Inverse generative restoration        |
| U-Net Restore      | `unet_restore.py`      | Skip-connected restoration network    |
| Diffusion Restore  | `diffusion_restore.py` | Conditional diffusion restoration     |
| DDIM Inversion     | `ddim_inversion.py`    | Latent space reconstruction           |
| Keypoint Detector  | `keypoint_detector.py` | Watermark ROI detection               |
| Latent Embedding   | `latent_embed.py`      | WIND-style Fourier noise injection    |
| Pixel Embedding    | `pixel_embed.py`       | Pixel-domain HF watermarking          |
| iIWN               | `iiwn.py`              | Normalizing flow reversible embedding |
| Perceptual Hash    | `perceptual_hash.py`   | DINOv2 ViT feature extraction         |
| ECDSA Signer       | `ecdsa_signer.py`      | secp256k1 signing & verification      |
| Forgery Detector   | `forgery_detector.py`  | Transplantation attack detection      |
| Discrepancy        | `discrepancy.py`       | Dual-layer attack classification      |
| Error Correction   | `crypto.py`            | BCH + Hamming error codes             |
| Model Loader       | `model_loader.py`      | Unified model loading utility         |

---

## Tech Stack

| Layer           | Technology                                    | Version          |
| --------------- | --------------------------------------------- | ---------------- |
| **Frontend**    | React · TypeScript · Vite                     | 18.3 · 5.6 · 6.4 |
| **Frontend FX** | Framer Motion · Web Audio API                 | 11.x             |
| **API**         | FastAPI · Pydantic v2 · Uvicorn               | 0.111+           |
| **Database**    | PostgreSQL · SQLAlchemy 2.0 (async) · Alembic | 16 · 2.0         |
| **Task Queue**  | Celery · Redis                                | 5.4 · 7          |
| **ML**          | PyTorch · torchvision · Swin Transformer      | 2.x              |
| **Auth**        | JWT (python-jose) · bcrypt · ECDSA            | —                |
| **Infra**       | Docker Compose · GitHub Actions               | —                |

---

## Repository Structure

```
ai-watermark-recovery/
├── .env                           # Environment variables (copy .env → edit)
├── .github/workflows/             # CI/CD pipelines
├── .pre-commit-config.yaml        # Code quality hooks
├── alembic.ini                    # Alembic migration config
├── docker-compose.yml             # PostgreSQL + Redis + API + Worker
├── Dockerfile                     # Multi-stage build (api + worker targets)
├── pyproject.toml                 # Python 3.11+ dependencies
├── DEVELOPMENT_PLAN.md            # Full 9-phase roadmap
├── copilot-instructions.md        # GitHub Copilot domain context
├── LICENSE                        # MIT
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app, lifespan, CORS
│   │   ├── worker.py              # Celery app factory
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic Settings (env-driven)
│   │   │   ├── database.py        # Async SQLAlchemy engine + session
│   │   │   └── security.py        # JWT + password hashing
│   │   ├── models/
│   │   │   ├── user.py            # User ORM model
│   │   │   ├── image.py           # Image metadata ORM
│   │   │   └── job.py             # Job tracking ORM
│   │   ├── schemas/
│   │   │   ├── user.py            # UserCreate, UserOut, Token
│   │   │   ├── image.py           # ImageOut, ImageUploadResponse
│   │   │   └── job.py             # JobCreate, JobOut
│   │   ├── routes/
│   │   │   ├── health.py          # GET /health
│   │   │   ├── images.py          # POST/GET /api/v1/images
│   │   │   └── jobs.py            # POST/GET /api/v1/jobs
│   │   └── tasks/
│   │       └── image_tasks.py     # Celery tasks: embed, extract, recover
│   │
│   ├── ml/                        # 21 ML modules (see table above)
│   │   ├── encoder.py             # HiDDeN encoder
│   │   ├── decoder.py             # Swin Transformer decoder
│   │   ├── asl.py                 # Attack Simulation Layer
│   │   ├── igrm.py                # Inverse Generative Restoration
│   │   ├── crypto.py              # BCH + Hamming error correction
│   │   ├── ecdsa_signer.py        # ECDSA cryptographic signing
│   │   ├── perceptual_hash.py     # DINOv2 perceptual hashing
│   │   ├── forgery_detector.py    # Transplantation detection
│   │   └── ...                    # (see full list above)
│   │
│   └── training/                  # Training scripts & configs
│
├── frontend/                      # React + TypeScript + Vite
│   ├── package.json               # Node.js dependencies
│   ├── vite.config.ts             # Vite config (proxy /api → :8000)
│   ├── tsconfig.json              # TypeScript config
│   ├── index.html                 # Entry HTML (Google Fonts)
│   └── src/
│       ├── main.tsx               # React entry point
│       ├── App.tsx                # Root component + routing
│       ├── api/client.ts          # Axios API client + JWT auth
│       ├── hooks/
│       │   └── useWatermarkJob.ts # Job polling hook
│       ├── utils/
│       │   └── sfx.ts            # Web Audio synthesized sounds
│       ├── styles/
│       │   └── globals.css        # Dark theme + animations
│       └── components/
│           ├── Hero.tsx           # Landing page (parallax, stats)
│           ├── EmbedWorkflow.tsx   # Watermark embedding page
│           ├── DetectWorkflow.tsx  # Detection & recovery page
│           ├── ImageUploader.tsx   # Drag-and-drop upload
│           ├── Navbar.tsx         # Navigation bar
│           ├── GooeyNav.tsx/.css   # Particle-morphing nav tabs
│           ├── SpotlightCard.tsx/.css # Mouse-tracking spotlight
│           ├── ClickSpark.tsx     # Click particle sparks
│           ├── DecryptedText.tsx   # Text scramble animation
│           ├── Effects.tsx        # Parallax, scroll-reveal, orbs
│           └── Interactive.tsx    # CountUp, TiltCard, Magnetic, etc.
│
├── alembic/                       # Database migrations
├── datasets/                      # Dataset scripts & data
├── notebooks/                     # Jupyter experiments
└── tests/                         # pytest test suite (246 tests)
```

---

## Prerequisites

Before you begin, ensure your system has:

| Requirement        | Version                                | Check Command            |
| ------------------ | -------------------------------------- | ------------------------ |
| **Git**            | Any                                    | `git --version`          |
| **Docker Desktop** | 24+                                    | `docker --version`       |
| **Docker Compose** | v2+                                    | `docker compose version` |
| **Node.js**        | 18+ (for frontend)                     | `node --version`         |
| **npm**            | 9+ (for frontend)                      | `npm --version`          |
| **Python**         | 3.11+ (only if running without Docker) | `python --version`       |

> **Note:** If you only want to run the backend, Docker is sufficient. Node.js is only needed for the frontend.

---

## Getting Started (Step-by-Step)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Hell-Phoenix/ai-watermark-recovery.git
cd ai-watermark-recovery
```

### Step 2 — Configure Environment Variables

The `.env` file is included with development defaults. Review and edit if needed:

```bash
# On Linux/macOS:
cat .env

# On Windows (PowerShell):
Get-Content .env
```

Key variables (defaults work out-of-the-box with Docker):

| Variable                | Default                                                       | Notes                            |
| ----------------------- | ------------------------------------------------------------- | -------------------------------- |
| `DATABASE_URL`          | `postgresql+asyncpg://postgres:postgres@db:5432/watermark_db` | Uses Docker service name `db`    |
| `REDIS_URL`             | `redis://redis:6379/0`                                        | Uses Docker service name `redis` |
| `CELERY_BROKER_URL`     | `redis://redis:6379/1`                                        | Celery message broker            |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2`                                        | Celery result store              |
| `SECRET_KEY`            | `hackathon2026xyzabc123random`                                | **Change in production**         |
| `UPLOAD_DIR`            | `./uploads`                                                   | Image storage path               |
| `MAX_UPLOAD_SIZE_MB`    | `20`                                                          | Upload file size limit           |

---

### Option A — Docker (Recommended)

This is the fastest way to get everything running. One command starts all 4 services.

```bash
# Build and start all services
docker compose up --build
```

This will start:

| Service    | Port                    | Description                                |
| ---------- | ----------------------- | ------------------------------------------ |
| **api**    | `http://localhost:8000` | FastAPI server                             |
| **db**     | `localhost:5432`        | PostgreSQL 16                              |
| **redis**  | `localhost:6379`        | Redis 7                                    |
| **worker** | —                       | Celery worker (processes background tasks) |

**Verify it's running:**

```bash
# Check all containers are up
docker compose ps

# Test the health endpoint
curl http://localhost:8000/health
# Expected: {"status":"healthy","database":"connected","redis":"connected"}
```

**Stop services:**

```bash
docker compose down           # Stop containers
docker compose down -v        # Stop + delete data volumes
```

---

### Option B — Manual Local Setup

If you prefer running without Docker, you need PostgreSQL and Redis installed locally.

#### 1. Install PostgreSQL & Redis

- **PostgreSQL:** Download from [postgresql.org](https://www.postgresql.org/download/) — create a database called `watermark_db`
- **Redis:** Download from [redis.io](https://redis.io/download/) or use WSL on Windows

#### 2. Update `.env` for local hostnames

```bash
# Change service names from Docker names to localhost:
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/watermark_db
DATABASE_URL_SYNC=postgresql+psycopg2://postgres:postgres@localhost:5432/watermark_db
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

#### 3. Create Python virtual environment & install

```bash
# Create and activate virtual environment
python -m venv .venv

# Activate:
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat

# Install all Python dependencies
pip install -e ".[dev]"
```

#### 4. Run database migrations

```bash
alembic upgrade head
```

> On first run, the FastAPI lifespan hook auto-creates tables. Use Alembic for subsequent schema changes.

#### 5. Start the FastAPI server

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

#### 6. Start the Celery worker (separate terminal)

```bash
celery -A backend.app.worker:celery_app worker --loglevel=info
```

#### 7. Verify

```bash
curl http://localhost:8000/health
```

---

## Frontend Setup

The frontend is a **React 18 + TypeScript + Vite** application with interactive visual effects.

### Step 1 — Install Node.js dependencies

```bash
cd frontend
npm install
```

### Step 2 — Start the development server

```bash
npm run dev
```

The frontend starts at **`http://localhost:5173`** and proxies API calls to `http://localhost:8000`.

### Step 3 — Build for production

```bash
npm run build     # Outputs to frontend/dist/
npm run preview   # Preview the production build
```

### Frontend + Backend Together

1. Start the backend first (Docker or manually as described above)
2. Start the frontend dev server (`npm run dev` in the `frontend/` directory)
3. Open **`http://localhost:5173`** in your browser

The Vite dev server proxies all `/api/*` requests to the backend at port 8000 automatically.

---

## API Documentation

Once the backend is running, interactive API docs are available at:

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoints

| Method | Path                  | Description                                  |
| ------ | --------------------- | -------------------------------------------- |
| `GET`  | `/health`             | Health check (database + Redis connectivity) |
| `POST` | `/api/v1/images/`     | Upload an image                              |
| `GET`  | `/api/v1/images/`     | List uploaded images                         |
| `GET`  | `/api/v1/images/{id}` | Get image metadata                           |
| `POST` | `/api/v1/jobs/`       | Submit an async processing job               |
| `GET`  | `/api/v1/jobs/{id}`   | Poll job status & results                    |
| `GET`  | `/api/v1/jobs/`       | List all jobs                                |

### Job Types

| Job Type            | Description                                 |
| ------------------- | ------------------------------------------- |
| `embed_watermark`   | Embed a watermark payload into an image     |
| `extract_watermark` | Extract watermark bits from an image        |
| `recover_watermark` | Run IGRM-based recovery on a degraded image |

### Example Usage

```bash
# 1. Upload an image
curl -X POST http://localhost:8000/api/v1/images/ \
  -F "file=@photo.png"
# Response: {"id": "uuid-here", "filename": "photo.png", ...}

# 2. Submit a watermark embedding job
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"image_id": "<uuid>", "job_type": "embed_watermark"}'
# Response: {"id": "job-uuid", "status": "pending", ...}

# 3. Poll job status
curl http://localhost:8000/api/v1/jobs/<job-uuid>
# Response: {"id": "...", "status": "completed", "result": {...}}
```

---

## Environment Variables

All settings are loaded from the `.env` file via Pydantic Settings:

| Variable                      | Default                                                        | Description             |
| ----------------------------- | -------------------------------------------------------------- | ----------------------- |
| `APP_NAME`                    | `AI Watermark Recovery`                                        | Application name        |
| `DEBUG`                       | `true`                                                         | Debug mode              |
| `DATABASE_URL`                | `postgresql+asyncpg://postgres:postgres@db:5432/watermark_db`  | Async PostgreSQL URL    |
| `DATABASE_URL_SYNC`           | `postgresql+psycopg2://postgres:postgres@db:5432/watermark_db` | Sync fallback (Alembic) |
| `DB_ECHO`                     | `false`                                                        | Log SQL queries         |
| `REDIS_URL`                   | `redis://redis:6379/0`                                         | Redis connection        |
| `CELERY_BROKER_URL`           | `redis://redis:6379/1`                                         | Celery message broker   |
| `CELERY_RESULT_BACKEND`       | `redis://redis:6379/2`                                         | Celery result backend   |
| `SECRET_KEY`                  | _(set in .env)_                                                | JWT signing key         |
| `ALGORITHM`                   | `HS256`                                                        | JWT algorithm           |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `60`                                                           | Token expiry            |
| `UPLOAD_DIR`                  | `./uploads`                                                    | Image upload directory  |
| `MAX_UPLOAD_SIZE_MB`          | `20`                                                           | Maximum upload size     |

---

## Running Tests

The project has **246 passing tests** covering all ML modules, API endpoints, schemas, and task processing.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=backend --cov-report=term-missing

# Run specific test file
pytest tests/test_health.py

# Run tests inside Docker
docker compose exec api pytest
```

### Linting & Type Checking

```bash
# Lint with ruff
ruff check backend/

# Type check with mypy
mypy backend/

# Auto-format
ruff format backend/
```

---

## Target Performance

Performance targets based on the research paper blueprint:

| Attack Vector                  | Current SOTA   | Our Target        | Key Component              |
| ------------------------------ | -------------- | ----------------- | -------------------------- |
| Clean detection                | 90–96% TPR     | **>99.9%**        | BCH + ECC + crypto binding |
| Generative regeneration (D2RA) | 0–10% survival | **>95% survival** | Dual-stage latent + WIND   |
| Extreme crop (<10% retained)   | Fails          | **>90% accuracy** | ADN + IGRM                 |
| JPEG QF = 10                   | Signal loss    | **BER < 0.05**    | JSNet + 16-bit BCH         |
| Forgery / transplantation      | Misattribution | **0% success**    | ViT perceptual binding     |

---

## Frontend Features

The frontend is designed as a visually immersive hackathon demo with:

### Visual Effects

- **Floating particles** — ambient colored dots drifting through background with connecting lines
- **Animated background orbs** — large blurred gradient spheres floating
- **Grain overlay** — subtle film-grain texture for depth
- **Parallax scrolling** — multi-speed scroll layers
- **Scroll-reveal** — elements animate in from different directions

### Interactive Elements

- **GooeyNav** — particle-morphing navigation tabs with SVG gooey filter
- **SpotlightCard** — mouse-tracking radial glow effect on cards
- **3D TiltCard** — perspective tilt following cursor + glare overlay
- **Magnetic buttons** — buttons subtly pull toward cursor on hover
- **Ripple effect** — material-design ripple animation on click
- **CountUp** — animated number counters on scroll into view
- **ClickSpark** — spark particles burst from every click
- **Cursor trail** — softly glowing dots following the mouse
- **Tooltips** — animated tooltips on pipeline architecture items
- **Page transitions** — fade + slide when switching pages

### Sound Effects (Web Audio API)

Synthesized sounds (no audio files needed):

- Click, hover, toggle, navigation tones
- File drop, success, error feedback sounds

---

## Troubleshooting

### Docker Issues

| Problem                     | Solution                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------- |
| Port 5432 already in use    | Stop local PostgreSQL: `sudo systemctl stop postgresql` or change port in `docker-compose.yml` |
| Port 8000 already in use    | Change the API port: modify `ports: "8001:8000"` in `docker-compose.yml`                       |
| Containers crash on start   | Check logs: `docker compose logs api` or `docker compose logs worker`                          |
| Database connection refused | Wait for PostgreSQL to initialize: `docker compose logs db`                                    |
| Out of disk space           | Prune unused images: `docker system prune -a`                                                  |

### Frontend Issues

| Problem               | Solution                                                                       |
| --------------------- | ------------------------------------------------------------------------------ |
| `npm install` fails   | Ensure Node.js 18+ is installed: `node --version`                              |
| API calls return 404  | Make sure the backend is running on port 8000                                  |
| Blank page in browser | Check browser console for errors; clear cache                                  |
| Sound not playing     | Click anywhere on the page first (browsers require user interaction for audio) |

### Backend Issues

| Problem                         | Solution                                                                                                |
| ------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError`           | Activate venv and run `pip install -e ".[dev]"`                                                         |
| Alembic migration fails         | Ensure DATABASE_URL_SYNC points to a running PostgreSQL instance                                        |
| Celery tasks stuck as "pending" | Verify Redis is running and CELERY_BROKER_URL is correct                                                |
| PyTorch CUDA errors             | Install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |

---

## Development Roadmap

| Phase       | Description                                                  | Status      |
| ----------- | ------------------------------------------------------------ | ----------- |
| **Phase 0** | Repository setup, Docker, CI/CD skeleton                     | ✅ Complete |
| **Phase 1** | Core encoder-decoder watermarking network (Swin Transformer) | ✅ Complete |
| **Phase 2** | Attack Simulation Layer with differentiable JPEG             | ✅ Complete |
| **Phase 3** | Inverse Generative Restoration Module (IGRM)                 | ✅ Complete |
| **Phase 4** | Dual-domain latent + pixel watermarking                      | ✅ Complete |
| **Phase 5** | Cryptographic binding — MetaSeal (ViT + ECDSA)               | ✅ Complete |
| **Phase 6** | FastAPI backend & async pipeline                             | ✅ Complete |
| **Phase 7** | React frontend dashboard                                     | ✅ Complete |
| **Phase 8** | Model training & dataset pipeline                            | 🔜 Planned  |
| **Phase 9** | Evaluation, benchmarking & CI/CD                             | 🔜 Planned  |

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the full detailed development plan.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <b>WatermarkAI</b> — AI-Powered Watermark Recovery System · AMD Hackathon 2025
</p>


<p align="center">
  <b>WatermarkAI</b> — AI-Powered Watermark Recovery System · AMD Hackathon 2025
</p>
