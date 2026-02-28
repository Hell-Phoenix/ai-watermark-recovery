# AI Watermark Recovery

> **Forensic Reconstruction of Hidden AI Watermarks — Deep Learning Solutions for Degradation Recovery**

A full-stack system for embedding, detecting, and recovering invisible watermarks from images — even after severe degradation (JPEG compression, cropping, diffusion regeneration attacks). Built with **FastAPI**, **PyTorch**, **Celery**, **Redis**, and **PostgreSQL**.

---

## Highlights

| Capability | Description |
|---|---|
| **Dual-domain watermarking** | Latent-space (WIND / Fourier noise) + pixel-space (iIWN normalizing flows) |
| **Blind recovery (IGRM)** | Inverse Generative Restoration Module recovers watermark bits from heavily degraded images |
| **Attack Simulation Layer** | Differentiable JPEG, crop, rotation, and diffusion attacks for end-to-end training |
| **Cryptographic binding** | DINOv2 ViT perceptual hash + ECDSA signing detects forgery & transplantation |
| **Error correction** | BCH + Hamming codes expand a 48-bit payload to 256 bits for burst-error resilience |
| **Async processing** | Celery workers handle heavy ML inference; poll job status via REST API |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API** | FastAPI · Pydantic v2 · Uvicorn |
| **Database** | PostgreSQL 16 · SQLAlchemy 2.0 (async) · Alembic |
| **Task Queue** | Celery 5 · Redis 7 |
| **ML** | PyTorch 2.x · torchvision · timm · Swin Transformer |
| **Auth** | JWT (python-jose) · bcrypt · ECDSA (cryptography) |
| **Frontend** | React · TypeScript · Vite *(planned)* |
| **Infra** | Docker Compose · GitHub Actions |

---

## Repository Structure

```
ai-watermark-recovery/
├── pyproject.toml                 # Python 3.11+, all dependencies
├── Dockerfile                     # Multi-stage build (api + worker targets)
├── docker-compose.yml             # Postgres, Redis, API, Celery worker
├── alembic.ini                    # Alembic migration config
├── .env                      # Environment configuration (copy from .env.example and edit)
│
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI entry-point, lifespan, CORS
│   │   ├── worker.py              # Celery app factory
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic Settings (env-driven)
│   │   │   ├── database.py        # Async SQLAlchemy engine & session
│   │   │   └── security.py        # Password hashing & JWT helpers
│   │   ├── models/
│   │   │   ├── user.py            # User ORM model
│   │   │   ├── image.py           # Image metadata ORM model
│   │   │   └── job.py             # Job ORM model (Celery task tracking)
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
│   ├── ml/                        # ML modules (encoder, decoder, IGRM, ASL, …)
│   └── training/                  # Training scripts & configs
│
├── alembic/
│   ├── env.py                     # Async-aware Alembic environment
│   ├── script.py.mako             # Migration template
│   └── versions/                  # Auto-generated migrations
│
├── frontend/                      # React + TypeScript + Vite (planned)
├── datasets/                      # Dataset download scripts & data
├── notebooks/                     # Jupyter experiments
├── tests/
│   └── test_health.py             # Smoke test
├── .github/workflows/             # CI/CD pipelines
├── copilot-instructions.md        # GitHub Copilot domain context
├── DEVELOPMENT_PLAN.md            # Full 9-phase development roadmap
└── LICENSE                        # MIT
```

---

## Getting Started

### Prerequisites

- **Python 3.11+**
- **Docker** & **Docker Compose** (recommended for Postgres + Redis)
- **Git**

### 1. Clone & configure

```bash
git clone https://github.com/Hell-Phoenix/ai-watermark-recovery.git
cd ai-watermark-recovery
cp .env.example .env          # copy template and edit with Docker service names
```

### 2a. Run with Docker (recommended)

```bash
docker compose up --build
```

This starts **PostgreSQL**, **Redis**, the **FastAPI** server (`http://localhost:8000`), and a **Celery worker** — all wired together.

### 2b. Run locally (without Docker)

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -e ".[dev]"

# Make sure Postgres & Redis are running, then:
uvicorn backend.app.main:app --reload
```

Start the Celery worker in a separate terminal:

```bash
celery -A backend.app.worker:celery_app worker --loglevel=info
```

### 3. Run database migrations

```bash
alembic upgrade head
```

> On first local run, tables are auto-created by the FastAPI lifespan hook. Use Alembic for subsequent schema changes.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/images/` | Upload an image |
| `GET` | `/api/v1/images/` | List uploaded images |
| `GET` | `/api/v1/images/{id}` | Get image metadata |
| `POST` | `/api/v1/jobs/` | Submit an async processing job |
| `GET` | `/api/v1/jobs/{id}` | Poll job status |
| `GET` | `/api/v1/jobs/` | List all jobs |

Interactive API docs are available at **`http://localhost:8000/docs`** (Swagger UI) and **`/redoc`**.

### Job Types

| Job Type | Description |
|---|---|
| `embed_watermark` | Embed a watermark payload into an image |
| `extract_watermark` | Extract watermark bits from an image |
| `recover_watermark` | Run IGRM-based recovery on a degraded image |
| `detect_forgery` | Check for transplantation attacks *(planned)* |

### Example: submit a job

```bash
# Upload an image
curl -X POST http://localhost:8000/api/v1/images/ \
  -F "file=@photo.png"

# Submit a watermark embedding job (use the image ID from the upload response)
curl -X POST http://localhost:8000/api/v1/jobs/ \
  -H "Content-Type: application/json" \
  -d '{"image_id": "<uuid>", "job_type": "embed_watermark"}'

# Poll job status
curl http://localhost:8000/api/v1/jobs/<job-uuid>
```

---

## Development

### Install dev dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
pytest
```

### Lint & type-check

```bash
ruff check backend/
mypy backend/
```

### Generate a new Alembic migration

```bash
alembic revision --autogenerate -m "describe the change"
alembic upgrade head
```

---

## Environment Variables

All settings are loaded from a `.env` file. Key variables:

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` | Async Postgres connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection (use `redis` service name in Docker) |
| `CELERY_BROKER_URL` | `redis://redis:6379/1` | Celery broker (use `redis` service name in Docker) |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/2` | Celery result store (use `redis` service name in Docker) |
| `SECRET_KEY` | `CHANGE-ME-in-production` | JWT signing key |
| `UPLOAD_DIR` | `./uploads` | Image upload directory |
| `DEBUG` | `true` | Enable debug mode |

---

## Target Performance (Research Paper Goals)

| Attack | Current SOTA | Target | Key Component |
|---|---|---|---|
| Clean detection | 90–96% TPR | >99.9% | BCH + ECC + crypto binding |
| Diffusion regeneration | 0–10% survival | >95% survival | Dual-stage latent + WIND |
| Extreme crop (<10%) | Fails | >90% accuracy | ADN + IGRM |
| JPEG QF = 10 | Signal loss | BER < 0.05 | JSNet + 16-bit BCH |
| Forgery / transplantation | Misattribution | 0% success | ViT perceptual binding |

---

## Roadmap

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the full 9-phase build plan, including:

- **Phase 1** — Core encoder-decoder watermarking network (Swin Transformer)
- **Phase 2** — Attack Simulation Layer with differentiable JPEG
- **Phase 3** — Inverse Generative Restoration Module (IGRM)
- **Phase 4** — Dual-domain latent + pixel watermarking
- **Phase 5** — Cryptographic binding (ViT + ECDSA)
- **Phase 6** — FastAPI backend & async pipeline ✅
- **Phase 7** — React frontend dashboard
- **Phase 8** — Model training & dataset pipeline
- **Phase 9** — Evaluation, benchmarking & CI/CD

---

## License

[MIT](LICENSE)

---

*Based on: "Forensic Reconstruction of Hidden AI Watermarks: Deep Learning Solutions for Degradation Recovery" (2026)*