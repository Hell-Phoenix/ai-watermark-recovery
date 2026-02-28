FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for psycopg2, Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libpq-dev libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

COPY . .

# ---------- API image ----------
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- Worker image ----------
FROM base AS worker
CMD ["celery", "-A", "backend.app.worker:celery_app", "worker", "--loglevel=info", "--concurrency=2"]
