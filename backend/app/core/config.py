"""Application-wide settings loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # repo root


class Settings(BaseSettings):
    """Central configuration — values come from env vars or a `.env` file."""

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "AI Watermark Recovery"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # --- Database (PostgreSQL) ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/watermark_db"
    database_url_sync: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/watermark_db"
    db_echo: bool = False

    # --- Redis ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Celery ---
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # --- Auth / JWT ---
    secret_key: str = "CHANGE-ME-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60

    # --- Storage ---
    upload_dir: Path = BASE_DIR / "uploads"
    max_upload_size_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings singleton."""
    return Settings()
