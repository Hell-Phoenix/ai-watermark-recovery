"""Pydantic schemas for watermark detection & embedding results.

Implements the Phase 6 response contract from DEVELOPMENT_PLAN.md:

.. code-block:: json

    {
      "payload": "hex string",
      "confidence": 0.97,
      "attack_type": "JPEG_QF_10 | CROP_95 | D2RA | GUID_DIFFUSION | CLEAN",
      "tamper_mask": "base64 PNG",
      "latent_layer_intact": true,
      "pixel_layer_intact": false,
      "forgery_detected": false
    }
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Attack-type enum
# ---------------------------------------------------------------------------

class AttackType(StrEnum):
    """Recognised attack / distortion types detected by the pipeline."""

    CLEAN = "CLEAN"
    JPEG_QF_5 = "JPEG_QF_5"
    JPEG_QF_10 = "JPEG_QF_10"
    JPEG_QF_20 = "JPEG_QF_20"
    JPEG_QF_30 = "JPEG_QF_30"
    CROP_50 = "CROP_50"
    CROP_75 = "CROP_75"
    CROP_90 = "CROP_90"
    CROP_95 = "CROP_95"
    D2RA = "D2RA"
    DAWN = "DAWN"
    GUID_DIFFUSION = "GUID_DIFFUSION"
    COMBINED = "COMBINED"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Embed request / response
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    """Body for ``POST /embed``."""

    image_id: uuid.UUID = Field(..., description="UUID of the previously-uploaded image")
    payload_hex: str = Field(
        default="000000000000",
        min_length=1,
        max_length=12,
        pattern=r"^[0-9a-fA-F]+$",
        description="Hex-encoded watermark payload (up to 48 bits = 12 hex chars)",
    )
    sign: bool = Field(
        default=False,
        description="If true, also ECDSA-sign the image content hash and embed the signature.",
    )


class EmbedResult(BaseModel):
    """Serialised result of a completed embed job."""

    watermarked_image_path: str = Field(..., description="Server path to the watermarked image")
    payload_hex: str
    signature_hex: str | None = Field(
        None, description="Compact ECDSA signature hex (128 chars = 64 bytes)"
    )
    psnr_db: float | None = Field(None, description="PSNR between cover and watermarked (dB)")


class EmbedResponse(BaseModel):
    """Returned by ``POST /embed`` — job accepted."""

    job_id: uuid.UUID
    status: str = "pending"
    message: str = "Watermark embedding job enqueued"


# ---------------------------------------------------------------------------
# Detect request / response
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    """Body for ``POST /detect``."""

    image_id: uuid.UUID = Field(..., description="UUID of the image to analyse")
    verify_signature: bool = Field(
        default=False,
        description="If true, also verify ECDSA content binding (requires public key on file).",
    )


class DetectResult(BaseModel):
    """Full watermark detection result — matches the Phase 6 JSON spec."""

    payload: str = Field(..., description="Recovered watermark payload as hex string")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregate confidence score")
    attack_type: AttackType = Field(
        default=AttackType.UNKNOWN,
        description="Classified attack / distortion type",
    )
    tamper_mask: str | None = Field(
        None,
        description="Base64-encoded PNG of per-pixel tamper heatmap",
    )
    latent_layer_intact: bool = Field(
        ..., description="True if the latent (WIND) watermark layer survived"
    )
    pixel_layer_intact: bool = Field(
        ..., description="True if the pixel (iIWN) watermark layer survived"
    )
    forgery_detected: bool = Field(
        ..., description="True if transplantation / ECDSA verification failed"
    )
    bit_error_rate: float | None = Field(
        None, ge=0.0, le=1.0, description="Estimated BER of the extracted payload"
    )
    ecdsa_valid: bool | None = Field(
        None, description="ECDSA signature verification result (if requested)"
    )


class DetectResponse(BaseModel):
    """Returned by ``POST /detect`` — job accepted."""

    job_id: uuid.UUID
    status: str = "pending"
    message: str = "Watermark detection job enqueued"


# ---------------------------------------------------------------------------
# Job polling — enriched version of the base JobOut
# ---------------------------------------------------------------------------

class JobStatusResponse(BaseModel):
    """Returned by ``GET /job/{id}``."""

    id: uuid.UUID
    job_type: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None
    error_message: str | None = None

    # Populated once the job reaches SUCCESS:
    result: DetectResult | EmbedResult | None = None

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Audit log entry
# ---------------------------------------------------------------------------

class AuditEntry(BaseModel):
    """One row from the forensic audit trail."""

    job_id: uuid.UUID
    job_type: str
    status: str
    created_at: datetime
    finished_at: datetime | None = None


class AuditResponse(BaseModel):
    """Returned by ``GET /audit/{image_hash}``."""

    image_hash: str
    entries: list[AuditEntry]
