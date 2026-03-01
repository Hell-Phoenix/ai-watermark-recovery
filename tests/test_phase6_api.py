"""Phase 6 tests — FastAPI endpoints, Pydantic schemas, JWT auth, rate limiter.

Tests are organised into:
  1. Schema validation (Pydantic models)
  2. Auth & security helpers
  3. Rate limiter logic
  4. Route integration tests (ASGI TestClient with mocked DB / auth / Celery)
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pydantic schema tests (no network / DB required)
# ---------------------------------------------------------------------------
from backend.app.schemas.detection import (
    AttackType,
    AuditEntry,
    AuditResponse,
    DetectRequest,
    DetectResponse,
    DetectResult,
    EmbedRequest,
    EmbedResponse,
    EmbedResult,
    JobStatusResponse,
)
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError


class TestAttackTypeEnum:
    def test_values(self) -> None:
        assert AttackType.CLEAN == "CLEAN"
        assert AttackType.D2RA == "D2RA"
        assert AttackType.JPEG_QF_10 == "JPEG_QF_10"
        assert AttackType.CROP_95 == "CROP_95"
        assert AttackType.GUID_DIFFUSION == "GUID_DIFFUSION"

    def test_membership(self) -> None:
        assert "CLEAN" in [e.value for e in AttackType]
        assert "UNKNOWN" in [e.value for e in AttackType]


class TestEmbedRequest:
    def test_defaults(self) -> None:
        r = EmbedRequest(image_id=uuid.uuid4())
        assert r.payload_hex == "000000000000"
        assert r.sign is False

    def test_custom(self) -> None:
        uid = uuid.uuid4()
        r = EmbedRequest(image_id=uid, payload_hex="deadbeef", sign=True)
        assert r.image_id == uid
        assert r.payload_hex == "deadbeef"
        assert r.sign is True

    def test_hex_validation_rejects_non_hex(self) -> None:
        with pytest.raises(ValidationError):
            EmbedRequest(image_id=uuid.uuid4(), payload_hex="xyz123")

    def test_hex_max_length(self) -> None:
        # 12 chars is max (48 bits)
        r = EmbedRequest(image_id=uuid.uuid4(), payload_hex="a" * 12)
        assert len(r.payload_hex) == 12

    def test_hex_too_long(self) -> None:
        with pytest.raises(ValidationError):
            EmbedRequest(image_id=uuid.uuid4(), payload_hex="a" * 13)


class TestEmbedResult:
    def test_construction(self) -> None:
        r = EmbedResult(
            watermarked_image_path="/uploads/test.png",
            payload_hex="deadbeef",
            psnr_db=42.5,
        )
        assert r.watermarked_image_path == "/uploads/test.png"
        assert r.signature_hex is None
        assert r.psnr_db == 42.5


class TestEmbedResponse:
    def test_defaults(self) -> None:
        r = EmbedResponse(job_id=uuid.uuid4())
        assert r.status == "pending"
        assert "enqueued" in r.message.lower()


class TestDetectRequest:
    def test_defaults(self) -> None:
        r = DetectRequest(image_id=uuid.uuid4())
        assert r.verify_signature is False

    def test_with_signature(self) -> None:
        r = DetectRequest(image_id=uuid.uuid4(), verify_signature=True)
        assert r.verify_signature is True


class TestDetectResult:
    def test_full_construction(self) -> None:
        r = DetectResult(
            payload="deadbeef0000",
            confidence=0.97,
            attack_type=AttackType.JPEG_QF_10,
            tamper_mask="iVBORw0KGgoA==",
            latent_layer_intact=True,
            pixel_layer_intact=False,
            forgery_detected=False,
            bit_error_rate=0.02,
            ecdsa_valid=True,
        )
        assert r.confidence == 0.97
        assert r.attack_type == AttackType.JPEG_QF_10
        assert r.latent_layer_intact is True
        assert r.pixel_layer_intact is False
        assert r.forgery_detected is False
        assert r.ecdsa_valid is True

    def test_minimal(self) -> None:
        r = DetectResult(
            payload="000000000000",
            confidence=0.5,
            latent_layer_intact=True,
            pixel_layer_intact=True,
            forgery_detected=False,
        )
        assert r.attack_type == AttackType.UNKNOWN
        assert r.tamper_mask is None
        assert r.ecdsa_valid is None

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DetectResult(
                payload="00", confidence=1.5,
                latent_layer_intact=True, pixel_layer_intact=True,
                forgery_detected=False,
            )

    def test_json_roundtrip(self) -> None:
        r = DetectResult(
            payload="abcdef123456",
            confidence=0.85,
            attack_type=AttackType.D2RA,
            latent_layer_intact=False,
            pixel_layer_intact=True,
            forgery_detected=True,
            ecdsa_valid=False,
        )
        data = json.loads(r.model_dump_json())
        assert data["attack_type"] == "D2RA"
        assert data["forgery_detected"] is True
        r2 = DetectResult(**data)
        assert r2 == r


class TestDetectResponse:
    def test_defaults(self) -> None:
        r = DetectResponse(job_id=uuid.uuid4())
        assert r.status == "pending"


class TestJobStatusResponse:
    def test_with_result(self) -> None:
        embed = EmbedResult(
            watermarked_image_path="/test.png",
            payload_hex="aabb",
        )
        r = JobStatusResponse(
            id=uuid.uuid4(),
            job_type="embed_watermark",
            status="success",
            created_at=datetime.now(UTC),
            result=embed,
        )
        assert r.result is not None
        assert isinstance(r.result, EmbedResult)

    def test_pending_no_result(self) -> None:
        r = JobStatusResponse(
            id=uuid.uuid4(),
            job_type="extract_watermark",
            status="pending",
            created_at=datetime.now(UTC),
        )
        assert r.result is None


class TestAuditResponse:
    def test_construction(self) -> None:
        entries = [
            AuditEntry(
                job_id=uuid.uuid4(),
                job_type="embed_watermark",
                status="success",
                created_at=datetime.now(UTC),
            )
        ]
        r = AuditResponse(image_hash="abc123", entries=entries)
        assert len(r.entries) == 1


# ---------------------------------------------------------------------------
# Security / JWT tests
# ---------------------------------------------------------------------------

from backend.app.core.security import (
    create_access_token,
    decode_access_token,
    get_password_hash,
    verify_password,
)


class TestPasswordHashing:
    def test_hash_and_verify(self) -> None:
        pw = "supersecret123!"
        hashed = get_password_hash(pw)
        assert hashed != pw
        assert verify_password(pw, hashed) is True

    def test_wrong_password_fails(self) -> None:
        hashed = get_password_hash("correct")
        assert verify_password("wrong", hashed) is False


class TestJWT:
    def test_create_and_decode(self) -> None:
        user_id = str(uuid.uuid4())
        token = create_access_token({"sub": user_id})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == user_id

    def test_invalid_token(self) -> None:
        assert decode_access_token("garbage.token.value") is None

    def test_token_has_expiry(self) -> None:
        token = create_access_token({"sub": "test"})
        payload = decode_access_token(token)
        assert "exp" in payload


# ---------------------------------------------------------------------------
# Route integration tests (mocked DB / Celery)
# ---------------------------------------------------------------------------

# We mock the DB dependency and auth dependency so we don't need a real
# PostgreSQL instance for unit tests.

from backend.app.core.auth import require_auth
from backend.app.core.database import get_db
from backend.app.main import app
from backend.app.models.image import Image
from backend.app.models.job import Job, JobStatus, JobType
from backend.app.models.user import User

_TEST_USER_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_TEST_IMAGE_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
_TEST_JOB_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")

_fake_user = MagicMock(spec=User)
_fake_user.id = _TEST_USER_ID
_fake_user.email = "test@example.com"
_fake_user.is_active = True


def _make_fake_image() -> MagicMock:
    img = MagicMock(spec=Image)
    img.id = _TEST_IMAGE_ID
    img.owner_id = _TEST_USER_ID
    img.filepath = "/app/uploads/test.png"
    img.filename = "test.png"
    return img


def _make_fake_job(
    job_type: str = JobType.EMBED_WATERMARK,
    status: str = JobStatus.PENDING,
    result_path: str | None = None,
) -> MagicMock:
    job = MagicMock(spec=Job)
    job.id = _TEST_JOB_ID
    job.job_type = job_type
    job.status = status
    job.celery_task_id = "celery-task-id-123"
    job.image_id = _TEST_IMAGE_ID
    job.created_at = datetime.now(UTC)
    job.finished_at = None
    job.error_message = None
    job.result_path = result_path
    return job


# Overridden dependencies
async def _override_auth() -> User:
    return _fake_user


# We need a mock DB session that supports `.get()` and `.execute()` etc.
class _MockDBSession:
    def __init__(self) -> None:
        self._store: dict[tuple, object] = {}

    def add_entity(self, model_class: type, pk: uuid.UUID, obj: object) -> None:
        self._store[(model_class, pk)] = obj

    async def get(self, model_class: type, pk: uuid.UUID) -> object | None:
        return self._store.get((model_class, pk))

    async def execute(self, stmt):
        """Return a minimal result proxy for SELECT queries."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_result.scalars.return_value.first.return_value = None
        return mock_result

    def add(self, obj):
        pass

    async def flush(self) -> None:
        pass

    async def refresh(self, obj) -> None:
        # Give the job a UUID so the response is valid
        if hasattr(obj, "id") and obj.id is None:
            obj.id = uuid.uuid4()
        if hasattr(obj, "created_at") and obj.created_at is None:
            obj.created_at = datetime.now(UTC)
        if hasattr(obj, "is_active") and obj.is_active is None:
            obj.is_active = True

    async def commit(self) -> None:
        pass

    async def rollback(self) -> None:
        pass


_mock_db = _MockDBSession()
_mock_db.add_entity(Image, _TEST_IMAGE_ID, _make_fake_image())


async def _override_db():
    yield _mock_db


@pytest.fixture(autouse=True)
def _apply_overrides():
    """Override auth + DB dependencies for all integration tests."""
    app.dependency_overrides[require_auth] = _override_auth
    app.dependency_overrides[get_db] = _override_db
    # Reset the Redis singleton between tests to avoid event-loop conflicts
    import backend.app.core.auth as _auth_mod
    _auth_mod._redis_pool = None
    yield
    app.dependency_overrides.clear()
    _auth_mod._redis_pool = None


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Health check still works
# ---------------------------------------------------------------------------

class TestHealthWithOverrides:
    @pytest.mark.asyncio
    async def test_health(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /embed
# ---------------------------------------------------------------------------

class TestEmbedEndpoint:
    @pytest.mark.asyncio
    @patch("backend.app.routes.watermark.pipeline_embed")
    async def test_embed_accepted(self, mock_task: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.id = "celery-test-id"
        mock_task.delay.return_value = mock_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/embed",
                json={"image_id": str(_TEST_IMAGE_ID), "payload_hex": "abcdef"},
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "pending"

    @pytest.mark.asyncio
    @patch("backend.app.routes.watermark.pipeline_embed")
    async def test_embed_with_sign(self, mock_task: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.id = "celery-test-id"
        mock_task.delay.return_value = mock_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/embed",
                json={
                    "image_id": str(_TEST_IMAGE_ID),
                    "payload_hex": "aabb",
                    "sign": True,
                },
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 202
        # Verify the task was called with sign=True
        mock_task.delay.assert_called_once()
        call_args = mock_task.delay.call_args
        assert call_args[0][2] == "aabb"  # payload
        assert call_args[0][3] is True    # sign

    @pytest.mark.asyncio
    async def test_embed_image_not_found(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/embed",
                json={"image_id": str(uuid.uuid4())},
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 404

    @pytest.mark.asyncio
    async def test_embed_bad_hex(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/embed",
                json={"image_id": str(_TEST_IMAGE_ID), "payload_hex": "xyz"},
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# POST /detect
# ---------------------------------------------------------------------------

class TestDetectEndpoint:
    @pytest.mark.asyncio
    @patch("backend.app.routes.watermark.pipeline_detect")
    async def test_detect_accepted(self, mock_task: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.id = "celery-detect-id"
        mock_task.delay.return_value = mock_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/detect",
                json={"image_id": str(_TEST_IMAGE_ID)},
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] == "pending"

    @pytest.mark.asyncio
    @patch("backend.app.routes.watermark.pipeline_detect")
    async def test_detect_with_verify_sig(self, mock_task: MagicMock) -> None:
        mock_result = MagicMock()
        mock_result.id = "celery-detect-id"
        mock_task.delay.return_value = mock_result

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/detect",
                json={
                    "image_id": str(_TEST_IMAGE_ID),
                    "verify_signature": True,
                },
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 202
        call_args = mock_task.delay.call_args
        assert call_args[0][2] is True  # verify_signature

    @pytest.mark.asyncio
    async def test_detect_image_not_found(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/detect",
                json={"image_id": str(uuid.uuid4())},
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /job/{id}
# ---------------------------------------------------------------------------

class TestJobStatusEndpoint:
    @pytest.mark.asyncio
    async def test_job_pending(self) -> None:
        fake_job = _make_fake_job()
        _mock_db.add_entity(Job, _TEST_JOB_ID, fake_job)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(
                f"/api/v1/job/{_TEST_JOB_ID}",
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "pending"
        assert body["result"] is None

    @pytest.mark.asyncio
    async def test_job_success_with_embed_result(self, tmp_path: Path) -> None:
        # Write a fake result JSON
        result_data = {
            "watermarked_image_path": "/uploads/test.png",
            "payload_hex": "aabbcc",
            "signature_hex": None,
            "psnr_db": 41.5,
        }
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(result_data))

        fake_job = _make_fake_job(
            status=JobStatus.SUCCESS,
            result_path=str(result_file),
        )
        _mock_db.add_entity(Job, _TEST_JOB_ID, fake_job)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(
                f"/api/v1/job/{_TEST_JOB_ID}",
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["result"]["watermarked_image_path"] == "/uploads/test.png"

    @pytest.mark.asyncio
    async def test_job_success_with_detect_result(self, tmp_path: Path) -> None:
        result_data = {
            "payload": "deadbeef0000",
            "confidence": 0.93,
            "attack_type": "JPEG_QF_10",
            "tamper_mask": "base64data==",
            "latent_layer_intact": True,
            "pixel_layer_intact": False,
            "forgery_detected": False,
            "bit_error_rate": 0.03,
            "ecdsa_valid": None,
        }
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps(result_data))

        fake_job = _make_fake_job(
            job_type=JobType.EXTRACT_WATERMARK,
            status=JobStatus.SUCCESS,
            result_path=str(result_file),
        )
        _mock_db.add_entity(Job, _TEST_JOB_ID, fake_job)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(
                f"/api/v1/job/{_TEST_JOB_ID}",
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["result"]["attack_type"] == "JPEG_QF_10"
        assert body["result"]["payload"] == "deadbeef0000"
        assert body["result"]["latent_layer_intact"] is True
        assert body["result"]["pixel_layer_intact"] is False

    @pytest.mark.asyncio
    async def test_job_not_found(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(
                f"/api/v1/job/{uuid.uuid4()}",
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /audit/{image_hash}
# ---------------------------------------------------------------------------

class TestAuditEndpoint:
    @pytest.mark.asyncio
    async def test_audit_returns_entries(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get(
                "/api/v1/audit/sha256_abc123",
                headers={"Authorization": "Bearer fake-token"},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["image_hash"] == "sha256_abc123"
        assert isinstance(body["entries"], list)


# ---------------------------------------------------------------------------
# Auth route tests
# ---------------------------------------------------------------------------

class TestAuthRoutes:
    @pytest.mark.asyncio
    async def test_register_endpoint_exists(self) -> None:
        """Verify the register endpoint is wired."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            # This will fail because our mock DB doesn't support User creation
            # properly, but we verify the route exists (not 404/405)
            r = await c.post(
                "/api/v1/auth/register",
                json={
                    "email": "new@test.com",
                    "password": "pass123",
                    "full_name": "Test User",
                },
            )
        # Could be 201 (success) or 500 (mock limitation) — NOT 404/405
        assert r.status_code != 404
        assert r.status_code != 405

    @pytest.mark.asyncio
    async def test_login_endpoint_exists(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/auth/login",
                json={"email": "test@test.com", "password": "wrong"},
            )
        # 401 (auth failure) is expected — NOT 404
        assert r.status_code != 404


# ---------------------------------------------------------------------------
# Auth dependency tests
# ---------------------------------------------------------------------------

class TestRequireAuth:
    @pytest.mark.asyncio
    async def test_missing_token_returns_401(self) -> None:
        # Remove the auth override so the real dependency runs
        app.dependency_overrides.pop(require_auth, None)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/embed",
                json={"image_id": str(_TEST_IMAGE_ID)},
            )
        assert r.status_code == 401

        # Restore override for other tests
        app.dependency_overrides[require_auth] = _override_auth

    @pytest.mark.asyncio
    async def test_invalid_token_returns_401(self) -> None:
        app.dependency_overrides.pop(require_auth, None)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.post(
                "/api/v1/embed",
                json={"image_id": str(_TEST_IMAGE_ID)},
                headers={"Authorization": "Bearer invalid-garbage"},
            )
        assert r.status_code == 401

        app.dependency_overrides[require_auth] = _override_auth


# ---------------------------------------------------------------------------
# Rate limiter unit tests (no Redis required)
# ---------------------------------------------------------------------------

from backend.app.core.auth import rate_limit


class TestRateLimitFactory:
    def test_returns_callable(self) -> None:
        dep = rate_limit(max_requests=10, window_seconds=60)
        assert callable(dep)

    def test_custom_params(self) -> None:
        dep = rate_limit(max_requests=5, window_seconds=30)
        assert callable(dep)


# ---------------------------------------------------------------------------
# Pipeline task helpers (unit tests)
# ---------------------------------------------------------------------------

from backend.app.tasks.pipeline_tasks import (
    _bits_to_hex,
    _compute_psnr,
    _hex_to_bits,
)


class TestTaskHelpers:
    def test_hex_to_bits_shape(self) -> None:
        with patch("backend.ml.model_loader.DEVICE", new="cpu"):
            bits = _hex_to_bits("abcdef123456")
        assert bits.shape == (1, 48)

    def test_bits_to_hex_roundtrip(self) -> None:
        # 48 bits → hex → bits
        original_hex = "abcdef123456"
        with patch("backend.ml.model_loader.DEVICE", new="cpu"):
            bits = _hex_to_bits(original_hex)
        recovered = _bits_to_hex(bits.squeeze(0))
        assert recovered == original_hex

    def test_compute_psnr_identical(self) -> None:
        import torch
        t = torch.rand(1, 3, 64, 64)
        assert _compute_psnr(t, t) == 100.0

    def test_compute_psnr_different(self) -> None:
        import torch
        a = torch.zeros(1, 3, 64, 64)
        b = torch.ones(1, 3, 64, 64)
        psnr = _compute_psnr(a, b)
        assert psnr == 0.0  # mse = 1.0 → psnr = 0 dB


# ---------------------------------------------------------------------------
# OpenAPI schema smoke test
# ---------------------------------------------------------------------------

class TestOpenAPISchema:
    @pytest.mark.asyncio
    async def test_openapi_includes_endpoints(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        paths = schema["paths"]
        assert "/api/v1/embed" in paths
        assert "/api/v1/detect" in paths
        assert "/api/v1/job/{job_id}" in paths
        assert "/api/v1/audit/{image_hash}" in paths
        assert "/api/v1/auth/register" in paths
        assert "/api/v1/auth/login" in paths

    @pytest.mark.asyncio
    async def test_openapi_detect_result_schema(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/openapi.json")
        schema = r.json()
        schemas = schema["components"]["schemas"]
        assert "DetectResult" in schemas
        detect_props = schemas["DetectResult"]["properties"]
        assert "payload" in detect_props
        assert "confidence" in detect_props
        assert "attack_type" in detect_props
        assert "tamper_mask" in detect_props
        assert "latent_layer_intact" in detect_props
        assert "pixel_layer_intact" in detect_props
        assert "forgery_detected" in detect_props

    @pytest.mark.asyncio
    async def test_openapi_attack_type_enum(self) -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/openapi.json")
        schema = r.json()
        schemas = schema["components"]["schemas"]
        assert "AttackType" in schemas
        enum_vals = schemas["AttackType"]["enum"]
        assert "CLEAN" in enum_vals
        assert "D2RA" in enum_vals
        assert "JPEG_QF_10" in enum_vals
