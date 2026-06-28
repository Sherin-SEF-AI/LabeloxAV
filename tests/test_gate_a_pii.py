"""Gate A (PII anonymization) tests. Unit tests prove blur is local + irreversible; an infra-gated
test proves a pii_audit row is written per ingested frame. No GPU; stub detectors (no weights)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from core.config import PiiSettings, Settings, get_settings
from services.anonymize.anonymizer import PiiAnonymizer


class _StubFace:
    available = True

    def __init__(self, box):
        self.box = box

    def detect(self, img):
        return [self.box]


class _StubPlate:
    available = False

    def detect(self, img):
        return []


def test_pii_enabled_by_default():
    assert Settings().pii.enabled is True


def test_blur_is_local_and_irreversible():
    img = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    orig = img.copy()
    region = (50.0, 50.0, 120.0, 120.0, 0.9)
    anon = PiiAnonymizer(PiiSettings(plate_mandatory=False), face_detector=_StubFace(region),
                         plate_detector=_StubPlate())

    res = anon.anonymize(img)
    assert res.n_faces == 1 and res.n_plates == 0
    assert res.method_version

    # inside the region: changed and smoothed (blur reduces variance) -> irreversible
    assert not np.array_equal(img[50:120, 50:120], orig[50:120, 50:120])
    assert img[50:120, 50:120].var() < orig[50:120, 50:120].var()
    # outside the region: byte-identical (locality)
    assert np.array_equal(img[:50, :], orig[:50, :])
    assert np.array_equal(img[120:, :], orig[120:, :])
    # audit region recorded
    assert res.regions[0]["type"] == "face"


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


@requires_infra
@pytest.mark.asyncio
async def test_ingest_writes_pii_audit_per_frame():
    from sqlalchemy import func, select

    from core.timebase import now_ns, seconds_to_ns
    from db.models import PiiAudit
    from db.session import get_sessionmaker
    from services.ingest.run import ingest
    from services.ingest.types import RawFrame

    rng = np.random.default_rng(3)
    start = now_ns()
    frames = [
        RawFrame(ts_ns=start + seconds_to_ns(i), cam_id="cam_f",
                 image_bgr=rng.integers(30, 220, (480, 640, 3), dtype=np.uint8))
        for i in range(3)
    ]
    anon = PiiAnonymizer(PiiSettings(plate_mandatory=False),
                         face_detector=_StubFace((100.0, 100.0, 180.0, 180.0, 0.95)),
                         plate_detector=_StubPlate())

    result = await ingest(
        frame_iter=iter(frames), vehicle="TIGOR-07", city="BLR", route="pii-test",
        raw_uri=None, mcap_uri=None, source_streams=["cam_f"], anonymizer=anon,
    )
    assert result["n_frames"] == 3
    assert result["pii"]["enabled"] is True
    assert result["pii"]["n_faces"] == 3

    sid = uuid.UUID(result["session_id"])
    async with get_sessionmaker()() as db:
        n = (await db.execute(select(func.count()).select_from(PiiAudit).where(PiiAudit.session_id == sid))).scalar_one()
        assert n == 3
        row = (await db.execute(select(PiiAudit).where(PiiAudit.session_id == sid).limit(1))).scalar_one()
        assert row.n_faces == 1 and row.method_version
