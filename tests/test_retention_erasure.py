"""Retention is enforced and a subject can be erased.

Two halves of the same gap. `retention_status()` returned `{"action": "purge"}` for an expired record and its
docstring said "a retention sweep acts on this"; no sweep existed. And `retention_until` could not be set
through the API at all, since the request model had no field and the handler dropped the parameter, so every
record carried a null deadline and the function answered "retain" forever. The feature was computed,
documented, and unreachable.

There was also no erasure path at all. The only deletion in the system was a per-object endpoint; nothing
walked from a subject to its data, and nothing touched object storage, which database cascades do not reach.

These tests use a real database and a real object store, because the thing worth proving is that the rows and
the blobs are actually gone."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.timebase import now_ns

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    from core.config import get_settings
    try:
        import redis as redis_lib
        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


async def _seed(db, *, n_frames: int = 2, retention_until: datetime | None = None) -> uuid.UUID:
    """A session with real frames, real blobs, objects, and a PII audit."""
    import cv2

    from core.storage import get_object_store
    from db.models import ConsentRecord, Frame, Object, PiiAudit
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    store = get_object_store()
    store.ensure_bucket()
    onto = get_ontology()
    sid, ts = uuid.uuid4(), now_ns()
    db.add(DbSession(session_id=sid, vehicle_id="ERASE-01", start_ts_ns=ts, end_ts_ns=ts + 1,
                     city="BLR", sensors={}, ontology_version=onto.version))
    await db.flush()

    img = np.random.default_rng(3).integers(30, 220, size=(64, 64, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    for i in range(n_frames):
        fid = uuid.uuid4()
        uri = store.put_bytes(f"frames/{sid}/cam_f/{ts + i}.jpg", buf.tobytes(), "image/jpeg")
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f", img_uri=uri,
                     width=64, height=64, quality=0.9))
        await db.flush()
        db.add(Object(frame_id=fid, class_id=onto.by_name("sedan").id, bbox=[1, 1, 20, 20],
                      conf=0.9, attrs={}, source="fused", state="review"))
        db.add(PiiAudit(frame_id=fid, session_id=sid, n_faces=1, n_plates=0, regions=[],
                        method_version="test", ts_ns=ts + i))

    db.add(ConsentRecord(session_id=sid, consent_status="granted", legal_basis="contract",
                         retention_until=retention_until))
    await db.commit()
    return sid


@requires_infra
async def test_erasure_removes_rows_and_blobs():
    # Deleting annotations while the images remain is not erasure, it is a metadata edit that leaves the
    # personal data in place. This asserts the frames, the objects, the audits, and the blobs are all gone.
    from sqlalchemy import func, select

    from core.storage import get_object_store
    from db.models import Frame, Object, PiiAudit
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session

    async with get_sessionmaker()() as db:
        sid = await _seed(db, n_frames=2)
        uris = [u for (u,) in (await db.execute(
            select(Frame.img_uri).where(Frame.session_id == sid))).all()]
        assert len(uris) == 2

        res = await erase_session(db, sid, reason="test")
        assert res["erased"] is True
        assert res["certificate"]["frames"] == 2
        assert res["certificate"]["objects"] == 2
        assert res["certificate"]["pii_audits"] == 2
        assert res["certificate"]["blobs_deleted"] == 2

        assert await db.get(DbSession, sid) is None
        assert (await db.execute(select(func.count()).select_from(Frame)
                                 .where(Frame.session_id == sid))).scalar_one() == 0
        assert (await db.execute(select(func.count()).select_from(PiiAudit)
                                 .where(PiiAudit.session_id == sid))).scalar_one() == 0
        assert (await db.execute(select(func.count()).select_from(Object)
                                 .where(Object.frame_id.in_(
                                     select(Frame.frame_id).where(Frame.session_id == sid)
                                 )))).scalar_one() == 0

    store = get_object_store()
    for uri in uris:
        # The image itself is gone, not merely dereferenced. Asserting the read fails (rather than catching
        # any exception) is what distinguishes a deleted blob from one that simply cannot be parsed.
        try:
            store.get_bytes(uri)
            raise AssertionError(f"blob still readable after erasure: {uri}")
        except AssertionError:
            raise
        except Exception:  # noqa: BLE001 - the store raises a backend-specific not-found
            pass


@requires_infra
async def test_dry_run_reports_the_footprint_without_deleting():
    # The first thing anyone sensible does before an irreversible bulk delete is ask what it would touch.
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session

    async with get_sessionmaker()() as db:
        sid = await _seed(db, n_frames=3)
        res = await erase_session(db, sid, reason="test", dry_run=True)
        assert res["dry_run"] is True and res["frames"] == 3 and res["objects"] == 3
        assert await db.get(DbSession, sid) is not None      # nothing removed

        await erase_session(db, sid, reason="cleanup")       # tidy up


@requires_infra
async def test_erasing_an_unknown_session_is_an_error_not_a_silent_success():
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session

    async with get_sessionmaker()() as db:
        res = await erase_session(db, uuid.uuid4(), reason="test")
        assert "error" in res


@requires_infra
async def test_expired_sessions_lists_only_those_past_the_deadline():
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session, expired_sessions

    past = datetime.now(UTC) - timedelta(days=1)
    future = datetime.now(UTC) + timedelta(days=365)
    async with get_sessionmaker()() as db:
        due_sid = await _seed(db, n_frames=1, retention_until=past)
        keep_sid = await _seed(db, n_frames=1, retention_until=future)
        none_sid = await _seed(db, n_frames=1, retention_until=None)

        ids = {s["session_id"] for s in await expired_sessions(db)}
        assert str(due_sid) in ids
        assert str(keep_sid) not in ids
        # A record with no deadline is retained indefinitely; treating null as "expired" would erase
        # everything the moment a sweep first ran.
        assert str(none_sid) not in ids

        for s in (due_sid, keep_sid, none_sid):
            await erase_session(db, s, reason="cleanup")


@requires_infra
async def test_sweep_defaults_to_dry_run_and_erases_only_when_asked():
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session, run_retention_sweep

    past = datetime.now(UTC) - timedelta(days=1)
    async with get_sessionmaker()() as db:
        sid = await _seed(db, n_frames=1, retention_until=past)

        preview = await run_retention_sweep(db)
        assert preview["dry_run"] is True and preview["due"] >= 1
        assert await db.get(DbSession, sid) is not None       # a preview must not delete

        done = await run_retention_sweep(db, dry_run=False)
        assert done["dry_run"] is False and done["erased"] >= 1
        assert await db.get(DbSession, sid) is None

        # anything else the sweep caught is already gone; nothing to clean up for this session
        assert (await erase_session(db, sid, reason="cleanup")).get("error")


@requires_infra
async def test_erasure_is_audited_and_the_certificate_is_tamper_evident():
    # A deletion that leaves no evidence cannot be shown to a regulator, and re-running it would be
    # indistinguishable from never having run it.
    import hashlib
    import json

    from sqlalchemy import select

    from db.models import AuditDecision
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session

    async with get_sessionmaker()() as db:
        sid = await _seed(db, n_frames=1)
        res = await erase_session(db, sid, reason="subject request")
        cert = res["certificate"]

        body = {k: v for k, v in cert.items() if k != "digest"}
        assert cert["digest"] == hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

        rows = (await db.execute(
            select(AuditDecision).where(AuditDecision.subject == str(sid)))).scalars().all()
        assert any(r.decision == "erase_session" for r in rows)


@requires_infra
async def test_retention_until_survives_the_api_model():
    # The field the request model was missing. Without it the value never reached the record and the whole
    # retention path was dead.
    from services.api.routers.govern import ConsentIn

    deadline = datetime.now(UTC) + timedelta(days=30)
    payload = ConsentIn(session_id=uuid.uuid4(), consent_status="granted", retention_until=deadline)
    assert payload.retention_until == deadline


@requires_infra
@pytest.mark.asyncio
async def test_erasure_removes_every_stored_file_not_only_the_frame_images():
    """The frame images were the only blobs erasure deleted. The uploaded source video, which is the
    unredacted copy, every mask and every raster stayed in the bucket under a certificate saying the
    session was gone."""
    from sqlalchemy import select

    from core.storage import get_object_store
    from db.models import DrivableMask, Frame, Object
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session

    store = get_object_store()
    async with get_sessionmaker()() as db:
        sid = await _seed(db, n_frames=2)
        frames = (await db.execute(select(Frame).where(Frame.session_id == sid))).scalars().all()
        fid = frames[0].frame_id

        raw = store.put_bytes(f"uploads/{uuid.uuid4().hex}/drive.mp4", b"raw unredacted video", "video/mp4")
        sess = await db.get(DbSession, sid)
        sess.raw_uri = raw
        mask = store.put_bytes(f"masks/{sid}/{fid}/{uuid.uuid4()}.json", b"{}", "application/json")
        obj = (await db.execute(select(Object).where(Object.frame_id == fid))).scalars().first()
        obj.mask_uri = mask
        drivable = store.put_bytes(f"masks/drivable/{sid}/{fid}.json", b"{}", "application/json")
        db.add(DrivableMask(frame_id=fid, mask_uri=drivable, coverage=0.5, source="proposed",
                            model_version="test"))
        # A blob with no row left pointing at it, as a re-label leaves behind.
        orphan = store.put_bytes(f"masks/{sid}/{fid}/superseded.json", b"{}", "application/json")
        # Somebody else's data, which must survive.
        other = store.put_bytes(f"frames/{uuid.uuid4()}/cam_f/1.jpg", b"not ours", "image/jpeg")
        await db.commit()

        dry = await erase_session(db, sid, reason="test", dry_run=True)
        assert dry["blobs"] == 6, dry      # 2 frames, raw video, mask, drivable, orphan
        assert store.exists(raw), "a dry run deleted something"

        res = await erase_session(db, sid, reason="test")
        assert res["certificate"]["blobs_deleted"] == 6

    for name, uri in (("source video", raw), ("mask", mask), ("drivable raster", drivable),
                      ("orphaned mask", orphan), *[("frame image", f.img_uri) for f in frames]):
        assert not store.exists(uri), f"the {name} survived erasure: {uri}"
    assert store.exists(other), "erasure deleted another session's frame"
    store.remove(other)


@requires_infra
@pytest.mark.asyncio
async def test_source_media_shared_with_another_session_is_kept_until_the_last_one_goes():
    from core.storage import get_object_store
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.govern.retention import erase_session

    store = get_object_store()
    async with get_sessionmaker()() as db:
        a, b = await _seed(db, n_frames=1), await _seed(db, n_frames=1)
        raw = store.put_bytes(f"uploads/{uuid.uuid4().hex}/shared.mp4", b"video", "video/mp4")
        for sid in (a, b):
            (await db.get(DbSession, sid)).raw_uri = raw
        await db.commit()

        await erase_session(db, a, reason="test")
        assert store.exists(raw), "the second session's source video was deleted with the first"
        await erase_session(db, b, reason="test")
        assert not store.exists(raw)


@requires_infra
@pytest.mark.asyncio
async def test_erasure_never_deletes_shared_artifacts_or_files_outside_the_bucket():
    from sqlalchemy import select

    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.govern.retention import _SHARED_URI_COLUMNS, session_blob_keys

    assert "weights_uri" in _SHARED_URI_COLUMNS
    async with get_sessionmaker()() as db:
        sid = await _seed(db, n_frames=1)
        sess = await db.get(DbSession, sid)
        sess.raw_uri = "/home/someone/drive.mp4"                 # a local path the importer recorded
        sess.mcap_uri = "s3://another-bucket/drives/x.mcap"      # not ours to delete
        await db.commit()
        fids = [f for (f,) in (await db.execute(select(Frame.frame_id).where(Frame.session_id == sid))).all()]
        keys = await session_blob_keys(db, sid, fids)
    assert len(keys) == 1 and keys[0].startswith(f"frames/{sid}/")
    assert not any("drive.mp4" in k or "x.mcap" in k for k in keys)


def test_the_store_knows_what_is_its_own():
    from core.storage import get_object_store

    store = get_object_store()
    assert store.owns(store.uri("frames/a/b.jpg")) and store.owns("frames/a/b.jpg")
    assert not store.owns("s3://another-bucket/frames/a/b.jpg")
    assert not store.owns("/etc/passwd") and not store.owns("http://x/y") and not store.owns("s3://nokey")
