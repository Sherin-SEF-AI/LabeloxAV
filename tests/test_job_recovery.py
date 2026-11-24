"""An import or export a restart interrupted is continued, not failed and not repeated.

Background jobs run inside the API process, so a restart ended them and the reaper failed the row ten
minutes later. Starting the import again made a second session holding the same frames. These pin the
three things that make continuing safe: a resumed import writes into the session it already has and
stores no frame twice; a job is picked up only when no live process holds its lock; and a job that keeps
taking the process down stops being picked up.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
import pytest
from sqlalchemy import func, select

from core.config import get_settings
from db.models import ExportJob, Frame, ImportJob, Object
from db.session import get_sessionmaker
from services.imports.records import ImportSpec

pytestmark = pytest.mark.db


def _infra_up() -> bool:
    try:
        import redis as redis_lib

        return bool(redis_lib.Redis.from_url(get_settings().redis.url).ping())
    except Exception:
        return False


requires_infra = pytest.mark.skipif(not _infra_up(), reason="infra not up (make up)")


def _coco_dir(root: Path, n: int = 5) -> Path:
    """A COCO folder of `n` real, distinct images, one box each, with no timestamps of their own."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    images, anns = [], []
    for i in range(n):
        name = f"img_{i:03d}.jpg"
        cv2.imwrite(str(root / name), rng.integers(30, 220, (120, 160, 3), dtype=np.uint8))
        images.append({"id": i + 1, "file_name": name, "width": 160, "height": 120})
        anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1, "bbox": [10, 10, 40, 30]})
    (root / "annotations.json").write_text(json.dumps(
        {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "car"}]}))
    return root


async def _import_job(source: Path, *, resumable: bool = True, attempts: int = 0,
                      status: str = "pending") -> uuid.UUID:
    jid = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(ImportJob(job_id=jid, status=status, format="coco", source_uri=str(source),
                         target_vehicle="RECOVER-01", progress=0.0, counts={},
                         spec={"resumable": True, "attempts": attempts, "options": {}} if resumable else {}))
        await db.commit()
    return jid


async def _frames_and_objects(session_id) -> tuple[int, int, int]:
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(Frame.cam_id, Frame.ts_ns)
                                 .where(Frame.session_id == session_id))).all()
        n_obj = (await db.execute(select(func.count()).select_from(Object)
                                  .join(Frame, Frame.frame_id == Object.frame_id)
                                  .where(Frame.session_id == session_id))).scalar()
    return len(rows), len(set(rows)), int(n_obj or 0)


@pytest.fixture
def no_pii(monkeypatch):
    # Redaction is not what is under test, and the detectors are slow to load.
    monkeypatch.setattr(get_settings().pii, "enabled", False)


@requires_infra
@pytest.mark.asyncio
async def test_an_interrupted_import_continues_into_its_own_session(tmp_path, no_pii, monkeypatch):
    from services.imports import run as imp

    src = _coco_dir(tmp_path / "coco", n=5)
    jid = await _import_job(src)
    spec = ImportSpec(format="coco", source_uri=str(src), target_vehicle="RECOVER-01")

    real = imp._prepare_frame
    seen = {"n": 0}

    def dies_on_the_fourth(*a, **k):
        seen["n"] += 1
        if seen["n"] == 4:
            raise RuntimeError("the process went away")
        return real(*a, **k)

    monkeypatch.setattr(imp, "_prepare_frame", dies_on_the_fourth)
    with pytest.raises(RuntimeError):
        await imp.import_dataset(spec, jid)

    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, jid)
        resume = (j.spec or {}).get("resume") or {}
    assert resume.get("session_id"), "the first run must record the session it created"
    sid = uuid.UUID(resume["session_id"])
    stored, _distinct, _objs = await _frames_and_objects(sid)
    assert 0 < stored < 5, "the interruption must have left a partial session for this test to mean anything"

    monkeypatch.setattr(imp, "_prepare_frame", real)
    result = await imp.import_dataset(spec, jid)

    assert result["session_id"] == str(sid), "a resumed import must not open a second session"
    stored, distinct, objs = await _frames_and_objects(sid)
    assert (stored, distinct, objs) == (5, 5, 5), "every frame once, each with its object"
    assert result["counts"]["frames"] == 5 and result["counts"]["objects"] == 5
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, jid)
        assert j.status == "done" and j.session_id == sid


@requires_infra
@pytest.mark.asyncio
async def test_a_resume_refuses_a_source_that_changed(tmp_path, no_pii, monkeypatch):
    from services.imports import run as imp

    src = _coco_dir(tmp_path / "coco", n=4)
    jid = await _import_job(src)
    spec = ImportSpec(format="coco", source_uri=str(src), target_vehicle="RECOVER-01")
    real = imp._prepare_frame
    calls = {"n": 0}

    def dies_on_the_third(*a, **k):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("gone")
        return real(*a, **k)

    monkeypatch.setattr(imp, "_prepare_frame", dies_on_the_third)
    with pytest.raises(RuntimeError):
        await imp.import_dataset(spec, jid)
    monkeypatch.setattr(imp, "_prepare_frame", real)

    # The same folder now lists its images in a different order.
    doc = json.loads((src / "annotations.json").read_text())
    doc["images"].reverse()
    (src / "annotations.json").write_text(json.dumps(doc))
    with pytest.raises(imp.ImportNotResumable, match="cannot be continued"):
        await imp.import_dataset(spec, jid)


@requires_infra
@pytest.mark.asyncio
async def test_a_failed_import_that_stored_nothing_leaves_no_session(tmp_path, no_pii, monkeypatch):
    from db.models import Session as DbSession
    from services.imports import run as imp

    src = _coco_dir(tmp_path / "coco", n=2)
    jid = await _import_job(src)

    def always_dies(*a, **k):
        raise RuntimeError("cannot decode")

    monkeypatch.setattr(imp, "_prepare_frame", always_dies)
    with pytest.raises(RuntimeError):
        await imp.import_dataset(ImportSpec(format="coco", source_uri=str(src),
                                            target_vehicle="RECOVER-EMPTY"), jid)
    async with get_sessionmaker()() as db:
        left = (await db.execute(select(func.count()).select_from(DbSession)
                                 .where(DbSession.vehicle_id == "RECOVER-EMPTY"))).scalar()
        j = await db.get(ImportJob, jid)
    assert left == 0
    assert not (j.spec or {}).get("resume")


@requires_infra
@pytest.mark.asyncio
async def test_recovery_continues_an_orphaned_import_and_counts_the_attempt(tmp_path, no_pii):
    from services.job_recovery import recover_interrupted_jobs

    src = _coco_dir(tmp_path / "coco", n=3)
    jid = await _import_job(src, status="running")
    out = await recover_interrupted_jobs()
    assert str(jid) in out["import"]
    for _ in range(100):
        async with get_sessionmaker()() as db:
            j = await db.get(ImportJob, jid)
            if j.status in ("done", "error"):
                break
        await asyncio.sleep(0.1)
    assert j.status == "done", j.error
    assert j.spec["attempts"] == 1
    assert (await _frames_and_objects(j.session_id))[0] == 3


@requires_infra
@pytest.mark.asyncio
async def test_recovery_leaves_a_job_that_a_live_process_is_running(tmp_path):
    from services.job_lock import job_lock
    from services.job_recovery import recover_interrupted_jobs

    jid = await _import_job(_coco_dir(tmp_path / "coco", n=1), status="running")
    async with job_lock(jid) as mine:
        assert mine
        out = await recover_interrupted_jobs()
    assert str(jid) not in out["import"]
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, jid)
        assert j.status == "running" and j.spec["attempts"] == 0
        j.status = "canceled"
        await db.commit()


@requires_infra
@pytest.mark.asyncio
async def test_recovery_gives_up_after_the_attempt_cap(tmp_path):
    from services.job_recovery import recover_interrupted_jobs

    cap = get_settings().jobs.max_attempts
    jid = await _import_job(_coco_dir(tmp_path / "coco", n=1), status="running", attempts=cap)
    out = await recover_interrupted_jobs()
    assert str(jid) not in out["import"]
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, jid)
    assert j.status == "error" and "not continued again" in j.error


@requires_infra
@pytest.mark.asyncio
async def test_recovery_ignores_rows_that_do_not_say_they_may_be_continued(tmp_path):
    from services.job_recovery import recover_interrupted_jobs

    jid = await _import_job(_coco_dir(tmp_path / "coco", n=1), resumable=False, status="running")
    eid = uuid.uuid4()
    async with get_sessionmaker()() as db:
        db.add(ExportJob(job_id=eid, name="unmarked", status="running", spec={"name": "unmarked"}))
        await db.commit()
    out = await recover_interrupted_jobs()
    assert str(jid) not in out["import"] and str(eid) not in out["export"]
    async with get_sessionmaker()() as db:
        assert (await db.get(ImportJob, jid)).status == "running"
        assert (await db.get(ExportJob, eid)).status == "running"
        for model, key in ((ImportJob, jid), (ExportJob, eid)):
            (await db.get(model, key)).status = "canceled"
        await db.commit()


@requires_infra
@pytest.mark.asyncio
async def test_recovery_can_be_switched_off(tmp_path, monkeypatch):
    from services.job_recovery import recover_interrupted_jobs

    monkeypatch.setattr(get_settings().jobs, "auto_resume", False)
    jid = await _import_job(_coco_dir(tmp_path / "coco", n=1), status="running")
    assert await recover_interrupted_jobs() == {"import": [], "export": []}
    async with get_sessionmaker()() as db:
        j = await db.get(ImportJob, jid)
        j.status = "canceled"
        await db.commit()


@requires_infra
@pytest.mark.asyncio
async def test_the_lock_is_free_again_once_its_holder_is_done():
    from services.job_lock import is_held, job_lock

    jid = uuid.uuid4()
    assert not await is_held(jid)
    async with job_lock(jid) as a:
        assert a
        async with job_lock(jid) as b:
            assert not b, "a second holder must be refused while the first is alive"
        assert await is_held(jid)
    assert not await is_held(jid)
