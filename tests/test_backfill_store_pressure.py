"""A backfill that dies on a busy store loses the run, and it is the run that is expensive.

The plate re-redaction over 41,718 frames ran for 65 minutes at 8 frames/second and stopped at 31,976 with
`SlowDownWrite ... reached max retries: 3` from MinIO. Nothing was corrupted, because the design leaves a
frame's old audit in place until the new image is written, so the remaining frames were still selected by
the stale-method query. But the operator lost an hour to a store that was asking to be asked again more
slowly, and botocore's own three retries all land inside a second, which is not what "slow down" means.

Two behaviours are load-bearing here and neither existed:

  1. A refused write must not produce an audit row. The audit is the record that says a frame is clean, so
     writing one for an image that never reached storage marks an unredacted plate as redacted. That is a
     DPDPA exposure created by the tool meant to remove one.
  2. One refused frame must not end the run, and a store that refuses everything must end it promptly rather
     than grinding through the remaining ten thousand, each at the cost of a decode and an inference.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest
from botocore.exceptions import ClientError
from sqlalchemy import func, select

from core.timebase import now_ns, seconds_to_ns
from db.models import Frame, OntologyVersion, PiiAudit
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.anonymize import backfill as bf

OLD, NEW = "test-pressure-v1", "test-pressure-v2"


def _throttle() -> ClientError:
    return ClientError({"Error": {"Code": "SlowDownWrite",
                                  "Message": "Resource requested is unwritable, please reduce your "
                                             "request rate"}}, "PutObject")


class FakeStore:
    """Accepts writes, refuses the first `refuse` of them, or refuses everything."""

    def __init__(self, refuse: int = 0, always: bool = False):
        self.refuse, self.always, self.calls, self.written = refuse, always, 0, []

    def parse_uri(self, uri: str):
        return "frames", uri.split("/", 3)[-1]

    def put_bytes(self, key: str, data: bytes, content_type: str = "") -> str:
        self.calls += 1
        if self.always or self.calls <= self.refuse:
            raise _throttle()
        self.written.append(key)
        return f"s3://frames/{key}"


class DeniedStore(FakeStore):
    def put_bytes(self, key: str, data: bytes, content_type: str = "") -> str:
        self.calls += 1
        raise ClientError({"Error": {"Code": "AccessDenied", "Message": "no"}}, "PutObject")


class FakeAnonymizer:
    """Finds two plates on every frame, so every frame needs a write."""

    method_version = NEW

    def anonymize(self, img):
        class Result:
            n_faces, n_plates = 0, 2
            regions = [{"kind": "plate", "bbox": [1, 1, 10, 10]}]
            method_version = NEW
        return Result()


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch):
    """The delays are real seconds in production and would make this module 40 seconds slower."""
    monkeypatch.setattr(bf, "_WRITE_RETRY_DELAYS_S", (0.0, 0.0, 0.0))
    monkeypatch.setattr(bf, "load_image_bgr", lambda store, uri: np.zeros((720, 1280, 3), np.uint8))


async def _seed(n: int = 1) -> tuple[uuid.UUID, list[Frame]]:
    """n frames in their own session, each already audited by a superseded method."""
    ts = now_ns()
    sid = uuid.uuid4()
    frames: list[Frame] = []
    async with get_sessionmaker()() as db:
        version = (await db.execute(select(OntologyVersion.version).limit(1))).scalar()
        if version is None:
            version = "pressure-onto"
            db.add(OntologyVersion(version=version, hierarchy_levels=3, attributes={}))
            await db.flush()
        db.add(DbSession(session_id=sid, vehicle_id="PRESSURE-1", start_ts_ns=ts,
                         end_ts_ns=ts + seconds_to_ns(1), city="BLR", sensors={},
                         ontology_version=version))
        await db.flush()
        for i in range(n):
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f",
                         img_uri=f"s3://frames/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            db.add(PiiAudit(frame_id=fid, session_id=sid, n_faces=0, n_plates=0, regions=[],
                            method_version=OLD, ts_ns=ts + i))
        await db.commit()
        frames = list((await db.execute(select(Frame).where(Frame.session_id == sid))).scalars().all())
    return sid, frames


class TestARefusedWrite:
    async def test_is_retried_rather_than_abandoned(self):
        """Two refusals then acceptance is the ordinary shape of a store under momentary pressure."""
        _sid, frames = await _seed()
        store = FakeStore(refuse=2)
        async with get_sessionmaker()() as db:
            res = await bf.backfill_frame(db, store, FakeAnonymizer(), frames[0], redo=True)
            await db.commit()
        assert res == {"n_faces": 0, "n_plates": 2}
        assert len(store.written) == 1

    async def test_never_records_the_frame_as_clean(self):
        """The audit says "this frame has been redacted". It may only exist if the image was stored."""
        _sid, frames = await _seed()
        async with get_sessionmaker()() as db:
            with pytest.raises(bf.StoreRefused):
                await bf.backfill_frame(db, FakeStore(always=True), FakeAnonymizer(), frames[0], redo=True)
        async with get_sessionmaker()() as db:
            audits = (await db.execute(select(PiiAudit)
                                       .where(PiiAudit.frame_id == frames[0].frame_id))).scalars().all()
        assert len(audits) == 1
        assert audits[0].method_version == OLD, "the old audit must survive so the frame is re-selected"

    async def test_a_real_fault_is_not_swallowed_as_pressure(self):
        """AccessDenied means the credentials are wrong, and waiting 40 seconds will not fix that."""
        _sid, frames = await _seed()
        store = DeniedStore()
        async with get_sessionmaker()() as db:
            with pytest.raises(ClientError):
                await bf.backfill_frame(db, store, FakeAnonymizer(), frames[0], redo=True)
        assert store.calls == 1, "a permanent error must not be retried at all"


class TestTheRun:
    async def test_counts_refusals_instead_of_dying_on_one(self, monkeypatch):
        """One frame the store would not take cost 31,976 already-processed frames."""
        sid, frames = await _seed(3)
        attempts_per_frame = len(bf._WRITE_RETRY_DELAYS_S) + 1

        class OneBadFrame(FakeStore):
            def put_bytes(self, key, data, content_type=""):
                self.calls += 1
                if self.calls <= attempts_per_frame:     # every attempt at the first frame
                    raise _throttle()
                self.written.append(key)
                return key

        store = OneBadFrame()
        monkeypatch.setattr(bf, "get_object_store", lambda: store)
        monkeypatch.setattr(bf, "get_anonymizer", lambda: FakeAnonymizer())

        totals = await bf.backfill_unaudited(limit=500, session_id=str(sid), stale_method=True)
        assert totals["refused"] == 1
        assert totals["frames"] == 2, "the frames after the refused one must still be processed"

        async with get_sessionmaker()() as db:
            done = (await db.execute(
                select(func.count()).select_from(PiiAudit)
                .where(PiiAudit.session_id == sid, PiiAudit.method_version == NEW))).scalar()
        assert done == 2
        assert len(frames) == 3

    async def test_stops_early_when_the_store_refuses_everything(self, monkeypatch):
        """A store that is down or full will not improve over ten thousand more attempts."""
        sid, _frames = await _seed(4)
        store = FakeStore(always=True)
        monkeypatch.setattr(bf, "get_object_store", lambda: store)
        monkeypatch.setattr(bf, "get_anonymizer", lambda: FakeAnonymizer())
        monkeypatch.setattr(bf, "_CONSECUTIVE_FAILURE_LIMIT", 2)

        totals = await bf.backfill_unaudited(limit=500, session_id=str(sid), stale_method=True)
        assert totals["frames"] == 0
        assert totals["refused"] == 2, "it stopped at the limit rather than working through the corpus"
