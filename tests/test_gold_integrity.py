"""A sealed gold set goes on claiming a size it no longer has.

A gold set is a list of object ids. Deleting the objects, which is what re-importing a session or rebuilding
a corpus does, takes the truth with them and leaves the row claiming its original count. This deployment
carries five such sets, one listing 171 objects of which zero survive, and on the quality page they were
indistinguishable from the two that are intact.

That is not cosmetic. A rotted set seeds no honeypots, so a project configured for quality measurement
silently gets none, and a quality sheet measured against it has no truth to compare with. Both failures are
silent, which is how five of them accumulated without anyone noticing.
"""

from __future__ import annotations

import uuid

import pytest

from db.models import Frame, GoldSet, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.analytics.quality import list_gold_sets

pytestmark = pytest.mark.db


async def _seal(db, *, alive: int, dead: int) -> str:
    """A gold set of `alive` objects that exist and `dead` ids that no longer do."""
    sess = DbSession(session_id=uuid.uuid4(), vehicle_id="GOLD-ROT", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test")
    db.add(sess)
    frame_id = uuid.uuid4()
    db.add(Frame(frame_id=frame_id, session_id=sess.session_id, ts_ns=1, cam_id="cam_f",
                 img_uri="s3://labeloxav/x.jpg", width=1920, height=1080))
    await db.flush()
    ids = []
    for _ in range(alive):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=frame_id, class_id=1, bbox=[1, 1, 9, 9],
                      conf=0.9, source="human", state="accepted"))
        ids.append(str(oid))
    ids += [str(uuid.uuid4()) for _ in range(dead)]
    gid = f"gold-{uuid.uuid4().hex[:16]}"
    db.add(GoldSet(gold_id=gid, name=f"rot-{gid[-6:]}", object_ids=ids, n_objects=len(ids),
                   n_frames=1, ontology_version="test"))
    await db.commit()
    return gid


async def _row(gid: str) -> dict:
    return next(g for g in await list_gold_sets() if g["gold_id"] == gid)


class TestGoldIntegrityIsReported:
    async def test_an_intact_set_is_usable_and_missing_nothing(self):
        async with get_sessionmaker()() as db:
            gid = await _seal(db, alive=3, dead=0)
        g = await _row(gid)
        assert g["usable"] is True and g["n_alive"] == 3 and g["n_missing"] == 0

    async def test_a_wholly_rotted_set_is_reported_unusable(self):
        """The fleet-v1 case: 171 listed, zero alive, and it looked healthy on the page."""
        async with get_sessionmaker()() as db:
            gid = await _seal(db, alive=0, dead=5)
        g = await _row(gid)
        assert g["usable"] is False and g["n_alive"] == 0 and g["n_missing"] == 5

    async def test_a_partly_rotted_set_reports_how_much_is_left(self):
        """The verify-mq1-blr case: 400 sealed, 47 surviving. Usable, but not what it claims."""
        async with get_sessionmaker()() as db:
            gid = await _seal(db, alive=2, dead=6)
        g = await _row(gid)
        assert g["usable"] is True and g["n_alive"] == 2 and g["n_missing"] == 6

    async def test_the_sealed_count_is_still_reported_unchanged(self):
        # `n_objects` is a historical fact about the seal and must not be quietly rewritten to the survivors.
        async with get_sessionmaker()() as db:
            gid = await _seal(db, alive=1, dead=4)
        g = await _row(gid)
        assert g["n_objects"] == 5

    async def test_an_empty_set_does_not_divide_by_anything(self):
        async with get_sessionmaker()() as db:
            gid = await _seal(db, alive=0, dead=0)
        g = await _row(gid)
        assert g["n_alive"] == 0 and g["n_missing"] == 0 and g["usable"] is False


class TestHoneypotSeedingSaysWhyItDidNothing:
    async def test_seeding_from_a_rotted_set_returns_zero_and_says_so(self, capsys):
        """It returned 0 in silence, so a project configured for quality measurement looked exactly like one
        that was not."""
        from db.models import LabelJob, LabelProject, LabelTask
        from services.labelops.quality import seed_honeypots

        async with get_sessionmaker()() as db:
            gid = await _seal(db, alive=0, dead=5)
            project = LabelProject(name=f"rot-{uuid.uuid4().hex[:8]}", honeypot_frac=0.2, gold_id=gid)
            db.add(project)
            await db.flush()
            task = LabelTask(project_id=project.project_id, name="t", predicate={})
            db.add(task)
            await db.flush()
            job = LabelJob(task_id=task.task_id, frame_ids=[str(uuid.uuid4())],
                           stage="annotation", state="new")
            db.add(job)
            await db.flush()

            n = await seed_honeypots(db, job, project)
            assert n == 0

        # structlog writes to stdout rather than through the logging module, so the record is read there.
        out = capsys.readouterr().out
        assert "honeypot_gold_rotted" in out
        assert gid in out, "the message has to name the gold set, or an operator cannot act on it"
