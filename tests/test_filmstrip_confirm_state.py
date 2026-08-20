"""The filmstrip showed how much work a frame holds and nothing about whether it was done.

A reviewer working through a session sees twenty-five neighbouring frames under the canvas, each with its
object count. Nothing distinguished a frame they had confirmed ten minutes ago from one nobody had opened,
so stepping back to check something meant reopening finished frames, and there was no way to see where they
had stopped.

The count has to be per state rather than a boolean, because the partly confirmed frame is the interesting
one: it is where somebody was interrupted, and a boolean cannot show it.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _session_with_frames(db, onto, *, states: list[list[str]]) -> list[str]:
    """One frame per entry in `states`, each carrying objects in the states named."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    cid = next(c.id for c in onto.classes if c.name == "rider")
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()

    ts = now_ns()
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="STRIP-1", start_ts_ns=ts,
                     end_ts_ns=ts + seconds_to_ns(60), city="BLR", sensors={},
                     ontology_version=onto.version))
    await db.flush()

    frame_ids: list[str] = []
    for i, frame_states in enumerate(states):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + i, cam_id="cam_f",
                     img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        for j, state in enumerate(frame_states):
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=cid,
                          bbox=[10.0 + j, 10.0, 110.0 + j, 210.0], conf=0.5, source="fused",
                          state=state, attrs={}, provenance={}, version=1))
        await db.flush()
        frame_ids.append(str(fid))
    return frame_ids


@pytest.mark.asyncio
async def test_the_strip_says_how_much_of_each_frame_is_confirmed():
    from db.session import get_sessionmaker
    from services.api.routers.objects import frame_filmstrip
    from services.autolabel.ontology import get_ontology

    async with get_sessionmaker()() as db:
        # An untouched frame, a half-finished one, a finished one, and an empty one.
        ids = await _session_with_frames(db, get_ontology(), states=[
            ["review", "review", "review", "review"],
            ["accepted", "rejected", "review", "review"],
            ["accepted", "accepted"],
            [],
        ])
        out = await frame_filmstrip(ids[0], span=12, db=db)
        tiles = {t["frame_id"]: t for t in out["frames"]}

        assert tiles[ids[0]]["n_objects"] == 4 and tiles[ids[0]]["n_confirmed"] == 0
        assert tiles[ids[1]]["n_confirmed"] == 2, "a frame left halfway must read as halfway"
        assert tiles[ids[2]]["n_confirmed"] == tiles[ids[2]]["n_objects"] == 2
        assert tiles[ids[3]]["n_objects"] == 0 and tiles[ids[3]]["n_confirmed"] == 0
        await db.rollback()
