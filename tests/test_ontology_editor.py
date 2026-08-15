"""The ontology could be added to and never repaired.

`merge_class`, `revert_merge`, `rename_in_sidecar` and `retire_from_sidecar` have existed in
`services/agent/ontology_merge.py` and been reachable from nothing. The only ontology write in the whole
application was minting a class, and that route had no auth on it at all, so anybody the API let in could
add to the vocabulary every subsequent label is drawn from and nobody could take it back.

The refusals are the interesting part. Retiring a class objects still carry would leave them on a name
nothing offers, which is how a corpus ends up with labels no picker can select and no reviewer can correct,
and merging into a class that does not exist would strand them entirely.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns, seconds_to_ns
from db.models import AgentRun, Frame, Object, OntologyClass, OntologyVersion
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.agent.ontology_merge import KIND, MergeError, merge_class, revert_merge
from services.autolabel.ontology import get_ontology


async def _two_classes_with_objects(db, n_a: int = 3):
    """Two real ontology classes, with objects on the first."""
    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    a = onto.by_name("rider").id
    b = onto.by_name("pedestrian").id

    ts = now_ns()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="ONT-1", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(5),
                     city="BLR", sensors={}, ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                 img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    oids = []
    for i in range(n_a):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=a, bbox=[float(i), 0.0, 10.0, 10.0],
                      conf=0.5, source="fused", state="review", attrs={}, provenance={}, version=1))
        oids.append(oid)
    await db.flush()
    await db.commit()
    return a, b, oids


class TestMerging:
    async def test_every_object_moves_and_the_run_can_undo_it(self):
        async with get_sessionmaker()() as db:
            a, b, oids = await _two_classes_with_objects(db)
            out = await merge_class(db, from_id=a, to_id=b, created_by=None)

            moved = (await db.execute(
                select(Object.class_id).where(Object.object_id.in_(oids)))).scalars().all()
            assert set(moved) == {b}, "an object was left on the class that was merged away"

            run = await db.get(AgentRun, uuid.UUID(out["run_id"]))
            assert run.kind == KIND
            await revert_merge(db, run)

            back = (await db.execute(
                select(Object.class_id).where(Object.object_id.in_(oids)))).scalars().all()
        assert set(back) == {a}, "the undo did not put them back"

    async def test_merging_into_a_class_that_does_not_exist_is_refused(self):
        """The objects would have nowhere to go, and a foreign key would fail after the fact."""
        async with get_sessionmaker()() as db:
            a, _b, _o = await _two_classes_with_objects(db)
            with pytest.raises(MergeError, match="nowhere to go"):
                await merge_class(db, from_id=a, to_id=999_999)

    async def test_a_class_cannot_be_merged_into_itself(self):
        async with get_sessionmaker()() as db:
            a, _b, _o = await _two_classes_with_objects(db)
            with pytest.raises(MergeError, match="into itself"):
                await merge_class(db, from_id=a, to_id=a)


class TestTheRoutesThatDidNotExist:
    def test_the_editor_routes_are_reachable(self):
        """All four service functions existed and none had a route, so the vocabulary was write-only."""
        from services.api.routers import meta

        paths = {getattr(r, "path", "") for r in meta.router.routes}
        for p in ("/ontology/classes/merge", "/ontology/classes/rename",
                  "/ontology/classes/retire", "/ontology/merges/{run_id}/revert"):
            assert p in paths, f"{p} is still unreachable"

    def test_they_are_admin_gated_and_minting_is_reviewer_gated(self):
        """Minting a class changes the vocabulary every later label is drawn from; merging rewrites the
        class of every object carrying it. Neither had any floor at all."""
        from services.api.routers import meta

        by_path = {getattr(r, "path", ""): r for r in meta.router.routes}
        for p in ("/ontology/classes/merge", "/ontology/classes/rename", "/ontology/classes/retire"):
            assert by_path[p].dependencies, f"{p} has no role floor"
        assert by_path["/ontology/classes"].dependencies, "anyone could mint a class"


class TestRetiring:
    async def test_a_class_objects_still_carry_cannot_be_retired(self):
        """It would leave them on a name nothing offers: no picker can select it and no reviewer can
        correct it."""
        from services.api.routers.meta import retire_classes

        async with get_sessionmaker()() as db:
            a, _b, _o = await _two_classes_with_objects(db)
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc:
                await retire_classes([a], db=db)
        assert exc.value.status_code == 400
        assert "merge them first" in str(exc.value.detail)

    async def test_retiring_leaves_the_database_row_alone(self):
        """`prediction` and `eval_patch` hold immutable history pointing at the class, so deleting the row
        would either fail on a foreign key or take that history with it."""
        from services.agent.ontology_merge import retire_from_sidecar

        async with get_sessionmaker()() as db:
            _a, b, _o = await _two_classes_with_objects(db)
            retire_from_sidecar({b})
            still_there = await db.get(OntologyClass, b)
        assert still_there is not None
