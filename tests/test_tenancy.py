"""The corpus had no tenant boundary anywhere.

Five tables carried `project_id` and all five sat in the labelling spine. The corpus spine, Session -> Frame
-> Object, carried none, so nothing scoped the frames or the labels and no query could be shown to stay
inside one customer's data. `Session.pack_id` is the domain pack, not a tenant.

Only `session` gets the column: everything below it reaches its tenant through `session_id`, so three tables
inherit the scope from one indexed hop rather than from a backfill of 576,393 rows.

This is a boundary only where it is applied, and with one tenant it is a convention. What these tests pin is
the part that has to be right before it can become more than that: a scoped query never returns another
tenant's rows, and an unassigned row is nobody's rather than everybody's.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from core.timebase import now_ns, seconds_to_ns
from db.models import Frame, LabelProject, Object, OntologyClass, OntologyVersion
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.api.tenancy import (
    DEFAULT_PROJECT_NAME,
    assign_session,
    default_project_id,
    scope_frames,
    scope_objects,
    scope_sessions,
    unassigned_sessions,
)
from services.autolabel.ontology import get_ontology

pytestmark = pytest.mark.db


async def _tenant(db, name: str) -> LabelProject:
    p = LabelProject(name=f"{name}-{uuid.uuid4().hex[:8]}")
    db.add(p)
    await db.flush()
    return p


async def _drive(db, project: LabelProject | None, *, n_objects: int = 2):
    """One session with a frame and some objects, in a project or in none."""
    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts = now_ns()
    sid, fid = uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="TEN-1", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(5),
                     city="BLR", sensors={}, ontology_version=onto.version,
                     project_id=project.project_id if project else None))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f",
                 img_uri=f"s3://missing/{fid}.jpg", width=1920, height=1080, quality=0.9, scene={}))
    await db.flush()
    oids = []
    for i in range(n_objects):
        oid = uuid.uuid4()
        db.add(Object(object_id=oid, frame_id=fid, class_id=onto.by_name("rider").id,
                      bbox=[float(i), 0.0, 10.0 + i, 10.0], conf=0.5, source="fused", state="review",
                      attrs={}, provenance={}, version=1))
        oids.append(oid)
    await db.flush()
    await db.commit()
    return sid, fid, oids


class TestScoping:
    async def test_one_tenant_cannot_see_another_tenants_sessions(self):
        async with get_sessionmaker()() as db:
            a, b = await _tenant(db, "acme"), await _tenant(db, "globex")
            sid_a, _f, _o = await _drive(db, a)
            sid_b, _f, _o = await _drive(db, b)

            seen = (await db.execute(
                scope_sessions(select(DbSession.session_id), a.project_id))).scalars().all()

        assert sid_a in seen
        assert sid_b not in seen

    async def test_frames_inherit_the_scope_through_their_session(self):
        """The whole reason only `session` carries the column."""
        async with get_sessionmaker()() as db:
            a, b = await _tenant(db, "acme"), await _tenant(db, "globex")
            _s, fid_a, _o = await _drive(db, a)
            _s, fid_b, _o = await _drive(db, b)

            seen = (await db.execute(
                scope_frames(select(Frame.frame_id), a.project_id))).scalars().all()

        assert fid_a in seen and fid_b not in seen

    async def test_objects_inherit_it_through_frame_and_session(self):
        async with get_sessionmaker()() as db:
            a, b = await _tenant(db, "acme"), await _tenant(db, "globex")
            _s, _f, oids_a = await _drive(db, a)
            _s, _f, oids_b = await _drive(db, b)

            seen = set((await db.execute(
                scope_objects(select(Object.object_id), a.project_id))).scalars().all())

        assert set(oids_a) <= seen
        assert not (set(oids_b) & seen), "another tenant's labels were visible"

    async def test_no_project_means_no_restriction(self):
        """The single-tenant deployment, which is every caller today. Stated rather than implied."""
        async with get_sessionmaker()() as db:
            a = await _tenant(db, "acme")
            sid_a, _f, _o = await _drive(db, a)
            sid_none, _f, _o = await _drive(db, None)

            seen = (await db.execute(
                scope_sessions(select(DbSession.session_id), None))).scalars().all()

        assert sid_a in seen and sid_none in seen


class TestTheUnassigned:
    async def test_a_session_with_no_project_is_nobodys_not_everybodys(self):
        """An unassigned row visible to every scoped caller would be a hole in the boundary shaped exactly
        like the rows nobody remembered to assign."""
        async with get_sessionmaker()() as db:
            a = await _tenant(db, "acme")
            sid_none, _f, _o = await _drive(db, None)

            seen = (await db.execute(
                scope_sessions(select(DbSession.session_id), a.project_id))).scalars().all()

        assert sid_none not in seen

    async def test_they_are_reported_rather_than_swept_up(self):
        """Assigning them automatically would hide the ingest path that is not setting a project."""
        async with get_sessionmaker()() as db:
            sid_none, _f, _o = await _drive(db, None)
            listed = await unassigned_sessions(db, limit=500)

        assert str(sid_none) in listed

    async def test_assigning_one_puts_it_where_a_scoped_caller_can_see_it(self):
        async with get_sessionmaker()() as db:
            a = await _tenant(db, "acme")
            sid, _f, _o = await _drive(db, None)
            out = await assign_session(db, sid, a.project_id)

            seen = (await db.execute(
                scope_sessions(select(DbSession.session_id), a.project_id))).scalars().all()

        assert out["project_id"] == str(a.project_id)
        assert sid in seen

    async def test_assigning_an_unknown_session_is_refused(self):
        async with get_sessionmaker()() as db:
            with pytest.raises(ValueError, match="not found"):
                await assign_session(db, uuid.uuid4())


class TestTheDefaultProject:
    """The suite empties the corpus once per session, which removes the `default` project the migration
    created, so these ensure it exists rather than depending on migration residue. That the backfill
    actually ran is a fact about a deployment, not about this database: on the live corpus it assigned
    377 of 377 sessions.
    """

    @staticmethod
    async def _ensure_default(db):
        pid = await default_project_id(db)
        if pid is None:
            db.add(LabelProject(name=DEFAULT_PROJECT_NAME))
            await db.commit()
            pid = await default_project_id(db)
        return pid

    async def test_it_is_reachable_by_name(self):
        """Everything that predates tenancy went into it, so it has to be found by name rather than by
        whoever happens to be first in the table."""
        async with get_sessionmaker()() as db:
            pid = await self._ensure_default(db)
        assert pid is not None

    async def test_a_session_assigned_to_it_is_visible_under_its_scope(self):
        """What the migration's backfill bought: everything that predates tenancy is in one project and a
        caller scoped to that project can see all of it. Asserted through the same scope helper the
        application uses rather than by reading the column, because the column being right and the scope
        reading it are two different claims."""
        async with get_sessionmaker()() as db:
            pid = await self._ensure_default(db)
            sid, _f, _o = await _drive(db, None)
            await assign_session(db, sid)          # no project given: it goes to the default

            seen = (await db.execute(
                scope_sessions(select(DbSession.session_id), pid))).scalars().all()

        assert pid is not None
        assert sid in seen
