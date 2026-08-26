"""Frame-level context, on the column that already holds it.

The brief asked for a `frame_context` table. `Frame.scene` already carries frame-level facts, written at
ingest by the scene classifier with a confidence per axis, so a second table would be the parallel attribute
mechanism this repo forbids for objects. What `scene` lacked is provenance: a value a person set and a value
a classifier guessed were the same JSON, so a correction could not survive the next classifier pass and
nothing afterwards could tell that it had been overwritten.
"""

import uuid

import pytest

from core.timebase import now_ns, seconds_to_ns

pytestmark = pytest.mark.db


async def _frame(db):
    from db.models import Frame, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts, sid, fid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="CTX-1", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(1),
                     city="BLR", sensors={}, ontology_version=onto.version))
    await db.flush()
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/a.jpg",
                 width=1920, height=1080, quality=0.9,
                 scene={"weather": "overcast", "confidence_per_axis": {"weather": 0.4}}))
    await db.flush()
    return fid


def _user(role="reviewer"):
    from types import SimpleNamespace

    return SimpleNamespace(name=f"ctx-{uuid.uuid4().hex[:6]}", user_id=None, role=role)


class TestTheVocabulary:
    def test_the_pack_owns_it_and_a_domain_may_have_none(self):
        """A static-camera domain counting entries is complete without one, and the engine offers no context
        panel rather than inventing weather categories for a warehouse."""
        from packs.base import ContextSpec
        from services.domain import context_spec

        assert context_spec("av") is not None
        assert ContextSpec().validate({"anything": 1}) == ["unknown context attribute 'anything'"]

    def test_an_unknown_axis_and_an_unknown_value_are_both_refused(self):
        from services.domain import validate_context

        assert validate_context({"night_lighting": "sort of"}) != []
        assert validate_context({"not_an_axis": True}) != []
        assert validate_context({"night_lighting": "unlit", "dust": True}) == []

    def test_a_bool_axis_refuses_a_string(self):
        """`dust: "yes"` is the shape a hand-written client sends, and it must not become truthy."""
        from services.domain import validate_context

        assert validate_context({"dust": "yes"}) != []


class TestWriting:
    @pytest.mark.asyncio
    async def test_a_write_merges_rather_than_replacing(self):
        """Only the keys sent are touched: setting the weather must not clear somebody's note about the
        lighting, and must not drop the classifier's confidence block."""
        from db.models import Frame
        from db.session import get_sessionmaker
        from services.api.routers.objects import FrameContextIn, set_frame_context

        async with get_sessionmaker()() as db:
            fid = await _frame(db)
            await set_frame_context(str(fid), FrameContextIn(attrs={"haze": "dense"}), db, _user())
            f = await db.get(Frame, fid)
            assert f.scene["haze"] == "dense"
            assert f.scene["weather"] == "overcast", "an untouched key was dropped"
            assert f.scene["confidence_per_axis"] == {"weather": 0.4}
            await db.rollback()

    @pytest.mark.asyncio
    async def test_who_set_each_key_is_recorded(self):
        """Without this the next classifier pass overwrites a person's correction and nothing can tell."""
        from db.models import Frame
        from db.session import get_sessionmaker
        from services.api.routers.objects import FrameContextIn, set_frame_context

        async with get_sessionmaker()() as db:
            fid = await _frame(db)
            u = _user()
            await set_frame_context(str(fid), FrameContextIn(attrs={"haze": "light"}), db, u)
            f = await db.get(Frame, fid)
            assert f.scene_provenance["haze"]["by"] == u.name
            assert f.scene_provenance["haze"]["ts_ns"] > 0
            # The classifier's own key is NOT claimed by the person who set a different one.
            assert "weather" not in f.scene_provenance
            await db.rollback()

    @pytest.mark.asyncio
    async def test_an_invalid_write_changes_nothing(self):
        from fastapi import HTTPException

        from db.models import Frame
        from db.session import get_sessionmaker
        from services.api.routers.objects import FrameContextIn, set_frame_context

        async with get_sessionmaker()() as db:
            fid = await _frame(db)
            with pytest.raises(HTTPException) as err:
                await set_frame_context(str(fid), FrameContextIn(attrs={"haze": "soupy"}), db, _user())
            assert err.value.status_code == 400
            assert "context_errors" in err.value.detail
            f = await db.get(Frame, fid)
            assert "haze" not in (f.scene or {})
            await db.rollback()

    @pytest.mark.asyncio
    async def test_a_missing_frame_is_a_404_not_a_silent_no_op(self):
        from fastapi import HTTPException

        from db.session import get_sessionmaker
        from services.api.routers.objects import FrameContextIn, set_frame_context

        async with get_sessionmaker()() as db:
            with pytest.raises(HTTPException) as err:
                await set_frame_context(str(uuid.uuid4()), FrameContextIn(attrs={"dust": True}),
                                        db, _user())
            assert err.value.status_code == 404
            await db.rollback()


class TestReading:
    @pytest.mark.asyncio
    async def test_the_editor_is_told_the_context_and_who_set_it(self):
        from db.session import get_sessionmaker
        from services.api.routers.objects import get_frame

        async with get_sessionmaker()() as db:
            fid = await _frame(db)
            await db.commit()
        try:
            async with get_sessionmaker()() as db:
                out = await get_frame(str(fid), None, db, _user())
            assert out["context"]["weather"] == "overcast"
            assert out["context_provenance"] == {}
        finally:
            await _cleanup(fid)


async def _cleanup(fid):
    from sqlalchemy import select

    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    async with get_sessionmaker()() as db:
        f = await db.get(Frame, fid)
        if f is not None:
            sid = f.session_id
            await db.delete(f)
            await db.flush()
            for s in (await db.execute(select(DbSession).where(DbSession.session_id == sid))).scalars():
                await db.delete(s)
        await db.commit()
