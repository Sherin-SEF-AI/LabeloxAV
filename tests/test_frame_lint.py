"""Checking a frame as it is saved, and refusing to fire rules whose data nobody is collecting.

The rules existed as a corpus sweep with one caller and no way to reach a single frame, and no machine
rule had ever created an `Issue`, so the editor's Issues panel was blind to every detector in the repo.

The interesting part is not the rules, it is the arming, and it exists because of a measurement. All three
of the relation and attribute rules fire on 100% of their scope today: `object_relationship` holds 98 rows
in the whole corpus, 56,410 riders have no `rider_of`, and `occupant_count` is set on 0 objects. "Rider
with no mount" is not 56,410 findings, it is one fact about what nobody has annotated yet, and a linter
that opened it as 56,410 issues would bury every real finding on its first run.

So a rule declares a precondition over the frame in front of it and stays dormant until the data it needs
is actually being collected there. Measured after the arming landed, over 150 frames and 2,878 objects:
every armed rule fires between 0.6% and 10.5% of objects, none is systemic, and all three unarmed rules
are dormant on 150 of 150 frames.

The tests below pin that behaviour from both sides: dormant when the data is absent, and firing when the
same frame does carry it.
"""

import uuid

import pytest

from core.timebase import now_ns

pytestmark = pytest.mark.db


async def _frame(db, onto, *, n_objects=0, width=1920, height=1080):
    from db.models import Frame, OntologyClass, OntologyVersion
    from db.models import Session as DbSession

    if await db.get(OntologyVersion, onto.version) is None:
        db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
        await db.flush()
        for c in onto.classes:
            db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                 india=c.india, map_to={}))
        await db.flush()
    ts, sid, fid = now_ns(), uuid.uuid4(), uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="LNT-1", start_ts_ns=ts, end_ts_ns=ts + 1,
                     city="BLR", sensors={}, ontology_version=onto.version))
    db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://lint/1.jpg",
                 width=width, height=height, quality=0.9, scene={}))
    await db.flush()
    return sid, fid


async def _obj(db, fid, onto, class_name, bbox, *, attrs=None, source="fused"):
    from db.models import Object

    o = Object(object_id=uuid.uuid4(), frame_id=fid, class_id=onto.by_name(class_name).id,
               bbox=list(bbox), conf=0.8, source=source, state="review",
               attrs=attrs or {}, provenance={}, version=1)
    db.add(o)
    await db.flush()
    return o


@pytest.mark.asyncio
async def test_a_relation_rule_stays_dormant_when_nobody_is_drawing_relations():
    """The claim the whole design rests on.

    A rider overlapping a motorcycle with no `rider_of` between them is the textbook finding, and on this
    corpus it is true of 56,410 riders because relations have never been annotated. Reporting it as a
    finding would be technically correct and completely useless.
    """
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        await _obj(db, fid, onto, "rider", [100, 100, 200, 300])
        await _obj(db, fid, onto, "motorcycle", [90, 200, 210, 380])

        res = await lint_frame(db, fid)
        rules = {f["rule"] for f in res["findings"]}
        assert "rider_without_mount" not in rules
        dormant = {d["rule"]: d["reason"] for d in res["dormant"]}
        assert "rider_without_mount" in dormant
        assert "coverage gap" in dormant["rider_without_mount"], (
            "a dormant rule has to say why, or it is indistinguishable from a rule that passed")
        await db.rollback()


@pytest.mark.asyncio
async def test_the_same_rule_fires_once_relations_are_being_drawn_on_that_frame():
    """The other half. Arming is a delay, not a permanent exemption: the moment somebody starts recording
    relations on a frame, a missing one there is a real omission."""
    from db.models import ObjectRelationship
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        rider = await _obj(db, fid, onto, "rider", [100, 100, 200, 300])
        bike = await _obj(db, fid, onto, "motorcycle", [90, 200, 210, 380])
        # A second, unrelated rider on the same frame, correctly linked. That confirmed relation is what
        # arms the rule: somebody is doing this work here.
        rider2 = await _obj(db, fid, onto, "rider", [600, 100, 700, 300])
        bike2 = await _obj(db, fid, onto, "motorcycle", [590, 200, 710, 380])
        db.add(ObjectRelationship(from_object_id=rider2.object_id, to_object_id=bike2.object_id,
                                  frame_id=fid, kind="rider_of", status="confirmed",
                                  source="human", evidence={}, conf=1.0))
        await db.flush()

        res = await lint_frame(db, fid)
        assert not any(d["rule"] == "rider_without_mount" for d in res["dormant"])
        hits = [f for f in res["findings"] if f["rule"] == "rider_without_mount"]
        assert [f["object_id"] for f in hits] == [str(rider.object_id)], (
            "only the unlinked rider should be reported; the linked one is exactly what armed the rule")
        assert str(bike.object_id) not in {f["object_id"] for f in hits}
        await db.rollback()


@pytest.mark.asyncio
async def test_a_rule_firing_on_most_of_the_frame_is_counted_not_listed():
    """The second guard, and the same one reanalyze.py already applies at 80%. A rule that objects to
    every object on a frame is reporting one fact, and the queue is the wrong place to say it a hundred
    times."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        # Eight specks, all below the minimum box size.
        for i in range(8):
            await _obj(db, fid, onto, "sedan", [i * 20.0, 0.0, i * 20.0 + 4.0, 4.0])

        res = await lint_frame(db, fid)
        assert res["systemic"].get("min_box_size") == 8
        assert not any(f["rule"] == "min_box_size" for f in res["findings"]), (
            "a rule that fired on every object should be counted once, not queued eight times")
        await db.rollback()


@pytest.mark.asyncio
async def test_a_few_bad_objects_are_still_listed_individually():
    """The guard must not swallow ordinary findings. Below the object floor, "fires on 80%" is two boxes
    and says nothing; above it, a minority of bad objects is exactly what a reviewer wants listed."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        for i in range(8):
            await _obj(db, fid, onto, "sedan", [i * 200.0, 0.0, i * 200.0 + 150.0, 150.0])
        await _obj(db, fid, onto, "sedan", [1800.0, 900.0, 1804.0, 904.0])   # one speck

        res = await lint_frame(db, fid)
        assert not res["systemic"]
        assert sum(1 for f in res["findings"] if f["rule"] == "min_box_size") == 1
        await db.rollback()


@pytest.mark.asyncio
async def test_the_linter_checks_a_humans_own_edit():
    """`detect_policy_violations` filters `source != "human"`, which is right for a sweep hunting machine
    mistakes and exactly wrong here: a linter that runs on save exists to check the edit just made."""
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        for i in range(6):
            await _obj(db, fid, onto, "sedan", [i * 200.0, 0.0, i * 200.0 + 150.0, 150.0])
        await _obj(db, fid, onto, "pedestrian", [10.0, 500.0, 400.0, 540.0], source="human")

        res = await lint_frame(db, fid)
        assert any(f["rule"] == "degenerate_aspect" for f in res["findings"]), (
            "a person's own box, wider than tall for a pedestrian, must still be reported")
        await db.rollback()


@pytest.mark.asyncio
async def test_a_helmet_array_is_only_checked_where_occupant_counts_are_answered():
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        await _obj(db, fid, onto, "motorcycle", [100, 100, 300, 400],
                   attrs={"helmet": [True, False]})
        res = await lint_frame(db, fid)
        assert any(d["rule"] == "helmet_without_occupants" for d in res["dormant"])

        # Now somebody answers the occupant count on a different object, which is the signal that this
        # frame is having occupancy annotated at all.
        await _obj(db, fid, onto, "motorcycle", [700, 100, 900, 400],
                   attrs={"occupant_count": 2, "helmet": [True, True]})
        res2 = await lint_frame(db, fid)
        assert not any(d["rule"] == "helmet_without_occupants" for d in res2["dormant"])
        hits = [f for f in res2["findings"] if f["rule"] == "helmet_without_occupants"]
        assert len(hits) == 1, "only the object with a helmet array and no count should be reported"
        await db.rollback()


@pytest.mark.asyncio
async def test_an_empty_frame_is_not_an_error():
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        res = await lint_frame(db, fid)
        assert res["n_objects"] == 0 and res["findings"] == []
        await db.rollback()


@pytest.mark.asyncio
async def test_opening_issues_twice_does_not_stack_duplicates():
    """The editor autosaves. A lint that opened a second copy of the same finding on every save would make
    the Issues panel unusable within a minute."""
    from sqlalchemy import func, select

    from db.models import Issue
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.quality.lint import lint_frame, open_issues_for

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        _sid, fid = await _frame(db, onto)
        for i in range(6):
            await _obj(db, fid, onto, "sedan", [i * 200.0, 0.0, i * 200.0 + 150.0, 150.0])
        await _obj(db, fid, onto, "sedan", [1800.0, 900.0, 1804.0, 904.0])

        res = await lint_frame(db, fid)
        first = await open_issues_for(db, fid, res["findings"])
        await db.flush()
        assert first["opened"] >= 1
        second = await open_issues_for(db, fid, res["findings"])
        await db.flush()
        assert second["opened"] == 0, "re-linting an unchanged frame must not open the same issue again"
        n = (await db.execute(select(func.count()).select_from(Issue)
                              .where(Issue.frame_id == fid))).scalar_one()
        assert n == first["opened"]
        await db.rollback()
