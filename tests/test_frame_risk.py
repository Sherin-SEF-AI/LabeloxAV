"""Ranking the objects on one frame by how likely they are to be wrong, and saying how much is known.

Every ingredient existed and none were composed. `triage.py` ranks by uncertainty times rarity with no
frame filter, `selector.py` composes seven signals and stops at session scope, `quality_score.py` scores
one object, `CliqueBandit` never touches an individual object, `ego_mask.py` answers yes/no and deletes.
`GET /frames/{id}/objects` returned rows in whatever order Postgres produced.

The tests that matter here are not about the ordering, which is arithmetic. They are about the two ways a
composed score of unevenly available signals goes quietly wrong.

The first is treating an absent signal as a zero. `quality_score` is set on 11.9% of the corpus and
`clique_bandit` is empty, so a score that averages over all eight components ranks an object nobody has
measured as safer than one that has been measured and looks fine. That is the worst direction to be wrong
in for a queue whose entire purpose is surfacing what nobody has checked.

The second is reordering the objects. `web/components/editor/properties/objectGroups.ts` records that the
server's order is drawing order, and that re-sorting the rows breaks the correspondence between the list
and the canvas without any visible symptom.
"""

import uuid

import pytest

from core.timebase import now_ns

pytestmark = pytest.mark.db


class _Obj:
    """Enough of an Object for the pure scorer, which never touches the session."""

    def __init__(self, class_id, conf=0.9, provenance=None, quality_score=None,
                 bbox=(0.0, 0.0, 100.0, 100.0)):
        self.object_id = uuid.uuid4()
        self.class_id = class_id
        self.conf = conf
        self.provenance = provenance or {}
        self.quality_score = quality_score
        self.bbox = list(bbox)


def _onto():
    from services.autolabel.ontology import get_ontology

    return get_ontology()


def test_an_unmeasured_object_is_not_reported_as_a_confident_zero():
    """The central claim. An object carrying no signals at all scores 0, and it must say that it scored 0
    on almost nothing, so a client can tell "checked and fine" from "never looked at"."""
    from services.activelearn.frame_risk import score_object

    onto = _onto()
    cid = onto.by_name("sedan").id
    bare = score_object(_Obj(cid, conf=0.95, provenance={}), onto)
    measured = score_object(
        _Obj(cid, conf=0.95, quality_score=0.9,
             provenance={"entropy": 0.05, "mask_box_disagree": False,
                         "proposals": [{"class_name": "sedan", "verdict": "agree"},
                                       {"class_name": "sedan", "verdict": "agree"}]}), onto)

    assert bare.risk == pytest.approx(measured.risk, abs=0.05), (
        "both are low risk, which is fine; the difference has to show up in coverage, not in the score")
    assert bare.coverage < measured.coverage, (
        f"the unmeasured object reported coverage {bare.coverage} against {measured.coverage}; without "
        "that gap nothing distinguishes 'checked and fine' from 'never checked'")
    comps = {c.name: c for c in bare.components}
    assert comps["quality"].present is False and comps["entropy"].present is False


def test_a_missing_component_does_not_drag_the_score_down_either():
    """The other direction. Averaging over all eight and counting absences as zero would make a genuinely
    risky object with sparse provenance look mild."""
    from services.activelearn.frame_risk import score_object

    onto = _onto()
    cid = onto.by_name("sedan").id
    # Only two signals, and both are as bad as they get.
    sparse = score_object(_Obj(cid, conf=0.1, provenance={"mask_box_disagree": True}), onto)
    assert sparse.risk > 0.5, (
        f"risk {sparse.risk:.3f}: a very low confidence plus a mask/box disagreement is a strong signal "
        "and must not be diluted by the six components that were never measured")


def test_coverage_is_the_fraction_of_weight_that_actually_fired():
    from services.activelearn.frame_risk import WEIGHTS, score_object

    onto = _onto()
    cid = onto.by_name("sedan").id
    r = score_object(_Obj(cid, conf=0.5, provenance={}), onto)
    # conf and rarity are always available; nothing else is, without provenance or an ego mask.
    expected = (WEIGHTS["calibrated_conf"] + WEIGHTS["rarity"]) / sum(WEIGHTS.values())
    assert r.coverage == pytest.approx(expected, abs=0.001)


def test_a_class_the_ontology_does_not_know_is_ranked_high_not_skipped():
    """Drift between the stored class ids and the loaded ontology has already caused a user-visible 500
    here. An object the ranker cannot name is a thing to look at, not a thing to pass over."""
    from services.activelearn.frame_risk import score_object

    r = score_object(_Obj(999_999, conf=0.95), _onto())
    rarity = {c.name: c for c in r.components}["rarity"]
    assert rarity.present is True and rarity.value == 1.0
    assert "not in the ontology" in rarity.detail


def test_disagreeing_paths_outrank_agreeing_ones():
    from services.activelearn.frame_risk import score_object

    onto = _onto()
    cid = onto.by_name("sedan").id
    agree = score_object(_Obj(cid, provenance={"proposals": [
        {"class_name": "sedan", "verdict": "agree"}, {"class_name": "sedan", "verdict": "agree"}]}), onto)
    disagree = score_object(_Obj(cid, provenance={"proposals": [
        {"class_name": "sedan", "verdict": "agree"}, {"class_name": "bus", "verdict": "overruled"}]}), onto)
    assert disagree.risk > agree.risk
    assert any("2 classes proposed" in d for d in disagree.reasons)


def test_a_box_half_on_the_bonnet_outranks_one_fully_on_or_fully_off_it():
    """Peaked, not monotone. A box entirely on the hood is already deleted by the cleanup sweep, and one
    entirely off it is unremarkable; the half-overlapping box is the reflection nobody caught."""
    from services.activelearn.frame_risk import _ego_edge

    class _Mask:
        def __init__(self, frac):
            self._f = frac

        def ego_fraction(self, bbox, w, h):
            return self._f

    half = _ego_edge(_Mask(0.5), [0, 0, 10, 10], 100, 100)
    full = _ego_edge(_Mask(1.0), [0, 0, 10, 10], 100, 100)
    none = _ego_edge(_Mask(0.0), [0, 0, 10, 10], 100, 100)
    assert half.value > full.value and half.value > none.value
    assert full.present is True, "measured and unremarkable is not the same as unmeasured"


def test_no_ego_mask_is_an_absent_component_rather_than_a_zero():
    from services.activelearn.frame_risk import _ego_edge

    c = _ego_edge(None, [0, 0, 10, 10], 1920, 1080)
    assert c.present is False and "no ego mask" in c.detail


def test_a_true_boolean_is_not_read_as_an_entropy_of_one():
    """`isinstance(True, int)` is true in Python, so a provenance key holding a bool would otherwise score
    as maximum entropy and put that object at the top of every frame it appears on."""
    from services.activelearn.frame_risk import _entropy

    assert _entropy({"entropy": True}).present is False
    assert _entropy({"entropy": 0.5}).present is True


@pytest.mark.asyncio
async def test_ranking_is_a_separate_list_and_the_objects_are_not_reordered():
    """objectGroups.ts: the server's order is drawing order, and re-sorting the rows silently breaks the
    correspondence between the object list and the canvas. So risk ships as ids, not as a new array."""
    from db.models import Frame, Object, OntologyClass, OntologyVersion
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.activelearn.frame_risk import rank_frame_objects
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        sid, fid, ts = uuid.uuid4(), uuid.uuid4(), now_ns()
        db.add(DbSession(session_id=sid, vehicle_id="RSK-1", start_ts_ns=ts, end_ts_ns=ts + 1,
                         city="BLR", sensors={}, ontology_version=onto.version))
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts, cam_id="cam_f", img_uri="s3://x/1.jpg",
                     width=1920, height=1080, quality=0.9, scene={}))
        await db.flush()
        safe = Object(frame_id=fid, class_id=onto.by_name("sedan").id, bbox=[1.0, 1.0, 50.0, 50.0],
                      conf=0.98, source="fused", state="review", attrs={},
                      provenance={"entropy": 0.01, "mask_box_disagree": False}, version=1)
        risky = Object(frame_id=fid, class_id=onto.by_name("sedan").id, bbox=[2.0, 2.0, 60.0, 60.0],
                       conf=0.15, source="fused", state="review", attrs={},
                       provenance={"entropy": 0.95, "mask_box_disagree": True,
                                   "proposals": [{"class_name": "sedan", "verdict": "agree"},
                                                 {"class_name": "bus", "verdict": "overruled"}]}, version=1)
        db.add_all([safe, risky])
        await db.flush()

        res = await rank_frame_objects(db, fid, onto)
        assert res["n"] == 2
        assert res["ranking"][0] == str(risky.object_id), "the risky object should rank first"
        # The payload carries ids, not objects: a client cannot accidentally render this as the object
        # list, because there is no object list in it.
        assert all(isinstance(x, str) for x in res["ranking"])
        assert set(res["objects"]) == {str(safe.object_id), str(risky.object_id)}
        # And it says what it could not see, so a weak ordering is legible as one.
        assert "quality" in res["components_missing"]
        assert 0.0 < res["coverage"] <= 1.0
        await db.rollback()
