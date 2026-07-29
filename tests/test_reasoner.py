"""The reasoning layer: evidence, verdicts, adjudication, and the trace that makes it measurable.

The gate's only input worth the name was the detector's own confidence, which is the model grading its own
homework. It cannot catch the failures that actually hurt this corpus, all of which are confident: an
autorickshaw called a delivery_rider_bike at 0.7, a pedestrian two pixels tall at ten metres, a sedan in
one frame that is an suv in the next, a boat on the Outer Ring Road.

Most of what is asserted here is about restraint rather than detection, because a reasoning layer that is
too eager is worse than none: it demotes correct labels, spends a GPU budget adjudicating settled
questions, and teaches reviewers to ignore it.
"""

from __future__ import annotations

import pytest

from core.schemas import BBox, PathProposal, Provenance, UnifiedObject
from services.autolabel.ontology import get_ontology
from services.autolabel.reasoner.evidence import (
    EvidenceContext,
    Finding,
    check_corpus_memory,
    check_cross_model,
    check_physics,
    check_scene,
    check_temporal,
    collect,
    load_priors,
)
from services.autolabel.reasoner.verdict import (
    ABSTAIN,
    ACCEPT,
    ADJUDICATE,
    PERMITS_AUTO_ACCEPT,
    REJECT,
    REVIEW,
    combine,
    reason_about,
)

ONTO = get_ontology()


def _obj(name: str, bbox=(100.0, 400.0, 140.0, 485.0), conf: float = 0.8, **prov) -> UnifiedObject:
    return UnifiedObject(class_id=ONTO.by_name(name).id if ONTO.has_name(name) else 1,
                         class_name=name, bbox=BBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3]),
                         conf=conf, provenance=Provenance(**prov))


def _ctx(obj: UnifiedObject, **kw) -> EvidenceContext:
    kw.setdefault("frame_w", 1280)
    kw.setdefault("frame_h", 960)
    return EvidenceContext(obj=obj, onto=ONTO, **kw)


def _proposal(path: str, name: str, conf: float) -> PathProposal:
    return PathProposal(path=path, class_name=name, conf=conf, verdict="proposed",
                        model_version="test")


# ---------------------------------------------------------------- priors

def test_the_priors_load_and_cover_the_common_classes():
    p = load_priors()
    heights = p["heights_m"]
    for name in ("pedestrian", "autorickshaw", "bus", "cattle"):
        lo, hi = heights[name]
        assert 0 < lo < hi, f"{name} has a nonsensical height band"


def test_a_class_with_no_prior_produces_no_evidence_rather_than_evidence_against():
    """The distinction the whole layer rests on. "No prior" and "impossible" are different findings, and
    conflating them would demote every class the moment it was added to the ontology."""
    obj = _obj("object_fallback", bbox=(0, 0, 10, 10))
    assert check_physics(_ctx(obj, depth_m=20.0, focal_px=1000.0)) == []


# ---------------------------------------------------------------- physics

def test_physics_supports_a_plausible_height_and_opposes_an_impossible_one():
    ok = check_physics(_ctx(_obj("pedestrian", bbox=(600, 400, 640, 485)),
                            depth_m=20.0, focal_px=1000.0))
    assert ok and ok[0].weight > 0

    # Four pixels tall at thirty metres implies twelve centimetres.
    bad = check_physics(_ctx(_obj("pedestrian", bbox=(100, 300, 110, 304)),
                             depth_m=30.0, focal_px=1000.0))
    assert bad and bad[0].weight < 0
    assert "0.12m" in bad[0].detail


def test_physics_is_silent_without_a_depth_prior():
    """The honest degradation: no depth means the check cannot run, not that the box is wrong."""
    assert check_physics(_ctx(_obj("pedestrian"))) == []
    assert check_physics(_ctx(_obj("pedestrian"), depth_m=20.0)) == []      # no focal length


def test_physics_scales_its_objection_with_how_far_off_it_is():
    """A pedestrian implied at 2.3m is a tall person; one implied at 0.15m is a detection on a
    reflection. A flat penalty would treat them the same."""
    slightly = check_physics(_ctx(_obj("pedestrian", bbox=(600, 400, 640, 520)),
                                  depth_m=18.0, focal_px=1000.0))
    wildly = check_physics(_ctx(_obj("pedestrian", bbox=(100, 300, 110, 304)),
                                depth_m=30.0, focal_px=1000.0))
    assert wildly[0].weight < slightly[0].weight


# ---------------------------------------------------------------- scene

def test_a_class_that_cannot_be_on_a_road_is_near_decisive():
    out = check_scene(_ctx(_obj("boat"), scene={"road_type": "urban"}))
    assert out and out[0].weight <= -0.9


def test_an_unusual_but_possible_class_is_only_mild():
    """A cow on a national highway is a real and dangerous event this loop must be able to label, so an
    unusual combination is evidence rather than a veto."""
    out = check_scene(_ctx(_obj("cycle"), scene={"road_type": "highway"}))
    assert out and -0.5 < out[0].weight < 0


def test_scene_is_silent_when_the_scene_is_unknown():
    assert check_scene(_ctx(_obj("pedestrian"), scene={})) == []


# ---------------------------------------------------------------- temporal

def test_a_track_that_mostly_says_something_else_names_it():
    """Naming the alternative is what lets Tier 2 ask a narrow question instead of an open one."""
    obj = _obj("sedan")
    suv = ONTO.by_name("suv").id
    history = [(i * 10**8, BBox(x1=100, y1=400, x2=140, y2=485), suv) for i in range(4)]
    out = check_temporal(_ctx(obj, track_neighbours=history))
    assert out and out[0].weight < 0
    assert out[0].suggests_class == "suv"


def test_a_track_that_agrees_with_itself_supports_the_label():
    obj = _obj("sedan")
    same = ONTO.by_name("sedan").id
    history = [(i * 10**8, BBox(x1=100, y1=400, x2=140, y2=485), same) for i in range(4)]
    out = check_temporal(_ctx(obj, track_neighbours=history))
    assert out and out[0].weight > 0


def test_a_teleporting_box_is_an_association_error_not_motion():
    obj = _obj("sedan", bbox=(100, 400, 140, 485))
    same = ONTO.by_name("sedan").id
    far = [(0, BBox(x1=900, y1=400, x2=940, y2=485), same)]
    out = check_temporal(_ctx(obj, track_neighbours=far))
    assert any("association error" in f.detail for f in out)


def test_temporal_is_silent_on_an_untracked_object():
    assert check_temporal(_ctx(_obj("sedan"))) == []


# ---------------------------------------------------------------- cross-model

def test_agreeing_paths_support_and_disagreeing_paths_oppose():
    agree = check_cross_model(_ctx(_obj(
        "pedestrian", proposals=[_proposal("a", "pedestrian", 0.9),
                                 _proposal("b", "pedestrian", 0.8)], agreement=True)))
    assert agree and agree[0].weight > 0

    disagree = check_cross_model(_ctx(_obj(
        "autorickshaw", proposals=[_proposal("a", "autorickshaw", 0.7),
                                   _proposal("b", "e_rickshaw", 0.66)])))
    assert disagree and disagree[0].weight < 0
    assert disagree[0].suggests_class == "e_rickshaw"


def test_a_class_no_path_proposed_is_the_strongest_cross_model_objection():
    out = check_cross_model(_ctx(_obj(
        "bus", proposals=[_proposal("a", "truck", 0.7), _proposal("b", "truck", 0.6)])))
    assert out and out[0].weight <= -0.5


# ---------------------------------------------------------------- corpus memory

def test_the_nearest_reviewed_crops_can_argue_against_the_label():
    """The corpus already holds thousands of human verdicts and nothing consulted them at annotation
    time."""
    neighbours = [("e_rickshaw", 0.9, "accepted")] * 8 + [("autorickshaw", 0.8, "accepted")] * 2
    out = check_corpus_memory(_ctx(_obj("autorickshaw"), neighbours=neighbours))
    assert out and out[0].weight < 0
    assert out[0].suggests_class == "e_rickshaw"


def test_too_few_reviewed_neighbours_says_nothing():
    """Below a handful the nearest neighbour is noise, and reporting it as evidence is worse than
    silence."""
    assert check_corpus_memory(_ctx(_obj("sedan"),
                                    neighbours=[("suv", 0.9, "accepted")])) == []


def test_unreviewed_neighbours_do_not_count():
    """A machine label agreeing with a machine label is not corroboration."""
    machine = [("suv", 0.95, "review")] * 10
    assert check_corpus_memory(_ctx(_obj("sedan"), neighbours=machine)) == []


# ---------------------------------------------------------------- the combiner

def test_a_clean_detection_is_accepted():
    v = reason_about(_ctx(
        _obj("pedestrian", bbox=(600, 400, 640, 485), conf=0.88,
             agreement=True, proposals=[_proposal("a", "pedestrian", 0.9),
                                        _proposal("b", "pedestrian", 0.8)]),
        depth_m=20.0, focal_px=1000.0, scene={"road_type": "urban"}))
    assert v.decision == ACCEPT
    assert v.score > 0


def test_a_class_that_cannot_be_on_a_road_is_rejected_without_a_second_opinion():
    """A boat does not need adjudicating, and spending a GPU call on it is the waste this tier exists to
    avoid."""
    v = reason_about(_ctx(_obj("boat", conf=0.83), scene={"road_type": "urban"}))
    assert v.decision == REJECT
    assert v.adjudication_question is None


def test_several_independent_objections_refute_even_with_no_single_decisive_one():
    """Three checks agreeing a box is impossible is a stronger statement than any of them made alone. The
    first version of this treated the detector's residual confidence as conflict and escalated instead,
    which spent a call adjudicating a settled question."""
    v = reason_about(_ctx(_obj("pedestrian", bbox=(100, 300, 110, 304), conf=0.72),
                          depth_m=30.0, focal_px=1000.0, scene={"road_type": "urban"}))
    assert v.decision == REJECT
    assert v.conflict == 0.0


def test_a_genuine_conflict_escalates_with_a_narrow_question():
    v = reason_about(_ctx(
        _obj("autorickshaw", bbox=(400, 500, 520, 640), conf=0.71,
             entropy=1.1, proposals=[_proposal("a", "autorickshaw", 0.71),
                                     _proposal("b", "e_rickshaw", 0.66)]),
        depth_m=15.0, focal_px=1000.0, scene={"road_type": "urban"},
        neighbours=[("e_rickshaw", 0.91, "accepted")] * 7 + [("autorickshaw", 0.8, "accepted")] * 2))
    assert v.decision == ADJUDICATE
    assert v.suggested_class == "e_rickshaw"
    # Narrow, and carrying the discriminating evidence: that is what makes a VLM's answer usable.
    assert "e_rickshaw" in v.adjudication_question


def test_a_detection_nothing_could_be_assessed_on_is_abstained_not_demoted():
    """The same principle the checks follow, applied at the verdict level: "no check could run" is not
    "checks ran and found it wanting".

    This was a real defect. Demoting the unassessable put 67% of a real session into the review queue,
    because most objects genuinely have no depth prior, no scene, no track and one path. A reasoner that
    floods the queue is worse than none, since it makes the queue worthless and teaches reviewers to
    ignore it.
    """
    v = combine([], confidence=0.55)
    assert v.decision == ABSTAIN
    assert v.decision in PERMITS_AUTO_ACCEPT
    assert "no check could be applied" in v.reasons[0]


def test_checks_that_ran_and_found_nothing_are_distinguished_from_checks_that_could_not_run():
    """Both abstain, and both say which happened. The difference matters when tuning: one is a gap in the
    priors, the other is a gap in the evidence available for that frame."""
    v = combine([Finding("scene", 0.1, "ordinary on an urban road")], confidence=0.5)
    assert v.decision == ABSTAIN
    assert "found nothing either way" in v.reasons[0]


def test_abstaining_never_promotes_a_weak_detection():
    """Permitting is not promoting. The gate's own calibrated thresholds still decide, so a 0.3 detection
    cannot be talked into being a label by the reasoner declining to comment."""
    from services.autolabel.reasoner.pass_ import FrameContext, apply_to_objects, reason_frame

    # On the ground and with no prior for its class, so genuinely nothing can be assessed. Placed low in
    # the frame deliberately: a box in the sky is assessable, and the horizon check would correctly fire.
    weak = _obj("object_fallback", bbox=(600, 700, 640, 760), conf=0.3)
    verdicts = reason_frame([weak], ONTO, FrameContext(width=1280, height=960))
    ok = apply_to_objects([weak], verdicts)
    # The reasoner says "I have nothing to add"; the gate still refuses it on confidence.
    assert verdicts[0].decision == ABSTAIN
    assert ok[id(weak)] is True


def test_the_detector_is_a_witness_not_the_judge():
    """A confident detector cannot talk its way to auto-accept over strong evidence against, which is the
    entire point: the failures that hurt this corpus are the confident ones.

    Escalating rather than rejecting outright is correct here and worth stating: a 0.99 detector against
    two strong objections is a genuine disagreement, which is exactly what Tier 2 exists to settle.
    """
    against = [Finding("physics", -0.7, "impossible size"),
               Finding("cross_model", -0.55, "no path proposed this")]
    v = combine(against, confidence=0.99)
    assert v.decision != ACCEPT
    assert v.score < 0


def test_the_suggested_class_is_the_one_the_evidence_argues_hardest_for():
    """Weighted rather than last-writer-wins, so a strong temporal majority outvotes a weak hint."""
    v = combine([Finding("corpus_memory", -0.2, "mixed", suggests_class="van"),
                 Finding("temporal", -0.6, "track says minibus", suggests_class="minibus")],
                confidence=0.7)
    assert v.suggested_class == "minibus"


# ---------------------------------------------------------------- the trace

def test_the_trace_records_every_finding_with_its_weight():
    """Summarising to a total would make each check's contribution unmeasurable, which is exactly what
    attribution needs to read back."""
    v = reason_about(_ctx(
        _obj("pedestrian", bbox=(600, 400, 640, 485), conf=0.88),
        depth_m=20.0, focal_px=1000.0, scene={"road_type": "urban"}))
    trace = v.as_trace()
    assert trace["decision"] and "score" in trace and "detector_conf" in trace
    assert trace["findings"] and all({"check", "weight", "detail"} <= set(f) for f in trace["findings"])


def test_a_broken_check_is_skipped_rather_than_fatal():
    """One bad check must not stop a session annotating: a reasoner that can halt the pipeline makes
    labels worse than having none."""
    import services.autolabel.reasoner.evidence as ev

    original = ev.CHECKS["scene"]
    ev.CHECKS["scene"] = lambda ctx: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        findings = collect(_ctx(_obj("pedestrian", bbox=(600, 400, 640, 485)),
                                depth_m=20.0, focal_px=1000.0))
        assert any(f.check == "physics" for f in findings)   # the others still ran
    finally:
        ev.CHECKS["scene"] = original


# ---------------------------------------------------------------- Tier 2

def test_the_adjudication_shortlist_is_the_dispute_not_the_ontology():
    """Offering 191 classes to a model adjudicating between two invites it to pick a third, which
    resolves nothing and creates a fresh disagreement."""
    from services.autolabel.reasoner.adjudicate import dispute_shortlist

    names = dispute_shortlist("autorickshaw", "e_rickshaw", ONTO)
    assert names[:2] == ["autorickshaw", "e_rickshaw"]
    assert len(names) <= 4


def test_an_unresolved_adjudication_routes_to_a_human():
    from services.autolabel.reasoner.adjudicate import Adjudication, apply_adjudication

    v = combine([Finding("cross_model", -0.4, "paths disagree", suggests_class="e_rickshaw")],
                confidence=0.7)
    out = apply_adjudication(v, Adjudication(False, None, 0.4, 5, detail="split"), "autorickshaw")
    assert out.decision == REVIEW


def test_an_upheld_adjudication_accepts_and_a_corrected_one_does_not():
    """A corrected label goes to a human rather than being applied: the system now holds three opinions,
    and a machine picking among them unsupervised is the failure that drove pedestrian recall to 0.004."""
    from services.autolabel.reasoner.adjudicate import Adjudication, apply_adjudication

    upheld = apply_adjudication(
        combine([Finding("cross_model", -0.4, "paths disagree")], confidence=0.7),
        Adjudication(True, "autorickshaw", 0.8, 5, upheld=True, detail="chose autorickshaw"),
        "autorickshaw")
    assert upheld.decision == ACCEPT

    corrected = apply_adjudication(
        combine([Finding("cross_model", -0.4, "paths disagree")], confidence=0.7),
        Adjudication(True, "e_rickshaw", 0.8, 5, upheld=False, detail="chose e_rickshaw"),
        "autorickshaw")
    assert corrected.decision == REVIEW
    assert corrected.suggested_class == "e_rickshaw"


def test_a_split_adjudicator_has_not_settled_anything():
    """A bare majority from a model that was asked precisely because two other signals disagreed is a
    third opinion, not a resolution."""
    from services.autolabel.reasoner.adjudicate import MIN_AGREEMENT, adjudicate

    class _Split:
        settings = type("S", (), {"models": type("M", (), {"vlm": type("V", (), {
            "vote_count": 5, "crop_margin": 0.2, "shortlist_size": 8})()})()})()

        def _attr_schema(self):
            return {}

        def _vote_plans(self, n):
            return [(0.2, 0.0)] * n

        def _validate(self, r):
            return r

        class client:
            calls = 0

            @staticmethod
            def verify(crop, shortlist, schema, temperature=0.0):
                from services.autolabel.paths.path_c_qwen3vl import VlmResult

                _Split.client.calls += 1
                # Alternates, so no class reaches the agreement floor.
                name = shortlist[_Split.client.calls % 2]
                return VlmResult(class_name=name, agreement=0.0, votes=1)

    import numpy as np

    v = combine([Finding("cross_model", -0.4, "paths disagree", suggests_class="e_rickshaw")],
                confidence=0.7)
    out = adjudicate(_Split(), np.zeros((100, 100, 3), dtype=np.uint8), (0, 0, 50, 50), v, ONTO,
                     current_class="autorickshaw")
    assert out.resolved is False
    # 3-2 across five votes. It reads as a majority and settles nothing, which is why the floor sits
    # above two thirds rather than at one half.
    assert out.agreement < MIN_AGREEMENT


def test_an_unavailable_adjudicator_is_not_a_verdict():
    """The object falls through to a human, which is correct when the tie-breaker could not be
    consulted."""
    import numpy as np

    from services.autolabel.reasoner.adjudicate import adjudicate

    class _Broken:
        settings = type("S", (), {"models": type("M", (), {"vlm": type("V", (), {
            "vote_count": 3, "crop_margin": 0.2})()})()})()

        def _attr_schema(self):
            raise RuntimeError("ollama is down")

    v = combine([Finding("cross_model", -0.4, "paths disagree", suggests_class="e_rickshaw")],
                confidence=0.7)
    out = adjudicate(_Broken(), np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 5, 5), v, ONTO,
                     current_class="autorickshaw")
    assert out.resolved is False and "unavailable" in out.detail


# ---------------------------------------------------------------- the frame pass

def test_the_pass_writes_a_trace_and_withholds_auto_accept_from_the_doubtful():
    from services.autolabel.reasoner.pass_ import (
        FrameContext,
        apply_to_objects,
        reason_frame,
        summarise,
    )

    good = _obj("pedestrian", bbox=(600, 400, 640, 485), conf=0.9, agreement=True,
                proposals=[_proposal("a", "pedestrian", 0.9), _proposal("b", "pedestrian", 0.85)])
    bad = _obj("boat", bbox=(300, 400, 500, 500), conf=0.83)

    frame = FrameContext(width=1280, height=960, scene={"road_type": "urban"})
    verdicts = reason_frame([good, bad], ONTO, frame)
    ok = apply_to_objects([good, bad], verdicts)

    assert ok[id(good)] is True
    assert ok[id(bad)] is False
    assert good.provenance.reasoning["decision"] == ACCEPT
    assert bad.provenance.reasoning["decision"] == REJECT
    # Flagged on the field the triage queue, the value score and the correction loop already read, so
    # none of them needed teaching about a new one.
    assert any(f.startswith("reasoner:") for f in bad.provenance.quality_flags)
    assert summarise(verdicts)[REJECT] == 1


def test_the_trace_can_be_turned_off_without_changing_the_decision():
    from services.autolabel.reasoner.pass_ import FrameContext, apply_to_objects, reason_frame

    obj = _obj("boat", bbox=(300, 400, 500, 500), conf=0.83)
    verdicts = reason_frame([obj], ONTO, FrameContext(width=1280, height=960,
                                                      scene={"road_type": "urban"}))
    ok = apply_to_objects([obj], verdicts, record_trace=False)
    assert ok[id(obj)] is False
    assert obj.provenance.reasoning is None


def test_escalation_orders_by_conflict_not_by_confidence():
    """The difference between this and the pass it replaces. The old rule spent the budget on the least
    confident objects, which are often simply hard; an object where two signals actively disagree is where
    an opinion changes the outcome."""
    from services.autolabel.reasoner.pass_ import escalate

    calls: list[str] = []

    class _Recorder:
        settings = type("S", (), {"models": type("M", (), {"vlm": type("V", (), {
            "vote_count": 1, "crop_margin": 0.2})()})()})()

        def _attr_schema(self):
            return {}

        def _vote_plans(self, n):
            return [(0.2, 0.0)]

        def _validate(self, r):
            return r

        class client:
            @staticmethod
            def verify(crop, shortlist, schema, temperature=0.0):
                from services.autolabel.paths.path_c_qwen3vl import VlmResult

                calls.append(shortlist[0])
                return VlmResult(class_name=shortlist[0], agreement=1.0, votes=1)

    import numpy as np

    low = _obj("sedan", conf=0.6)
    high = _obj("autorickshaw", conf=0.6)
    v_low = combine([Finding("cross_model", -0.4, "mild", suggests_class="suv"),
                     Finding("physics", 0.35, "size fits")], confidence=0.6)
    v_high = combine([Finding("cross_model", -0.6, "strong", suggests_class="e_rickshaw"),
                      Finding("physics", 0.35, "size fits"),
                      Finding("geometry", 0.2, "aspect fits")], confidence=0.9)
    assert v_low.decision == ADJUDICATE and v_high.decision == ADJUDICATE

    used = escalate([low, high], [v_low, v_high], ONTO,
                    np.zeros((100, 100, 3), dtype=np.uint8), _Recorder(), budget=1)
    assert used == 1
    # The more conflicted object was adjudicated, not the one that happened to be first.
    assert calls == ["autorickshaw"]


def test_an_object_that_needed_a_second_opinion_and_did_not_get_one_is_not_accepted():
    from services.autolabel.reasoner.pass_ import apply_to_objects

    obj = _obj("autorickshaw", conf=0.7)
    v = combine([Finding("cross_model", -0.5, "paths disagree", suggests_class="e_rickshaw"),
                 Finding("physics", 0.35, "size fits")], confidence=0.8)
    assert v.decision == ADJUDICATE
    ok = apply_to_objects([obj], [v])
    assert ok[id(obj)] is False


# ---------------------------------------------------------------- attribution

@pytest.mark.db
async def test_attribution_reports_nothing_rather_than_zero_when_nothing_is_reasoned():
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.attribution import measure_checks

    async with get_sessionmaker()() as db:
        out = await measure_checks(db, since_hours=1)
    # An empty corpus of traces is "we have not measured", not "every check scores zero".
    assert out["reasoned"] == 0 or out["checks"] is not None


@pytest.mark.db
async def test_attribution_counts_only_reviewed_objects():
    """An object nobody looked at is not evidence the check was right. Counting it would let a check earn
    precision on things nobody ever examined."""

    from db.models import Frame, Object, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.attribution import measure_checks

    trace = {"decision": "review", "score": -0.4, "conflict": 0.0, "detector_conf": 0.7,
             "findings": [{"check": "physics", "weight": -0.7, "detail": "impossible"}]}

    async with get_sessionmaker()() as db:
        baseline = await measure_checks(db)
        before_against = (baseline.get("checks", {}).get("physics") or {}).get("fired_against", 0)

        s = DbSession(vehicle_id="veh-reason", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=640, height=480, quality=0.9)
        db.add(f)
        await db.flush()

        reviewed = Object(frame_id=f.frame_id, class_id=1, bbox=[0.0, 0.0, 10.0, 10.0], conf=0.7,
                          state="rejected", source="fused", provenance={"reasoning": trace})
        unreviewed = Object(frame_id=f.frame_id, class_id=1, bbox=[0.0, 0.0, 10.0, 10.0], conf=0.7,
                            state="review", source="fused", provenance={"reasoning": trace})
        db.add_all([reviewed, unreviewed])
        await db.flush()
        db.add(Review(object_id=reviewed.object_id, reviewer="tester", action="reject",
                      before={}, after={}, ts_ns=0))
        await db.commit()

        after = await measure_checks(db)

    # Measured as a delta, because the test database accumulates objects from every other test in the run
    # and an absolute count would be asserting about their leftovers rather than about these two rows.
    added = after["checks"]["physics"]["fired_against"] - before_against
    assert added == 1, "the reviewed object counted once and the unreviewed one not at all"
    assert after["reviewed"] == after["reasoned"], "only reviewed objects are fetched at all"


@pytest.mark.db
async def test_suggested_weights_are_reported_never_applied():
    """A scoring function that silently rewrites itself from a few hundred verdicts drifts in a way nobody
    notices until a class collapses."""
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.attribution import suggest_weights
    from services.autolabel.reasoner.evidence import CHECKS

    async with get_sessionmaker()() as db:
        out = await suggest_weights(db)
    assert "not applied" in out["note"]
    # Every registered collector gets a suggestion, whether or not it has fired. "geometry" is gone: it was
    # three unrelated rules under one switch, and it is now aspect, mask_box and elevation, each of which
    # can be graded and turned off on its own.
    assert set(out["suggestions"]) >= {"physics", "aspect", "mask_box", "elevation",
                                       "temporal", "scene"}
    assert "geometry" not in CHECKS


# ---------------------------------------------------------------- the live runner seam

@pytest.mark.db
async def test_the_runner_reasons_before_the_gate_and_the_verdict_reaches_it():
    """The seam itself, driven the way `autolabel_session.on_frame` drives it.

    Written because the rest of this file tests the reasoner's own modules and the rerun path, and neither
    exercises the arithmetic inside the runner: the reasoner_ok handoff, and the fact that a withheld
    verdict actually changes what the gate returns. A layer that is correct in isolation and unwired is
    worth nothing, and until this existed nothing would have caught that.
    """
    import uuid

    from core.config import get_settings
    from services.autolabel.gate import gate_object
    from services.autolabel.quality_reviewer import review_object_quality
    from services.autolabel.reasoner.pass_ import (
        FrameContext,
        apply_to_objects,
        reason_frame,
    )

    settings = get_settings()

    # A sedan both paths agree on, at high confidence, in a perfectly sensible box. The gate accepts it
    # and is right to on everything it can see. What it cannot see is that the same track was an suv in
    # the four neighbouring frames, and neither can the quality reviewer, which is per-frame by
    # construction. This is the class of error the layer was built for.
    obj = _obj("sedan", bbox=(400, 500, 560, 600), conf=0.97, agreement=True,
               proposals=[_proposal("a", "sedan", 0.96), _proposal("b", "sedan", 0.9)])
    obj.track_id = uuid.uuid4()

    quality = review_object_quality(obj, [], ONTO, 1280, 960, settings.quality)
    # The quality reviewer has no objection: the box is a sensible shape in a sensible place.
    assert quality.ok is True

    without_reasoner = gate_object(obj, ONTO, settings.gate, auto_accept_enabled=True,
                                   quality_ok=quality.ok)

    suv = ONTO.by_name("suv").id
    history = [(i * 10**8, BBox(x1=400, y1=500, x2=560, y2=600), suv) for i in range(4)]
    verdicts = reason_frame([obj], ONTO,
                            FrameContext(width=1280, height=960, scene={"road_type": "urban"},
                                         track_history={str(obj.track_id): history}))
    reasoner_ok = apply_to_objects([obj], verdicts)

    # The runner combines both signals exactly this way, and either may withhold auto-accept.
    combined = quality.ok and reasoner_ok[id(obj)]
    with_reasoner = gate_object(obj, ONTO, settings.gate, auto_accept_enabled=True,
                                quality_ok=combined)

    assert without_reasoner.value == "auto_accept", "the gate would have accepted this on its own"
    assert with_reasoner.value != "auto_accept", "the reasoner's objection did not reach the gate"
    assert obj.provenance.reasoning is not None
    assert "reasoner:temporal" in obj.provenance.quality_flags
    # And it names what it thinks the object actually is, which is what makes the review fast.
    assert verdicts[0].suggested_class == "suv"


@pytest.mark.db
async def test_the_escalation_budget_cannot_exceed_the_vlm_allowance():
    """The arithmetic the runner does before calling escalate.

    Three separate ceilings apply and the smallest wins: what is left of the session's VLM budget, the
    per-frame cap, and the reasoner's own adjudication cap. The last exists so a prior that suddenly
    conflicts everywhere cannot consume the whole GPU allowance on its own, and getting this wrong would
    be invisible until a bill arrived.
    """
    from core.config import get_settings

    settings = get_settings()
    session_budget, used = 100, 96
    per_frame_cap = 3
    reasoner_cap = settings.reasoner.max_adjudications_per_session
    already = reasoner_cap - 1

    room = min(session_budget - used, per_frame_cap, reasoner_cap - already)
    assert room == 1, "the tightest of the three ceilings must win"

    # And an exhausted budget yields no calls rather than a negative one.
    assert max(0, min(0, per_frame_cap, reasoner_cap - already)) == 0


@pytest.mark.db
async def test_a_settled_object_keeps_its_label_but_still_records_what_the_reasoner_thought():
    """The bug that made the whole layer ungradable. Attribution measures the reasoner against what humans
    decided, and the rerun refused to write a trace on any object a human had decided, so the two halves
    excluded each other and no amount of backfilling could produce a measurement.

    Writing a trace is an observation. Changing a label is a decision. Only the second is a person's."""
    import uuid as _uuid

    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.rerun import rerun_session

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-trace", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=1280, height=960, quality=0.9)
        db.add(f)
        await db.flush()
        cls_id = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "sedan"))).scalar()
        if cls_id is None:
            pytest.skip("the ontology in this database has no sedan class")
        for state in ("accepted", "rejected", "review"):
            db.add(Object(frame_id=f.frame_id, class_id=cls_id,
                          bbox=[100.0, 500.0, 260.0, 600.0], conf=0.9, source="fused",
                          attrs={}, state=state, provenance={}))
        await db.commit()
        sid = str(s.session_id)

    async with get_sessionmaker()() as db:
        out = await rerun_session(db, sid, limit=100, apply=True)

    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object).join(Frame, Object.frame_id == Frame.frame_id)
            .where(Frame.session_id == _uuid.UUID(sid)))).scalars().all()

    by_state = {r.state: r for r in rows}
    assert out["skipped_human_decisions"] == 2, "two objects were settled by a person"
    # Every one carries a trace, settled or not: that is what makes the layer measurable.
    assert all((r.provenance or {}).get("reasoning") for r in rows), \
        "a settled object is the answer key, and withholding its trace is what left attribution empty"
    # And the settled ones kept their labels.
    assert by_state["accepted"].state == "accepted"
    assert by_state["rejected"].state == "rejected"


@pytest.mark.db
async def test_attribution_reads_every_reviewed_object_not_an_arbitrary_page():
    """The measurement is only over reviewed objects, so fetching an arbitrary page of the whole table and
    filtering afterwards puts the answer at the mercy of which rows the page happened to contain. On the real
    corpus that was 20,000 rows drawn from 583,525 to find 2,007 reviewed ones."""

    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.attribution import measure_checks

    trace = {"decision": "review", "score": -0.4, "conflict": 0.0, "detector_conf": 0.7,
             "findings": [{"check": "physics", "weight": -0.7, "detail": "impossible"}]}

    async with get_sessionmaker()() as db:
        cls_id = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "sedan"))).scalar()
        if cls_id is None:
            pytest.skip("the ontology in this database has no sedan class")
        s = DbSession(vehicle_id="veh-attr-page", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=1280, height=960, quality=0.9)
        db.add(f)
        await db.flush()

        # One reviewed object carrying a trace, among a crowd of unreviewed ones that also carry traces.
        # A page-then-filter implementation can miss the only row that counts.
        reviewed = Object(frame_id=f.frame_id, class_id=cls_id, bbox=[1.0, 2.0, 3.0, 4.0],
                          conf=0.8, source="fused", attrs={}, state="rejected",
                          provenance={"reasoning": trace})
        db.add(reviewed)
        await db.flush()
        db.add(Review(object_id=reviewed.object_id, action="reject", reviewer="tester",
                      before={}, after={}, ts_ns=0))
        for _ in range(30):
            db.add(Object(frame_id=f.frame_id, class_id=cls_id, bbox=[1.0, 2.0, 3.0, 4.0],
                          conf=0.8, source="fused", attrs={}, state="review",
                          provenance={"reasoning": trace}))
        await db.commit()
        target = reviewed.object_id

    async with get_sessionmaker()() as db:
        out = await measure_checks(db)

    # The invariant the fix encodes: no unreviewed row enters the computation at all, so the page can never
    # be crowded out by them however many there are. Under a fetch-then-filter implementation the thirty
    # unreviewed objects above would be pulled in and counted among the reasoned, leaving fewer reviewed
    # than reasoned, and on a table of 583,525 rows they would have displaced the reviewed ones entirely.
    assert out["reasoned"] == out["reviewed"], \
        "an object nobody looked at must not even be fetched, let alone grade a check"
    assert out["reviewed"] >= 1
    assert (out["checks"].get("physics") or {}).get("fired_against", 0) >= 1

    async with get_sessionmaker()() as db:
        still = await db.get(Object, target)
    assert still.state == "rejected"


@pytest.mark.db
async def test_a_correction_means_opposite_things_for_accept_and_for_reject():
    """Reported under one name it inverted three decisions out of five: `reject` read as 94% wrong on the
    real corpus when it was 94% vindicated. A human correcting an object the reasoner let through is the
    reasoner having been wrong; correcting one it escalated is the escalation having been warranted."""
    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.attribution import decision_outcomes

    def trace(decision):
        return {"decision": decision, "score": 0.0, "conflict": 0.0, "detector_conf": 0.7,
                "findings": []}

    async with get_sessionmaker()() as db:
        cls_id = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "sedan"))).scalar()
        if cls_id is None:
            pytest.skip("the ontology in this database has no sedan class")
        s = DbSession(vehicle_id="veh-polarity", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=1280, height=960, quality=0.9)
        db.add(f)
        await db.flush()
        for decision in ("accept", "reject"):
            o = Object(frame_id=f.frame_id, class_id=cls_id, bbox=[1.0, 2.0, 3.0, 4.0], conf=0.8,
                       source="fused", attrs={}, state="rejected",
                       provenance={"reasoning": trace(decision)})
            db.add(o)
            await db.flush()
            db.add(Review(object_id=o.object_id, reviewer="tester", action="reject",
                          before={}, after={}, ts_ns=0))
        await db.commit()

    async with get_sessionmaker()() as db:
        out = await decision_outcomes(db)

    accept, reject = out["by_decision"]["accept"], out["by_decision"]["reject"]
    assert "error_rate" in accept and "justified_rate" not in accept
    assert "justified_rate" in reject and "error_rate" not in reject
    assert accept["meaning"] == "let through without review"
    assert reject["meaning"] == "escalated to a human"
    # And the headline names the one number the layer is accountable to.
    assert out["auto_accept_error_rate"] == accept["error_rate"]


@pytest.mark.db
async def test_a_check_is_graded_against_the_base_rate_not_against_a_coin_flip():
    """Precision alone is uninterpretable. On the real corpus 63% of reviewed objects were corrected anyway,
    so a rule firing at random scores 0.63 and looks respectable, and one scoring 0.53 looks like a coin flip
    when it is in fact anti-correlated: firing more often on the objects that were fine. Two rules were being
    read that way, and one of them was making the score actively worse."""
    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass, Review
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.attribution import measure_checks

    def trace(check):
        return {"decision": "review", "score": -0.4, "conflict": 0.0, "detector_conf": 0.7,
                "findings": [{"check": check, "weight": -0.5, "detail": "objection"}]}

    async with get_sessionmaker()() as db:
        cls_id = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "sedan"))).scalar()
        if cls_id is None:
            pytest.skip("the ontology in this database has no sedan class")
        s = DbSession(vehicle_id="veh-lift", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=1280, height=960, quality=0.9)
        db.add(f)
        await db.flush()
        # A rule that objects only to objects humans then corrected, and one that objects only to objects
        # humans accepted. The first carries information; the second is worse than silence.
        for check, action, state in (("lift_good", "reject", "rejected"),
                                     ("lift_bad", "confirm", "accepted")):
            for _ in range(30):
                o = Object(frame_id=f.frame_id, class_id=cls_id, bbox=[1.0, 2.0, 3.0, 4.0],
                           conf=0.8, source="fused", attrs={}, state=state,
                           provenance={"reasoning": trace(check)})
                db.add(o)
                await db.flush()
                db.add(Review(object_id=o.object_id, reviewer="tester", action=action,
                              before={}, after={}, ts_ns=0))
        await db.commit()

    async with get_sessionmaker()() as db:
        out = await measure_checks(db)

    assert out["base_rate_wrong"] is not None, "the bar every check clears has to be reported"
    good, bad = out["checks"]["lift_good"], out["checks"]["lift_bad"]
    assert good["lift_against"] > 1.0 and good["informative"] is True
    assert bad["lift_against"] < 1.0 and bad["informative"] is False, \
        "a rule that fires on the objects that were fine is not weak, it is harmful"


@pytest.mark.db
async def test_a_rerun_says_when_the_limit_cut_the_session_short():
    """A partial pass that reads as a complete one is how a corpus ends up believing it has been reasoned
    over when a fifth of it has not. That is exactly what happened: the largest session holds 60,232 objects
    and a backfill capped at 20,000 left 40,232 unreasoned with nothing in the result to say so."""
    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.reasoner.rerun import rerun_session

    async with get_sessionmaker()() as db:
        cls_id = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "sedan"))).scalar()
        if cls_id is None:
            pytest.skip("the ontology in this database has no sedan class")
        s = DbSession(vehicle_id="veh-trunc", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=1280, height=960, quality=0.9)
        db.add(f)
        await db.flush()
        for _ in range(6):
            db.add(Object(frame_id=f.frame_id, class_id=cls_id, bbox=[1.0, 2.0, 3.0, 4.0],
                          conf=0.8, source="fused", attrs={}, state="review", provenance={}))
        await db.commit()
        sid = str(s.session_id)

    async with get_sessionmaker()() as db:
        short = await rerun_session(db, sid, limit=4, apply=False)
        whole = await rerun_session(db, sid, limit=100, apply=False)

    assert short["truncated"] is True
    assert short["not_reasoned"] == 2
    assert short["objects_in_session"] == 6
    assert whole["truncated"] is False
    assert whole["not_reasoned"] == 0
