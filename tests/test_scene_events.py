"""Lane behaviour, signal phases, and the dense-raster edit path.

The geometry rules are tested against constructed series rather than against a real session, because a real
session cannot demonstrate the cases that matter: the jitter that must not count as a crossing, the
transition the phase graph forbids, the gap that must not be bridged. Those are the rules the whole feature
rests on, and each of them is a decision to reject something, which real data supplies only by accident.
"""

from __future__ import annotations

import uuid

import pytest

from services.intelligence.event_taxonomy import TaxonomyError, kind_spec, severity_of, validate
from services.intelligence.lane_events import (
    Observation,
    _ObsSeries,
    derive_lane_events,
    find_crossings,
    find_straddle,
    find_weave,
    lane_x_at,
    signed_offset,
)
from services.intelligence.lane_linking import (
    LaneRow,
    curve_distance,
    link_session_lanes,
    match_frames,
)
from services.intelligence.signal_events import (
    derive_for_track,
    find_flicker,
    find_invalid_transitions,
    segment_phases,
)

MS = 1_000_000
FRAME_W = 1280


def _obs(offsets: list[float], step_ns: int = 100 * MS) -> list[Observation]:
    return [Observation(ts_ns=i * step_ns, frame_id=f"f{i}", offset=o)
            for i, o in enumerate(offsets)]


# ---- taxonomy ----------------------------------------------------------------------------------------

def test_an_unknown_kind_cannot_be_written():
    with pytest.raises(TaxonomyError, match="unknown event kind"):
        validate("vehicle_did_a_thing", t_start_ns=0, t_end_ns=1, track_id="t")


def test_a_deriver_cannot_propose_a_human_only_kind():
    """No geometry we have can see intent, so a machine-proposed near_miss would be an unfalsifiable claim
    landing in the training set."""
    assert kind_spec("near_miss")["derived"] is False
    with pytest.raises(TaxonomyError, match="human-only"):
        validate("near_miss", t_start_ns=0, t_end_ns=10, track_id="t", source="auto")
    # The same kind is fine from a person.
    validate("near_miss", t_start_ns=0, t_end_ns=10, track_id="t", source="human")


def test_shape_and_anchor_are_enforced():
    with pytest.raises(TaxonomyError, match="needs a t_end_ns"):
        validate("lane_change", t_start_ns=0, t_end_ns=None, track_id="t")
    with pytest.raises(TaxonomyError, match="cannot have a distinct t_end_ns"):
        validate("signal_transition_invalid", t_start_ns=5, t_end_ns=9, track_id="t")
    with pytest.raises(TaxonomyError, match="needs a track_id"):
        validate("lane_change", t_start_ns=0, t_end_ns=10, track_id=None)


def test_crossing_a_solid_line_is_a_violation_and_a_dashed_one_is_not():
    """The distinction the whole severity axis exists for: a manoeuvre and an offence look identical in
    geometry and differ only in what was crossed."""
    assert severity_of("lane_change_illegal") == "violation"
    assert severity_of("lane_change") == "notable"


# ---- lane geometry -----------------------------------------------------------------------------------

def test_lane_x_is_interpolated_and_refuses_to_extrapolate():
    cps = [[100.0, 0.0], [200.0, 100.0]]
    assert lane_x_at(cps, 50.0) == pytest.approx(150.0)
    # Past the last control point the lane was never annotated, and inventing it is exactly where a distant
    # vehicle would be judged to have crossed something that was not drawn.
    assert lane_x_at(cps, 150.0) is None
    assert lane_x_at(cps, -10.0) is None


def test_signed_offset_says_which_side():
    cps = [[100.0, 0.0], [100.0, 100.0]]
    assert signed_offset((160.0, 50.0), cps) == pytest.approx(60.0)
    assert signed_offset((40.0, 50.0), cps) == pytest.approx(-60.0)
    assert signed_offset((160.0, 500.0), cps) is None


def test_a_committed_crossing_is_found():
    obs = _obs([-40, -30, -12, 25, 40, 55, 60])
    got = find_crossings(obs, commit_offset=20.0, min_frames_after=3)
    assert len(got) == 1
    assert (got[0].from_side, got[0].to_side) == ("left", "right")


def test_jitter_across_the_line_is_not_a_lane_change():
    """A box edge wobbling on the boundary flips sign every other frame. Without the commit rule this is a
    lane change per flip, and on a real session the noise outnumbers the signal."""
    obs = _obs([-3, 2, -4, 3, -2, 4, -3])
    assert find_crossings(obs, commit_offset=20.0, min_frames_after=3) == []


def test_a_crossing_that_never_commits_is_not_counted():
    """The actor got across the line and stayed within a couple of pixels of it. That is a drift in the fit,
    not a decision to change lane."""
    obs = _obs([-8, 3, 4, 5, 6])
    assert find_crossings(obs, commit_offset=30.0, min_frames_after=3) == []


def test_a_crossing_at_the_very_end_is_not_confirmed():
    """The series ends before the actor could be seen to stay across. Reporting it would be asserting
    something the data does not show."""
    obs = _obs([-40, -30, 25, 40])
    assert find_crossings(obs, commit_offset=20.0, min_frames_after=3) == []


def test_weaving_is_two_crossings_that_return_to_where_they_started():
    obs = _obs([-40, -30, 25, 40, 45, 50, 30, -25, -40, -45, -50])
    crossings = find_crossings(obs, commit_offset=20.0, min_frames_after=3)
    assert len(crossings) == 2
    weaves = find_weave(crossings, window_ns=4_000_000_000)
    assert len(weaves) == 1
    assert weaves[0]["crossings"] == 2


def test_straddling_needs_to_persist():
    assert find_straddle(_obs([2, -1, 3, -2]), band=5.0, min_frames=6) == []
    runs = find_straddle(_obs([2, -1, 3, -2, 1, -3, 2]), band=5.0, min_frames=6)
    assert len(runs) == 1 and runs[0]["frames"] == 7


def test_a_weave_is_reported_once_and_not_also_as_two_lane_changes():
    """Listing the weave and both crossings that make it up would triple-count one behaviour in every rate
    the mining surfaces compute."""
    series = {("track-a", "lane-a"): _ObsSeries(
        lane_type="dashed", lane_id="lane-a", is_ego=False,
        obs=_obs([-40, -30, 25, 40, 45, 50, 30, -25, -40, -45, -50]))}
    kinds = [e["kind"] for e in derive_lane_events(series, frame_width=FRAME_W)]
    assert kinds.count("lane_weave") == 1
    assert "lane_change" not in kinds


def test_the_lane_type_decides_whether_a_crossing_is_an_offence():
    obs = _obs([-40, -30, -12, 25, 40, 55, 60])
    for lane_type, expected in (("dashed", "lane_change"), ("solid", "lane_change_illegal")):
        # A measured type. Without one, no crossing is an offence whatever the line says it is, which the
        # next test covers.
        series = {("t", "l"): _ObsSeries(lane_type=lane_type, lane_id="l", is_ego=False,
                                         type_conf=0.9, obs=obs)}
        got = [e for e in derive_lane_events(series, frame_width=FRAME_W)
               if e["kind"].startswith("lane_change")]
        assert len(got) == 1 and got[0]["kind"] == expected
        assert got[0]["payload"]["direction"] == "right"


# ---- lane identity across frames --------------------------------------------------------------------

def _line(x: float) -> list[list[float]]:
    return [[x, 400.0], [x, 600.0], [x, 800.0]]


def test_curve_distance_is_horizontal_separation_over_shared_height():
    d = curve_distance(_line(100.0), _line(140.0))
    assert d is not None and d[0] == pytest.approx(40.0)
    # No shared height at all is not a large distance, it is no comparison.
    assert curve_distance([[100.0, 0.0], [100.0, 100.0]], [[100.0, 500.0], [100.0, 600.0]]) is None


def test_the_match_budget_scales_with_the_gap_between_frames():
    """The bug this feature shipped broken on. At 3fps a real lane moves 40 to 105 pixels between frames; a
    fixed budget tight enough for 30fps rejects every one, which left track_ref null on the whole corpus."""
    prev = [LaneRow("p1", "f0", 0, _line(100.0), "solid", False)]
    cur = [LaneRow("c1", "f1", 0, _line(190.0), "solid", False)]
    assert match_frames(prev, cur, frame_width=FRAME_W, dt_ns=33 * MS) == {}
    assert match_frames(prev, cur, frame_width=FRAME_W, dt_ns=333 * MS) == {"c1": "p1"}


def test_lanes_of_different_types_are_never_the_same_lane():
    prev = [LaneRow("p1", "f0", 0, _line(100.0), "solid", False)]
    cur = [LaneRow("c1", "f1", 0, _line(104.0), "dashed", False)]
    assert match_frames(prev, cur, frame_width=FRAME_W, dt_ns=333 * MS) == {}


def test_linking_chains_a_lane_across_frames_and_keeps_a_second_lane_apart():
    rows = []
    for i in range(5):
        rows.append(LaneRow(f"L{i}", f"f{i}", i * 100 * MS, _line(300.0 + i * 6), "solid", False))
        rows.append(LaneRow(f"R{i}", f"f{i}", i * 100 * MS, _line(900.0 + i * 6), "solid", False))
    ident = link_session_lanes(rows, frame_width=FRAME_W)
    left = {ident[f"L{i}"] for i in range(5)}
    right = {ident[f"R{i}"] for i in range(5)}
    assert len(left) == 1, "the left lane should keep one identity across all five frames"
    assert len(right) == 1
    assert left != right, "the two lanes must not collapse into one identity"


def test_a_lane_that_matches_nothing_still_gets_an_identity():
    rows = [LaneRow("only", "f0", 0, _line(300.0), "solid", False)]
    ident = link_session_lanes(rows, frame_width=FRAME_W)
    assert ident["only"]


# ---- signals -----------------------------------------------------------------------------------------

def test_phases_are_contiguous_runs_and_a_gap_breaks_them():
    got = segment_phases([(0, "G"), (1 * MS, "G"), (2 * MS, None), (3 * MS, "G")])
    assert [p["state"] for p in got] == ["G", "G"]
    assert got[1]["after_gap"] is True, "an unlabelled frame is not evidence the light stayed green"


def test_an_impossible_transition_is_flagged():
    """Amber straight to green is not a signal, it is a mislabelled frame, and it is invisible on the crop
    because each individual label looks perfectly reasonable on its own."""
    phases = segment_phases([(0, "Y"), (1 * MS, "Y"), (2 * MS, "G"), (3 * MS, "G")])
    got = find_invalid_transitions(phases)
    assert len(got) == 1
    assert (got[0]["from_state"], got[0]["to_state"]) == ("Y", "G")


def test_red_straight_to_green_is_legal_here():
    """India does not use a red-amber phase, so the graph permits it and flagging it would bury the real
    errors under thousands of correct labels."""
    phases = segment_phases([(0, "R"), (1 * MS, "G")])
    assert find_invalid_transitions(phases) == []


def test_a_transition_across_a_gap_is_never_flagged():
    phases = segment_phases([(0, "Y"), (1 * MS, None), (2 * MS, "G")])
    assert find_invalid_transitions(phases) == []


def test_flicker_needs_the_state_to_revert():
    """A genuinely short amber is short and then goes red. A mislabelled frame goes red, green once, red
    again, and requiring the revert is what separates them."""
    short = [(0, "R"), (100 * MS, "R"), (200 * MS, "G"), (300 * MS, "R"), (400 * MS, "R")]
    assert len(find_flicker(segment_phases(short), min_phase_ns=400 * MS)) == 1
    progressing = [(0, "R"), (100 * MS, "R"), (200 * MS, "G"), (300 * MS, "Y"), (400 * MS, "Y")]
    assert find_flicker(segment_phases(progressing), min_phase_ns=400 * MS) == []


def test_a_signal_track_produces_phases_and_its_anomalies_together():
    samples = [(0, "Y", "f0"), (100 * MS, "Y", "f1"), (200 * MS, "G", "f2"), (300 * MS, "G", "f3")]
    kinds = [e["kind"] for e in derive_for_track("track-1", samples)]
    assert kinds.count("signal_phase") == 2
    assert kinds.count("signal_transition_invalid") == 1
    for e in derive_for_track("track-1", samples):
        # Every derived event must survive the taxonomy it claims to belong to, or the persister drops it.
        validate(e["kind"], t_start_ns=e["t_start_ns"], t_end_ns=e.get("t_end_ns"),
                 track_id=e.get("track_id"), frame_id=e.get("frame_id"), source="auto")


# ---- dense raster editing ----------------------------------------------------------------------------

def test_polygons_paint_a_class_raster_and_report_names_the_ontology_lost():
    from services.segment2d.edit import coverage_of, rasterize_class_polygons

    def name_to_id(n):
        return {"road": 7}.get(n)

    labels, unknown = rasterize_class_polygons(
        [{"class_name": "road", "polygons": [[0, 0, 10, 0, 10, 10, 0, 10]]},
         {"class_name": "unicorn_lane", "polygons": [[0, 0, 5, 0, 5, 5, 0, 5]]}],
        width=20, height=20, name_to_id=name_to_id)

    assert labels[5][5] == 7
    assert labels[15][15] == 0
    assert unknown == ["unicorn_lane"], "the rest of the edit still lands"
    assert coverage_of(labels, lambda cid: "road" if cid == 7 else None)["road"] > 0


def test_an_edit_lays_over_the_existing_raster_rather_than_erasing_it():
    """The canvas only ever sends back what the annotator drew. Treating zero as 'erase' would mean
    correcting one car silently deleted the road."""
    import numpy as np

    from services.segment2d.edit import merge_onto_existing

    base = np.full((4, 4), 7, dtype=np.int32)
    edit = np.zeros((4, 4), dtype=np.int32)
    edit[0][0] = 3
    out = merge_onto_existing(base, edit)
    assert out[0][0] == 3
    assert out[3][3] == 7


def test_a_degenerate_ring_is_skipped_rather_than_closed():
    from services.segment2d.edit import rasterize_class_polygons

    labels, _ = rasterize_class_polygons(
        [{"class_name": "road", "polygons": [[1, 1, 2, 2]]}], width=8, height=8,
        name_to_id=lambda n: 7)
    assert int(labels.sum()) == 0


# ---- persistence -------------------------------------------------------------------------------------

@pytest.mark.db
async def test_deriving_twice_does_not_duplicate_and_never_overrules_a_person():
    """Deriving is cheap and will be re-run after every lane correction. If a re-run appended, every rate
    the safety surfaces compute would drift upward with the number of times somebody pressed the button."""
    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass, TimelineEvent, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.intelligence.driving_events import persist_driving_events

    async with get_sessionmaker()() as db:
        signal_cls = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "traffic_signal"))).scalar()
        if signal_cls is None:
            pytest.skip("the ontology in this database has no traffic_signal class")

        s = DbSession(vehicle_id="veh-events", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        track = Track(session_id=s.session_id, class_id=signal_cls,
                      first_ts_ns=0, last_ts_ns=300 * MS)
        db.add(track)
        await db.flush()
        # Amber then green: one impossible transition plus two phases.
        for i, state in enumerate(["Y", "Y", "G", "G"]):
            f = Frame(session_id=s.session_id, ts_ns=i * 100 * MS, cam_id="cam_f",
                      img_uri="s3://x", width=1280, height=960, quality=0.9)
            db.add(f)
            await db.flush()
            db.add(Object(frame_id=f.frame_id, track_id=track.track_id, class_id=signal_cls,
                          bbox=[10.0, 10.0, 40.0, 90.0], conf=0.9, source="fused",
                          attrs={"signal_state": state}, state="review"))
        await db.commit()
        sid = s.session_id

    async with get_sessionmaker()() as db:
        first = await persist_driving_events(db, sid)
    assert first["inserted"] == 3
    assert first["by_kind"] == {"signal_phase": 2, "signal_transition_invalid": 1}

    async with get_sessionmaker()() as db:
        again = await persist_driving_events(db, sid)
        rows = (await db.execute(
            select(TimelineEvent).where(TimelineEvent.session_id == sid))).scalars().all()
    assert again["inserted"] == 0, "a re-derivation must not append"
    assert len(rows) == 3

    # A person confirms one. The next derivation must leave it exactly as it is.
    async with get_sessionmaker()() as db:
        row = (await db.execute(select(TimelineEvent).where(
            TimelineEvent.session_id == sid,
            TimelineEvent.kind == "signal_transition_invalid"))).scalars().first()
        row.state = "confirmed"
        await db.commit()
        confirmed_id = row.event_id

    async with get_sessionmaker()() as db:
        third = await persist_driving_events(db, sid)
        still = await db.get(TimelineEvent, confirmed_id)
    assert third["skipped_reviewed"] >= 1
    assert still.state == "confirmed"


@pytest.mark.db
async def test_a_candidate_the_geometry_no_longer_implies_is_pruned():
    """A corrected lane that no longer produces a crossing must stop showing the crossing. Leaving the stale
    candidate means the correction visibly did nothing."""
    from sqlalchemy import select

    from db.models import Session as DbSession
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.driving_events import persist_driving_events

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-prune", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        db.add(TimelineEvent(session_id=s.session_id, kind="lane_change", modality="driving",
                             t_start_ns=0, t_end_ns=100, track_id=None, conf=0.7,
                             payload={}, source="auto", state="review"))
        # A human ruling is not a candidate and survives whatever the geometry says now.
        db.add(TimelineEvent(session_id=s.session_id, kind="lane_change", modality="driving",
                             t_start_ns=500, t_end_ns=600, track_id=None, conf=0.7,
                             payload={}, source="auto", state="confirmed"))
        await db.commit()
        sid = s.session_id

    async with get_sessionmaker()() as db:
        out = await persist_driving_events(db, sid)
        rows = (await db.execute(
            select(TimelineEvent).where(TimelineEvent.session_id == sid))).scalars().all()
    assert out["pruned_stale"] == 1
    assert [r.state for r in rows] == ["confirmed"]


@pytest.mark.db
async def test_the_summary_counts_violations_separately_from_everything_else():
    from db.models import Session as DbSession
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.driving_events import session_event_summary

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-sum", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        for kind in ("lane_change", "lane_change_illegal", "lane_change_illegal", "signal_phase"):
            db.add(TimelineEvent(session_id=s.session_id, kind=kind, modality="driving",
                                 t_start_ns=0, t_end_ns=1, payload={}, source="auto",
                                 state="review"))
        await db.commit()
        sid = s.session_id

    async with get_sessionmaker()() as db:
        out = await session_event_summary(db, sid)
    assert out["total"] == 4
    assert out["violations"] == 2
    assert out["by_severity"]["info"] == 1


@pytest.mark.db
async def test_lane_linking_writes_identities_the_deriver_can_then_use():
    from sqlalchemy import select

    from db.models import Frame, Lane
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.intelligence.lane_linking import link_lanes_for_session

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-lanes", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        for i in range(5):
            f = Frame(session_id=s.session_id, ts_ns=i * 100 * MS, cam_id="cam_f",
                      img_uri="s3://x", width=1280, height=960, quality=0.9)
            db.add(f)
            await db.flush()
            db.add(Lane(frame_id=f.frame_id, session_id=s.session_id,
                        control_points=[[300.0 + i * 6, 400.0], [300.0 + i * 6, 800.0]],
                        lane_type="solid", is_ego=True, source="proposed"))
        await db.commit()
        sid = s.session_id

    async with get_sessionmaker()() as db:
        out = await link_lanes_for_session(db, sid, apply=True)
        refs = (await db.execute(select(Lane.track_ref).where(Lane.session_id == sid))).scalars().all()

    assert out["linked"] == 5
    assert out["multi_frame_identities"] == 1
    assert len({str(r) for r in refs}) == 1, "one lane across five frames is one lane"
    assert all(isinstance(r, uuid.UUID) for r in refs)


@pytest.mark.db
async def test_a_severity_filter_is_not_defeated_by_the_page_limit():
    """Filtering severity after the fetch returns whatever fraction of the first page happened to match. On
    a real session that read "no violations" while eleven sat below the limit."""
    from db.models import Session as DbSession
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.timeline_events import list_events

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-filter", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        # Twenty harmless events first, then the violation. A limit of three sees only the harmless ones.
        for i in range(20):
            db.add(TimelineEvent(session_id=s.session_id, kind="signal_phase", modality="driving",
                                 t_start_ns=i, t_end_ns=i + 1, payload={}, source="auto",
                                 state="review"))
        db.add(TimelineEvent(session_id=s.session_id, kind="lane_change_illegal", modality="driving",
                             t_start_ns=100, t_end_ns=101, payload={}, source="auto", state="review"))
        await db.commit()
        sid = s.session_id

    out = await list_events(sid, modality="driving", severity="violation", limit=3)
    assert out["count"] == 1
    assert out["events"][0]["kind"] == "lane_change_illegal"


@pytest.mark.db
async def test_the_list_carries_the_session_origin_so_a_time_can_be_read():
    """Absolute capture time is what the pipeline needs and what no reviewer can read. Without the origin
    travelling with the page, every row shows an epoch timestamp instead of an offset into the drive."""
    from db.models import Session as DbSession
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.timeline_events import list_events

    origin = 1_782_458_400_000_000_000
    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-origin", city="BLR", start_ts_ns=origin,
                      end_ts_ns=origin + 10**10, sensors={},
                      ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        db.add(TimelineEvent(session_id=s.session_id, kind="signal_phase", modality="driving",
                             t_start_ns=origin + 12_300_000_000, t_end_ns=origin + 13_000_000_000,
                             payload={}, source="auto", state="review"))
        await db.commit()
        sid = s.session_id

    out = await list_events(sid, modality="driving")
    assert out["session_start_ns"] == origin
    offset_s = (out["events"][0]["t_start_ns"] - out["session_start_ns"]) / 1e9
    assert offset_s == pytest.approx(12.3)


@pytest.mark.db
async def test_reclassifying_an_event_corrects_it_instead_of_filing_a_second_one():
    """The bug the lane classifier exposed. Identity keyed on the kind meant a crossing that stopped being an
    offence was recorded again as an ordinary lane change, next to the offence it replaced, and the corpus
    held one crossing twice with contradictory severities."""
    from sqlalchemy import select

    from db.models import Session as DbSession
    from db.models import TimelineEvent
    from db.session import get_sessionmaker
    from services.intelligence.driving_events import _identity, persist_driving_events

    # Same occurrence, different reading: the identity must not distinguish them.
    assert _identity("lane_change_illegal", "t", 5) == _identity("lane_change", "t", 5)
    # Genuinely different occurrences still are.
    assert _identity("lane_change", "t", 5) != _identity("lane_weave", "t", 5)

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-reclass", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        db.add(TimelineEvent(session_id=s.session_id, kind="lane_change_illegal",
                             modality="driving", t_start_ns=0, t_end_ns=100, payload={},
                             source="auto", state="review"))
        await db.commit()
        sid = s.session_id

    # The session has no lanes, so nothing derives and the stale candidate is pruned rather than duplicated.
    async with get_sessionmaker()() as db:
        out = await persist_driving_events(db, sid)
        rows = (await db.execute(
            select(TimelineEvent).where(TimelineEvent.session_id == sid))).scalars().all()
    assert out["pruned_stale"] == 1
    assert rows == []


@pytest.mark.db
async def test_a_ruling_whose_evidence_changed_is_flagged_not_overwritten_and_not_hidden():
    """A person confirmed a violation when every lane was typed solid by default. Once the paint was actually
    read the line may not be one. Overwriting their verdict undoes their work; keeping it silently preserves a
    conclusion whose premise is gone."""
    from sqlalchemy import select

    from db.models import Frame, Lane, Object, OntologyClass, TimelineEvent, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.intelligence.driving_events import persist_driving_events

    async with get_sessionmaker()() as db:
        cls_id = (await db.execute(
            select(OntologyClass.id).where(OntologyClass.name == "sedan"))).scalar()
        if cls_id is None:
            pytest.skip("the ontology in this database has no sedan class")
        s = DbSession(vehicle_id="veh-evid", city="BLR", start_ts_ns=0, end_ts_ns=10**10,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        track = Track(session_id=s.session_id, class_id=cls_id, first_ts_ns=0,
                      last_ts_ns=10 * 100 * MS)
        db.add(track)
        await db.flush()

        ref = uuid.uuid4()
        offsets = [-60.0, -50.0, -20.0, 40.0, 60.0, 70.0, 80.0]
        first_ts = None
        for i, off in enumerate(offsets):
            f = Frame(session_id=s.session_id, ts_ns=i * 100 * MS, cam_id="cam_f",
                      img_uri="s3://x", width=1280, height=960, quality=0.9)
            db.add(f)
            await db.flush()
            if first_ts is None:
                first_ts = 0
            # A lane the paint says is dashed, but with real confidence, so crossing it is not an offence.
            db.add(Lane(frame_id=f.frame_id, session_id=s.session_id, track_ref=ref,
                        control_points=[[640.0, 300.0], [640.0, 900.0]], lane_type="dashed",
                        is_ego=False, source="proposed", marking_conf=0.9))
            cx = 640.0 + off
            db.add(Object(frame_id=f.frame_id, track_id=track.track_id, class_id=cls_id,
                          bbox=[cx - 30, 700.0, cx + 30, 800.0], conf=0.9, source="fused",
                          attrs={}, state="review"))
        await db.commit()
        sid = s.session_id

    async with get_sessionmaker()() as db:
        first = await persist_driving_events(db, sid)
    assert first["by_kind"].get("lane_change") == 1, "a dashed crossing is a manoeuvre, not an offence"

    # Somebody confirms it, then the line is retyped as solid.
    async with get_sessionmaker()() as db:
        row = (await db.execute(select(TimelineEvent).where(
            TimelineEvent.session_id == sid))).scalars().first()
        row.state = "confirmed"
        await db.execute(
            Lane.__table__.update().where(Lane.session_id == sid).values(lane_type="solid"))
        await db.commit()
        event_id = row.event_id

    async with get_sessionmaker()() as db:
        second = await persist_driving_events(db, sid)
        still = await db.get(TimelineEvent, event_id)
        rows = (await db.execute(
            select(TimelineEvent).where(TimelineEvent.session_id == sid))).scalars().all()

    assert len(rows) == 1, "the reclassification must not file a second record"
    assert still.kind == "lane_change", "a person's ruling is not overwritten"
    assert still.state == "confirmed"
    assert still.provenance["evidence_changed"]["would_now_be"] == "lane_change_illegal"
    assert second["reviewed_with_changed_evidence"], "and it is reported rather than hidden"
