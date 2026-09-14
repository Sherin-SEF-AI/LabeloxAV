"""Finding where a signal changes, and refusing to find one where it only slopes.

There was no changepoint detection anywhere in this repo. The nearest existing things answer a different
question: `inertial_events.py::_event_runs` finds runs above a threshold and the anomaly pass scores
spikes by median absolute deviation, and a threshold crossing is where a behaviour got big enough to
notice rather than where it started.

Two things had to be true before this was worth having, and the second took two attempts.

**It must not fire on noise.** Per-object speed from `ObjectDynamics` is the only motion signal this
corpus has at scale, and its measured frame-to-frame change is 9.1 km/h against a real hard brake of about
12. So every shift is weighed against the series' own scatter rather than against a number in km/h.

**It must not fire on a slope.** A vehicle decelerating steadily has no moment it started braking in the
middle of the ramp, and binary segmentation cuts one into eight pieces if you let it. Measuring the shift
over a short window instead of the whole segment was the obvious fix and was not enough: a ramp of 10 to
50 over 40 samples still moves about 4 units across four samples either side, which clears any threshold
calibrated to see a real brake. What works is comparing two models over the same window, a straight line
against two constants, and requiring the step to explain the window clearly better.

Over 400 real tracks that leaves 86% with no changepoint at all, 14% with one to three, and none with four
or more, which is what a set of snap targets should look like.
"""

import numpy as np
import pytest

from services.temporal.changepoint import (
    MIN_SEGMENT,
    find_changepoints,
    robust_scatter,
    snap_index,
)


def _rng(seed: int = 0):
    return np.random.default_rng(seed)


def test_a_clean_step_is_found_exactly_once_and_in_the_right_place():
    v = np.concatenate([np.full(20, 40.0), np.full(20, 10.0)]) + _rng().normal(0, 1.5, 40)
    cps = find_changepoints(v)
    assert len(cps) == 1
    assert cps[0].index == 20
    assert cps[0].shift < 0, "the sign should say the speed dropped"


def test_a_flat_series_has_no_changepoint():
    """The common and correct answer. Most tracks here are a vehicle at a roughly constant speed, and a
    detector that always finds something makes every snap target meaningless."""
    assert find_changepoints(np.full(40, 30.0) + _rng().normal(0, 1.5, 40)) == []


def test_loud_noise_alone_is_not_a_changepoint():
    """Sigma 12 is larger than a real hard brake. Weighing the shift against a fixed number would make
    this series nothing but events; weighing it against the series' own scatter makes it nothing."""
    assert find_changepoints(np.full(40, 30.0) + _rng(3).normal(0, 12.0, 40)) == []


def test_a_steady_ramp_is_not_a_string_of_changepoints():
    """The failure that took two attempts to fix.

    Binary segmentation cuts a smooth 10-to-50 ramp into eight pieces, because the segment means either
    side of every split differ by most of the rise. Shrinking the measurement window is not enough: four
    samples of that ramp still move by four units. Requiring a step to beat a straight line over the same
    window is what settles it.
    """
    counts = [len(find_changepoints(np.linspace(10, 50, 40) + _rng(s).normal(0, 1.0, 40)))
              for s in range(20)]
    assert max(counts) <= 1, f"a smooth ramp produced up to {max(counts)} changepoints"
    assert sum(counts) <= 4, f"a smooth ramp produced {sum(counts)} changepoints across 20 noise draws"


def test_two_steps_are_both_found():
    v = np.concatenate([np.full(15, 10.0), np.full(15, 45.0), np.full(15, 20.0)])
    cps = find_changepoints(v + _rng().normal(0, 1.5, 45))
    assert sorted(c.index for c in cps) == [15, 30]


def test_a_step_at_the_end_of_a_ramp_is_still_a_step():
    """The case that must survive the anti-ramp rule: a vehicle rolling steadily and then braking hard."""
    v = np.concatenate([np.linspace(10, 30, 20), np.full(20, 5.0)]) + _rng().normal(0, 1.0, 40)
    cps = find_changepoints(v)
    assert any(abs(c.index - 20) <= 1 for c in cps), f"the brake at 20 was missed: {[c.index for c in cps]}"


def test_the_scatter_is_measured_from_successive_differences_not_from_the_spread():
    """A ramp has a large spread and small noise. Measuring the spread would call the ramp noise and hide
    every real step inside it."""
    ramp = np.linspace(0, 100, 50)
    assert robust_scatter(ramp) < np.std(ramp) / 10


def test_a_series_too_short_to_split_returns_nothing_rather_than_guessing():
    assert find_changepoints(np.array([1.0, 5.0, 1.0])) == []
    assert find_changepoints(np.arange(2 * MIN_SEGMENT - 1, dtype=float)) == []


def test_missing_samples_are_skipped_not_treated_as_zero():
    """A None in a speed series means nobody measured that frame. Reading it as zero would invent a stop."""
    v = [30.0, 31.0, None, 29.0, 30.0, 31.0, 30.0, 29.0]
    assert find_changepoints(v) == []


def test_snapping_only_pulls_an_edge_that_is_already_close():
    """An edge dragged into the middle of a steady stretch means what it says. Hauling it several frames
    to the nearest event would move a span placed deliberately."""
    cps = find_changepoints(
        np.concatenate([np.full(20, 40.0), np.full(20, 10.0)]) + _rng().normal(0, 1.5, 40))
    assert snap_index(22, cps, window=3) is not None
    assert snap_index(22, cps, window=3).index == 20
    assert snap_index(35, cps, window=3) is None


def test_the_nearest_changepoint_wins_a_tie_on_strength():
    """Two events inside the window: the one the annotator was aiming at is the near one, not the loud one."""
    v = np.concatenate([np.full(12, 10.0), np.full(12, 60.0), np.full(12, 55.0)])
    cps = find_changepoints(v + _rng().normal(0, 0.5, 36))
    got = snap_index(25, cps, window=4)
    assert got is not None and got.index == 24


@pytest.mark.db
@pytest.mark.asyncio
async def test_the_ego_source_refuses_with_the_count_rather_than_falling_back():
    """The gate that keeps this honest.

    Ego speed is the natural signal for a braking event and it is set on 6 frames of 41,752 in this
    corpus. Silently using per-object speed instead would put span edges where nobody could explain them,
    so the ego source says how many samples it actually found.
    """
    import uuid

    from core.timebase import now_ns, seconds_to_ns
    from db.models import Frame, Object, OntologyClass, OntologyVersion, Track
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.temporal.changepoint import track_changepoints

    onto = get_ontology()
    async with get_sessionmaker()() as db:
        if await db.get(OntologyVersion, onto.version) is None:
            db.add(OntologyVersion(version=onto.version, hierarchy_levels=3, attributes={}))
            await db.flush()
            for c in onto.classes:
                db.add(OntologyClass(id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1,
                                     india=c.india, map_to={}))
            await db.flush()
        ts, sid, tid = now_ns(), uuid.uuid4(), uuid.uuid4()
        cid = onto.by_name("sedan").id
        db.add(DbSession(session_id=sid, vehicle_id="CP-1", start_ts_ns=ts, end_ts_ns=ts + seconds_to_ns(20),
                         city="BLR", sensors={}, ontology_version=onto.version))
        await db.flush()
        db.add(Track(track_id=tid, session_id=sid, class_id=cid, first_ts_ns=ts,
                     last_ts_ns=ts + seconds_to_ns(20), trajectory={}, id_switch_flags={},
                     tracker_version="test", intents={}))
        await db.flush()
        for i in range(12):
            fid = uuid.uuid4()
            db.add(Frame(frame_id=fid, session_id=sid, ts_ns=ts + seconds_to_ns(i), cam_id="cam_f",
                         img_uri=f"s3://cp/{i}.jpg", width=1920, height=1080, quality=0.9, scene={}))
            await db.flush()
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, track_id=tid, class_id=cid,
                          bbox=[1.0, 1.0, 50.0, 90.0], conf=0.8, source="fused", state="review",
                          attrs={}, provenance={}, version=1))
        await db.flush()

        res = await track_changepoints(db, tid, source="ego_speed")
        assert res["changepoints"] == []
        assert res["samples"] == 0
        assert "ego speed" in res["reason"]
        assert "12 frames" in res["reason"], "the refusal has to say how much was actually there"
        await db.rollback()


@pytest.mark.db
@pytest.mark.asyncio
async def test_an_unknown_source_is_refused_rather_than_defaulted():
    """Defaulting would answer a question that was not asked, from a signal the caller did not choose."""
    import uuid

    from db.session import get_sessionmaker
    from services.temporal.changepoint import track_changepoints

    async with get_sessionmaker()() as db:
        with pytest.raises(ValueError):
            await track_changepoints(db, uuid.uuid4(), source="imu")
