"""Reading a lane's type off the paint.

Every case here is a synthetic frame rather than a real one, because the cases that matter are the refusals
and real data supplies those only by accident: the solid line behind a parked car that must not read dashed,
the worn line that must read unknown rather than guess, and the kerb that must not read as a painted line
just because one side of it is brighter.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.autolabel.lane.linetype import (
    MIN_PAINT_CONTRAST,
    classify_lane,
    classify_profile,
    drop_short_runs,
    interior_gaps,
    lateral_peaks,
    paint_response,
    resample_curve,
    runs_of,
    sample_strip,
)

H, W = 720, 1280
CPS = [[640.0, 300.0], [640.0, 500.0], [640.0, 700.0]]


def _road(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (np.full((H, W, 3), 60, np.int16)
            + rng.integers(-4, 5, (H, W, 3))).clip(0, 255).astype(np.uint8)


def _line(img, x=640, width=5, y0=300, y1=700, value=210):
    import cv2
    cv2.line(img, (x, y0), (x, y1), (value,) * 3, width)
    return img


def _dashes(img, x=640, on=28, off=28):
    import cv2
    y = 300
    while y < 700:
        cv2.line(img, (x, y), (x, min(y + on, 700)), (210,) * 3, 5)
        y += on + off
    return img


# ---- the whole point: what it says about a real-looking frame ----------------------------------------

def test_a_solid_line_reads_solid():
    kind, conf, ev = classify_lane(_line(_road()), CPS, frame_width=W)
    assert kind == "solid"
    assert conf > 0.5
    assert ev.duty > 0.9


def test_a_dashed_line_reads_dashed():
    kind, conf, ev = classify_lane(_dashes(_road()), CPS, frame_width=W)
    assert kind == "dashed"
    assert ev.paint_runs >= 3
    assert ev.gap_cv is not None and ev.gap_cv < 0.5, "real dashes are evenly spaced"


def test_a_double_line_reads_double():
    img = _road()
    _line(img, x=633, width=4)
    _line(img, x=647, width=4)
    kind, _conf, ev = classify_lane(img, CPS, frame_width=W)
    assert kind == "double"
    assert ev.lateral_peaks == 2


def test_a_solid_line_behind_parked_cars_is_not_dashed():
    """The case the whole regularity test exists for. Occlusion gives the same duty cycle as dashes, and
    calling it dashed invents a permission to cross a line nobody may cross."""
    import cv2

    img = _line(_road())
    cv2.rectangle(img, (600, 420), (680, 500), (55,) * 3, -1)
    cv2.rectangle(img, (600, 620), (680, 655), (55,) * 3, -1)
    kind, _conf, _ev = classify_lane(img, CPS, frame_width=W)
    assert kind == "solid"


def test_worn_paint_reads_unknown_rather_than_guessing():
    kind, conf, ev = classify_lane(_line(_road(), value=68), CPS, frame_width=W)
    assert kind == "unknown"
    assert conf < 0.3
    assert ev.contrast < MIN_PAINT_CONTRAST


def test_bare_asphalt_reads_unknown():
    kind, _conf, _ev = classify_lane(_road(), CPS, frame_width=W)
    assert kind == "unknown"


def test_a_surface_step_is_a_road_edge_not_a_painted_line():
    """A kerb is bright on one side and dark on the other. Taking the brightest pixel across the strip and
    calling the excess paint reported it as a solid line, which would then make crossing it an offence."""
    img = _road()
    img[:, 640:] = 110
    kind, _conf, ev = classify_lane(img, CPS, frame_width=W)
    assert kind == "road_edge"
    assert ev.contrast < MIN_PAINT_CONTRAST, "a step must not register as a paint ridge"
    assert ev.cross_surface_delta > MIN_PAINT_CONTRAST


def test_a_uniformly_bright_line_is_not_mistaken_for_bare_road():
    """The first version measured contrast as the spread of the profile along the line, which is zero for a
    solid line precisely because it is uniform. Every solid line in the corpus read as asphalt."""
    _kind, _conf, ev = classify_lane(_line(_road()), CPS, frame_width=W)
    assert ev.contrast > 100, "contrast is paint above road, not variation along the run"


# ---- the pieces ---------------------------------------------------------------------------------------

def test_the_curve_is_sampled_along_its_near_half_only():
    """Dashes shrink toward the horizon until they are sub-pixel, and sampling up there adds noise to the
    profile that no rule can use.

    Two control points, which is how most lanes are stored. Restricting the control points rather than the
    densified curve left nothing to keep and silently sampled the whole lane.
    """
    curve = resample_curve([[640.0, 100.0], [640.0, 700.0]], n=32)
    assert curve.shape == (32, 4)
    assert curve[:, 1].min() >= 300.0, "the far part of the lane is not sampled"
    assert curve[:, 1].max() == pytest.approx(700.0)


def test_a_lane_with_one_point_yields_no_curve():
    assert resample_curve([[10.0, 10.0]], n=16).size == 0
    assert resample_curve([], n=16).size == 0


def test_out_of_frame_samples_are_nan_not_the_edge_pixel():
    """Clamping repeats the edge pixel, which manufactures a bright constant run at exactly the place a lane
    leaves the image, and that run reads as paint."""
    gray = np.full((100, 100), 50.0)
    curve = np.array([[1.0, 50.0, 0.0, 1.0]])
    strip = sample_strip(gray, curve, half_width=10)
    assert np.isnan(strip).any()
    assert not np.isnan(strip).all()


def test_paint_response_rejects_a_step_and_accepts_a_ridge():
    ridge = np.full((10, 21), 50.0)
    ridge[:, 9:12] = 200.0
    along_ridge, _ = paint_response(ridge)
    assert along_ridge.min() > 100

    step = np.full((10, 21), 50.0)
    step[:, 11:] = 200.0
    along_step, _ = paint_response(step)
    assert along_step.max() <= 0, "brighter on one side only is not a ridge"


def test_runs_and_short_run_removal():
    mask = np.array([1, 1, 1, 0, 1, 0, 0, 0], dtype=bool)
    paint, gaps = runs_of(mask)
    assert paint == [3, 1]
    assert gaps == [1, 3]
    # At min_len 2 the one-sample gap is filled and the one-sample paint run is removed, so the leading run
    # absorbs the gap and the speckle after it disappears: three plus one becomes four.
    cleaned = drop_short_runs(mask, 2)
    assert runs_of(cleaned)[0] == [4]


def test_end_gaps_are_not_counted_as_gaps_between_dashes():
    """A gap at either end is the lane leaving the frame, not a space between two dashes, and counting it
    makes evenly spaced dashes look irregular."""
    mask = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    _paint, gaps = runs_of(mask)
    assert gaps == [2, 2, 4]
    assert interior_gaps(mask, gaps) == [2]


def test_one_lumpy_ridge_is_not_two_lines():
    """A single wide line with a slight dip in the middle must not read as a double, or every worn line
    becomes a no-overtaking boundary."""
    lateral = np.array([0, 0, 40, 90, 80, 88, 40, 0, 0], dtype=np.float64)
    peaks, _sep = lateral_peaks(lateral, contrast=90.0)
    assert peaks == 1

    genuine = np.array([0, 0, 90, 20, 5, 20, 90, 0, 0], dtype=np.float64)
    peaks2, sep = lateral_peaks(genuine, contrast=90.0)
    assert peaks2 == 2
    assert sep is not None and sep > 0


def test_too_few_samples_refuses_rather_than_deciding():
    kind, conf, ev = classify_profile(np.array([100.0, 100.0]), np.array([100.0]))
    assert kind == "unknown"
    assert conf == 0.0
    assert "too few samples" in " ".join(ev.notes)


def test_confidence_rises_with_the_number_of_dashes_seen():
    few = np.array(([200.0] * 4 + [0.0] * 4) * 3, dtype=np.float64)
    many = np.array(([200.0] * 4 + [0.0] * 4) * 8, dtype=np.float64)
    lateral = np.array([0, 0, 200, 0, 0], dtype=np.float64)
    kind_few, conf_few, _ = classify_profile(few, lateral)
    kind_many, conf_many, _ = classify_profile(many, lateral)
    assert kind_few == kind_many == "dashed"
    assert conf_many > conf_few


# ---- persistence --------------------------------------------------------------------------------------

@pytest.mark.db
async def test_typing_never_overwrites_a_human_and_is_idempotent():
    from sqlalchemy import select

    from db.models import Frame, Lane
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.intelligence.lane_typing import classify_session_lanes

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-type", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://missing",
                  width=1280, height=720, quality=0.9)
        db.add(f)
        await db.flush()
        db.add(Lane(frame_id=f.frame_id, session_id=s.session_id,
                    control_points=[[640.0, 300.0], [640.0, 700.0]], lane_type="dashed",
                    is_ego=True, source="human"))
        db.add(Lane(frame_id=f.frame_id, session_id=s.session_id,
                    control_points=[[300.0, 300.0], [300.0, 700.0]], lane_type="solid",
                    is_ego=False, source="proposed"))
        await db.commit()
        sid = s.session_id

    async with get_sessionmaker()() as db:
        out = await classify_session_lanes(db, sid, apply=True)
        rows = {r.source: r for r in (await db.execute(
            select(Lane).where(Lane.session_id == sid))).scalars().all()}

    # The image does not exist, so nothing could be measured. That must leave confidences null rather than
    # writing a confident answer about paint nobody saw.
    assert out["lanes"] == 1, "the human lane is never a candidate"
    assert out["unreadable_frames_lanes"] == 1
    assert rows["human"].lane_type == "dashed"
    assert rows["human"].marking_conf is None
    assert rows["proposed"].marking_conf is None, "an unreadable frame leaves it unmeasured, not guessed"


@pytest.mark.db
async def test_an_unmeasured_lane_type_cannot_make_a_crossing_an_offence():
    """Accusing an actor of an offence rests on knowing what it crossed. Every lane in the corpus carried
    the literal string solid before the classifier existed, and treating those as evidence turned every
    crossing into a violation."""
    from services.intelligence.lane_events import Observation, _ObsSeries, derive_lane_events

    obs = [Observation(ts_ns=i * 10**8, frame_id=f"f{i}", offset=o)
           for i, o in enumerate([-40, -30, -12, 25, 40, 55, 60])]

    def kinds(type_conf):
        series = {("t", "l"): _ObsSeries(lane_type="solid", lane_id="l", is_ego=False,
                                         type_conf=type_conf, obs=obs)}
        return [e["kind"] for e in derive_lane_events(series, frame_width=1280)]

    assert "lane_change" in kinds(None), "never measured is not evidence of an offence"
    assert "lane_change" in kinds(0.2), "measured and unsure is not evidence either"
    assert "lane_change_illegal" in kinds(0.9), "measured and sure is"


@pytest.mark.db
async def test_typing_does_not_clobber_the_model_that_proposed_the_geometry():
    """model_version says which detector drew the line. The typing pass draws nothing, so writing its own
    version there erases the provenance of the proposer, which is what the first backfill did to 4,554
    lanes before this was caught."""
    from sqlalchemy import select

    from db.models import Frame, Lane
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.intelligence.lane_typing import classify_session_lanes

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-prov", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://missing",
                  width=1280, height=720, quality=0.9)
        db.add(f)
        await db.flush()
        db.add(Lane(frame_id=f.frame_id, session_id=s.session_id,
                    control_points=[[300.0, 300.0], [300.0, 700.0]], lane_type="solid",
                    is_ego=False, source="proposed", model_version="clrernet:local"))
        await db.commit()
        sid = s.session_id

    async with get_sessionmaker()() as db:
        await classify_session_lanes(db, sid, apply=True, reclassify=True)
        lane = (await db.execute(select(Lane).where(Lane.session_id == sid))).scalars().first()

    assert lane.model_version == "clrernet:local", "the proposer's identity survives typing"
