"""Which frames deserve an open-vocabulary pass, when running one everywhere is the thing to avoid.

`run_recall` with no frame list runs its model channels over every frame in the session. Across 36,905
frames that is a second pre-labelling pass, not a recovery assist.

Two candidate triggers were measured against the corpus before either was built, and the obvious one is
wrong. Frames tagged `sparse` hold a median of 36 objects and frames tagged `dense` hold 13, so the density
axis is inverted or mislabelled, and only 1,780 of 36,905 frames carry the tag at all. A selector built on it
would have targeted the opposite of what it meant to.

The frame's own neighbours do carry it. At 3fps from a moving vehicle the scene does not empty between
consecutive frames, so a frame holding far fewer objects than the frames either side of it is a detection
failure rather than a quiet road. Measured, that selects 248 frames, 0.67% of the corpus, twelve of them
holding nothing at all while their neighbours average five or more, and the worst holding zero between frames
averaging 87.9.

The tests below are mostly about the ways a selector like this stops being targeted: a thin neighbourhood
that cannot support the inference, a session boundary that would compare two different vehicles, and the
selection rate itself, which is the only thing that distinguishes an assist from the blanket pass it
replaces.
"""

from __future__ import annotations

import uuid

import pytest

from core.timebase import now_ns
from db.models import Frame, Object
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.recall.targeting import (
    deficit,
    frames_worth_recovering,
    is_suspicious,
)

pytestmark = pytest.mark.db

HZ = 333_000_000  # 3fps in ns, which is the rate this reasoning depends on


async def _seed_session(db, counts: list[int], *, vehicle: str = "TEST-RT",
                        img: str = "s3://labeloxav/f.jpg") -> tuple[uuid.UUID, list[uuid.UUID]]:
    """One session whose frames hold `counts[i]` objects each, in time order."""
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id=vehicle, start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test"))
    await db.flush()
    fids = []
    base = now_ns()
    for i, n in enumerate(counts):
        fid = uuid.uuid4()
        db.add(Frame(frame_id=fid, session_id=sid, ts_ns=base + i * HZ, cam_id="cam_f",
                     img_uri=img, width=1920, height=1080))
        await db.flush()
        for _ in range(n):
            db.add(Object(object_id=uuid.uuid4(), frame_id=fid, class_id=1,
                          bbox=[10, 10, 100, 100], conf=0.5, source="auto_accept", state="review"))
        fids.append(fid)
    await db.commit()
    return sid, fids


# ------------------------------------------------------------------------------- the score

def test_a_frame_holding_nothing_amid_busy_neighbours_scores_highest():
    """The measured worst case: zero objects between frames averaging 87.9."""
    assert deficit(0, 87.9) == 1.0


def test_the_score_is_a_fraction_so_quiet_and_crowded_sessions_rank_together():
    """A raw difference would put every busy session above every quiet one regardless of how wrong it is."""
    assert deficit(5, 10) == deficit(50, 100)


def test_a_frame_matching_its_neighbours_has_no_deficit():
    assert deficit(20, 20.0) == 0.0
    assert deficit(30, 20.0) == 0.0


def test_no_neighbours_means_no_claim():
    assert deficit(0, 0.0) == 0.0


# ------------------------------------------------------------------------------- the trigger

def test_a_thin_neighbourhood_cannot_support_the_inference():
    """Two objects dropping to zero is noise. A frame is only anomalous against neighbours that had
    something to lose."""
    assert is_suspicious(0, 2.0) is False
    assert is_suspicious(0, 20.0) is True


def test_the_threshold_is_a_proportion_not_a_count():
    assert is_suspicious(3, 20.0) is True
    assert is_suspicious(15, 20.0) is False


# ------------------------------------------------------------------------------- selection

async def test_the_frame_that_lost_its_objects_is_selected():
    async with get_sessionmaker()() as db:
        sid, fids = await _seed_session(db, [20, 20, 20, 0, 20, 20, 20])
        out = await frames_worth_recovering(db, session_id=sid)
    picked = {f["frame_id"] for f in out["frames"]}
    assert str(fids[3]) in picked


async def test_a_steady_session_selects_nothing():
    """The property that keeps this a shortlist. A selector that fires everywhere is the blanket pass."""
    async with get_sessionmaker()() as db:
        sid, _ = await _seed_session(db, [20] * 8)
        out = await frames_worth_recovering(db, session_id=sid)
    assert out["selected"] == 0


async def test_a_quiet_session_is_not_suspected_of_missing_everything():
    """Every frame holding one object is a road with one object on it, not a detector that failed eight
    times in a row."""
    async with get_sessionmaker()() as db:
        sid, _ = await _seed_session(db, [1, 1, 0, 1, 1, 0, 1, 1])
        out = await frames_worth_recovering(db, session_id=sid)
    assert out["selected"] == 0


async def test_frames_are_ranked_worst_first():
    """The budget is spent from the top, so the order is the feature."""
    async with get_sessionmaker()() as db:
        sid, fids = await _seed_session(db, [30, 30, 30, 0, 30, 30, 12, 30, 30, 30])
        out = await frames_worth_recovering(db, session_id=sid)
    assert out["frames"][0]["frame_id"] == str(fids[3])
    assert out["frames"][0]["deficit"] >= out["frames"][-1]["deficit"]


async def test_the_selection_rate_is_reported():
    """"Targeted" is a claim about proportion, and an unreported one cannot be checked."""
    async with get_sessionmaker()() as db:
        sid, _ = await _seed_session(db, [20, 20, 20, 0, 20, 20, 20])
        out = await frames_worth_recovering(db, session_id=sid)
    assert out["considered"] == 7
    assert 0 < out["selected_pct"] <= 100
    assert "of the corpus" in out["detail"]


async def test_a_frame_whose_image_cannot_be_fetched_is_not_shortlisted():
    """It can never succeed, and it would take a slot in a list whose whole point is that it is short."""
    async with get_sessionmaker()() as db:
        sid, _ = await _seed_session(db, [20, 20, 20, 0, 20, 20, 20], img="s3://x.jpg")
        out = await frames_worth_recovering(db, session_id=sid)
    assert out["selected"] == 0


async def test_neighbours_do_not_cross_a_session_boundary():
    """Otherwise a busy session ending beside a quiet one makes the quiet one's first frames look like
    detection failures, when they are a different vehicle on a different day."""
    async with get_sessionmaker()() as db:
        quiet_sid, quiet_fids = await _seed_session(db, [1, 1, 1, 1], vehicle="TEST-RT-QUIET")
        await _seed_session(db, [80, 80, 80, 80], vehicle="TEST-RT-BUSY")
        out = await frames_worth_recovering(db, session_id=quiet_sid)
    assert out["selected"] == 0
    assert str(quiet_fids[0]) not in {f["frame_id"] for f in out["frames"]}


async def test_the_shortlist_respects_its_budget():
    async with get_sessionmaker()() as db:
        sid, _ = await _seed_session(db, [30, 30, 0, 30, 0, 30, 0, 30, 0, 30, 30])
        out = await frames_worth_recovering(db, session_id=sid, limit=2)
    assert out["selected"] == 2
