"""Feeding the inspector's 3D panel, where every decision is about size.

The corpus holds 154 clouds across 124 sessions, 39 of them real LiDAR. They run from 80 points to 500,000,
averaging 9,573. Half a million points is about 6MB of raw float32 and several times that as JSON, so the
payload is binary and the browser maps it into a Float32Array with no parse step.

The sampling is the part worth pinning. A cloud is stored in scan order, so taking the first N points of a
500,000-point sweep returns one sector of one rotation, and a panel drawing that looks entirely functional
while showing a quarter of the scene. Stride keeps the shape of the whole cloud at any budget.

And the nearest cloud is chosen by time rather than exact match, because clouds are sparse against frames and
the clock lands wherever somebody scrubbed. How far away it was has to travel with it, or a panel implies a
cloud from four seconds ago is what the camera is looking at now.
"""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from db.models import PointCloud
from db.models import Session as DbSession
from db.session import get_sessionmaker
from services.inspector.cloud import (
    HARD_MAX_POINTS,
    nearest_cloud,
    pack,
    stride_indices,
)

pytestmark = pytest.mark.db


# ------------------------------------------------------------------------------- sampling

def test_a_small_cloud_is_returned_whole():
    assert list(stride_indices(50, 200)) == list(range(50))


def test_a_large_cloud_is_cut_to_the_budget():
    assert len(stride_indices(500_000, 200_000)) == 200_000


def test_sampling_spans_the_whole_cloud_not_its_first_sector():
    """The failure that looks like success. In scan order the first 20,000 points of a 500,000-point sweep
    are one sector of one rotation, and the panel would render a quarter scene convincingly."""
    idx = stride_indices(500_000, 1_000)
    assert idx[0] == 0
    assert idx[-1] == 499_999
    # Evenly spread, so no region is over-represented.
    gaps = np.diff(idx)
    assert gaps.max() - gaps.min() <= 1


def test_sampling_is_reproducible():
    """Two viewers scrubbed to the same timestamp must draw the same points, or the difference is a bug
    report nobody can act on."""
    assert list(stride_indices(9_573, 1_000)) == list(stride_indices(9_573, 1_000))


def test_indices_are_unique_and_in_range():
    idx = stride_indices(1_000, 300)
    assert len(set(idx.tolist())) == len(idx)
    assert idx.min() >= 0 and idx.max() < 1_000


def test_an_empty_cloud_asks_for_nothing():
    assert len(stride_indices(0, 200)) == 0


def test_the_budget_is_bounded_rather_than_trusted():
    """The caller is a URL anybody can edit, and the ceiling is what stops a panel refresh becoming a
    denial-of-service against our own API."""
    assert len(stride_indices(10_000_000, 99_999_999)) == HARD_MAX_POINTS


def test_a_nonsense_budget_still_returns_something_drawable():
    assert len(stride_indices(1_000, 0)) == 1


# ------------------------------------------------------------------------------- the wire format

def test_points_are_interleaved_little_endian_float32():
    """The layout a buffer geometry wants, in the byte order every browser this runs in uses."""
    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    raw = pack(xyz, np.array([0.5, 0.25], dtype=np.float32))
    assert len(raw) == 2 * 4 * 4
    back = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)
    assert back[0].tolist() == [1.0, 2.0, 3.0, 0.5]
    assert back[1].tolist() == [4.0, 5.0, 6.0, 0.25]


def test_a_cloud_without_intensity_still_packs():
    """Pseudo-LiDAR from monocular depth has no return strength, and it is 96 of the 154 clouds here."""
    raw = pack(np.zeros((3, 3), dtype=np.float32), None)
    back = np.frombuffer(raw, dtype="<f4").reshape(-1, 4)
    assert back.shape == (3, 4) and back[:, 3].tolist() == [0.0, 0.0, 0.0]


def test_mismatched_intensity_is_ignored_rather_than_misaligned():
    """Zipping a short intensity array against xyz would colour every point by its neighbour's return."""
    raw = pack(np.zeros((3, 3), dtype=np.float32), np.array([1.0], dtype=np.float32))
    assert np.frombuffer(raw, dtype="<f4").reshape(-1, 4)[:, 3].tolist() == [0.0, 0.0, 0.0]


# ------------------------------------------------------------------------------- choosing one

async def _seed(db, ts_list: list[int]) -> uuid.UUID:
    sid = uuid.uuid4()
    db.add(DbSession(session_id=sid, vehicle_id="TEST-3D", start_ts_ns=0, end_ts_ns=1,
                     ontology_version="test"))
    await db.flush()
    for ts in ts_list:
        db.add(PointCloud(cloud_id=uuid.uuid4(), session_id=sid, ts_ns=ts, source="pseudo",
                          cloud_uri=f"s3://labeloxav/clouds/{sid}/{ts}.npz", point_count=100,
                          bounds={"n": 100, "min": [0, 0, 0], "max": [1, 1, 1]}))
    await db.commit()
    return sid


async def test_the_closest_cloud_in_time_is_chosen():
    """Clouds are sparse against frames, so an exact match is the rare case rather than the normal one."""
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [1_000, 5_000, 9_000])
        got = await nearest_cloud(db, sid, 5_400)
    assert int(got.ts_ns) == 5_000


async def test_the_nearest_cloud_before_the_playhead_wins_when_it_is_closer():
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [1_000, 9_000])
        got = await nearest_cloud(db, sid, 2_000)
    assert int(got.ts_ns) == 1_000


async def test_a_session_with_no_clouds_returns_nothing_rather_than_guessing():
    """124 of the corpus's sessions have clouds. The rest must produce an empty panel, not another
    session's geometry."""
    async with get_sessionmaker()() as db:
        sid = await _seed(db, [])
        assert await nearest_cloud(db, sid, 1_000) is None


async def test_clouds_do_not_leak_across_sessions():
    async with get_sessionmaker()() as db:
        await _seed(db, [1_000, 2_000])
        empty = await _seed(db, [])
        assert await nearest_cloud(db, empty, 1_000) is None
