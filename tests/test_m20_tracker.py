"""M2.0 BoT-SORT + DINOv3 tracker: a pure unit test (no DB) that two crossing objects keep distinct,
stable ids, and an object briefly occluded re-acquires its SAME id by appearance (not a new track)."""

from __future__ import annotations

import uuid

import numpy as np

from services.autolabel.track.tracker import track_camera_botsort
from services.intelligence.tracking import Det


def _emb(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(768).astype("float32")
    return v / np.linalg.norm(v)


def test_botsort_persists_id_through_crossing_and_occlusion():
    embA, embB = _emb(1), _emb(2)  # distinct appearances
    fid = uuid.uuid4()
    dets: list[Det] = []
    for f in range(12):
        ts = f * 1000
        ax, bx = 100 + 20 * f, 340 - 20 * f  # A rightward, B leftward, cross near the middle
        if f < 4 or f > 6:  # A occluded on frames 4,5,6 (no detection)
            dets.append(Det(uuid.uuid4(), fid, ts, "cam_f", (ax, 100, ax + 40, 200), 5, embA))
        dets.append(Det(uuid.uuid4(), fid, ts, "cam_f", (bx, 100, bx + 40, 200), 6, embB))

    keep, results, switches = track_camera_botsort(dets)

    assert len(results) == 2  # exactly two tracks (A and B), not split by the crossing/occlusion
    a_tracks = {keep[d.object_id] for d in dets if d.class_id == 5 and d.object_id in keep}
    b_tracks = {keep[d.object_id] for d in dets if d.class_id == 6 and d.object_id in keep}
    assert len(a_tracks) == 1 and len(b_tracks) == 1  # each object kept ONE id through occlusion
    assert a_tracks != b_tracks                        # no id swap at the crossing
    assert isinstance(switches, dict)                  # switch-flag channel present (0 here = clean recovery)
