"""The rig reader against a session that lies about itself in the four ways a real one does.

A BluRabbit capture ships an `alignment/` index precisely because the raw session is self-describing and
wrong. Every one of these traps produces a plausible corpus rather than an error, which is why they are
pinned here rather than left to a smoke test: a reader that fell for any of them would still emit frames,
still ingest, and still look synchronised on a contact sheet.

The fixture is deliberately tiny and deliberately hostile. Each frame is a solid grey whose value encodes
its position in the camera's own sequence, so a test can assert which frame was decoded rather than merely
that something decoded.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from services.ingest.reader_rig import cam_id_for, read_index, rig_frames

EPOCH_NS = 1_785_492_214_257_109_142
FRAME_PERIOD_NS = 33_333_333

# The cameras are free-running rather than trigger-synced, so the rear camera exposes a fixed offset after
# the front one. This is the spread the reader must preserve rather than average away.
CAM_OFFSET_NS = {"cam0": 0, "cam1": 28_000_000, "cam2": 12_000_000}

# Corrected position and role, as `frames.csv` records them.
CAM_TRUTH = {"cam0": ("front", "wide"), "cam1": ("rear", "narrow"), "cam2": ("left", "wide")}

# What `manifest.json` claims, which is wrong on every channel. Chosen not to collide with any camera's
# corrected id, so that finding a manifest name in the output is unambiguous evidence the manifest was read.
CAM_MANIFEST_LIE = {"cam0": ("right", "narrow"), "cam1": ("left", "narrow"), "cam2": ("rear", "wide")}

SEG_FRAMES = 4
N_INSTANTS = 8
# cam2 is in the index with no video on disk at all, which is the real session's reference camera.
NO_VIDEO = "cam2"
# The instant at which cam1 dropped a frame, so a group legitimately has fewer members.
CAM1_DROPS_AT = 2


def _grey(gi: int) -> int:
    """A frame's global index encoded as a grey level, spaced far enough apart to survive mp4 compression."""
    return 20 + gi * 28


def _segment_of(gi: int) -> tuple[str, int]:
    return (f"seg{gi // SEG_FRAMES}.mp4", gi % SEG_FRAMES)


def _session_t(cam: str, gi: int) -> int:
    return gi * FRAME_PERIOD_NS + CAM_OFFSET_NS[cam]


@pytest.fixture(scope="module")
def rig_session(tmp_path_factory) -> Path:
    """A two-segment, three-camera session carrying all four traps at once."""
    root = tmp_path_factory.mktemp("rig") / "20260731T100333Z-fixture"
    (root / "alignment").mkdir(parents=True)

    for cam in CAM_TRUTH:
        (root / cam).mkdir()
        if cam == NO_VIDEO:
            continue
        for seg in range(N_INSTANTS // SEG_FRAMES):
            writer = cv2.VideoWriter(str(root / cam / f"seg{seg}.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
            assert writer.isOpened(), "fixture needs a working mp4 writer"
            for k in range(SEG_FRAMES):
                writer.write(np.full((64, 64, 3), _grey(seg * SEG_FRAMES + k), np.uint8))
            writer.release()

    (root / "manifest.json").write_text(json.dumps(
        {"cameras": [{"camera_id": c, "position": p, "role": r}
                     for c, (p, r) in CAM_MANIFEST_LIE.items()]}))

    with (root / "alignment" / "frames.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["camera_id", "file", "frame_in_file", "session_t_ns", "real_ns", "position", "role"])
        for cam, (pos, role) in CAM_TRUTH.items():
            # Rows are emitted newest-first per camera so that "row N of the sidecar" and "frame N of the
            # mp4" cannot coincide even by accident.
            for gi in reversed(range(N_INSTANTS)):
                if cam == "cam1" and gi == CAM1_DROPS_AT:
                    continue
                f, fi = _segment_of(gi)
                t = _session_t(cam, gi)
                w.writerow([cam, f, fi, t, EPOCH_NS + t, pos, role])

    with (root / "alignment" / "sync_groups.csv").open("w", newline="") as fh:
        cols = ["usable"] + [f"{c}__{k}" for c in CAM_TRUTH for k in ("file", "frame")]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for gi in range(N_INSTANTS):
            # The first instant is recorded while the cameras were still starting up staggered.
            row = {"usable": "0" if gi == 0 else "1"}
            for cam in CAM_TRUTH:
                if cam == "cam1" and gi == CAM1_DROPS_AT:
                    row[f"{cam}__file"], row[f"{cam}__frame"] = "", ""
                else:
                    row[f"{cam}__file"], row[f"{cam}__frame"] = _segment_of(gi)
            w.writerow(row)

    return root


CAM_BY_ID = {cam_id_for(*truth): cam for cam, truth in CAM_TRUTH.items()}


def _instant_of(frame) -> int:
    """Which rig instant a frame belongs to, backing out that camera's own offset.

    The offset has to come out here precisely because the reader keeps it in: two cameras at the same instant
    do not share a timestamp, and treating the spread as elapsed time would file the rear camera one instant
    late.
    """
    cam = CAM_BY_ID[frame.cam_id]
    return round((frame.ts_ns - EPOCH_NS - CAM_OFFSET_NS[cam]) / FRAME_PERIOD_NS)


def _by_instant(frames) -> dict[int, dict[str, object]]:
    """Emitted frames keyed by rig instant and camera id, for assertions that name an instant."""
    out: dict[int, dict[str, object]] = {}
    for fr in frames:
        out.setdefault(_instant_of(fr), {})[fr.cam_id] = fr
    return out


@pytest.fixture(scope="module")
def emitted(rig_session):
    return list(rig_frames(rig_session))


def test_the_epoch_makes_session_time_absolute(rig_session):
    """Session time is relative to a pipeline start. Only the epoch makes a frame comparable to anything."""
    assert read_index(rig_session)["epoch_ns"] == EPOCH_NS


def test_a_frame_carries_its_own_instant_not_the_groups(emitted):
    """The cameras are free-running, so stamping a group's members with the reference camera's time would
    erase real ego motion and let the corpus claim a sync it does not have."""
    front = {_instant_of(f): f for f in emitted if f.cam_id == "front_wide"}
    rear = {_instant_of(f): f for f in emitted if f.cam_id == "rear_narrow"}
    shared = set(front) & set(rear)
    assert shared, "the fixture must produce instants seen by both cameras"
    for gi in shared:
        assert rear[gi].ts_ns - front[gi].ts_ns == CAM_OFFSET_NS["cam1"], (
            "the physical spread between cameras must survive into ts_ns")


def test_absolute_time_is_epoch_plus_session_time(emitted):
    for fr in emitted:
        cam = CAM_BY_ID[fr.cam_id]
        assert fr.ts_ns == EPOCH_NS + _session_t(cam, _instant_of(fr))


def test_it_decodes_the_frame_the_index_names_not_the_sidecar_row(emitted):
    """The trap this index exists for. In 14 of 48 real segments a boundary frame is logged to the
    neighbouring segment's sidecar, so row N is not frame N. The fixture writes each camera's rows in
    reverse, which a reader seeking by row order would decode backwards while emitting a full stream."""
    seen = _by_instant(emitted)
    for gi, members in seen.items():
        for fr in members.values():
            got = float(np.asarray(fr.image_bgr).mean())
            assert abs(got - _grey(gi)) < 14, (
                f"instant {gi} decoded a frame whose grey level says it is index "
                f"{round((got - 20) / 28)}")


def test_camera_ids_come_from_the_corrected_position_not_the_manifest(emitted):
    """`manifest.json` names the wrong direction on every channel of the real session. A reader trusting it
    yields a corpus where `front_wide` means the rear camera, which nothing downstream can detect."""
    ids = {f.cam_id for f in emitted}
    assert ids == {"front_wide", "rear_narrow"}, ids
    for cam, (pos, role) in CAM_MANIFEST_LIE.items():
        if cam == NO_VIDEO:
            continue
        assert cam_id_for(pos, role) not in ids, f"{cam} was named from the manifest"


def test_startup_groups_are_skipped(emitted):
    """The real session flags its first 18 groups unusable. Admitting them would put a group claiming
    synchronisation into the corpus with a 428 ms spread."""
    assert 0 not in _by_instant(emitted), "the unusable startup instant must not be emitted"
    assert 1 in _by_instant(emitted)


def test_a_camera_with_no_video_is_absent_rather_than_invented(emitted):
    """The delivered sample has zero segments for its reference camera. A group must then be honestly
    smaller, never padded with a neighbouring camera's frame."""
    assert not any(f.cam_id == cam_id_for(*CAM_TRUTH[NO_VIDEO]) for f in emitted)


def test_a_dropped_frame_leaves_the_group_smaller(emitted):
    seen = _by_instant(emitted)
    assert set(seen[CAM1_DROPS_AT]) == {"front_wide"}
    assert set(seen[CAM1_DROPS_AT + 1]) == {"front_wide", "rear_narrow"}


def test_emission_is_monotonic_in_time(emitted):
    """Ingest records a session's start and end from the first and last frame it sees, so a reader that
    emitted a group's members in camera order rather than time order would misreport the session span."""
    ts = [f.ts_ns for f in emitted]
    assert ts == sorted(ts)


def test_stride_samples_whole_instants(rig_session):
    """Sampling groups rather than frames is what keeps each kept instant complete across the rig, which is
    the property multi-camera annotation depends on."""
    strided = _by_instant(list(rig_frames(rig_session, stride=2)))
    full = _by_instant(list(rig_frames(rig_session)))
    assert set(strided) == {1, 3, 5, 7}
    for gi in strided:
        assert set(strided[gi]) == set(full[gi]), "striding must not drop a camera from a kept instant"


def test_a_camera_can_be_selected_without_reading_the_others(rig_session):
    frames = list(rig_frames(rig_session, cameras=["cam1"]))
    assert frames and {f.cam_id for f in frames} == {"rear_narrow"}


def test_max_groups_bounds_the_read(rig_session):
    assert len(_by_instant(list(rig_frames(rig_session, max_groups=2)))) == 2


def test_a_directory_that_is_not_a_rig_session_is_refused(tmp_path):
    """Silently returning nothing would make a mistyped path look like an empty drive."""
    with pytest.raises(FileNotFoundError):
        read_index(tmp_path)
