"""Reading a synchronised multi-camera rig capture, using the session's own alignment index.

A BluRabbit rig session ships six cameras as mp4 segments plus sidecar CSVs, and an `alignment/` directory
that exists because the raw session describes itself wrongly in two ways. Both are the kind of thing that
produces a plausible corpus rather than an error, so they are worth stating.

**The mp4 timestamps are per-camera, not session-global.** Each container's PTS is relative to that camera's
own pipeline start, and the six pipelines started staggered by up to 667 ms. Every file claims to begin at
zero. Aligning on PTS or on frame number puts cameras up to twenty frames apart while looking perfectly
synchronised, so this reader uses `session_t_ns` from the index and never touches `pts_ns_original`.

**Sidecar row N of a segment is not frame N of that segment's mp4.** In 14 of 48 segments a boundary frame
is logged to the neighbouring segment's sidecar. Per-camera totals reconcile exactly and both sequences are
in capture order, so the index resolves it by walking segments with real decoded counts and publishing
`file` plus `frame_in_file`. Those are what this seeks on.

Two more things the index records that this reader honours rather than smooths over.

**Camera positions in `manifest.json` are wrong on all six channels**, corrected from the footage after the
fact. The corrected `position` and `role` in `frames.csv` are used, so `cam_id` reads `front_wide` rather
than a channel name that means the opposite of what it says.

**Each camera keeps its own instant, not the group's.** The residual spread is p50 28.2 ms against a 33.3 ms
frame period, because the cameras are free-running rather than trigger-synced. That is physical and the
index says so. Stamping all six members of a group with the reference camera's time would erase a real
28 ms of ego motion and let the corpus claim a sync it does not have, so each frame carries its own
`session_t_ns` and the grouping is left to `services/multicam/sync.py`, which records the spread it finds.

Six sequential readers advanced in step rather than random seeks: a rig session is 48 segments and seeking
per frame is both slow and unreliable across codecs, while the sync groups and each camera's frames both
advance monotonically, so stepping forward is a merge over six ordered streams.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

import cv2

from core.logging import get_logger
from services.ingest.types import RawFrame

log = get_logger("reader_rig")


def _session_dir(root: Path) -> Path:
    """The session directory, whether root is it or contains it."""
    if (root / "alignment" / "frames.csv").exists():
        return root
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "alignment" / "frames.csv").exists():
            return child
    raise FileNotFoundError(f"no alignment/frames.csv under {root}; this is not a rig session")


def read_index(root: Path) -> dict:
    """The alignment index: per-frame rows, sync groups, and the epoch that makes session time absolute.

    The epoch is derived as `real_ns - session_t_ns`, which the six cameras agree on to about 1.5
    microseconds. Taking the median rather than any one camera's value avoids inheriting whichever channel
    happened to be read first.
    """
    sdir = _session_dir(root)
    frames = list(csv.DictReader((sdir / "alignment" / "frames.csv").open()))
    groups = list(csv.DictReader((sdir / "alignment" / "sync_groups.csv").open()))
    if not frames or not groups:
        raise ValueError(f"alignment index under {sdir} is empty")

    epochs = sorted(int(r["real_ns"]) - int(r["session_t_ns"]) for r in frames)
    epoch_ns = epochs[len(epochs) // 2]

    cams: dict[str, dict] = {}
    for r in frames:
        cams.setdefault(r["camera_id"], {"position": r["position"], "role": r["role"]})

    return {"session_dir": sdir, "frames": frames, "groups": groups,
            "epoch_ns": epoch_ns, "cameras": cams}


def cam_id_for(position: str, role: str) -> str:
    """`front_wide`, `rear_narrow`, and so on.

    Built from the corrected position and role rather than the channel name, because the channel names in
    this session's manifest describe the wrong direction on every one of the six.
    """
    return f"{position}_{role}".lower()


class _CameraStream:
    """One camera's segments, read forward only.

    Holds a single open capture and advances to the requested frame with `grab()`, which pulls a frame
    without paying to decode it. A rig session is 48 segments and per-frame seeking is slow and unreliable
    across codecs, while both the groups and each camera's frames advance monotonically.
    """

    def __init__(self, cam_dir: Path) -> None:
        self.dir = cam_dir
        self.cap: cv2.VideoCapture | None = None
        self.file: str | None = None
        self.pos = -1

    def frame_at(self, file: str, index: int):
        if file != self.file:
            if self.cap is not None:
                self.cap.release()
            path = self.dir / file
            if not path.exists():
                return None
            self.cap = cv2.VideoCapture(str(path))
            self.file, self.pos = file, -1
        if self.cap is None:
            return None
        if index < self.pos:
            # The index went backwards, which should not happen within a camera. Reopening is correct and
            # cheap at segment scale, and silently returning the wrong frame would not be.
            self.cap.release()
            self.cap = cv2.VideoCapture(str(self.dir / file))
            self.pos = -1
        while self.pos < index - 1:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, img = self.cap.read()
        if not ok:
            return None
        self.pos = index
        return img

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def rig_frames(root: Path, *, stride: int = 1, max_groups: int | None = None,
               cameras: list[str] | None = None) -> Iterator[RawFrame]:
    """Frames from every camera of every usable sync group, in session-time order.

    `stride` samples groups: a rig at 30 fps yields 13,928 groups over eight minutes, and ingesting all six
    cameras of every one is 83,559 frames for a single drive. Sampling groups rather than frames keeps each
    instant complete across the rig, which is the property the multi-camera annotation depends on.

    Groups flagged `usable=0` are skipped. In this session that is the first 18, recorded while the cameras
    were still starting up staggered, and admitting them would put a "synchronised" group in the corpus with
    a 428 ms spread.
    """
    idx = read_index(root)
    sdir: Path = idx["session_dir"]
    epoch = idx["epoch_ns"]
    cam_meta = idx["cameras"]

    wanted = set(cameras) if cameras else set(cam_meta)
    streams = {c: _CameraStream(sdir / c) for c in wanted}

    # session_t_ns per (camera, file, frame_in_file), so each frame carries its own instant rather than the
    # group's reference time.
    t_of: dict[tuple[str, str, int], int] = {}
    for r in idx["frames"]:
        if r["camera_id"] in wanted:
            t_of[(r["camera_id"], r["file"], int(r["frame_in_file"]))] = int(r["session_t_ns"])

    emitted = skipped_unusable = missing = 0
    groups = [g for g in idx["groups"] if g.get("usable") == "1"]
    skipped_unusable = len(idx["groups"]) - len(groups)
    selected = groups[::max(1, stride)]
    if max_groups:
        selected = selected[:max_groups]

    try:
        for g in selected:
            members = []
            for cam in sorted(wanted):
                file = g.get(f"{cam}__file")
                raw_idx = g.get(f"{cam}__frame")
                if not file or raw_idx in (None, ""):
                    missing += 1
                    continue
                fi = int(raw_idx)
                ts = t_of.get((cam, file, fi))
                if ts is None:
                    missing += 1
                    continue
                members.append((ts, cam, file, fi))

            # Within a group the six cameras genuinely expose at different instants, so emitting in their
            # own time order keeps the stream monotonic for a consumer that records a session's start and
            # end from the first and last frame it sees.
            for ts, cam, file, fi in sorted(members):
                img = streams[cam].frame_at(file, fi)
                if img is None:
                    missing += 1
                    continue
                meta = cam_meta[cam]
                yield RawFrame(ts_ns=epoch + ts,
                               cam_id=cam_id_for(meta["position"], meta["role"]),
                               image_bgr=img)
                emitted += 1
    finally:
        for s in streams.values():
            s.close()
        log.info("reader_rig.done", emitted=emitted, groups=len(selected),
                 skipped_unusable=skipped_unusable, missing=missing, stride=stride)
