"""Live RTSP ingest for fixed cameras.

The Sec pack could ingest a recorded clip and nothing else, so every deployment was retrospective: footage
arrived as a file, was analysed later, and an incident was discovered after it mattered. A security
deployment whose cameras are live and whose analysis is not is doing half the job.

The hard part of live ingest is not reading the stream, it is deciding what to keep. A camera at 25 fps
produces 2.16 million frames a day, almost all of which show the same empty corridor, and storing them is
both ruinous and useless: the corpus fills with the background and the interesting minute is a rounding
error in it. So this samples, and the sampling is motion-aware:

- **A frame is kept when it differs enough from the running background**, which for a fixed camera is a
  genuinely good motion detector and costs almost nothing.
- **A slow heartbeat is kept regardless**, so a camera whose scene never changes still proves it was alive
  and the background model has something recent to rebuild from after a lighting change.
- **A burst limit caps a busy minute**, because a swaying tree in wind is motion and would otherwise fill
  the disk with it.

Reconnection is expected rather than exceptional. A camera on a building's network goes away: the stream
is retried with backoff and the gap is recorded on the session, so a hole in the timeline is visible as a
hole rather than as a period when nothing happened.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

from core.logging import get_logger

log = get_logger("sec_rtsp")


class RtspUnavailable(RuntimeError):
    """The stream could not be opened, and why."""


@dataclass
class SamplingPolicy:
    """What to keep out of a live stream.

    Every default here is a storage decision. At 25 fps a camera produces 2.16 million frames a day; the
    question is never "can we keep them" but "which fraction is worth keeping".
    """

    # Mean absolute difference from the background, 0..255, above which a frame is interesting.
    motion_threshold: float = 6.0
    # Keep at least one frame this often whatever happens, so a still scene still proves liveness and the
    # background model has something recent to rebuild from after a lighting change.
    heartbeat_seconds: float = 30.0
    # And at most this many per minute, so wind in a tree does not fill the disk.
    max_frames_per_minute: int = 60
    # How fast the background forgets. Higher adapts quicker to a light being switched on and is quicker to
    # absorb a person who stops moving, which is the trade.
    background_alpha: float = 0.02
    # Frames to prime the background before anything is judged against it. Without this the first frame is
    # the background and the second is "motion".
    warmup_frames: int = 20
    min_interval_seconds: float = 0.2


@dataclass
class StreamStats:
    read: int = 0
    kept: int = 0
    dropped_similar: int = 0
    dropped_rate_limited: int = 0
    reconnects: int = 0
    gaps: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"read": self.read, "kept": self.kept, "dropped_similar": self.dropped_similar,
                "dropped_rate_limited": self.dropped_rate_limited, "reconnects": self.reconnects,
                "gaps": self.gaps,
                "kept_fraction": round(self.kept / self.read, 4) if self.read else 0.0}


class MotionSampler:
    """Decides frame by frame whether to keep it. Pure and synchronous so it is testable without a camera."""

    def __init__(self, policy: SamplingPolicy | None = None) -> None:
        self.policy = policy or SamplingPolicy()
        self._background: np.ndarray | None = None
        self._seen = 0
        self._last_kept_at: float | None = None
        self._minute_start: float | None = None
        self._minute_count = 0

    def score(self, frame_gray: np.ndarray) -> float:
        """Mean absolute difference from the background, before it is updated."""
        if self._background is None:
            return 0.0
        return float(np.mean(np.abs(frame_gray.astype(np.float32) - self._background)))

    def consider(self, frame_gray: np.ndarray, now: float | None = None) -> tuple[bool, str, float]:
        """Returns (keep, reason, motion). The reason is carried so a session can report why it kept what
        it kept, which is the only way to tune a threshold after the fact."""
        now = time.time() if now is None else now
        motion = self.score(frame_gray)
        self._seen += 1
        self._update_background(frame_gray)

        if self._seen <= self.policy.warmup_frames:
            return False, "warmup", motion

        if self._minute_start is None or now - self._minute_start >= 60.0:
            self._minute_start, self._minute_count = now, 0

        since = None if self._last_kept_at is None else now - self._last_kept_at
        heartbeat = since is None or since >= self.policy.heartbeat_seconds

        if not heartbeat and since is not None and since < self.policy.min_interval_seconds:
            return False, "too soon", motion

        interesting = motion >= self.policy.motion_threshold
        if not interesting and not heartbeat:
            return False, "no motion", motion

        # The heartbeat is exempt from the rate limit on purpose: it is the frame that proves the camera is
        # alive, and dropping it during a busy minute is exactly when its absence would be misread.
        if interesting and not heartbeat and self._minute_count >= self.policy.max_frames_per_minute:
            return False, "rate limited", motion

        self._last_kept_at = now
        self._minute_count += 1
        return True, ("heartbeat" if heartbeat and not interesting else "motion"), motion

    def _update_background(self, frame_gray: np.ndarray) -> None:
        f = frame_gray.astype(np.float32)
        if self._background is None or self._background.shape != f.shape:
            self._background = f.copy()
            return
        a = self.policy.background_alpha
        self._background = (1 - a) * self._background + a * f


def open_stream(url: str, *, timeout_seconds: float = 10.0):
    """Open an RTSP URL with OpenCV, refusing by name rather than returning something unusable."""
    import cv2

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # A small buffer, because a live stream that buffers is a live stream that is behind: reconnecting to
    # a camera and receiving thirty seconds of stale video is worse than dropping it.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    except Exception:  # noqa: BLE001 - not every backend honours it
        pass

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if cap.isOpened():
            return cap
        time.sleep(0.25)
    cap.release()
    raise RtspUnavailable(
        f"could not open {url.split('@')[-1]!r} within {timeout_seconds}s. Check the camera is reachable "
        "and the credentials are correct; OpenCV needs FFMPEG support for RTSP.")


def sample_stream(url: str, *, policy: SamplingPolicy | None = None,
                  max_frames: int = 500, max_seconds: float = 300.0,
                  max_reconnects: int = 3) -> Iterator[tuple[np.ndarray, dict]]:
    """Yield the frames worth keeping, with why each was kept.

    A generator so the caller owns persistence: this module decides what is interesting and never decides
    where it goes, which is what lets the same sampler serve an ingest job and a live preview.
    """
    import cv2

    sampler = MotionSampler(policy)
    stats = StreamStats()
    started = time.time()
    reconnects = 0
    cap = open_stream(url)

    try:
        while stats.kept < max_frames and (time.time() - started) < max_seconds:
            ok, frame = cap.read()
            if not ok:
                if reconnects >= max_reconnects:
                    stats.gaps.append({"at": time.time(), "reason": "stream ended"})
                    break
                # Backoff, and record the gap. A hole in the timeline must be visible as a hole rather
                # than as a period when nothing happened.
                reconnects += 1
                stats.reconnects = reconnects
                gap_start = time.time()
                cap.release()
                time.sleep(min(2 ** reconnects, 10))
                try:
                    cap = open_stream(url)
                except RtspUnavailable:
                    stats.gaps.append({"at": gap_start, "reason": "reconnect failed"})
                    break
                stats.gaps.append({"at": gap_start, "seconds": round(time.time() - gap_start, 2),
                                   "reason": "reconnected"})
                continue

            stats.read += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            keep, reason, motion = sampler.consider(gray)
            if not keep:
                if reason == "rate limited":
                    stats.dropped_rate_limited += 1
                elif reason in ("no motion", "too soon"):
                    stats.dropped_similar += 1
                continue
            stats.kept += 1
            yield frame, {"reason": reason, "motion": round(motion, 3),
                          "ts_ns": int(time.time() * 1e9), "index": stats.read}
    finally:
        cap.release()
        log.info("sec.rtsp_sampled", **stats.as_dict())


async def ingest_stream(url: str, camera_id: str, *, city: str | None = None,
                        policy: SamplingPolicy | None = None, max_frames: int = 500,
                        max_seconds: float = 300.0, pack_id: str | None = None) -> dict:
    """Sample a live camera into a real session, using the Sec pack's own persistence path.

    Capability-gated on `rtsp`, which only the Sec pack declares. A live camera feeding an AV corpus would
    be a surveillance deployment wearing a driving product's clothes.
    """
    import cv2

    from services.anpr.recognize import AnprNotAuthorised
    from services.domain import default_pack_id, has_capability

    pid = pack_id or default_pack_id()
    if not has_capability("rtsp", pid):
        raise AnprNotAuthorised(
            f"live RTSP ingest is not authorised for pack {pid!r}; a pack must declare 'rtsp'.")

    from core.storage import get_object_store
    from db.models import Frame
    from db.models import Session as DbSession
    from db.session import get_sessionmaker

    store = get_object_store()
    store.ensure_bucket()
    kept: list[dict] = []
    first_ts = last_ts = None

    async with get_sessionmaker()() as db:
        sess = DbSession(vehicle_id=None, city=city, start_ts_ns=int(time.time() * 1e9),
                         end_ts_ns=int(time.time() * 1e9), sensors={"rtsp": True},
                         pack_id=pid, ontology_version="labelox-sec-0.1.0")
        db.add(sess)
        await db.flush()

        for frame, meta in sample_stream(url, policy=policy, max_frames=max_frames,
                                         max_seconds=max_seconds):
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                continue
            ts = int(meta["ts_ns"])
            first_ts = ts if first_ts is None else first_ts
            last_ts = ts
            uri = store.put_bytes(f"frames/{sess.session_id}/{ts}.jpg", buf.tobytes(), "image/jpeg")
            h, w = frame.shape[:2]
            db.add(Frame(session_id=sess.session_id, ts_ns=ts, cam_id=camera_id, img_uri=uri,
                         width=w, height=h, quality=0.9, ego_speed=None))
            kept.append(meta)

        if first_ts is not None:
            sess.start_ts_ns, sess.end_ts_ns = first_ts, last_ts
        await db.commit()
        session_id = str(sess.session_id)

    log.info("sec.rtsp_ingested", session=session_id, camera=camera_id, frames=len(kept))
    return {"session_id": session_id, "camera_id": camera_id, "frames": len(kept),
            "pack_id": pid,
            "reasons": {r: sum(1 for k in kept if k["reason"] == r)
                        for r in {k["reason"] for k in kept}},
            # Reported so a threshold can be tuned after the fact: a stream that kept everything and one
            # that kept nothing both need a different number, and only this says which.
            "start_ts_ns": first_ts, "end_ts_ns": last_ts}
