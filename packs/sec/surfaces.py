"""The Sec pack's ZonePolicy and StreamSource, binding this pack's geometry and camera IO to the contract.

These are thin on purpose. The engine used to reach into `packs.sec.zones` and `packs.sec.rtsp` directly,
which the import-linter contract forbids and which quietly made the security pack a hard dependency of the
engine core: an AV-only deployment still imported RTSP handling, and a third pack could not have supplied its
own spatial rules without editing engine code.

Nothing here is new behaviour. The functions being wrapped are unchanged; only the seam moved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from packs.base import Crossing
from packs.sec import rtsp, zones


class SecZonePolicy:
    """Polygon and tripwire rules on a fixed camera."""

    def validate(self, kind: str, rule: str, points: Sequence, dwell_seconds: float | None) -> None:
        zones.validate_zone(kind, rule, list(points), dwell_seconds)

    def evaluate_track(self, zone: Mapping[str, object],
                       samples: Sequence[Mapping[str, object]]) -> Sequence[Crossing]:
        return zones.evaluate_track(dict(zone), list(samples))


class SecStreamSource:
    """Sampling a live RTSP camera into a session."""

    # Exposed on the surface so the engine can answer "the camera was unreachable" (502) rather than
    # "this server broke" (500) without importing the pack's exception type.
    unavailable_error: type[Exception] = rtsp.RtspUnavailable

    def sampling_policy(self, **overrides: object) -> rtsp.SamplingPolicy:
        # Only the keys the caller actually set are forwarded, so the pack's own defaults stay authoritative
        # rather than being overwritten with a None the engine never meant to express.
        return rtsp.SamplingPolicy(**{k: v for k, v in overrides.items() if v is not None})

    async def ingest(self, url: str, camera_id: str, *, city: str | None = None,
                     policy: object | None = None, max_frames: int | None = None,
                     max_seconds: float | None = None, pack_id: str | None = None) -> dict:
        return await rtsp.ingest_stream(url, camera_id, city=city, policy=policy,
                                        max_frames=max_frames, max_seconds=max_seconds,
                                        pack_id=pack_id)
