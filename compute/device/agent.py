"""The on-rig agent: score, decide, queue, and tell the server what it never sent.

The selection itself is in selector.py and the descriptor in embed.py, both pure. This is the part with a
disk and a network, and the part where the interesting failure modes live.

**The dropped frames have to be declared.** A rig that uploads 8% of its day and says nothing about the
other 92% produces a corpus with an invisible sampling bias: every downstream rate, every class frequency,
every claim about coverage is computed over a filtered population that nothing records the shape of. So the
manifest carries the counts and the reasons, and the reason histogram is what makes a strange fleet
diagnosable ("busy scene" dominating means the density bar is wrong for this route). This is the same
argument as reporting what a quality certificate did not measure, applied to data collection.

**A signed manifest, because the rig is outside the trust boundary.** FORGYX already takes this position
deliberately: device telemetry cannot auto-demote a champion, precisely because devices are not trusted. The
same reasoning applies to a device claiming what it saw, so the manifest is HMAC-signed with the device key
and the server can tell a real report from an invented one.

**Uplink failure must not lose the decision.** A drive happens where there is no signal. Frames are queued
on disk with their manifest and drained when a link exists; the agent never blocks the camera loop on a
network call, because a frame missed while retrying an upload is a frame nobody can get back.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from compute.device.embed import describe
from compute.device.selector import Decision, SelectorConfig, StreamingSelector


@dataclass
class DeviceConfig:
    device_id: str
    secret: str
    vehicle_id: str = ""
    camera_id: str = "cam_f"
    backend: str = "tiled_histogram"
    spool_dir: str = "/var/spool/labelox"
    # Stop accepting new frames when the spool reaches this. A device that fills its disk stops recording
    # entirely, which is a far worse failure than dropping the marginal frame, so the cap is enforced here
    # rather than left to the filesystem.
    max_spool_mb: int = 4096
    selector: SelectorConfig = field(default_factory=SelectorConfig)


@dataclass
class FrameRecord:
    frame_ref: str
    ts_ns: int
    novelty: float
    reason: str


class DeviceAgent:
    """One camera's selection loop. Construct once per camera, feed it frames, drain the spool when online."""

    def __init__(self, cfg: DeviceConfig) -> None:
        self.cfg = cfg
        self.selector = StreamingSelector(cfg=cfg.selector)
        self.spool = Path(cfg.spool_dir)
        self.spool.mkdir(parents=True, exist_ok=True)
        self.kept: list[FrameRecord] = []
        self._dropped = 0
        self._spool_full = False

    def offer(self, image_bgr: np.ndarray, *, ts_ns: int, rare: bool = False,
              object_count: int = 0, frame_ref: str | None = None) -> Decision:
        """Offer one frame. Returns the decision; a kept frame is written to the spool.

        Never raises on a full spool. A device that starts throwing exceptions into the camera loop stops
        recording, and the drop is recorded and reported instead.
        """
        vec = describe(image_bgr, backend=self.cfg.backend)
        decision = self.selector.observe(vec, rare=rare, object_count=object_count)

        if decision.keep and self._spool_has_room():
            ref = frame_ref or f"{self.cfg.camera_id}_{ts_ns}"
            self._write(ref, image_bgr)
            self.kept.append(FrameRecord(ref, ts_ns, decision.novelty, decision.reason))
        elif decision.keep:
            # Wanted it, could not store it. Counted separately in the manifest, because a fleet dropping
            # frames for want of disk is a different problem from one selecting them away, and the two
            # would otherwise be indistinguishable in the reported keep rate.
            self._dropped += 1
            self._spool_full = True
            return Decision(False, "spool full", decision.novelty, decision.threshold)
        return decision

    def _spool_has_room(self) -> bool:
        used = sum(f.stat().st_size for f in self.spool.glob("*") if f.is_file())
        return used < self.cfg.max_spool_mb * 1024 * 1024

    def _write(self, ref: str, image_bgr: np.ndarray) -> None:
        import cv2

        ok, buf = cv2.imencode(".jpg", image_bgr)
        if ok:
            (self.spool / f"{ref}.jpg").write_bytes(buf.tobytes())

    def manifest(self) -> dict:
        """What this device saw, kept, and did not send.

        The dropped count and its reasons are the point. Without them the corpus records a filtered
        population and nothing anywhere records the shape of the filter.
        """
        stats = self.selector.stats
        return {
            "device_id": self.cfg.device_id,
            "vehicle_id": self.cfg.vehicle_id,
            "camera_id": self.cfg.camera_id,
            "descriptor_backend": self.cfg.backend,
            "frames_seen": stats["seen"],
            "frames_kept": len(self.kept),
            "frames_dropped": stats["seen"] - len(self.kept),
            "dropped_for_spool": self._dropped,
            "spool_full": self._spool_full,
            "keep_frac": stats["keep_frac"],
            "target_keep_frac": stats["target_keep_frac"],
            "final_threshold": stats["threshold"],
            "reasons": stats["reasons"],
            "kept": [asdict(r) for r in self.kept],
        }

    def sign(self, manifest: dict) -> str:
        """HMAC over the canonical manifest, with the device key.

        Devices sit outside the trust boundary, which FORGYX already assumes elsewhere. A signature does not
        make a device honest; it makes an unsigned or wrongly-signed report distinguishable from a real one.
        """
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        return hmac.new(self.cfg.secret.encode(), body, hashlib.sha256).hexdigest()

    def seal(self) -> dict:
        """The manifest and its signature, ready to upload."""
        m = self.manifest()
        return {"manifest": m, "signature": self.sign(m)}

    def drain(self, uploader) -> dict:
        """Send the spool, then clear only what was accepted.

        Clearing on send rather than on acknowledgement would lose a drive to one failed request. `uploader`
        returns the refs it accepted, so a partial upload leaves the rest queued for the next window.
        """
        sealed = self.seal()
        files = {r.frame_ref: self.spool / f"{r.frame_ref}.jpg" for r in self.kept}
        present = {ref: p for ref, p in files.items() if p.exists()}
        accepted = set(uploader(sealed, present))

        for ref in accepted:
            p = present.get(ref)
            if p and p.exists():
                p.unlink()
        self.kept = [r for r in self.kept if r.frame_ref not in accepted]
        return {"offered": len(present), "accepted": len(accepted), "remaining": len(self.kept)}

    def reset_spool(self) -> None:
        """Wipe the spool. Only for a device being re-provisioned; loses anything not yet uploaded."""
        shutil.rmtree(self.spool, ignore_errors=True)
        self.spool.mkdir(parents=True, exist_ok=True)
        self.kept = []
        self._spool_full = False


def verify_manifest(secret: str, manifest: dict, signature: str) -> bool:
    """Server side: check a device report before believing it. Constant-time."""
    body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")
