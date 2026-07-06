"""SANYX orchestration: run the six checks on a real session and persist a HealthReport (a session_health
row carrying the 0..100 score, the {pass, degraded, quarantine} decision, and the per-check evidence).

What runs depends on what the session actually has. Camera timestamps and sampled frames come from the Frame
rows (always present), so dropped-frames, exposure, and lens contamination always run. The inertial channels
(time-sync against the STM32 reference, IMU, GPS) come from the CHRONYX MCAP; when a session has no such
channels (e.g. an imported image dataset), those checks are omitted rather than fabricated, and the score
renormalizes over the checks that did run. Nothing here writes labels or mutates raw data; it writes only a
derived health verdict, exactly as the existing M-I.2 health check does."""

from __future__ import annotations

from uuid import UUID

import cv2
import numpy as np
from sqlalchemy import select

from core.config import get_settings
from core.logging import get_logger
from core.storage import get_object_store
from db.models import Frame, SessionHealth
from db.session import get_sessionmaker
from services.ingest.quality import score_frame
from services.sanyx import checks as C
from services.sanyx.score import aggregate

log = get_logger("sanyx.run")
INDEXER_VERSION = "sanyx-1"


def _well_exposed(res) -> bool:
    return res.accepted or ("blur" not in " ".join(res.reasons) and res.clip_fraction < 0.2)


def _sample_frames(store, frames: list[Frame], per_cam: int = 12) -> tuple[dict, dict, list]:
    """Group frame timestamps by camera and decode a temporal sample of images per camera. Returns
    (cam_ts_by_cam, {cam_id: [gray frames]})."""
    by_cam: dict[str, list[Frame]] = {}
    for f in frames:
        by_cam.setdefault(f.cam_id or "cam", []).append(f)
    cam_ts = {cam: sorted(int(f.ts_ns) for f in fs) for cam, fs in by_cam.items()}
    grays: dict[str, list[np.ndarray]] = {}
    exposures: list[dict] = []
    cfg = get_settings().ingest
    for cam, fs in by_cam.items():
        fs = sorted(fs, key=lambda f: f.ts_ns)
        step = max(1, len(fs) // per_cam)
        sample = fs[::step][:per_cam]
        gs: list[np.ndarray] = []
        for f in sample:
            try:
                buf = np.frombuffer(store.get_bytes(f.img_uri), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            except Exception as exc:  # noqa: BLE001  a missing blob must not abort the whole health run
                log.warning("sanyx.frame_read_failed", frame=str(f.frame_id), error=str(exc))
                continue
            if img is None:
                continue
            gs.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            res = score_frame(img, cfg)
            exposures.append({"well_exposed": _well_exposed(res), "mean_luma": res.mean_luma,
                              "clip_fraction": res.clip_fraction, "blur": res.blur})
        if gs:
            grays[cam] = gs
    return cam_ts, grays, exposures


def _pseudo_seq(ts: list[int]) -> list[int]:
    """Turn camera timestamps into a sequence index (round to the median frame interval), so a PTS gap of
    N missing frames shows up as a sequence gap in check_dropped_frames without stored sequence numbers."""
    if len(ts) < 2:
        return list(range(len(ts)))
    d = np.diff(sorted(ts))
    med = float(np.median(d)) or 1.0
    t0 = min(ts)
    return [int(round((t - t0) / med)) for t in sorted(ts)]


async def run_health(session_id: UUID, sample_per_cam: int = 12) -> dict:
    """Run SANYX over a session and persist a HealthReport. Returns the report payload."""
    cfg = get_settings().sanyx
    store = get_object_store()
    async with get_sessionmaker()() as db:
        frames = (await db.execute(
            select(Frame).where(Frame.session_id == session_id).order_by(Frame.ts_ns))).scalars().all()
        if not frames:
            report = {"score": 0.0, "decision": "quarantine", "verdict": "fail", "hard_failed": ["no_frames"],
                      "checks": [{"name": "file_integrity", "score": 0.0, "status": "quarantine",
                                  "detail": "no frames extracted for this session", "evidence": {}}]}
            db.add(SessionHealth(session_id=session_id, checks=report["checks"], verdict="fail",
                                 score=0.0, decision="quarantine", indexer_version=INDEXER_VERSION))
            await db.commit()
            return report

        cam_ts, grays, exposures = _sample_frames(store, frames, sample_per_cam)
        results: list[dict] = []

        # image + camera-timestamp checks (always available)
        seq_by_cam = {cam: _pseudo_seq(ts) for cam, ts in cam_ts.items()}
        results.append(C.check_dropped_frames(cfg, seq_by_cam))
        results.append(C.check_exposure(cfg, exposures))
        first_cam_grays = next(iter(grays.values()), [])
        results.append(C.check_lens_contamination(cfg, first_cam_grays))

        # inertial channels from the MCAP, when present; omitted (not fabricated) when the session has none
        inertial = await _load_inertial(db, session_id)
        if inertial is not None and inertial.get("imu_ts"):
            results.append(C.check_time_sync(cfg, inertial["imu_ts"], cam_ts,
                                             inertial.get("gnss_ts"), inertial.get("pps_ts")))
            if inertial.get("accel"):
                results.append(C.check_imu(cfg, inertial["accel"], inertial["gyro"], inertial["imu_ts"]))
            if inertial.get("fix_types"):
                results.append(C.check_gps(cfg, inertial["fix_types"], inertial["hdop"],
                                           inertial["gnss_ts"], inertial.get("rtk_flags")))

        report = aggregate(results, cfg)
        # M10: name the fault and hand the operator a remediation hint, from the same evidence
        from services.sanyx.rootcause import classify
        rc = classify(report["checks"]) if report["decision"] != "pass" else {"fault": None, "remediation": None}
        report["root_cause"] = rc["fault"]
        report["remediation"] = rc.get("remediation")
        db.add(SessionHealth(session_id=session_id, checks=report["checks"], verdict=report["verdict"],
                             score=report["score"], decision=report["decision"], root_cause=rc["fault"],
                             remediation=rc.get("remediation"), indexer_version=INDEXER_VERSION))
        await db.commit()
    log.info("sanyx.health", session=str(session_id), score=report["score"], decision=report["decision"],
             checks=len(report["checks"]), root_cause=report.get("root_cause"))
    return report


async def _load_inertial(db, session_id) -> dict | None:
    """Best-effort extraction of IMU/GNSS/PPS channels for a session from its CHRONYX MCAP. Returns None when
    the session has no inertial channels (an imported image dataset), so the inertial checks are simply not
    run rather than run on fabricated data."""
    try:
        from services.sanyx.inertial import extract_inertial
        return await extract_inertial(db, session_id)
    except Exception as exc:  # noqa: BLE001  missing/undecodable inertial channels are expected, not fatal
        log.info("sanyx.inertial_unavailable", session=str(session_id), reason=str(exc))
        return None
