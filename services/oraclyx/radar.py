"""ORACLYX radar Doppler velocity fusion (M14). A radar return carries a range-rate (Doppler) that is far more
accurate than the frame-to-frame differencing a camera-only track uses for velocity. This fuses radar returns
onto a camera track to give each pseudo-GT box a calibrated radial velocity, and validates that velocity
against the ego motion the track already knows (GNSS/CAN derived) so a mis-associated return is rejected rather
than trusted. Capability-honest: with no radar returns present it returns the camera-derived velocity unchanged
and flags source="camera", never fabricating a Doppler reading. Pure over dicts, so association and validation
are testable."""

from __future__ import annotations

import math


def _center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def associate_return(track_box: list[float], radar_returns: list[dict], gate_px: float = 60.0) -> dict | None:
    """Associate the nearest projected radar return to a track box within a pixel gate. Each return is
    {"px": [x, y], "range_rate": float}; px is the return projected into the image. Returns the associated
    return or None when nothing falls inside the gate."""
    cx, cy = _center(track_box)
    best, best_d = None, gate_px
    for r in radar_returns:
        px = r.get("px")
        if not px:
            continue
        d = math.hypot(px[0] - cx, px[1] - cy)
        if d <= best_d:
            best, best_d = r, d
    return best


def fuse_radar_velocity(track_box: list[float], camera_velocity: float, radar_returns: list[dict],
                        ego_speed: float = 0.0, tol_kmh: float = 12.0, gate_px: float = 60.0) -> dict:
    """Fuse a Doppler range-rate onto a track's velocity.

    camera_velocity and the returns' range_rate are radial speeds in km/h (closing positive). When a return
    associates inside the gate and its range-rate is consistent with the camera velocity offset by ego motion
    (within tol_kmh), the Doppler value is adopted (source="radar"); an inconsistent association is rejected as
    a likely mis-association and the camera velocity is kept (source="camera_rejected_radar"). With no
    association the camera velocity is kept (source="camera")."""
    r = associate_return(track_box, radar_returns, gate_px)
    if r is None:
        return {"velocity_kmh": round(camera_velocity, 3), "source": "camera", "residual_kmh": None}

    doppler = float(r["range_rate"])
    # the radar measures object radial speed in the ego frame; compare against the camera estimate net of ego
    expected = camera_velocity - ego_speed
    residual = abs(doppler - expected)
    if residual <= tol_kmh:
        return {"velocity_kmh": round(doppler + ego_speed, 3), "source": "radar",
                "residual_kmh": round(residual, 3)}
    return {"velocity_kmh": round(camera_velocity, 3), "source": "camera_rejected_radar",
            "residual_kmh": round(residual, 3)}


def fuse_track(track: list[dict], radar_by_frame: dict[int, list[dict]], ego_speed: float = 0.0,
               tol_kmh: float = 12.0) -> list[dict]:
    """Apply Doppler fusion frame by frame over a stitched track, returning per-frame velocity records."""
    out = []
    for b in track:
        returns = radar_by_frame.get(b["frame"], [])
        cam_v = float(b.get("camera_velocity", 0.0))
        rec = fuse_radar_velocity(b["bbox"], cam_v, returns, ego_speed, tol_kmh)
        out.append({"frame": b["frame"], **rec})
    return out


# --- 3D: Doppler onto a Track3D's velocity ------------------------------------------------------------------
#
# A radar measures one component of an object's velocity: the radial one, along the line of sight. A 3D track
# knows the whole vector, but from differencing box centres a tenth of a second apart, which is noisy where the
# radar is precise. Fusion keeps the track's tangential component and replaces its radial component with the
# radar's, after the same consistency gate as above: a Doppler reading that disagrees with the track by more
# than `tol_mps` is more likely a different object than a better measurement, and is rejected, not averaged.
# All speeds here are over the ground (ego motion removed) in m/s, and vectors are in one frame (the scan's
# ego frame), so the caller does the frame bookkeeping once.


def fuse_radial(track_vel: list[float], los: list[float], radar_radial: float | None,
                tol_mps: float = 3.0) -> dict:
    """Replace the radial component of `track_vel` (2D or 3D) with a Doppler reading along `los`.

    `los` is the line of sight from the radar to the object (any length). Returns the fused velocity, its
    speed, and which source the radial component came from: "radar", "track_rejected_radar" (the reading
    disagreed by more than `tol_mps`), or "track" (no reading)."""
    v = [float(x) for x in track_vel]
    u = [float(x) for x in los][: len(v)]
    n = math.sqrt(sum(x * x for x in u)) or 1.0
    u = [x / n for x in u]
    track_radial = sum(a * b for a, b in zip(v, u, strict=True))
    if radar_radial is None:
        return {"velocity": v, "speed_mps": round(math.sqrt(sum(x * x for x in v)), 3), "source": "track",
                "radial_track_mps": round(track_radial, 3), "residual_mps": None}
    residual = abs(float(radar_radial) - track_radial)
    if residual > tol_mps:
        return {"velocity": v, "speed_mps": round(math.sqrt(sum(x * x for x in v)), 3),
                "source": "track_rejected_radar", "radial_track_mps": round(track_radial, 3),
                "radial_radar_mps": round(float(radar_radial), 3), "residual_mps": round(residual, 3)}
    fused = [a + (float(radar_radial) - track_radial) * b for a, b in zip(v, u, strict=True)]
    return {"velocity": [round(x, 4) for x in fused], "speed_mps": round(math.sqrt(sum(x * x for x in fused)), 3),
            "source": "radar", "radial_track_mps": round(track_radial, 3),
            "radial_radar_mps": round(float(radar_radial), 3), "residual_mps": round(residual, 3)}


def fuse_track3d(points: list[dict], tol_mps: float = 3.0) -> dict:
    """Doppler fusion along one 3D track. Each point is {ts_ns, velocity (track, over ground), los, radial
    (radar, over ground) or None}. Returns per-point records and a summary saying how many points the radar
    spoke for, how many it was overruled on, and the median fused speed."""
    recs = []
    for p in points:
        r = fuse_radial(p["velocity"], p["los"], p.get("radial"), tol_mps)
        recs.append({"ts_ns": int(p["ts_ns"]), **r, "n_returns": int(p.get("n_returns", 0))})
    speeds = sorted(r["speed_mps"] for r in recs)
    by = {s: sum(1 for r in recs if r["source"] == s) for s in ("radar", "track_rejected_radar", "track")}
    return {"points": recs, "fused": by["radar"], "rejected": by["track_rejected_radar"], "no_radar": by["track"],
            "median_speed_mps": speeds[len(speeds) // 2] if speeds else None,
            "source": "radar" if by["radar"] else "track"}
