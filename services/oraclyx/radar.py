"""ORACLYX radar Doppler velocity fusion (M14). A radar return carries a range-rate (Doppler) that is far more
accurate than the frame-to-frame differencing a camera-only track uses for velocity. This fuses radar returns
onto a camera track to give each pseudo-GT box a calibrated radial velocity, and validates that velocity
against the ego motion the track already knows (GNSS/CAN derived) so a mis-associated return is rejected rather
than trusted. Capability-honest: with no radar returns present it returns the camera-derived velocity unchanged
and flags source="camera", never fabricating a Doppler reading. Pure over dicts, so association and validation
are testable."""

from __future__ import annotations

import math

from services.oraclyx.consensus import iou


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
