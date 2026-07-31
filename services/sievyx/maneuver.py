"""SIEVYX clip-level maneuver recognition (M12). Mining at the scenario level, not just the frame level: a
track's trajectory over a clip is reduced to a compact motion descriptor (net turn, lateral displacement,
speed profile, path curvature) and classified into the maneuvers that matter on Indian roads (cut-in,
unprotected turn, U-turn, lane change, jaywalk, straight). The descriptor doubles as the clip embedding for
scenario retrieval and clustering.

Pure over a trajectory (a list of timestamped positions), so it is testable with constructed maneuvers."""

from __future__ import annotations

import math

import numpy as np

# Below this a step is a difference of two noisy positions and nothing else, so the heading it implies is
# noise. The old floor was 1e-6 metres, which admitted a 14 cm step reading +90 degrees.
MIN_SEGMENT_M = 0.25

MANEUVERS = ["cut_in", "unprotected_turn", "u_turn", "lane_change", "jaywalk", "straight"]


def trajectory_features(traj: list[dict]) -> dict:
    """Reduce a trajectory (list of {t, x, y}, x lateral and y forward in a BEV/ego frame, metres) to a motion
    descriptor. Robust to short tracks."""
    if len(traj) < 3:
        return {"n": len(traj), "net_turn_deg": 0.0, "lateral_disp": 0.0, "forward_disp": 0.0,
                "path_len": 0.0, "curvature": 0.0, "lateral_ratio": 0.0}
    p = np.array([[float(t["x"]), float(t["y"])] for t in traj], dtype=np.float64)
    d = np.diff(p, axis=0)
    seg = np.linalg.norm(d, axis=1)
    path_len = float(seg.sum())
    net = p[-1] - p[0]
    # Turn accumulated along the path, from segments long enough to have a direction.
    #
    # Two things were wrong before. Every segment counted however short, so a 14 cm step between two noisy
    # positions contributed a heading of +90 degrees; and the turn was read as the unwrapped difference
    # between the first and last heading, which lets that noise drift without bound. A track that merely
    # approached the camera accumulated 438 degrees, and four fifths of the corpus classified as U-turns.
    #
    # Summing per-step turns, each wrapped to the shorter way round, is the honest measure: it is the total
    # curvature for a real turn, and for a jittering track the steps are small and signed and largely cancel
    # rather than accumulating. The length floor does the rest, since it is the short steps that carry no
    # direction at all.
    moving = seg > MIN_SEGMENT_M
    if moving.sum() >= 2:
        head = np.arctan2(d[moving][:, 0], d[moving][:, 1])   # heading relative to the forward axis
        steps = (np.diff(head) + math.pi) % (2 * math.pi) - math.pi
        net_turn = float(np.degrees(steps.sum()))
    else:
        net_turn = 0.0
    lateral = float(net[0])
    forward = float(net[1])
    lateral_ratio = abs(lateral) / (abs(forward) + abs(lateral) + 1e-6)
    curvature = abs(net_turn) / (path_len + 1e-6)
    return {"n": len(traj), "net_turn_deg": round(net_turn, 2), "lateral_disp": round(lateral, 3),
            "forward_disp": round(forward, 3), "path_len": round(path_len, 3),
            "curvature": round(curvature, 4), "lateral_ratio": round(lateral_ratio, 4)}


def classify_maneuver(feat: dict) -> dict:
    """Classify the motion descriptor into a maneuver with a confidence."""
    turn = abs(feat["net_turn_deg"])
    lat = abs(feat["lateral_disp"])
    fwd = feat["forward_disp"]
    lat_ratio = feat["lateral_ratio"]

    if turn >= 150:
        m, c = "u_turn", min(1.0, turn / 180)
    elif turn >= 45:
        m, c = "unprotected_turn", min(1.0, turn / 90)
    elif lat_ratio >= 0.7 and fwd < 2.0:
        m, c = "jaywalk", min(1.0, lat_ratio)                      # mostly-lateral, little forward: crossing
    elif 1.5 <= lat <= 5.0 and turn < 30 and fwd > lat:
        m, c = "cut_in" if lat < 3.0 else "lane_change", 0.7        # sideways shift with forward motion
    elif lat >= 1.0 and turn < 20:
        m, c = "lane_change", 0.6
    else:
        m, c = "straight", max(0.5, 1.0 - lat_ratio)
    return {"maneuver": m, "confidence": round(float(c), 3), "features": feat}


def recognize(traj: list[dict]) -> dict:
    return classify_maneuver(trajectory_features(traj))
