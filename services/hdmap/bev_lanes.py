"""Lanes in a top-down view, as a map: splines, ground coordinates, ego-pose propagation and map export (#7).

The BEV warp (`bev_view.py`) made lanes drawable on a flattened road. This module is what turns the lines drawn
there into something a map can hold:

- **Splines.** A lane is stored as control points (image pixels, like every other lane); what is drawn and what
  is exported is the Catmull-Rom curve through them. `catmull_rom` is the one definition, mirrored in
  web/lib/tools/drivingGeom.ts so the editor shows exactly the curve the export writes.
- **Ground coordinates.** Control points are lifted to the road plane through the resolved calibration (the same
  `ipm_pixel_to_vehicle` the warp uses), as (forward, left) metres from the camera's ground point. Points above
  the horizon have no ground point and are dropped, never clamped.
- **Propagation along the route.** Optical flow (the existing `/lanes/propagate`) follows paint; it drifts on a
  faded marking and cannot cross an occlusion. When the session has ego poses, a lane can instead be carried by
  geometry: lift it to the ground, place it in the world through the source frame's pose, and bring it back
  through the target frame's pose and calibration. The road does not move, so the lane lands where it is, as
  far as the flat-road assumption and the pose accuracy allow.
- **Export.** Lanes go to world coordinates through the frame's pose and the session's GNSS origin and out through
  `export.py` as Lanelet2 and OpenDRIVE, with the topology a person drew (`lane_relation`) as hints.
"""

from __future__ import annotations

import math

import numpy as np

from core.logging import get_logger

log = get_logger("bev_lanes")

LANE_TYPES = ("solid", "dashed", "double", "road_edge", "implicit", "fallback", "unknown", "stop_line")
# A propagated control point must land this far ahead of the camera to be kept: closer than the near edge of the
# BEV it is under the bonnet; further than the far cap it is drawn on interpolation (bev_view.MAX_FAR_M).
MIN_FORWARD_M = 2.0
_R_EARTH_M = 6_371_000.0


def catmull_rom(points: list[list[float]], samples: int = 8) -> list[list[float]]:
    """Uniform Catmull-Rom through the points, `samples` per segment, passing through every point.

    The end tangents come from mirroring the second point about the first (and the penultimate about the last),
    which keeps a two-point lane a straight line.
    """
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) < 3 or samples < 2:
        return p.tolist()
    ext = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    ts = np.linspace(0.0, 1.0, samples, endpoint=False)
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for t in ts:
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(p[-1])
    return [[round(float(a), 4), round(float(b), 4)] for a, b in out]


def image_to_ground(points: list[list[float]], cal) -> list[list[float]]:
    """Image pixels to (forward, left) metres on the road from the camera's ground point; above-horizon dropped."""
    from services.calibration.resolve import ipm_args
    from services.hdmap.georef import ipm_pixel_to_vehicle

    args = ipm_args(cal)
    out = []
    for p in points or []:
        if len(p) < 2:
            continue
        g = ipm_pixel_to_vehicle(float(p[0]), float(p[1]), **args)
        if g is not None:
            out.append([g[0], -g[1]])           # the IPM's lateral is to the right; the map's y is to the left
    return out


def ground_to_image(points: list[list[float]], cal, img_w: int | None = None, img_h: int | None = None,
                    margin_px: float = 0.0) -> list[list[float]]:
    """(forward, left) ground metres back to image pixels; behind-camera points (and, with a size, points
    outside the image by more than `margin_px`) are dropped."""
    from services.calibration.resolve import ipm_args
    from services.hdmap.georef import vehicle_to_ipm_pixel

    args = ipm_args(cal)
    out = []
    for f, left in points or []:
        uv = vehicle_to_ipm_pixel(float(f), -float(left), **args)
        if uv is None:
            continue
        if img_w is not None and img_h is not None and not (
                -margin_px <= uv[0] <= img_w + margin_px and -margin_px <= uv[1] <= img_h + margin_px):
            continue
        out.append([round(uv[0], 2), round(uv[1], 2)])
    return out


def ego_transform(pose, sensors: dict | None, cal) -> np.ndarray:
    """4x4 placing camera-ground (forward, left, 0) points in the session world frame at one instant.

    world <- pose unit <- ego (lever arm, KITTI's IMU -> Velodyne) <- camera ground point (the mount's x, y).
    """
    from services.lidar.aggregate.scene_frame import lever_arm, pose_matrix

    cam = np.eye(4)
    cam[0, 3], cam[1, 3] = float(cal.xyz_m[0]), float(cal.xyz_m[1])
    return pose_matrix(pose) @ lever_arm(sensors) @ cam


def transfer(points_ground: list[list[float]], t_src: np.ndarray, t_dst: np.ndarray) -> list[list[float]]:
    """Ground points seen from one instant, as seen from another (both 4x4 from `ego_transform`)."""
    if not points_ground:
        return []
    p = np.asarray(points_ground, dtype=float)
    h = np.c_[p, np.zeros(len(p)), np.ones(len(p))]
    world = h @ t_src.T
    # The road is the plane the lane was lifted onto; take the world point back at that plane's height in the
    # target frame rather than letting a pose's z or pitch lift it off the road.
    back = world @ np.linalg.inv(t_dst).T
    return [[round(float(x), 4), round(float(y), 4)] for x, y in back[:, :2]]


def carry_lane(control_points: list[list[float]], cal_src, cal_dst, t_src: np.ndarray, t_dst: np.ndarray,
               img_w: int, img_h: int, far_m: float = 60.0) -> list[list[float]] | None:
    """One lane's control points on the target frame, or None when fewer than two land in its view."""
    g = image_to_ground(control_points, cal_src)
    moved = [p for p in transfer(g, t_src, t_dst) if MIN_FORWARD_M <= p[0] <= far_m]
    pts = ground_to_image(moved, cal_dst, img_w, img_h, margin_px=2.0)
    return pts if len(pts) >= 2 else None


def travel_order(points_ground: list[list[float]]) -> list[list[float]]:
    """A lane seen ahead runs away from the vehicle: order its points by increasing forward distance."""
    if len(points_ground) >= 2 and points_ground[0][0] > points_ground[-1][0]:
        return points_ground[::-1]
    return points_ground


def enu_to_lonlat(east: float, north: float, lat0: float, lon0: float) -> tuple[float, float]:
    """The inverse of `services/intelligence/ego_pose.enu_offset` (same small-angle frame)."""
    lat = lat0 + math.degrees(north / _R_EARTH_M)
    lon = lon0 + math.degrees(east / (_R_EARTH_M * math.cos(math.radians(lat0))))
    return lon, lat


def lane_world_polyline(control_points: list[list[float]], cal, t_world: np.ndarray | None,
                        origin: tuple[float, float] | None, samples: int = 6) -> list[tuple[float, float]]:
    """A lane as a (lon, lat) polyline: lift, order, spline in metres, place in the world, georeference.

    With no pose the vehicle frame at that instant is the world; with no GNSS origin the local metres are written
    about (0, 0) and the caller must say the file is not georeferenced.
    """
    g = travel_order(image_to_ground(control_points, cal))
    if len(g) < 2:
        return []
    dense = catmull_rom(g, samples)
    if t_world is not None:
        p = np.asarray(dense, dtype=float)
        world = np.c_[p, np.zeros(len(p)), np.ones(len(p))] @ t_world.T
        dense = world[:, :2].tolist()
    lat0, lon0 = origin if origin else (0.0, 0.0)
    return [enu_to_lonlat(e, n, lat0, lon0) for e, n in dense]


def wkt_line(coords: list[tuple[float, float]]) -> str:
    return "LINESTRING(" + ", ".join(f"{lo:.9f} {la:.9f}" for lo, la in coords) + ")"


def hints_from_relations(relations: list[dict]) -> dict:
    """lane_relation rows (as dicts) to the export's topology hints: pairs, successors, regulatory elements.

    left_neighbour / right_neighbour are symmetric statements about one lane; both directions collapse to one
    (left, right) pair. predecessor is a successor read backwards.
    """
    pairs: set[tuple[str, str]] = set()
    succ: set[tuple[str, str]] = set()
    signals: dict[str, dict] = {}
    for r in relations:
        a, b, kind = str(r["lane_id"]), r.get("target_lane_id"), r["relation"]
        b = str(b) if b else None
        if kind == "right_neighbour" and b:
            pairs.add((a, b))
        elif kind == "left_neighbour" and b:
            pairs.add((b, a))
        elif kind == "successor" and b:
            succ.add((a, b))
        elif kind == "predecessor" and b:
            succ.add((b, a))
        elif kind in ("controlled_by", "stops_at"):
            reg = signals.setdefault(a, {"lane_ids": [a], "signal": None, "stop_line_id": None, "id": a})
            if kind == "stops_at" and b:
                reg["stop_line_id"] = b
            elif kind == "controlled_by":
                reg["signal_ref"] = str(r.get("target_object_id") or r.get("target_element_id"))
    # A boundary in two different pairs on the same side would build two lanelets over one line; keep the first.
    seen_left, seen_right, clean = set(), set(), []
    for left, right in sorted(pairs):
        if left in seen_left or right in seen_right:
            continue
        seen_left.add(left)
        seen_right.add(right)
        clean.append((left, right))
    return {"pairs": clean, "successors": sorted(succ), "regulatory": list(signals.values())}
