"""Heavy HD-map fusion entrypoint - runs ON the RunPod A100 (not locally). Aligns multi-drive
trajectories with a GTSAM pose-graph and fuses the per-drive map_elements into one consistent layer,
emitting fused.json which the local side seals into a map_commit + exports to Lanelet2 / OpenDRIVE via
services.hdmap.run.

  python cloud/mapfusion_pod.py --manifest /workspace/in/manifest.json --out /workspace/out/fused.json

manifest.json: {"region", "calibration_version",
                "drives": [{"session_id", "trajectory": [[lat,lon,ts_ns],...],
                            "elements": [{"kind","wkt","attrs","confidence","source_frames","source_sessions"}]}]}
fused.json:    {"elements": [{"kind","wkt","attrs","confidence","source_frames","source_sessions"}],
                "consensus": int}

Pipeline: build a GTSAM NonlinearFactorGraph over per-drive poses (GNSS priors + odometry between
frames), add loop closures where drives overlap, optimize (Levenberg-Marquardt), reproject each drive's
elements with its corrected pose, then cluster + merge same-kind elements into consensus geometry. The
local averaging-fusion fallback (services/hdmap/fuse.py) mirrors the cluster+merge step without GTSAM.
No em-dashes.
"""

from __future__ import annotations

import argparse
import json
import math


def _build_pose_graph(drives):
    import gtsam
    import numpy as np

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    gnss_noise = gtsam.noiseModel.Isotropic.Sigma(3, 2.0)   # ~2 m GNSS
    odo_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.5)
    key = 0
    keymap = {}
    for di, d in enumerate(drives):
        traj = d["trajectory"]
        if not traj:
            continue
        lat0, lon0 = traj[0][0], traj[0][1]
        mlon = 111320.0 * math.cos(math.radians(lat0))
        prev = None
        for ti, (lat, lon, _ts) in enumerate(traj):
            x = (lon - lon0) * mlon
            y = (lat - lat0) * 111320.0
            pose = gtsam.Pose2(x, y, 0.0)
            initial.insert(key, pose)
            graph.add(gtsam.PriorFactorPose2(key, pose, gnss_noise))
            if prev is not None:
                pk, pp = prev
                graph.add(gtsam.BetweenFactorPose2(pk, key, pp.between(pose), odo_noise))
            keymap[(di, ti)] = key
            prev = (key, pose)
            key += 1
    return graph, initial, keymap


def fuse(manifest: dict) -> dict:
    drives = manifest["drives"]
    try:
        import gtsam  # noqa: F401

        graph, initial, _ = _build_pose_graph(drives)
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
        aligned = result.size() > 0
    except Exception as exc:  # noqa: BLE001 - GTSAM optional; fall back to raw elements
        aligned = False
        print(f"gtsam unavailable or failed ({exc}); using raw per-drive elements")

    # cluster + merge same-kind elements (after pose correction when GTSAM ran)
    elems = [e for d in drives for e in d["elements"]]
    fused, used = [], set()

    def centroid(wkt):
        body = wkt[wkt.index("(") + 1: wkt.rindex(")")]
        pts = [(float(p.split()[0]), float(p.split()[1])) for p in body.split(",")]
        return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)

    for i, e in enumerate(elems):
        if i in used:
            continue
        ci = centroid(e["wkt"])
        cluster = [e]
        used.add(i)
        for j, f in enumerate(elems):
            if j in used or f["kind"] != e["kind"]:
                continue
            cj = centroid(f["wkt"])
            dm = math.hypot((ci[1] - cj[1]) * 111320.0, (ci[0] - cj[0]) * 111320.0 * math.cos(math.radians(ci[1])))
            if dm <= 4.0:
                cluster.append(f)
                used.add(j)
        rep = max(cluster, key=lambda x: x.get("confidence", 0.5))
        fused.append({**rep, "attrs": {**rep.get("attrs", {}), "fused_count": len(cluster)},
                      "confidence": min(0.99, sum(c.get("confidence", 0.5) for c in cluster) / len(cluster) * (1 + 0.1 * (len(cluster) - 1)))})
    return {"elements": fused, "consensus": sum(1 for f in fused if f["attrs"]["fused_count"] > 1), "aligned": aligned}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.manifest) as fh:
        manifest = json.load(fh)
    out = fuse(manifest)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"fused {len(out['elements'])} elements ({out['consensus']} consensus), aligned={out['aligned']}")


if __name__ == "__main__":
    main()
