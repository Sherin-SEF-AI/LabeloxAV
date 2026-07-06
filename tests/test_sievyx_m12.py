"""SIEVYX M12 tests: maneuver recognition from trajectories, unsupervised rare-cluster discovery, core-set
batch coverage beating naive selection, failure-driven mining, and ODD coverage-gap detection."""

import numpy as np

from services.sievyx.batch import select_coreset
from services.sievyx.discovery import discover_rare_clusters
from services.sievyx.failure_mining import mine_from_failures
from services.sievyx.maneuver import recognize
from services.sievyx.odd import coverage_gaps


def _traj_from_headings(headings_deg, ds=1.0):
    p = [(0.0, 0.0)]
    for h in headings_deg:
        x, y = p[-1]
        p.append((x + ds * np.sin(np.radians(h)), y + ds * np.cos(np.radians(h))))
    return [{"t": i, "x": x, "y": y} for i, (x, y) in enumerate(p)]


def test_maneuver_u_turn():
    traj = _traj_from_headings(list(np.linspace(0, 180, 20)))
    assert recognize(traj)["maneuver"] == "u_turn"


def test_maneuver_straight():
    traj = _traj_from_headings([0] * 20)
    assert recognize(traj)["maneuver"] == "straight"


def test_maneuver_turn():
    traj = _traj_from_headings(list(np.linspace(0, 80, 20)))
    assert recognize(traj)["maneuver"] == "unprotected_turn"


def test_maneuver_jaywalk_is_mostly_lateral():
    # a pedestrian crossing: large lateral, tiny forward
    traj = [{"t": i, "x": i * 0.5, "y": 0.05 * i} for i in range(12)]
    assert recognize(traj)["maneuver"] == "jaywalk"


def test_discover_rare_cluster():
    rng = np.random.default_rng(0)
    dense = np.array([1.0, 0, 0]) + rng.normal(0, 0.01, size=(12, 3))     # common blob
    rare = np.array([0.0, 0, 1.0]) + rng.normal(0, 0.01, size=(3, 3))     # small, far
    X = np.vstack([dense, rare])
    ids = [f"d{i}" for i in range(12)] + [f"r{i}" for i in range(3)]
    clusters = discover_rare_clusters(X, ids, min_size=2, sim_thr=0.6)
    assert clusters, "expected at least one cluster"
    rarest = clusters[0]
    assert all(i.startswith("r") for i in rarest["member_ids"])          # the far small cluster is rarest
    assert rarest["size"] == 3


def test_coreset_covers_better_than_contiguous():
    # points on a line 0..19; a diverse cover should spread out more than the first 4
    X = np.array([[float(i), 0.0] for i in range(20)])
    ids = [str(i) for i in range(20)]
    picked = set(select_coreset(X, ids, budget=4))
    picked_idx = sorted(int(i) for i in picked)

    def min_gap(idx):
        return min(np.diff(idx)) if len(idx) > 1 else 0

    assert min_gap(picked_idx) > min_gap([0, 1, 2, 3])                    # more spread than the naive first-k
    assert {0, 19}.issubset(set(picked_idx))                             # covers the extremes


def test_failure_mining_ranks_near_failures_first():
    fails = [[1.0, 0.0, 0.0]]
    pool = [[0.95, 0.05, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    ranked = mine_from_failures(fails, pool, ["a", "b", "c"], k=3)
    assert ranked[0]["id"] == "a" and ranked[0]["similarity"] > ranked[1]["similarity"]


def test_odd_coverage_gaps():
    fleet = {"day_clear": 900, "night_rain": 10}
    target = {"day_clear": 0.5, "night_rain": 0.3, "dust": 0.2}
    rep = coverage_gaps(fleet, target)
    cells = {g["cell"]: g for g in rep["gaps"]}
    assert "night_rain" in cells and "dust" in cells        # both underrepresented
    assert "day_clear" not in cells                          # over-covered, no gap
    assert cells["dust"]["missing"] is True                  # dust entirely absent
    assert rep["gaps"][0]["gap"] >= rep["gaps"][-1]["gap"]   # sorted by gap
