"""VERDYX M15 tests: track-level safety metrics (time-to-detection, fragmentation, ID switches), safety-weighted
recall (critical-object, TTC-weighted, near-miss slice), bootstrap CIs and paired significance, and continuous
shadow eval with protected-slice regression triage."""

from services.verdyx.safety_recall import critical_object_recall, near_miss_slice, ttc_weighted_recall
from services.verdyx.shadow import ShadowEval, regression_triage
from services.verdyx.stats import bootstrap_ci, paired_significance
from services.verdyx.track_metrics import id_switches, time_to_detection, track_fragmentation


def _gt(frames):
    return [{"frame": f, "bbox": [10, 10, 50, 50]} for f in frames]


def test_time_to_detection():
    gt = _gt([0, 1, 2, 3])
    # detected first at frame 2
    pred = {2: [{"bbox": [11, 11, 51, 51], "track_id": "a"}], 3: [{"bbox": [11, 11, 51, 51], "track_id": "a"}]}
    r = time_to_detection(gt, pred, fps=10.0)
    assert r["detected"] and r["frames_to_detect"] == 2 and abs(r["seconds_to_detect"] - 0.2) < 1e-6
    assert time_to_detection(gt, {}, fps=10.0)["detected"] is False


def test_fragmentation_and_id_switches():
    gt = _gt([0, 1, 2, 3, 4])
    # detected on 0,1 then gap then 3,4 -> two fragments; id changes a->b at frame 3 -> one switch
    pred = {
        0: [{"bbox": [11, 11, 51, 51], "track_id": "a"}],
        1: [{"bbox": [11, 11, 51, 51], "track_id": "a"}],
        3: [{"bbox": [11, 11, 51, 51], "track_id": "b"}],
        4: [{"bbox": [11, 11, 51, 51], "track_id": "b"}],
    }
    frag = track_fragmentation(gt, pred)
    assert frag["fragments"] == 2 and frag["matched_frames"] == 4
    assert id_switches(gt, pred) == 1


def test_safety_weighted_recall():
    objs = [
        {"object_id": "p1", "class_id": 0, "detected": True, "ttc_s": 1.0},    # imminent VRU, detected
        {"object_id": "p2", "class_id": 0, "detected": False, "ttc_s": 1.5},   # imminent VRU, MISSED
        {"object_id": "c1", "class_id": 4, "detected": True, "ttc_s": 5.0},    # distant car, detected
    ]
    crit = critical_object_recall(objs)
    assert crit["n_critical"] == 2 and crit["recall"] == 0.5
    ttc = ttc_weighted_recall(objs, ttc_horizon_s=6.0)
    # the missed imminent VRU drags weighted recall below the unweighted 2/3
    assert 0.0 < ttc["ttc_weighted_recall"] < 0.667
    near = near_miss_slice(objs, ttc_s=2.0)
    assert near["n_near_miss"] == 2 and near["missed_object_ids"] == ["p2"]


def test_bootstrap_ci_and_significance_deterministic():
    values = [1.0] * 80 + [0.0] * 20                  # recall 0.8
    ci = bootstrap_ci(values, n_boot=1000, seed=7)
    assert ci["mean"] == 0.8 and ci["lo"] < 0.8 < ci["hi"]
    assert bootstrap_ci(values, n_boot=1000, seed=7) == ci   # deterministic under seed
    # challenger strictly better on most objects -> significant
    champ = [0.0] * 50
    chall = [1.0] * 40 + [0.0] * 10
    sig = paired_significance(champ, chall, n_perm=1000, seed=7)
    assert sig["delta"] > 0 and sig["significant"] is True
    # identical models -> not significant
    tie = paired_significance(champ, champ, n_perm=1000, seed=7)
    assert tie["significant"] is False


def test_shadow_eval_and_protected_regression():
    sh = ShadowEval()
    for _ in range(8):
        sh.observe("vru_night", True)
    for _ in range(2):
        sh.observe("vru_night", False)
    assert sh.recall("vru_night") == 0.8
    triage = regression_triage({"vru_night": 0.9, "car_day": 0.9},
                               {"vru_night": 0.8, "car_day": 0.89},
                               margin=0.05, protected={"vru_night"})
    assert triage["protected_regression"] is True     # vru_night dropped 0.10 > margin
    assert triage["regressions"][0]["slice"] == "vru_night"
    assert triage["alarm"] is True
