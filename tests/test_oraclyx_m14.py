"""ORACLYX M14 tests: 4D track stitching with one identity and occlusion healing, radar Doppler fusion and
mis-association rejection, monocular metric-depth recovery against a synthetic scene, calibrated pseudo-GT
uncertainty monotonicity + soft targets, and the disagreement queue ranked by expected information gain."""

from services.oraclyx.mono_depth import depth_from_size, metric_depth
from services.oraclyx.radar import fuse_radar_velocity
from services.oraclyx.tracks4d import stitch_track
from services.oraclyx.uncertainty import (
    information_gain,
    pseudo_label_uncertainty,
    rank_disagreements,
    soft_target,
)


def test_stitch_track_single_identity_and_healing():
    obs = {0: [10, 10, 50, 50], 4: [30, 10, 70, 50], 10: [60, 10, 100, 50]}
    r = stitch_track(obs, n_frames=11, max_heal_gap=8)
    assert r["identities"] == 1                       # one stable track identity
    assert r["healed_gaps"] == 8                       # frames 1-3 (gap 0->4) and 5-9 (gap 4->10) interpolated
    sources = {b["source"] for b in r["boxes"]}
    assert sources == {"observed", "interpolated"}
    # a gap wider than max_heal_gap is left as a hole, not fabricated
    sparse = stitch_track({0: [0, 0, 10, 10], 20: [0, 0, 10, 10]}, n_frames=21, max_heal_gap=8)
    assert sparse["healed_gaps"] == 0


def test_radar_fusion_adopts_and_rejects():
    box = [100, 100, 140, 200]
    ret = [{"px": [120, 150], "range_rate": 30.0}]
    good = fuse_radar_velocity(box, camera_velocity=28.0, radar_returns=ret, ego_speed=0.0)
    assert good["source"] == "radar" and abs(good["velocity_kmh"] - 30.0) < 1e-6
    # a wildly inconsistent Doppler is treated as a mis-association and rejected
    bad = fuse_radar_velocity(box, camera_velocity=28.0, radar_returns=[{"px": [120, 150], "range_rate": 90.0}])
    assert bad["source"] == "camera_rejected_radar" and bad["velocity_kmh"] == 28.0
    # no return in the gate keeps the camera velocity
    none = fuse_radar_velocity(box, camera_velocity=28.0, radar_returns=[{"px": [900, 900], "range_rate": 30.0}])
    assert none["source"] == "camera"


def test_mono_depth_recovers_known_size():
    # A 1.7 m pedestrian, 340 px tall, focal 1000 -> depth ~ 5 m.
    #
    # The class id is resolved through the ontology rather than written as a literal. This test used to
    # pass class_id=0 and call it "a pedestrian", which asserted the height table against itself: it read
    # CLASS_HEIGHT_M[0], got 1.7, and confirmed 1.7. There is no id 0 in the governed ontology, so the one
    # test covering this prior could never detect that the whole table was keyed 0-based against 1-based
    # ids - which is how latent bug #1 survived after #2, #3 and #4 were fixed.
    from services.autolabel.ontology import get_ontology

    focal = 1000.0
    pedestrian = get_ontology().by_name("pedestrian").id
    d = depth_from_size([0, 0, 40, 340], class_id=pedestrian, focal_px=focal)
    assert abs(d - 5.0) < 0.01
    fused = metric_depth([0, 380, 40, 720], class_id=pedestrian, focal_px=focal, cy=360.0, cam_height_m=1.4,
                         pitch_rad=0.0)
    assert fused is not None and fused["depth_m"] > 0 and 0.0 <= fused["uncertainty"] <= 1.0
    # no calibration and no size prior -> refused, not guessed
    assert metric_depth([0, 0, 40, 100], class_id=999, focal_px=focal, cy=360.0, cam_height_m=None) is None


def test_every_size_prior_names_a_class_the_ontology_has():
    """The table is only correct if its names resolve. A typo silently drops a prior to None, which reads
    as "this class has no known size" rather than as a mistake."""
    from services.autolabel.ontology import get_ontology
    from services.oraclyx.mono_depth import CLASS_HEIGHT_M_BY_NAME, _height_by_id

    onto = get_ontology()
    missing = sorted(n for n in CLASS_HEIGHT_M_BY_NAME if not onto.has_name(n))
    assert missing == [], f"size priors naming classes the ontology does not have: {missing}"
    assert len(_height_by_id()) == len(CLASS_HEIGHT_M_BY_NAME)


def test_the_vru_and_animal_classes_have_a_size_prior_again():
    """pedestrian, rider and cattle are the classes this prior most needs, and under the 0-based table all
    three fell outside it and returned None - the size prior was silently off for every VRU."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    for name in ("pedestrian", "rider", "cattle"):
        d = depth_from_size([0, 0, 40, 340], class_id=onto.by_name(name).id, focal_px=1000.0)
        assert d is not None and d > 0, name


def test_a_delivery_bike_is_not_given_a_trucks_height():
    """id 5 was commented "truck" at 3.2 m and is actually delivery_rider_bike, so a scooter with a box on
    the back was placed at roughly twice its true distance."""
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    box, focal = [0, 0, 40, 340], 1000.0
    bike = depth_from_size(box, class_id=onto.by_name("delivery_rider_bike").id, focal_px=focal)
    truck = depth_from_size(box, class_id=onto.by_name("truck").id, focal_px=focal)
    assert bike < truck / 1.5, f"delivery bike {bike:.1f} m is still being modelled like a truck {truck:.1f} m"


def test_uncertainty_monotonic_and_soft_target():
    strong = pseudo_label_uncertainty(0.95, n_views=4, calib_confidence=0.95, depth_uncertainty=0.05)
    weak = pseudo_label_uncertainty(0.55, n_views=1, calib_confidence=0.6, depth_uncertainty=0.5)
    assert weak > strong                               # a weaker signal is strictly more uncertain
    assert soft_target(strong) > soft_target(weak)     # certain labels weigh more in distillation
    assert soft_target(1.0) >= 0.05                    # uncertain labels are damped, never dropped


def test_disagreement_ranked_by_info_gain():
    items = [
        {"id": "sure", "uncertainty": 0.05, "rarity": 0.0, "disagreement": 0.0},
        {"id": "rare_uncertain", "uncertainty": 0.8, "rarity": 0.9, "disagreement": 0.7, "safety_weight": 2.0},
        {"id": "mid", "uncertainty": 0.4, "rarity": 0.2, "disagreement": 0.3},
    ]
    ranked = rank_disagreements(items)
    assert ranked[0]["id"] == "rare_uncertain"         # most training value first
    assert ranked[-1]["id"] == "sure"                  # reviewing a sure label teaches least
    assert information_gain(0.0, 0.0, 0.0) == 0.0
