"""A trajectory recogniser fed noise returns a maneuver, confidently.

`trajectory_features` measured turn as the unwrapped difference between the first and last segment heading,
over every segment however short. Both halves fail on real monocular positions: a 14 cm step between two
noisy points implies a heading of +90 degrees, and unwrapping a sequence of such headings drifts without
bound. A track that merely approached the camera accumulated 438 degrees of turn, and four fifths of every
track in the corpus classified as a U-turn.

These tests cover the two properties that matter on noisy input, using synthetic paths where the right
answer is known: a jittering path must not read as a turn, and a real turn must still read as one.
"""

import math

import numpy as np

from services.sievyx.maneuver import MIN_SEGMENT_M, recognize, trajectory_features


def _path(points):
    return [{"t": i, "x": x, "y": y} for i, (x, y) in enumerate(points)]


def test_a_straight_path_with_jitter_is_not_a_turn():
    """The corpus case: an object approaching, with position noise on every sample."""
    rng = np.random.default_rng(0)
    pts = [(rng.normal(0, 0.3), 40.0 - 2.0 * i + rng.normal(0, 0.3)) for i in range(20)]
    f = trajectory_features(_path(pts))
    assert abs(f["net_turn_deg"]) < 90.0, \
        f"jitter on a straight approach read as {f['net_turn_deg']} degrees of turn"


def test_a_real_u_turn_is_still_a_u_turn():
    """Robustness must not cost the signal: the turn the measure exists to find."""
    heads = np.linspace(0, math.pi, 20)
    pts, x, y = [], 0.0, 0.0
    for h in heads:
        x += 2.0 * math.sin(h)
        y += 2.0 * math.cos(h)
        pts.append((x, y))
    assert recognize(_path(pts))["maneuver"] == "u_turn"


def test_a_real_gentle_turn_is_still_a_turn():
    heads = np.linspace(0, math.radians(80), 20)
    pts, x, y = [], 0.0, 0.0
    for h in heads:
        x += 2.0 * math.sin(h)
        y += 2.0 * math.cos(h)
        pts.append((x, y))
    assert recognize(_path(pts))["maneuver"] == "unprotected_turn"


def test_steps_too_short_to_have_a_direction_are_ignored():
    """A step below the floor is a difference of two noisy positions and nothing else."""
    # A clean straight run, then a cluster of sub-floor jitter that used to swing the heading by 90 degrees.
    pts = [(0.0, float(20 - i)) for i in range(8)]
    pts += [(0.05 * ((-1) ** i), 12.0 + 0.05 * ((-1) ** i)) for i in range(6)]
    f = trajectory_features(_path(pts))
    assert abs(f["net_turn_deg"]) < 45.0


def test_the_segment_floor_is_a_real_distance_not_an_epsilon():
    assert MIN_SEGMENT_M >= 0.1, "1e-6 metres admits pure noise as a heading"


def test_turn_is_bounded_by_the_path_rather_than_accumulating():
    """Wrapped per-step deltas cannot drift the way an unwrapped endpoint difference does.

    Reproduces the old measure beside the new one on the same noisy path, so the regression stays
    demonstrable rather than becoming a line in a commit message.
    """
    # The regime the corpus is actually in: position noise larger than the motion between frames. On real
    # tracks that meant 100 m jumps against a few metres of genuine movement. Mild noise does not separate
    # the two measures, which is why this uses the noise level that was really there.
    rng = np.random.default_rng(7)
    pts = [(rng.normal(0, 3.0), 30.0 - 1.5 * i + rng.normal(0, 3.0)) for i in range(24)]
    p = np.array(pts, dtype=np.float64)
    d = np.diff(p, axis=0)
    seg = np.linalg.norm(d, axis=1)

    old_moving = seg > 1e-6                       # the old floor: everything
    head = np.arctan2(d[old_moving][:, 0], d[old_moving][:, 1])
    old_turn = float(np.degrees(np.unwrap(head)[-1] - np.unwrap(head)[0]))

    new_turn = trajectory_features(_path(pts))["net_turn_deg"]
    assert abs(new_turn) < abs(old_turn), \
        f"old measure {old_turn:.0f} deg, new {new_turn:.0f} deg on the same jitter"
