"""The on-rig selector: hitting an uplink budget without being able to see the rest of the drive.

`extract_smart.py` makes this decision server-side, after every frame has been driven home and stored. The
device version has to make it in a stream, and the ways a stream version goes quietly wrong are the subject
of these tests: a threshold that means nothing on the descriptor actually installed, and a quantile that
ends up describing the selector's own behaviour instead of the road.
"""

from __future__ import annotations

import numpy as np
import pytest

from compute.device.embed import describe, tiled_histogram
from compute.device.selector import SelectorConfig, StreamingSelector


def _drive(sel: StreamingSelector, n: int, sigma: float, seed: int = 3, rare_at: set | None = None):
    """A synthetic drive: a scene drifting at `sigma` per frame."""
    rng = np.random.default_rng(seed)
    scene = rng.integers(40, 200, size=(120, 160, 3), dtype=np.uint8).astype(np.float32)
    out = []
    for i in range(n):
        scene = np.clip(scene + rng.normal(0, sigma, scene.shape), 0, 255)
        out.append(sel.observe(tiled_histogram(scene.astype(np.uint8)),
                               rare=bool(rare_at and i in rare_at)))
    return out


# --- the budget ---------------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0.05, 0.10, 0.25])
def test_the_keep_rate_tracks_the_uplink_budget(target):
    """The property the whole design is for. A rig that cannot hit its budget either fills its disk by 9am
    or throws away the afternoon."""
    sel = StreamingSelector(cfg=SelectorConfig(target_keep_frac=target, min_gap_frames=1))
    _drive(sel, 1500, sigma=14.0)
    kept = sel.stats["keep_frac"]
    assert abs(kept - target) < 0.03, f"kept {kept} against a {target} budget"


def test_ranking_on_step_change_rather_than_distance_from_the_last_kept_frame():
    """The bug that made the first version undershoot a 10% budget by a factor of eight.

    Difference from the last kept frame grows the longer it has been since a keep, so a window of those
    values describes the gaps the selector has been leaving rather than the road. Ranking against it is
    self-reinforcing: keep rarely, observe large differences, raise the bar, keep more rarely.
    """
    import inspect

    from compute.device import selector

    src = inspect.getsource(selector.StreamingSelector.observe)
    assert "self._recent.append(step)" in src, "the window must hold step change"
    assert "self._recent.append(novelty)" not in src, "not distance from the last kept frame"


def test_a_static_camera_uploads_almost_nothing_whatever_the_budget_says():
    """The budget is a ceiling, not a quota. A camera pointed at a wall should not spend it on the wall."""
    for target in (0.05, 0.25):
        sel = StreamingSelector(cfg=SelectorConfig(target_keep_frac=target, min_gap_frames=1))
        _drive(sel, 1200, sigma=0.02)
        assert sel.stats["keep_frac"] < 0.02, sel.stats


# --- what must never be dropped -------------------------------------------------------------------


def test_a_rare_class_is_kept_however_spent_the_budget_is():
    """A cow on a motorway is the frame the whole pipeline exists to find. The cost of keeping a boring
    frame is a few hundred kilobytes; the cost of dropping this one is the week."""
    sel = StreamingSelector(cfg=SelectorConfig(target_keep_frac=0.01, min_gap_frames=1))
    decisions = _drive(sel, 600, sigma=0.05, rare_at={200, 400})
    rare_decisions = [decisions[i] for i in (200, 400)]
    assert all(d.keep for d in rare_decisions)
    assert all(d.reason == "rare class" for d in rare_decisions)


def test_a_busy_scene_is_kept_even_when_the_pixels_barely_moved():
    """A junction with fifteen road users is worth having whatever its pixel novelty says."""
    sel = StreamingSelector(cfg=SelectorConfig(min_gap_frames=1, dense_objects=8))
    rng = np.random.default_rng(1)
    scene = rng.integers(40, 200, size=(120, 160, 3), dtype=np.uint8)
    sel.observe(tiled_histogram(scene))
    d = sel.observe(tiled_histogram(scene), object_count=12)
    assert d.keep and d.reason == "busy scene"


def test_a_motionless_stretch_still_leaves_a_heartbeat():
    sel = StreamingSelector(cfg=SelectorConfig(max_gap_frames=50, min_gap_frames=1))
    decisions = _drive(sel, 400, sigma=0.01)
    assert any(d.reason.startswith("heartbeat") for d in decisions)


def test_a_shake_burst_cannot_empty_the_budget_in_a_second():
    sel = StreamingSelector(cfg=SelectorConfig(min_gap_frames=5, target_keep_frac=0.9))
    decisions = _drive(sel, 300, sigma=40.0)
    kept_at = [i for i, d in enumerate(decisions) if d.keep]
    gaps = [b - a for a, b in zip(kept_at, kept_at[1:], strict=False)]
    assert all(g >= 5 for g in gaps), f"minimum gap violated: {sorted(gaps)[:5]}"


def test_two_identical_frames_never_both_get_uploaded():
    sel = StreamingSelector(cfg=SelectorConfig(min_gap_frames=1))
    rng = np.random.default_rng(2)
    img = rng.integers(40, 200, size=(120, 160, 3), dtype=np.uint8)
    first = sel.observe(tiled_histogram(img))
    second = sel.observe(tiled_histogram(img))
    assert first.keep and not second.keep
    assert second.reason == "identical to the last kept frame"


# --- the descriptor -------------------------------------------------------------------------------


def test_an_unknown_descriptor_backend_fails_loudly_rather_than_falling_back():
    """A silent fallback would split a fleet into devices that select well and devices that select badly,
    with nothing in the uploaded data saying which was which."""
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown descriptor backend"):
        describe(img, backend="dinov4")


def test_the_cheap_descriptor_is_unit_norm_and_stable():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    v = tiled_histogram(img)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    assert np.allclose(v, tiled_histogram(img))


def test_the_descriptor_sees_where_in_the_frame_something_changed():
    """A whole-frame histogram cannot tell a car entering on the left from one leaving on the right, because
    the two cancel. Tiling is what stops that reading as no change at all."""
    base = np.full((120, 160, 3), 100, dtype=np.uint8)
    left = base.copy()
    left[:, :40] = 200
    right = base.copy()
    right[:, 120:] = 200
    assert float(tiled_histogram(left) @ tiled_histogram(right)) < 0.999


def test_the_descriptor_survives_an_exposure_shift():
    """Exposure moves constantly on a road. A descriptor that reads it as a scene change would spend the
    whole budget on the sun coming out."""
    # Structured rather than uniform noise: two uniform-random images have near-identical histograms by
    # construction, so comparing against one would test nothing about the descriptor.
    def _scene(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.full((120, 160, 3), 90, dtype=np.uint8)
        for _ in range(12):
            y, x = rng.integers(0, 90), rng.integers(0, 130)
            img[y:y + 25, x:x + 25] = rng.integers(30, 220)
        return img

    img = _scene(9)
    brighter = np.clip(img.astype(np.int16) + 25, 0, 255).astype(np.uint8)
    same_scene = float(tiled_histogram(img) @ tiled_histogram(brighter))
    different_scene = float(tiled_histogram(img) @ tiled_histogram(_scene(10)))
    assert same_scene > different_scene, (same_scene, different_scene)


# --- the manifest -----------------------------------------------------------------------------------


def test_the_manifest_declares_what_was_dropped_and_why():
    """A rig that uploads 8% of its day and says nothing about the other 92% produces a corpus with an
    invisible sampling bias: every rate downstream is computed over a filtered population whose filter
    nothing records."""
    import tempfile

    from compute.device.agent import DeviceAgent, DeviceConfig, verify_manifest

    with tempfile.TemporaryDirectory() as tmp:
        agent = DeviceAgent(DeviceConfig(device_id="rig-1", secret="k", spool_dir=tmp,
                                         selector=SelectorConfig(min_gap_frames=1)))
        rng = np.random.default_rng(4)
        scene = rng.integers(40, 200, size=(60, 80, 3), dtype=np.uint8).astype(np.float32)
        for i in range(200):
            scene = np.clip(scene + rng.normal(0, 8.0, scene.shape), 0, 255)
            agent.offer(scene.astype(np.uint8), ts_ns=i * 333_000_000)

        sealed = agent.seal()
        m = sealed["manifest"]
        assert m["frames_seen"] == 200
        assert m["frames_dropped"] == 200 - m["frames_kept"]
        assert m["descriptor_backend"] == "tiled_histogram"
        assert sum(m["reasons"].values()) == 200
        assert verify_manifest("k", m, sealed["signature"])
        assert not verify_manifest("other-key", m, sealed["signature"])


def test_a_tampered_manifest_does_not_verify():
    """Devices sit outside the trust boundary, which FORGYX already assumes for telemetry. A signature does
    not make a device honest; it makes an invented report distinguishable from a real one."""
    import hashlib
    import hmac
    import json

    from compute.device.agent import verify_manifest

    m = {"device_id": "rig-1", "frames_seen": 100, "frames_kept": 8}
    sig = hmac.new(b"k", json.dumps(m, sort_keys=True, separators=(",", ":")).encode(),
                   hashlib.sha256).hexdigest()
    assert verify_manifest("k", m, sig)
    assert not verify_manifest("k", {**m, "frames_kept": 80}, sig)
