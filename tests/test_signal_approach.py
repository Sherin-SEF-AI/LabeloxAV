"""Was the ego still closing on a signal that was red?

Named approach rather than entry because that is what the evidence carries. Ego speed exists on 2% of frames
and on none of the sessions with signal phases, so a red-light-running detector would return nothing forever.
A signal's apparent geometry is the instrument the corpus does support, and the tests that matter are the
ones where it refuses: waiting at a light, and a distant signal merely resolving.
"""

from __future__ import annotations

from services.intelligence.signal_approach import (
    MIN_GROWTH_RATIO,
    classify_approach,
    growth_ratio,
    rise,
)

H = 1080


def _run(start, scale_per_step, drop_per_step, n):
    """A box growing by scale_per_step and drifting down by drop_per_step each frame."""
    x1, y1, x2, y2 = start
    out = []
    for i in range(n):
        w, h = (x2 - x1) * (scale_per_step ** i), (y2 - y1) * (scale_per_step ** i)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + drop_per_step * i
        out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def test_a_signal_that_grows_and_drops_is_being_approached():
    boxes = _run((900, 200, 940, 260), scale_per_step=1.18, drop_per_step=9, n=8)
    ok, ev = classify_approach(boxes, H)
    assert ok is True
    assert ev["growth_ratio"] > MIN_GROWTH_RATIO
    assert ev["rise_frac"] > 0
    assert "closing" in ev["reason"]


def test_waiting_at_a_light_is_not_an_approach():
    """The commonest thing a dashcam records at a junction. The box jitters and does not grow, and calling
    that an approach would make the finding meaningless."""
    boxes = _run((900, 200, 940, 260), scale_per_step=1.004, drop_per_step=0.2, n=10)
    ok, ev = classify_approach(boxes, H)
    assert ok is False
    assert "waiting at a light" in ev["reason"]


def test_a_distant_signal_resolving_is_not_an_approach():
    """A far signal detected better frame by frame also grows. What separates it from a real approach is
    that it does not drift down the image, because it is not passing overhead."""
    boxes = _run((900, 200, 940, 260), scale_per_step=1.2, drop_per_step=0.0, n=8)
    ok, ev = classify_approach(boxes, H)
    assert ok is False
    assert "resolving" in ev["reason"]


def test_a_short_run_is_refused_rather_than_guessed():
    """Two boxes can grow by chance. On this corpus 91% of signal phases are a single frame, so refusing
    short runs is the difference between a finding and noise."""
    boxes = _run((900, 200, 940, 260), scale_per_step=1.5, drop_per_step=20, n=3)
    ok, ev = classify_approach(boxes, H)
    assert ok is False
    assert "too few frames" in ev["reason"]


def test_growth_is_measured_over_thirds_not_endpoints():
    """A single mis-sized box at either end would otherwise decide the answer for the whole approach."""
    good = _run((900, 200, 940, 260), scale_per_step=1.15, drop_per_step=8, n=9)
    spiked = list(good)
    spiked[-1] = [0.0, 0.0, 4.0, 4.0]          # one bad box at the end
    assert growth_ratio(spiked) > 1.0, "one bad box must not invert the trend"
    assert classify_approach(spiked, H)[0] is True


def test_rise_is_zero_without_a_frame_height():
    assert rise(_run((900, 200, 940, 260), 1.2, 9, 5), 0) == 0.0


def test_an_empty_run_is_not_an_approach():
    ok, ev = classify_approach([], H)
    assert ok is False
    assert ev["frames"] == 0
