"""An autolabel job reported 0.05 from its first frame to its last.

That was harmless while the only reader was a table of job rows, which shows a status word. It became a
visible defect when the top bar grew a progress bar: a bar stuck at five percent for forty minutes says the
work is wedged, which is the opposite of what the bar was added to say. Observed on a live run: a 180 frame
session sat at progress 0.05 for its entire duration.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from services.autolabel.progress import ProgressBand


class TestBandFraction:
    def test_first_and_last_frame_sit_at_the_band_edges(self):
        b = ProgressBand(start=0.05, end=0.99)
        assert b.fraction(0, 10) == pytest.approx(0.05)
        assert b.fraction(10, 10) == pytest.approx(0.99)

    def test_never_reaches_one_while_frames_are_still_being_processed(self):
        # Only the code that knows the job finished may write 1.0. A frame loop reporting completion while
        # objects are still being persisted would tell every watcher to move on early.
        b = ProgressBand()
        assert b.fraction(10, 10) < 1.0

    def test_rises_monotonically(self):
        b = ProgressBand()
        seen = [b.fraction(i, 50) for i in range(51)]
        assert seen == sorted(seen)

    def test_a_session_with_no_frames_is_not_a_division_by_zero(self):
        # An empty session is a real thing that happens, and it is not worth failing a job over.
        assert ProgressBand().fraction(0, 0) == pytest.approx(0.05)

    def test_a_count_past_the_total_clamps_instead_of_overshooting(self):
        b = ProgressBand()
        assert b.fraction(99, 10) == pytest.approx(b.fraction(10, 10))

    def test_a_nonsensical_band_is_rejected_at_construction(self):
        with pytest.raises(ValueError):
            ProgressBand(start=0.9, end=0.5)
        with pytest.raises(ValueError):
            ProgressBand(step=0)


class TestBandRationing:
    def test_the_first_report_always_lands(self):
        # Otherwise a watcher waits a whole percent to learn the job started moving, which on a long session
        # is minutes.
        assert ProgressBand().next_if_due(1, 5000) is not None

    def test_a_write_costs_at_least_a_step_of_movement(self):
        b = ProgressBand(step=0.01)
        b.next_if_due(1, 1000)
        assert b.next_if_due(2, 1000) is None

    def test_a_long_session_writes_about_once_per_percent(self):
        b = ProgressBand()
        writes = sum(1 for i in range(1, 5001) if b.next_if_due(i, 5000) is not None)
        assert writes <= 100, f"{writes} writes for 5000 frames is a commit storm"
        assert writes >= 90, f"{writes} writes is too coarse for a bar to look alive"

    def test_the_end_of_the_band_is_reported_even_from_a_short_hop(self):
        # A 3 frame session moves in jumps far larger than a step, but the last frame must still land or the
        # bar stops short of where the job actually is.
        b = ProgressBand()
        vals = [b.next_if_due(i, 3) for i in (1, 2, 3)]
        assert vals[-1] == pytest.approx(0.99)

    def test_standing_still_is_never_worth_a_write(self):
        b = ProgressBand()
        b.next_if_due(10, 10)
        assert b.next_if_due(10, 10) is None


@dataclass
class _Frame:
    frame_id: str
    img_uri: str


class TestFrameLoopReportsProgress:
    """The loop is the only place that knows both counts, so it is the only place that can report them."""

    def test_process_session_reports_after_every_frame(self, monkeypatch):
        from services.autolabel import runner

        frames = [_Frame(f"f{i}", f"s3://x/{i}.jpg") for i in range(4)]

        class _Runner:
            def __init__(self, *a, **kw):
                self.guard = type("g", (), {"peak_mb": lambda self: 0.0})()
                self.settings = type("s", (), {"gpu": type("g", (), {"vram_total_mb": 10_000})()})()

            def open_stage1(self):
                pass

            def close_stage1(self):
                pass

            def run_stage1_frame(self, img):
                return [], []

        async def _fetch(session_id, limit):
            return frames

        monkeypatch.setattr(runner, "fetch_frames", _fetch)
        monkeypatch.setattr(runner, "load_image", lambda uri: object())
        monkeypatch.setattr(runner, "StagedRunner", _Runner)

        seen: list[tuple[int, int]] = []

        async def on_frame(fd):
            pass

        async def on_progress(done, total):
            seen.append((done, total))

        asyncio.run(runner.process_session(None, None, on_frame, on_progress=on_progress))
        assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]

    def test_a_caller_that_wants_no_progress_still_works(self, monkeypatch):
        # Every existing caller passes nothing, and the CLI has no job row to write to.
        from services.autolabel import runner

        class _Runner:
            def __init__(self, *a, **kw):
                self.guard = type("g", (), {"peak_mb": lambda self: 0.0})()
                self.settings = type("s", (), {"gpu": type("g", (), {"vram_total_mb": 10_000})()})()

            def open_stage1(self):
                pass

            def close_stage1(self):
                pass

            def run_stage1_frame(self, img):
                return [], []

        async def _fetch(session_id, limit):
            return [_Frame("f0", "s3://x/0.jpg")]

        monkeypatch.setattr(runner, "fetch_frames", _fetch)
        monkeypatch.setattr(runner, "load_image", lambda uri: object())
        monkeypatch.setattr(runner, "StagedRunner", _Runner)

        async def on_frame(fd):
            pass

        summary = asyncio.run(runner.process_session(None, None, on_frame))
        assert summary["frames"] == 1
