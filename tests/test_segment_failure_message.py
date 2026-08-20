"""A 503 that names a cause nobody checked.

The interactive segmentation route answered every GPU-shaped failure with "a training job is using the GPU",
and it never asked whether one was. Reported live against a machine whose card was idle (15 MiB of 16 GB, no
compute processes) and whose `training_job` table held no running row: the message sent somebody looking for
a job that did not exist. The branch also swallowed the exception without logging it, so the one fact that
would have explained the failure was gone.

The failure itself was a long-lived API process whose SAM/CUDA state had gone bad; a fresh process segments
the same frame fine. That is exactly the case the message hid.
"""

from __future__ import annotations

import pytest

from services.api.routers.objects import segment_failure_detail


class _GpuCapacityError(Exception):
    pass


class _CudaOutOfMemoryError(Exception):
    pass


class TestWhatTheCallerIsTold:
    def test_a_job_holding_the_card_is_named_only_when_one_does(self):
        detail = segment_failure_detail(_GpuCapacityError("CUDA not available"), gpu_busy=True)
        assert "training job" in detail

    def test_an_idle_card_is_never_blamed_on_a_training_job(self):
        """The report that started this: an idle GPU, no running job, and a message about a training job."""
        detail = segment_failure_detail(_GpuCapacityError("CUDA not available"), gpu_busy=False)
        assert "training job" not in detail
        assert "free" in detail

    def test_the_real_error_survives_into_the_message(self):
        """Without it the caller has a 503 and no way to tell a missing driver from a broken process."""
        detail = segment_failure_detail(_GpuCapacityError("CUDA not available; the autolabel plane "
                                                         "requires a GPU"), gpu_busy=False)
        assert "_GpuCapacityError" in detail
        assert "CUDA not available" in detail

    def test_it_says_what_still_works_either_way(self):
        """Box review needs no GPU, and a refusal that does not say so reads as the editor being down."""
        for busy in (True, False):
            assert "box review" in segment_failure_detail(_GpuCapacityError("CUDA"), gpu_busy=busy).lower()


class TestWhichFailuresItClaims:
    @pytest.mark.parametrize("exc", [
        _GpuCapacityError("CUDA not available; the autolabel plane requires a GPU"),
        _CudaOutOfMemoryError("tried to allocate 2.00 GiB"),
        RuntimeError("CUDA error: out of memory"),
    ])
    def test_a_gpu_failure_becomes_a_503(self, exc):
        assert segment_failure_detail(exc, gpu_busy=False) is not None

    @pytest.mark.parametrize("exc", [
        ValueError("box must be [x1,y1,x2,y2]"),
        KeyError("masks"),
        FileNotFoundError("sam weights"),
    ])
    def test_anything_else_is_left_alone(self, exc):
        """A 503 on an ordinary bug would tell the caller to wait for hardware that is not the problem, and
        would hide a real error behind a retry."""
        assert segment_failure_detail(exc, gpu_busy=False) is None

    def test_a_long_error_is_truncated_rather_than_returned_whole(self):
        detail = segment_failure_detail(_GpuCapacityError("CUDA " + "x" * 5_000), gpu_busy=False)
        assert len(detail) < 500


def test_the_route_logs_before_it_decides():
    """The branch left no trace at all, which is why a wrong message could survive unnoticed."""
    import inspect

    from services.api.routers import objects

    src = inspect.getsource(objects.segment)
    assert 'log.exception("segment.failed"' in src
    assert src.index('log.exception') < src.index("segment_failure_detail")


class TestAMalformedRequest:
    """A bad box reached SAM and came back as an unhandled 500, which reads as the service being broken
    rather than as the request being wrong. `/objects/classify` beside it has always checked its box."""

    def test_the_route_checks_the_box_shape(self):
        import inspect

        from services.api.routers import objects

        src = inspect.getsource(objects.segment)
        assert 'len(payload.box) != 4' in src
        assert '"box must be [x1,y1,x2,y2]"' in src

    def test_it_refuses_a_prompt_with_neither_a_point_nor_a_box(self):
        """SAM given nothing to prompt on segments whatever it likes, and the caller gets a polygon they
        never asked for rather than an error."""
        import inspect

        from services.api.routers import objects

        assert "a point or a box is required" in inspect.getsource(objects.segment)

    def test_it_refuses_labels_that_do_not_match_the_points(self):
        import inspect

        from services.api.routers import objects

        assert "labels must have one entry per point" in inspect.getsource(objects.segment)
