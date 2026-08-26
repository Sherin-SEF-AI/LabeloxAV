"""One card, one job, across every process that wants it.

The standing instruction is that nothing here may take the machine down. The failure this guards against is
not a crash with a stack trace: it is two GPU jobs on one card producing an out-of-memory part way through a
batch, which the caller counts as a failed frame rather than as contention, twenty of which trip a
consecutive-failure guard and stop a corpus pass that had nothing wrong with it.
"""

import inspect

import pytest


class TestTheLongRunningGpuJobsCoordinate:
    """A sweep that holds the card for hours must say so, or nothing else can wait its turn."""

    @pytest.mark.parametrize("mod,fn", [
        ("services.autolabel.runner", "autolabel_session"),
        ("services.agent.relabel_agent", "run_relabel_all"),
        ("services.labelops.class_precision", "judge_class"),
        ("services.forgyx.export", "export_and_benchmark"),
    ])
    def test_it_takes_the_gpu_slot(self, mod, fn):
        import importlib

        m = importlib.import_module(mod)
        assert "gpu_slot" in inspect.getsource(getattr(m, fn)), (
            f"{mod}.{fn} runs on the GPU for a long time and does not take the slot")

    def test_autolabel_holds_it_around_the_whole_pipeline_not_a_fragment(self):
        """The wrapper exists so the lock spans the pass. If the work moved back inline, the lock would be
        taken and released around setup and the frames would run unprotected."""
        from services.autolabel import runner

        src = inspect.getsource(runner.autolabel_session)
        assert "_autolabel_session_locked" in src
        assert "async with gpu_slot" in src

    def test_the_slot_is_taken_inside_the_pipeline_not_only_at_the_router(self):
        """The router is one of six callers. redetect, the ops agent, two fleet scripts and the CLI all
        reach the pipeline directly, and the router's gates never applied to any of them."""
        from services.autolabel import runner

        # The guard is in the module the callers share, not in services/api/routers/autolabel.py.
        assert "gpu_slot" in inspect.getsource(runner.autolabel_session)


class TestWorkIsBounded:
    def test_the_judge_sweep_rechecks_between_batches(self):
        """So a training job that starts mid-class waits seconds rather than the length of the class."""
        from services.labelops import class_precision

        src = inspect.getsource(class_precision.judge_class)
        assert "wait_for_headroom" in src
        assert class_precision.BATCH <= 50

    def test_the_embedding_daemon_yields_rather_than_competing(self):
        """It runs unattended off the scheduler, so it is the one most likely to start beside something."""
        from services.intelligence.embed import daemon

        assert "_free_vram_mb" in inspect.getsource(daemon)

    def test_a_headroom_floor_exists_and_is_not_zero(self):
        from services.labelops.class_precision import MIN_FREE_VRAM_MB

        assert MIN_FREE_VRAM_MB > 0
