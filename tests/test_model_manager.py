"""The shared model registry: one load per model, no eviction of work in flight, room made on demand.

Pure unit tests. VRAM is a number the test controls, so the eviction rules are checked exactly rather than
against whatever happens to be on the card.
"""

from __future__ import annotations

import threading
import time

import pytest

from core.model_manager import GpuCapacityError, ModelManager, device_index


class FakeCard:
    """A card with a fixed size whose free memory is whatever the loaded models have not taken."""

    def __init__(self, total_mb: float):
        self.total = total_mb
        self.used: dict[str, float] = {}
        self.cache_emptied = 0

    def free(self, _device=None) -> float:
        return self.total - sum(self.used.values())

    def empty(self) -> None:
        self.cache_emptied += 1

    def loader(self, key: str, mb: float):
        def _load():
            self.used[key] = mb
            return {"model": key}
        return _load

    def unloader(self, key: str):
        return lambda _m: self.used.pop(key, None)


def _mgr(card: FakeCard, clock=None) -> ModelManager:
    return ModelManager(free_mb=card.free, empty_cache=card.empty, clock=clock or time.monotonic)


def test_a_model_is_loaded_once_and_shared():
    card = FakeCard(16000)
    m = _mgr(card)
    n = {"loads": 0}

    def load():
        n["loads"] += 1
        return object()

    a = m.get("sam:x", load)
    b = m.get("sam:x", load)
    assert a is b and n["loads"] == 1
    assert m.stats()[0]["uses"] == 2


def test_two_threads_asking_for_the_same_model_load_it_once():
    card = FakeCard(16000)
    m = _mgr(card)
    n = {"loads": 0}
    gate = threading.Event()

    def slow():
        n["loads"] += 1
        gate.wait(2)
        return object()

    got: list = []
    threads = [threading.Thread(target=lambda: got.append(m.get("yolo:x", slow))) for _ in range(6)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    gate.set()
    for t in threads:
        t.join(5)
    assert n["loads"] == 1 and len({id(g) for g in got}) == 1


def test_a_different_model_is_not_held_up_behind_a_slow_load():
    card = FakeCard(16000)
    m = _mgr(card)
    gate = threading.Event()
    slow_started = threading.Event()

    def slow():
        slow_started.set()
        gate.wait(5)
        return "slow"

    t = threading.Thread(target=lambda: m.get("slow", slow))
    t.start()
    assert slow_started.wait(2)
    t0 = time.monotonic()
    assert m.get("fast", lambda: "fast") == "fast"
    assert time.monotonic() - t0 < 1.0, "an unrelated load waited on the registry lock"
    gate.set()
    t.join(5)


def test_a_load_that_does_not_fit_evicts_the_least_recently_used_idle_model():
    card = FakeCard(10000)
    now = {"t": 0.0}
    m = _mgr(card, clock=lambda: now["t"])
    m.get("a", card.loader("a", 4000), est_mb=4000, on_evict=card.unloader("a"))
    now["t"] = 1.0
    m.get("b", card.loader("b", 4000), est_mb=4000, on_evict=card.unloader("b"))
    now["t"] = 2.0
    m.get("a", card.loader("a", 4000))          # touch a: b is now the older one
    now["t"] = 3.0
    m.get("c", card.loader("c", 4000), est_mb=4000, headroom_mb=500, on_evict=card.unloader("c"))
    assert m.loaded("a") and m.loaded("c") and not m.loaded("b")
    assert card.cache_emptied >= 1, "evicting without emptying the cache frees nothing the driver can see"


def test_a_leased_model_is_never_evicted():
    card = FakeCard(10000)
    m = _mgr(card)
    with m.lease("a", card.loader("a", 6000), est_mb=6000, on_evict=card.unloader("a")):
        with pytest.raises(GpuCapacityError, match="nothing idle is left"):
            m.get("b", card.loader("b", 6000), est_mb=6000, on_evict=card.unloader("b"))
        assert m.loaded("a") and not m.loaded("b")
        assert not m.release("a"), "a release must be refused while the model is in use"
    # Once the lease ends it is fair game.
    m.get("b", card.loader("b", 6000), est_mb=6000, on_evict=card.unloader("b"))
    assert m.loaded("b") and not m.loaded("a")


def test_a_pinned_model_survives_make_room():
    card = FakeCard(10000)
    m = _mgr(card)
    m.get("keep", card.loader("keep", 6000), est_mb=6000, pinned=True, on_evict=card.unloader("keep"))
    assert m.make_room(6000) is False
    assert m.loaded("keep")


def test_require_passes_untouched_when_there_is_room():
    card = FakeCard(10000)
    m = _mgr(card)
    m.get("a", card.loader("a", 1000), est_mb=1000, on_evict=card.unloader("a"))
    m.require(2000, "b", headroom_mb=500)
    assert m.loaded("a"), "nothing should be evicted when the load already fits"


def test_models_on_another_device_are_not_evicted_for_this_one():
    free = {0: 1000.0, 1: 9000.0}
    m = ModelManager(free_mb=lambda d=None: free[device_index(d)], empty_cache=lambda: None)
    m.get("on1", lambda: "x", device="cuda:1")
    assert m.make_room(4000, device="cuda:0") is False
    assert m.loaded("on1"), "evicting a model on card 1 frees nothing on card 0"


def test_no_gpu_means_no_refusals():
    m = ModelManager(free_mb=lambda _d=None: None, empty_cache=lambda: None)
    m.get("cpu-model", lambda: "x", est_mb=99999)
    m.require(99999, "anything")
    assert m.make_room(99999) is True


def test_release_runs_the_hook_and_drops_the_model():
    card = FakeCard(10000)
    m = _mgr(card)
    m.get("a", card.loader("a", 3000), est_mb=3000, on_evict=card.unloader("a"))
    assert card.free() == 7000
    assert m.release("a") is True
    assert card.free() == 10000 and not m.loaded("a")
    assert m.release("a") is False


def test_a_failing_evict_hook_still_drops_the_model():
    m = ModelManager(free_mb=lambda _d=None: 100.0, empty_cache=lambda: None)

    def boom(_m):
        raise RuntimeError("hook")

    m.get("a", lambda: "x", on_evict=boom)
    assert m.release("a") is True and not m.loaded("a")


def test_release_prefix_frees_a_family():
    m = ModelManager(free_mb=lambda _d=None: None, empty_cache=lambda: None)
    for k in ("sam:a", "sam:b", "yolo:a"):
        m.get(k, lambda: "x")
    assert m.release_prefix("sam:") == 2
    assert [s["key"] for s in m.stats()] == ["yolo:a"]


@pytest.mark.parametrize("device,idx", [("cuda:0", 0), ("cuda:1", 1), ("cuda", 0), (None, 0), (2, 2),
                                        ("cuda:x", 0)])
def test_device_index(device, idx):
    assert device_index(device) == idx


def test_the_infer_lock_is_one_per_model():
    m = ModelManager(free_mb=lambda _d=None: None, empty_cache=lambda: None)
    assert m.infer_lock("sam:a") is m.infer_lock("sam:a")
    assert m.infer_lock("sam:a") is not m.infer_lock("yolo:a")


class _FakeBoxes:
    """The slice of an Ultralytics Boxes object the paths read."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        import torch

        for x1, y1, x2, y2, conf, cls in self._rows:
            yield type("B", (), {"xyxy": torch.tensor([[x1, y1, x2, y2]]), "conf": torch.tensor([conf]),
                                 "cls": torch.tensor([cls])})()


class _FakeYolo:
    """Returns one box per frame whose x1 is the frame's own marker, so order mix-ups are visible."""

    def __init__(self):
        self.calls: list[int] = []

    def predict(self, source, **_kw):
        frames = source if isinstance(source, list) else [source]
        self.calls.append(len(frames))
        return [type("R", (), {"names": {0: "car"},
                               "boxes": _FakeBoxes([(float(f[0, 0, 0]), 0.0, 10.0, 10.0, 0.9, 0)])})()
                for f in frames]


def test_batched_detection_is_one_call_and_keeps_each_frames_results_with_that_frame():
    pytest.importorskip("torch")
    import numpy as np

    from services.autolabel.paths.path_a_detect import YoloPath

    path = YoloPath("fake.pt")
    fake = _FakeYolo()
    path._model = fake
    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in (3, 7, 5)]
    out = path.infer_batch(frames)
    assert fake.calls == [3], "three frames must be one forward pass, not three"
    assert [d[0].bbox[0] for d in out] == [3.0, 7.0, 5.0], "results must stay with the frame they came from"
    assert path.infer(frames[1])[0].bbox[0] == 7.0
    assert path.infer_batch([]) == []


def test_a_detector_that_returns_the_wrong_number_of_results_is_an_error_not_a_misalignment():
    pytest.importorskip("torch")
    import numpy as np

    from services.autolabel.paths.path_a_detect import YoloPath

    path = YoloPath("fake.pt")
    path._model = type("Short", (), {"predict": lambda self, source, **kw: []})()
    with pytest.raises(RuntimeError, match="returned 0 results for 2 frames"):
        path.infer_batch([np.zeros((4, 4, 3), np.uint8)] * 2)


@pytest.mark.parametrize("before", [None, "0"])
def test_a_cpu_predict_does_not_hide_the_gpu_from_the_rest_of_the_process(monkeypatch, before):
    # Ultralytics writes CUDA_VISIBLE_DEVICES="-1" on a CPU predict. The plate detector runs on the CPU, so
    # an ingest followed by auto-label in one process refused with "CUDA not available".
    from core.model_manager import keep_gpu_visible

    if before is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", before)
    import os

    with pytest.raises(RuntimeError), keep_gpu_visible():
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        raise RuntimeError("predict failed")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


def test_cuda_is_initialised_before_a_cpu_predict_can_hide_it(monkeypatch):
    # The CUDA runtime reads CUDA_VISIBLE_DEVICES once, on first touch. A first touch inside the CPU predict
    # hides the GPU for the life of the process, so the guard touches CUDA first, while it is still visible.
    import os

    import core.model_manager as mm

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    seen = []
    monkeypatch.setattr(mm, "cuda_available", lambda: seen.append(os.environ.get("CUDA_VISIBLE_DEVICES")) or True)
    with mm.keep_gpu_visible():
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    assert seen == [None]
