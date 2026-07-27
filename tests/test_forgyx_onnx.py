"""FORGYX edge export is reachable code, not a dead branch.

The defect was not in the capability gate, which is well designed: each target checks that its backend is
importable and raises rather than fabricating a result. The defect was that onnx and onnxruntime appeared in
no dependency group at all, so on a stock install every target failed that gate and the entire export,
quantization, and benchmarking subsystem was unreachable. Both are pure-CPU wheels that install anywhere, so
there was never a hardware reason for that. They are now in the `edge` extra.

The remaining targets (TensorRT, LiteRT, Hailo) genuinely need a device toolchain and stay out on purpose;
those tests assert the gate still refuses them honestly.

Marked `gpu` because the export loads real model weights (downloading them on a cold machine)."""
from __future__ import annotations

import importlib.util

import pytest

from services.forgyx.capabilities import CapabilityError, available_targets, require

pytestmark = pytest.mark.gpu

_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None
requires_ort = pytest.mark.skipif(not _HAS_ORT, reason="edge extra not installed")


def test_onnx_backends_are_installable_and_detected():
    # The regression this guards: these two dropping out of the dependency set again would silently turn the
    # whole subsystem back into unreachable code, with the gate dutifully reporting it as unavailable.
    targets = available_targets()
    assert targets["onnx"] is True
    assert targets["onnxruntime"] is True


def test_device_targets_still_refuse_honestly():
    # These need a toolchain or a device. The point is that they raise rather than pretend, and that the
    # error names what is missing instead of failing somewhere deep in an export.
    for target in ("agx_orin_trt", "orin_nano_trt", "pi_hailo", "sentrixai_litert"):
        if available_targets()[target]:
            continue          # a machine that genuinely has the toolchain is not a failure
        with pytest.raises(CapabilityError) as exc:
            require(target)
        assert target in str(exc.value)


def test_unknown_target_is_refused():
    with pytest.raises(CapabilityError, match="unknown target"):
        require("not_a_real_target")


@requires_ort
def test_export_produces_a_model_that_actually_runs(tmp_path):
    """Export a real detector and run real inference through it.

    Asserting the file exists would not prove much: a truncated or structurally invalid graph is still a
    file. Loading it in ONNX Runtime and getting a correctly shaped detection tensor back is what shows the
    export path produces something usable.
    """
    import numpy as np
    import onnxruntime as ort

    from services.forgyx.optimize import export_onnx

    out = tmp_path / "model.onnx"
    export_onnx("yolo11n.pt", out, imgsz=320)
    assert out.exists() and out.stat().st_size > 1_000_000, "an exported detector is megabytes, not bytes"

    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    result = sess.run(None, {inp.name: np.random.rand(1, 3, 320, 320).astype(np.float32)})
    # YOLO detection head: (batch, 4 box + n_classes, anchors)
    assert result[0].ndim == 3 and result[0].shape[0] == 1
    assert result[0].shape[1] >= 5


@requires_ort
def test_benchmark_returns_ordered_percentiles(tmp_path):
    from services.forgyx.benchmark import benchmark_onnx_latency
    from services.forgyx.optimize import export_onnx

    out = tmp_path / "model.onnx"
    export_onnx("yolo11n.pt", out, imgsz=320)

    r = benchmark_onnx_latency(str(out), (1, 3, 320, 320), n_iters=20, warmup=3)
    assert r["p50"] > 0
    assert r["p50"] <= r["p95"] <= r["p99"], "percentiles must be ordered or the measurement is wrong"
    assert r["throughput_fps"] > 0
    assert r["provider"] == "CPUExecutionProvider"
