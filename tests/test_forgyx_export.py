"""FORGYX's three benchmarks described artifacts that do not exist.

The table held exactly three rows, all for a model named `demo-challenger`, on `agx_orin_trt`,
`orin_nano_trt` and `sentrixai_litert`. Their artifact URIs point at `s3://labeloxav/models/demo/*.bin`, and
`ObjectStore.exists` returns False for every one of them. The latencies beside them (p50 4.2ms, 178 fps, 22W)
were typed in.

Everything above rested on that: the Pareto gate ranked those three, the deployment table takes a
`benchmark_ref`, and the manifest signer signs a statement about an artifact it never opens. There was also
no export function in the module at all, so nothing could have produced a real artifact even in principle.

The tests here are about the two claims that keep a benchmark honest: the artifact has to exist, and the
target name has to be the runtime that actually served the measurement.
"""

from __future__ import annotations

import uuid

import pytest

from services.forgyx.export import artifact_exists, percentiles, sha256_file


class _FakeStore:
    def __init__(self, present: set[str]):
        self.present = present

    def exists(self, uri: str) -> bool:
        return uri in self.present


class _BrokenStore:
    def exists(self, uri: str) -> bool:
        raise ConnectionError("object store unreachable")


# ------------------------------------------------------------------------------- the artifact guard

def test_a_missing_artifact_is_not_treated_as_present(monkeypatch):
    """The check the three demo rows fail. This is the whole reason they survived."""
    import services.forgyx.export as ex

    monkeypatch.setattr(ex, "get_object_store", lambda: _FakeStore(set()))
    assert artifact_exists("s3://labeloxav/models/demo/agx_orin_trt.bin") is False


def test_a_real_artifact_is_recognised(monkeypatch):
    import services.forgyx.export as ex

    uri = "s3://labeloxav/artifacts/m/onnx/model-abc.onnx"
    monkeypatch.setattr(ex, "get_object_store", lambda: _FakeStore({uri}))
    assert artifact_exists(uri) is True


def test_no_uri_is_not_an_artifact():
    assert artifact_exists(None) is False
    assert artifact_exists("") is False


def test_an_unreachable_store_does_not_report_the_artifact_as_fine(monkeypatch):
    """Failing open here would let a network blip certify an artifact nobody can fetch."""
    import services.forgyx.export as ex

    monkeypatch.setattr(ex, "get_object_store", lambda: _BrokenStore())
    assert artifact_exists("s3://labeloxav/whatever.onnx") is False


# ------------------------------------------------------------------------------- the measurement

def test_latency_is_reported_as_a_distribution_not_a_mean():
    """A device budget is spent on the tail. A mean hides exactly the part that blows it."""
    out = percentiles([10.0, 10.1, 10.2, 30.0])
    assert out["p50"] < out["p95"] <= out["p99"]
    assert out["max"] == 30.0 and out["n"] == 4


def test_percentiles_do_not_depend_on_the_order_measured():
    assert percentiles([3.0, 1.0, 2.0]) == percentiles([1.0, 2.0, 3.0])


def test_no_samples_reports_nothing_rather_than_zero():
    """A zero latency reads as an infinitely fast model, which is the most flattering possible lie."""
    assert percentiles([]) == {}


def test_a_single_sample_still_produces_a_usable_shape():
    out = percentiles([7.5])
    assert out["p50"] == out["p99"] == 7.5 and out["n"] == 1


# ------------------------------------------------------------------------------- content addressing

def test_the_hash_is_of_the_bytes_so_a_manifest_signs_something_real(tmp_path):
    a = tmp_path / "a.onnx"
    b = tmp_path / "b.onnx"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert sha256_file(a) == sha256_file(b)
    b.write_bytes(b"different")
    assert sha256_file(a) != sha256_file(b)


# ------------------------------------------------------------------------------- refusals

@pytest.mark.db
async def test_an_unregistered_model_is_refused_rather_than_benchmarked():
    from db.session import get_sessionmaker
    from services.forgyx.export import export_and_benchmark

    async with get_sessionmaker()() as db:
        out = await export_and_benchmark(db, f"nope-{uuid.uuid4().hex[:8]}")
    assert out["ok"] is False and "not registered" in out["reason"]


@pytest.mark.db
async def test_a_caller_hosted_model_has_nothing_to_export():
    """`external://caller-hosted` means the caller runs it. There is no artifact for us to make or time, and
    recording a target's numbers against it would attribute somebody else's hardware to ours."""
    from db.models import ModelRegistry
    from db.session import get_sessionmaker
    from services.forgyx.export import export_and_benchmark

    mv = f"ext-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        db.add(ModelRegistry(model_version=mv, weights_uri="external://caller-hosted"))
        await db.commit()
        out = await export_and_benchmark(db, mv)
    assert out["ok"] is False and "nothing local to export" in out["reason"]


# ------------------------------------------------------------------------------- recording

@pytest.mark.db
async def test_a_benchmark_naming_a_missing_artifact_is_refused(monkeypatch):
    """Exactly what the three demo rows did. Nothing in the system would ever have said so."""
    import services.forgyx.export as ex
    from db.session import get_sessionmaker
    from services.forgyx.run import record_benchmark

    monkeypatch.setattr(ex, "get_object_store", lambda: _FakeStore(set()))
    async with get_sessionmaker()() as db:
        out = await record_benchmark(db, f"m-{uuid.uuid4().hex[:8]}", "agx_orin_trt",
                                     {"p50": 4.2, "p95": 5.6, "p99": 6.8},
                                     artifact_uri="s3://labeloxav/models/demo/agx_orin_trt.bin")
    assert out["ok"] is False and out["benchmark_id"] is None
    assert "cannot be verified" in out["reason"]


@pytest.mark.db
async def test_a_device_with_nothing_to_upload_can_still_report(monkeypatch):
    """A board measuring hardware this system does not host has no artifact to offer, and saying so is
    honest. Refusing it would push real measurements out of the table to keep fabricated ones out."""
    import services.forgyx.export as ex
    from db.models import ModelRegistry
    from db.session import get_sessionmaker
    from services.forgyx.run import record_benchmark

    monkeypatch.setattr(ex, "get_object_store", lambda: _FakeStore(set()))
    mv = f"m-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        db.add(ModelRegistry(model_version=mv, weights_uri="s3://w.pt"))
        await db.commit()
        out = await record_benchmark(db, mv, "orin_nano_trt", {"p50": 11.0, "p95": 14.3, "p99": 16.1})
    assert out["ok"] is True and out["benchmark_id"]


@pytest.mark.db
async def test_the_audit_names_the_benchmarks_that_cannot_be_checked(monkeypatch):
    import services.forgyx.export as ex
    from db.models import Benchmark, ModelRegistry
    from db.session import get_sessionmaker
    from services.forgyx.run import audit_benchmarks

    monkeypatch.setattr(ex, "get_object_store", lambda: _FakeStore(set()))
    mv = f"m-{uuid.uuid4().hex[:8]}"
    async with get_sessionmaker()() as db:
        db.add(ModelRegistry(model_version=mv, weights_uri="s3://w.pt"))
        await db.flush()
        db.add(Benchmark(model_version=mv, target="agx_orin_trt", latency_ms={"p50": 4.2},
                         artifact_uri="s3://labeloxav/models/demo/gone.bin"))
        await db.commit()
        out = await audit_benchmarks(db)
    assert out["n_unverifiable"] >= 1
    assert any(u["artifact_uri"].endswith("gone.bin") for u in out["unverifiable"])
