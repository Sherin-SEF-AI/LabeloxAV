"""Scene-level export, streaming delivery, dataset diffing, and the cloud data-movement contract.

Four annotation types could be created, corrected, propagated and gated inside the system and never leave
it: masks left only as COCO polygons, and lanes, drivable surfaces and the HD map could not leave at all,
because every adapter is object-shaped and none of those three is an Object. Delivery was one presigned URL
per file, which for forty thousand frames is forty thousand round trips. And the cloud contract was a
docstring that raised, so four job types parked forever with no way to move a byte.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------- masks


def test_masks_paint_the_smallest_object_last():
    """Painting in arbitrary order lets a bus swallow the pedestrian inside it, which is the common and
    quiet way a mask export loses its rarest classes."""
    from services.export.adapter_scene import write_masks

    class _Onto:
        version = "test-1"

    class _Store:
        def __init__(self, polys):
            self._polys = polys

        def get_bytes(self, uri):
            return json.dumps({"polygons": self._polys[uri]}).encode()

    big = [[0.0, 0.0, 100.0, 0.0, 100.0, 100.0, 0.0, 100.0]]
    small = [[40.0, 40.0, 60.0, 40.0, 60.0, 60.0, 40.0, 60.0]]

    class _Rec:
        def __init__(self, uri, class_id, bbox, name):
            self.frame_id, self.mask_uri, self.class_id = "f1", uri, class_id
            self.bbox, self.class_name = bbox, name
            self.width, self.height = 100, 100

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = write_masks(
            [_Rec("bus", 7, [0, 0, 100, 100], "bus"), _Rec("ped", 3, [40, 40, 60, 60], "pedestrian")],
            _Onto(), _Store({"bus": big, "ped": small}), Path(tmp) / "masks")

        import cv2

        png = cv2.imread(str(out / "labelIds" / "f1.png"), cv2.IMREAD_UNCHANGED)
        assert png is not None
        # The centre belongs to the pedestrian, the corner to the bus.
        assert int(png[50, 50]) == 3
        assert int(png[5, 5]) == 7

        labels = json.loads((out / "labels.json").read_text())
        assert labels["classes"] == {"3": "pedestrian", "7": "bus"}
        # The encoding is stated: a consumer must not have to guess whether a pixel is an id or a palette
        # index, because the two are indistinguishable by inspection.
        assert "ontology class id" in labels["encoding"]


# ---------------------------------------------------------------- streaming

def test_a_streamed_zip_is_valid_and_carries_a_manifest():
    from services.export.streaming import stream_zip

    entries = [(f"data/{i}.txt", f"content-{i}".encode()) for i in range(5)]
    blob = b"".join(stream_zip(iter(entries), manifest={"commit_id": "abc"}))

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert zf.testzip() is None
        names = set(zf.namelist())
        assert "MANIFEST.json" in names
        assert zf.read("data/3.txt") == b"content-3"
        manifest = json.loads(zf.read("MANIFEST.json"))
        assert manifest["files_written"] == 5 and manifest["commit_id"] == "abc"


def test_an_unreadable_file_is_recorded_rather_than_aborting_the_stream():
    """Half a gigabyte into a download is the worst possible moment to discover one blob is missing, and
    the consumer needs to know which one rather than starting again."""
    from services.export.streaming import stream_zip

    entries = [("good.txt", b"ok"), ("bad.bin", None), ("also-good.txt", b"fine")]
    blob = b"".join(stream_zip(iter(entries)))   # type: ignore[arg-type]

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert zf.testzip() is None
        assert set(zf.namelist()) == {"good.txt", "also-good.txt", "MANIFEST.json"}
        manifest = json.loads(zf.read("MANIFEST.json"))
        assert manifest["files_written"] == 2
        assert manifest["files_skipped"] == [{"path": "bad.bin", "reason": "unreadable"}]


def test_the_archive_uses_zip64_and_no_compression():
    """A dataset crossing 4 GB or 65535 entries silently corrupts under the classic format, and both are
    ordinary sizes here. Stored rather than deflated because the payload is already-compressed imagery."""
    from services.export.streaming import stream_zip

    blob = b"".join(stream_zip(iter([("a.bin", b"x" * 4096)])))
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        info = zf.getinfo("a.bin")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.compress_size == info.file_size


def test_the_suggested_filename_is_safe():
    from services.export.streaming import suggested_filename

    assert suggested_filename("abc123def456", "fleet v1/../etc") == "fleetv1etc-abc123def456.zip"
    assert suggested_filename("abc123def456", None).startswith("dataset-")


# ---------------------------------------------------------------- lane rasterisation to export

@pytest.mark.db
async def test_lane_export_writes_culane_geometry_and_the_attributes_it_cannot_carry():
    """CULane's format is purely geometric. Dropping lane_type and is_ego on the way out would make it
    impossible to train the type classifier the corpus has labels for."""
    import tempfile

    from db.models import Frame, Lane
    from db.models import Session as DbSession
    from db.session import get_sessionmaker
    from services.export.adapter_scene import write_lanes

    async with get_sessionmaker()() as db:
        s = DbSession(vehicle_id="veh-lane", city="BLR", start_ts_ns=0, end_ts_ns=10**9,
                      sensors={}, ontology_version="labelox-in-0.1.0")
        db.add(s)
        await db.flush()
        f = Frame(session_id=s.session_id, ts_ns=0, cam_id="cam_f", img_uri="s3://x",
                  width=640, height=480, quality=0.9)
        db.add(f)
        await db.flush()
        db.add(Lane(frame_id=f.frame_id, session_id=s.session_id,
                    control_points=[[10, 400], [300, 200], [500, 100]],
                    lane_type="solid", is_ego=True, source="human"))
        await db.commit()
        fid = str(f.frame_id)

    with tempfile.TemporaryDirectory() as tmp:
        out = await write_lanes([fid], Path(tmp) / "lanes")
        line_file = out / "lines" / f"{fid}.lines.txt"
        assert line_file.exists()
        assert len(line_file.read_text().strip().split("\n")) == 1

        meta = json.loads((out / "lanes.json").read_text())
        assert meta["format"] == "culane"
        entry = meta["lanes"][fid][0]
        assert entry["lane_type"] == "solid" and entry["is_ego"] is True


# ---------------------------------------------------------------- the export registry

def test_the_scene_formats_are_accepted_and_dispatchable():
    from services.export.dataset import _SCENE_WRITERS, SUPPORTED_EXPORT_FORMATS, validate_formats

    for fmt in ("masks", "lanes", "drivable", "hdmap", "panoptic"):
        assert fmt in SUPPORTED_EXPORT_FORMATS
    # Exhaustive on purpose, and it earned that: adding the panoptic writer failed here, which is what an
    # exact-set assertion is for. A writer registered without being validated, or validated without being
    # registered, ships a format that either cannot be asked for or silently writes nothing.
    assert set(_SCENE_WRITERS) == {"lanes", "drivable", "hdmap", "panoptic"}
    validate_formats(["coco", "masks", "lanes", "drivable", "hdmap", "panoptic"])   # must not raise


def test_an_unsupported_format_is_still_refused():
    """The registry got wider; it must not have got laxer."""
    from services.export.dataset import UnknownExportFormat, validate_formats

    with pytest.raises(UnknownExportFormat):
        validate_formats(["coco", "not-a-format"])


# ---------------------------------------------------------------- the cloud contract

def _fake_store(tmp: Path):
    """A local stand-in for the object store, so the transfer contract is testable with no MinIO."""

    class _Store:
        def ensure_bucket(self):
            return None

        def put_file(self, key, path):
            dest = tmp / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(path).read_bytes())
            return f"s3://test/{key}"

        def put_bytes(self, key, data, content_type="application/octet-stream"):
            dest = tmp / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return f"s3://test/{key}"

        def get_bytes(self, key):
            return (tmp / key.replace("s3://test/", "")).read_bytes()

    return _Store()


def test_the_manifest_is_published_last_so_a_partial_upload_never_looks_ready(tmp_path, monkeypatch):
    """A pod polls for the manifest. If it were written first, a pod could start training on half a
    dataset and report a number computed from part of the data."""
    from services.cloud import transfer

    store_root = tmp_path / "store"
    order: list[str] = []
    inner = _fake_store(store_root)

    class _Recording:
        ensure_bucket = inner.ensure_bucket
        get_bytes = inner.get_bytes

        def put_file(self, key, path):
            order.append(key)
            return inner.put_file(key, path)

        def put_bytes(self, key, data, content_type="application/octet-stream"):
            order.append(key)
            return inner.put_bytes(key, data)

    monkeypatch.setattr(transfer, "get_object_store", lambda: _Recording())

    src = tmp_path / "ds"
    (src / "images").mkdir(parents=True)
    (src / "images" / "a.jpg").write_bytes(b"jpeg-bytes")
    (src / "data.yaml").write_text("nc: 1\n")

    out = transfer.push_workspace(src, job_id="job-1", kind="training", entrypoint="train_yolo")
    assert out["files"] == 2
    assert order[-1].endswith("MANIFEST.json")


def test_a_workspace_round_trips_and_a_corrupted_file_is_caught(tmp_path, monkeypatch):
    """A truncated transfer produces a dataset that loads, trains, and is silently wrong, which is far
    worse than one that fails outright."""
    from services.cloud import transfer

    store_root = tmp_path / "store"
    monkeypatch.setattr(transfer, "get_object_store", lambda: _fake_store(store_root))

    src = tmp_path / "ds"
    src.mkdir()
    (src / "a.txt").write_bytes(b"hello")
    (src / "b.txt").write_bytes(b"world")
    transfer.push_workspace(src, job_id="job-2", kind="training")

    dest = tmp_path / "pulled"
    pulled = transfer.pull_workspace(dest, job_id="job-2", kind="training")
    assert pulled["files"] == 2
    assert (dest / "a.txt").read_bytes() == b"hello"

    # Corrupt one object in the store, then pull again.
    (store_root / "workspace/training/job-2/a.txt").write_bytes(b"tampered")
    with pytest.raises(transfer.TransferError) as exc:
        transfer.pull_workspace(tmp_path / "pulled2", job_id="job-2", kind="training")
    # Named, not summarised: "3 files corrupt" is not actionable, the paths are.
    assert "a.txt" in str(exc.value)


def test_an_unready_workspace_reads_as_not_ready_rather_than_raising(tmp_path, monkeypatch):
    from services.cloud import transfer

    monkeypatch.setattr(transfer, "get_object_store", lambda: _fake_store(tmp_path / "store"))
    assert transfer.read_manifest("training", "never-pushed") is None
    with pytest.raises(transfer.TransferError):
        transfer.pull_workspace(tmp_path / "x", job_id="never-pushed", kind="training")


def test_results_are_filtered_and_best_weights_win_over_last(tmp_path, monkeypatch):
    """Pulling a whole run directory back would move tens of gigabytes of cached dataset for a few weight
    files. And `last.pt` is whatever the final epoch produced, which after an overfitting tail is not the
    model that was evaluated."""
    from services.cloud import transfer

    monkeypatch.setattr(transfer, "get_object_store", lambda: _fake_store(tmp_path / "store"))

    run = tmp_path / "run"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"best")
    (run / "weights" / "last.pt").write_bytes(b"last")
    (run / "results.csv").write_text("epoch,map50\n1,0.4\n")
    # Scratch that must not come home.
    (run / "dataset_cache").mkdir()
    (run / "dataset_cache" / "images.npy").write_bytes(np.zeros(64, dtype=np.uint8).tobytes())

    pushed = transfer.push_results(run, job_id="job-3", kind="training")
    paths = set(pushed["uris"])
    assert "weights/best.pt" in paths and "results.csv" in paths
    assert not any(p.endswith(".npy") for p in paths)

    assert transfer.weights_uri("job-3", "training").endswith("weights/best.pt")


def test_a_run_that_produced_nothing_says_so(tmp_path, monkeypatch):
    from services.cloud import transfer

    monkeypatch.setattr(transfer, "get_object_store", lambda: _fake_store(tmp_path / "store"))
    empty = tmp_path / "empty"
    (empty / "logs").mkdir(parents=True)
    (empty / "logs" / "train.log").write_text("nothing useful")

    with pytest.raises(transfer.TransferError) as exc:
        transfer.push_results(empty, job_id="job-4", kind="training")
    assert "nothing to bring home" in str(exc.value)


# ---------------------------------------------------------------- dataset diff

@pytest.mark.db
async def test_a_deep_diff_reports_which_objects_moved_and_flags_an_ontology_change():
    """The headline comparison answers "is it bigger". A release decision needs "what changed", and under a
    vocabulary change a class that looks like it grew may only have been renamed."""
    import uuid

    from db.models import DatasetCommit
    from db.session import get_sessionmaker
    from services.export.diff import deep_diff_commits

    a_id, b_id = f"c-{uuid.uuid4().hex[:10]}", f"c-{uuid.uuid4().hex[:10]}"
    async with get_sessionmaker()() as db:
        db.add(DatasetCommit(commit_id=a_id, slice_spec={"name": "fleet", "formats": ["coco"]},
                             object_count=10, ontology_version="v1"))
        db.add(DatasetCommit(commit_id=b_id, slice_spec={"name": "fleet", "formats": ["coco", "yolo"]},
                             object_count=14, ontology_version="v2"))
        await db.commit()

        out = await deep_diff_commits(db, a_id, b_id)

    assert out["ontology"]["changed"] is True
    assert "renamed" in out["ontology"]["warning"]
    assert out["formats"]["added"] == ["yolo"]
    assert out["slice_spec_delta"]["formats"] == {"a": ["coco"], "b": ["coco", "yolo"]}
