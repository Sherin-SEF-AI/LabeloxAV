"""What the delivered files say about the split, and two bugs that were in them.

`data.yaml` declared `train: images` and `val: images` — one directory named twice — and no `images/`
directory was ever written, so a YOLO export shipped a split that was a fiction pointing at nothing.

Label files were named from the image basename, which is not unique across sessions: two drives whose
frames are both `000001.jpg` wrote the same `labels/000001.txt` and one silently overwrote the other. That
was already losing labels; splitting by session would have made it intermittent, because a colliding pair
sometimes lands in two directories and sometimes in one.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import yaml

from services.autolabel.ontology import get_ontology
from services.export.adapter_coco import write_coco
from services.export.adapter_yolo import write_yolo
from services.export.records import ExportRecord


def _rec(*, session: int, frame: int, basename: str, split: str = "train", cls: str = "rider") -> ExportRecord:
    onto = get_ontology()
    c = onto.by_name(cls)
    return ExportRecord(
        object_id=uuid.uuid4(), frame_id=uuid.UUID(int=frame), session_id=uuid.UUID(int=session),
        ts_ns=1, cam_id="cam_f", img_uri=f"s3://frames/{session}/{basename}", width=1920, height=1080,
        vehicle_id="V1", city="BLR", class_id=c.id, class_name=c.name,
        bbox=[10.0, 20.0, 110.0, 220.0], conf=0.9, state="accepted", source="human", split=split)


class TestYolo:
    def test_data_yaml_names_the_splits_that_exist(self, tmp_path: Path):
        recs = [_rec(session=1, frame=1, basename="a.jpg", split="train"),
                _rec(session=2, frame=2, basename="b.jpg", split="val"),
                _rec(session=3, frame=3, basename="c.jpg", split="test")]
        write_yolo(recs, get_ontology(), tmp_path)
        data = yaml.safe_load((tmp_path / "data.yaml").read_text())

        assert data["train"] == "train.txt" and data["val"] == "val.txt" and data["test"] == "test.txt"
        assert len({data["train"], data["val"], data["test"]}) == 3, "the old export named one dir twice"
        for name in ("train.txt", "val.txt", "test.txt"):
            assert (tmp_path / name).exists(), "a declared split must point at something that exists"

    def test_an_unsplit_export_declares_only_train(self, tmp_path: Path):
        """Two empty list files would read to a trainer as "validation found nothing"."""
        write_yolo([_rec(session=1, frame=1, basename="a.jpg")], get_ontology(), tmp_path)
        data = yaml.safe_load((tmp_path / "data.yaml").read_text())
        assert "train" in data and "val" not in data and "test" not in data
        assert not (tmp_path / "val.txt").exists()

    def test_labels_are_written_under_their_split(self, tmp_path: Path):
        recs = [_rec(session=1, frame=1, basename="a.jpg", split="train"),
                _rec(session=2, frame=2, basename="b.jpg", split="val")]
        write_yolo(recs, get_ontology(), tmp_path)
        assert list((tmp_path / "labels" / "train").glob("*.txt"))
        assert list((tmp_path / "labels" / "val").glob("*.txt"))

    def test_two_sessions_with_the_same_basename_do_not_overwrite_each_other(self, tmp_path: Path):
        """This lost labels before the split existed, and the split would have made it intermittent."""
        recs = [_rec(session=1, frame=1, basename="000001.jpg"),
                _rec(session=2, frame=2, basename="000001.jpg")]
        write_yolo(recs, get_ontology(), tmp_path)
        written = list((tmp_path / "labels" / "train").glob("*.txt"))
        assert len(written) == 2, "one frame's labels were overwritten by the other's"

    def test_the_list_file_carries_image_uris(self, tmp_path: Path):
        """This adapter is given no object store, so it cannot ship pixels and must not imply it does."""
        write_yolo([_rec(session=1, frame=1, basename="a.jpg")], get_ontology(), tmp_path)
        assert (tmp_path / "train.txt").read_text().strip() == "s3://frames/1/a.jpg"


class TestCoco:
    def test_the_split_is_on_the_image_not_the_annotation(self, tmp_path: Path):
        """A split is a fact about a frame. Per annotation it is N copies that can contradict each other."""
        recs = [_rec(session=1, frame=1, basename="a.jpg", split="val"),
                _rec(session=1, frame=1, basename="a.jpg", split="val")]
        out = write_coco(recs, get_ontology(), None, tmp_path)
        doc = json.loads(Path(out).read_text())

        assert len(doc["images"]) == 1
        assert doc["images"][0]["split"] == "val"
        assert all("split" not in a for a in doc["annotations"])

    def test_the_image_carries_its_frame_id_since_the_basename_is_not_unique(self, tmp_path: Path):
        recs = [_rec(session=1, frame=1, basename="000001.jpg"),
                _rec(session=2, frame=2, basename="000001.jpg")]
        out = write_coco(recs, get_ontology(), None, tmp_path)
        doc = json.loads(Path(out).read_text())

        assert len(doc["images"]) == 2, "two distinct frames must remain two images"
        assert len({im["frame_id"] for im in doc["images"]}) == 2
        assert len({im["file_name"] for im in doc["images"]}) == 1, "the basenames really do collide"
