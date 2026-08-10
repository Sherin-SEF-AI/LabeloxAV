"""A gold val split is rebuilt, not added to.

Both materialisers write one file per frame that has surviving, in-vocabulary objects and skip the rest, into
a directory keyed only on the gold set. They used to `mkdir(exist_ok=True)` and nothing else, so whatever an
earlier build left behind was still on disk and the val pass scored it as part of the next one.

This is the failure mode this repo keeps finding: it does not raise, it moves the number. The aligned split
is rebuilt per model in that model's own class order, so a leftover label file carries a different model's
class indices and the harness reads them as this one's. On the DashLab detector the residue reported mAP50
0.381 where a clean build of the same gold set in the same call gives 0.537.

The sealed split has the same shape with a different trigger: its indices are assigned from the classes
present in the object list, so gaining or losing one object renumbers every label while stale files keep the
old numbering.
"""

from __future__ import annotations

import numpy as np
import pytest

from services.training.gold import _materialize, _materialize_aligned, _reset_split


class _FakeStore:
    """Returns one small encoded JPEG for any uri. The images are irrelevant here; the file inventory is."""

    def __init__(self):
        import cv2
        self._blob = cv2.imencode(".jpg", np.full((32, 32, 3), 128, np.uint8))[1].tobytes()

    def get_bytes(self, uri):  # noqa: ARG002
        return self._blob


class _FakeOnto:
    def __init__(self, names: dict[int, str]):
        self._names = names

    def by_id(self, cid):
        return type("C", (), {"name": self._names[cid], "id": cid})()


def _objs(frame_ids, class_id=1):
    return [{"object_id": f"o{i}", "frame_id": f, "img_uri": "s3://x", "w": 32, "h": 32,
             "class_id": class_id, "bbox": [1.0, 1.0, 10.0, 10.0], "track_id": None}
            for i, f in enumerate(frame_ids)]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Point both materialisers at a temp scratch dir with a stub object store."""
    import services.training.gold as G

    monkeypatch.setattr(G, "get_object_store", lambda: _FakeStore())
    monkeypatch.setattr(G, "get_settings", lambda: type("S", (), {"scratch_path": lambda self=None: tmp_path})())
    return tmp_path


def _files(root, gold_id, aligned: bool):
    base = root / "gold" / gold_id / ("aligned" if aligned else "")
    return (sorted(p.name for p in (base / "images/val").glob("*.jpg")),
            sorted(p.name for p in (base / "labels/val").glob("*.txt")))


def test_a_smaller_aligned_rebuild_does_not_inherit_the_larger_one(patched):
    """The DashLab case. A second model that matches fewer frames must be scored on its own frames only."""
    onto = _FakeOnto({1: "truck", 2: "autorickshaw"})
    gold = "gold-residue"

    _materialize_aligned(gold, _objs(["f1", "f2", "f3"]), onto, ["truck"])
    assert _files(patched, gold, True)[0] == ["f1.jpg", "f2.jpg", "f3.jpg"]

    _materialize_aligned(gold, _objs(["f1"]), onto, ["truck"])
    imgs, labels = _files(patched, gold, True)
    assert imgs == ["f1.jpg"], f"stale frames were scored as part of this model: {imgs}"
    assert labels == ["f1.txt"]


def test_a_model_that_matches_nothing_leaves_an_empty_split_not_the_previous_one(patched):
    """The worst version: every class is out of vocabulary, so the honest result is an empty val set. With
    residue it would instead silently report the previous model's score as this model's."""
    onto = _FakeOnto({1: "truck"})
    gold = "gold-empty"

    _materialize_aligned(gold, _objs(["f1", "f2"]), onto, ["truck"])
    assert _files(patched, gold, True)[0] == ["f1.jpg", "f2.jpg"]

    _materialize_aligned(gold, _objs(["f1", "f2"]), onto, ["person", "car"])
    assert _files(patched, gold, True) == ([], []), "an unscorable model must not inherit a score"


def test_the_sealed_split_is_rebuilt_too(patched):
    """Not just the aligned one: `_materialize` renumbers classes from whichever are present, so residue
    there carries labels whose indices point at different classes."""
    onto = _FakeOnto({1: "truck", 2: "autorickshaw"})
    gold = "gold-sealed"

    _materialize(gold, _objs(["f1", "f2", "f3"]), onto)
    assert _files(patched, gold, False)[0] == ["f1.jpg", "f2.jpg", "f3.jpg"]

    _materialize(gold, _objs(["f2"]), onto)
    assert _files(patched, gold, False)[0] == ["f2.jpg"]


def test_the_two_splits_do_not_clear_each_other(patched):
    """`aligned` lives inside the sealed directory. Resetting one must not destroy the other, or sealing a
    set would wipe the per-model split and vice versa."""
    onto = _FakeOnto({1: "truck"})
    gold = "gold-nested"

    _materialize(gold, _objs(["f1", "f2"]), onto)
    _materialize_aligned(gold, _objs(["f1"]), onto, ["truck"])

    assert _files(patched, gold, False)[0] == ["f1.jpg", "f2.jpg"], "sealing was clobbered by the aligned build"
    assert _files(patched, gold, True)[0] == ["f1.jpg"]


def test_reset_creates_the_directories_when_absent(patched):
    """It replaces the old mkdir, so a first build on a clean machine must still work."""
    out = patched / "fresh"
    _reset_split(out)
    assert (out / "images/val").is_dir()
    assert (out / "labels/val").is_dir()
