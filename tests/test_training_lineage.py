"""A promoted champion has to be traceable to the exact data that produced it.

`register(..., dataset_commit=name)` stamped the human-readable build name, so two "loop-v1" builds a week
apart were different corpora carrying the same string. DatasetCommit.content_fingerprint already existed
and the export path already wrote one; training never did. The governance layer's strongest claim - this
model beat the champion on a sealed gold set - sat on a training set nobody could reconstruct.

Pure: the fingerprint is taken over a directory, so these build one on disk rather than training anything.
"""

from __future__ import annotations

from services.training.lineage import dataset_fingerprint


def _build(tmp_path, labels: dict[str, str], sub: str = "labels/train"):
    d = tmp_path / sub
    d.mkdir(parents=True, exist_ok=True)
    for name, body in labels.items():
        (d / name).write_text(body)
    return tmp_path


class TestItDescribesTheData:
    def test_the_same_labels_give_the_same_fingerprint(self, tmp_path):
        a = _build(tmp_path / "a", {"1.txt": "0 0.5 0.5 0.2 0.2\n"})
        b = _build(tmp_path / "b", {"1.txt": "0 0.5 0.5 0.2 0.2\n"})
        assert dataset_fingerprint(a, "v1") == dataset_fingerprint(b, "v1")

    def test_one_moved_box_changes_it(self, tmp_path):
        a = _build(tmp_path / "a", {"1.txt": "0 0.5 0.5 0.2 0.2\n"})
        b = _build(tmp_path / "b", {"1.txt": "0 0.6 0.5 0.2 0.2\n"})
        assert dataset_fingerprint(a, "v1") != dataset_fingerprint(b, "v1"), (
            "an edited annotation must produce a different training set identity; this is the whole point")

    def test_one_extra_object_changes_it(self, tmp_path):
        a = _build(tmp_path / "a", {"1.txt": "0 0.5 0.5 0.2 0.2\n"})
        b = _build(tmp_path / "b", {"1.txt": "0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.1 0.1\n"})
        assert dataset_fingerprint(a, "v1") != dataset_fingerprint(b, "v1")

    def test_an_extra_frame_changes_it(self, tmp_path):
        a = _build(tmp_path / "a", {"1.txt": "0 0.5 0.5 0.2 0.2\n"})
        b = _build(tmp_path / "b", {"1.txt": "0 0.5 0.5 0.2 0.2\n", "2.txt": "0 0.5 0.5 0.2 0.2\n"})
        assert dataset_fingerprint(a, "v1") != dataset_fingerprint(b, "v1")

    def test_the_ontology_version_is_part_of_the_identity(self, tmp_path):
        d = _build(tmp_path / "a", {"1.txt": "0 0.5 0.5 0.2 0.2\n"})
        assert dataset_fingerprint(d, "v1") != dataset_fingerprint(d, "v2"), (
            "the same class ids mean different things under a different ontology")

    def test_the_build_spec_is_part_of_the_identity(self, tmp_path):
        d = _build(tmp_path / "a", {"1.txt": "0 0.5 0.5 0.2 0.2\n"})
        assert dataset_fingerprint(d, "v1", {"states": []}) != dataset_fingerprint(d, "v1", {"states": ["accepted"]})

    def test_it_is_order_independent(self, tmp_path):
        # rglob order is filesystem-dependent; the identity must not be.
        a = _build(tmp_path / "a", {"1.txt": "x\n", "2.txt": "y\n"})
        b = _build(tmp_path / "b", {"2.txt": "y\n", "1.txt": "x\n"})
        assert dataset_fingerprint(a, "v1") == dataset_fingerprint(b, "v1")

    def test_train_and_val_are_both_covered(self, tmp_path):
        a = tmp_path / "a"
        _build(a, {"1.txt": "x\n"}, sub="labels/train")
        _build(a, {"9.txt": "y\n"}, sub="labels/val")
        b = tmp_path / "b"
        _build(b, {"1.txt": "x\n"}, sub="labels/train")
        _build(b, {"9.txt": "z\n"}, sub="labels/val")
        assert dataset_fingerprint(a, "v1") != dataset_fingerprint(b, "v1"), (
            "a different validation split is a different result, so it is a different dataset identity")


class TestItRefusesToInventLineage:
    def test_a_directory_with_no_labels_is_unfingerprinted(self, tmp_path):
        """A hash over nothing is a constant, and a constant makes every build look identical.

        That is worse than having no fingerprint, because it looks like lineage. Returning empty makes the
        caller fall back to the build name and leaves content_fingerprint NULL, which is what the column
        already means for pre-0061 commits.
        """
        (tmp_path / "images/train").mkdir(parents=True)
        assert dataset_fingerprint(tmp_path, "v1") == ""

    def test_a_missing_directory_is_unfingerprinted(self, tmp_path):
        assert dataset_fingerprint(tmp_path / "nope", "v1") == ""
