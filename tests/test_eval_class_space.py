"""A model is scored in its own class space, and the training gate compares like with like.

The champion knows 15 classes in its own order; the training set carries the ontology's 156. Ultralytics
matches by class index, and only index 0 named the same class in both, so the champion's baseline scored
mAP50 0.0001 and every candidate read as an improvement over it.
"""

from __future__ import annotations

import yaml

from services.training.eval import aligned_data_yaml, regression_gate


def _dataset(tmp_path, names):
    root = tmp_path / "ds"
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    (root / "images" / "val" / "f.jpg").write_bytes(b"x")
    (root / "labels" / "val" / "f.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n2 0.7 0.7 0.1 0.1\n")
    y = root / "data.yaml"
    y.write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                 "nc": len(names), "names": names}))
    return y


def test_labels_are_rewritten_into_the_model_indices_by_name(tmp_path):
    y = _dataset(tmp_path, {0: "pedestrian", 1: "scooter", 2: "sedan"})
    out, info = aligned_data_yaml({0: "sedan", 1: "pedestrian"}, str(y))

    assert info["remapped"] and info["shared_classes"] == 2
    assert info["labels_kept"] == 2 and info["labels_dropped"] == 1      # the model has no scooter
    data = yaml.safe_load(open(out))
    assert data["names"] == {0: "sedan", 1: "pedestrian"}
    root = tmp_path / "ds"
    aligned = (tmp_path / [p.name for p in tmp_path.iterdir() if p.name.startswith("ds__as-")][0])
    assert (aligned / "labels" / "val" / "f.txt").read_text() == "1 0.5 0.5 0.1 0.1\n0 0.7 0.7 0.1 0.1\n"
    assert (aligned / "images" / "val" / "f.jpg").samefile(root / "images" / "val" / "f.jpg")


def test_a_dataset_already_in_the_model_class_space_is_used_as_is(tmp_path):
    y = _dataset(tmp_path, {0: "pedestrian", 1: "sedan"})
    out, info = aligned_data_yaml(["pedestrian", "sedan"], str(y))
    assert out == str(y) and not info["remapped"]


def test_the_gate_compares_different_class_sets_on_the_classes_they_share():
    """The champion, scored on its own classes, beats a candidate whose headline mAP is averaged over far
    more (and rarer) classes; on the shared classes the candidate is worse, and the gate says so."""
    baseline = {"map50": 0.30, "per_class": {"pedestrian": 0.32, "sedan": 0.28}}
    candidate = {"map50": 0.05, "per_class": {"pedestrian": 0.02, "sedan": 0.30, "scooter": 0.0,
                                              "e_auto": 0.0}}
    g = regression_gate(candidate, baseline, max_class_drop=0.15)
    assert g["compared_on"] == "2 shared classes"
    assert g["map50_delta"] == round((0.02 + 0.30) / 2 - (0.32 + 0.28) / 2, 4)
    assert not g["promote"]
    assert [r["class"] for r in g["regressed_classes"]] == ["pedestrian"]


def test_identical_class_sets_keep_the_headline_comparison():
    base = {"map50": 0.40, "per_class": {"a": 0.4}}
    cand = {"map50": 0.45, "per_class": {"a": 0.45}}
    g = regression_gate(cand, base)
    assert g["compared_on"] == "all classes" and g["map50_delta"] == 0.05 and g["promote"]
