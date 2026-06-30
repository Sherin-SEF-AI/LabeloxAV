"""Milestone I: dataset snapshot diff. Count deltas between two snapshots, ontology-change detection, and
slice_spec field changes; identical snapshots diff to nothing."""

from __future__ import annotations

from services.export.snapshots import diff_commits


def _commit(cid, n, onto="v1", spec=None):
    return {"commit_id": cid, "object_count": n, "ontology_version": onto, "slice_spec": spec or {"name": "s"}}


def test_object_count_delta():
    d = diff_commits(_commit("a", 100), _commit("b", 150))
    assert d["object_count_delta"] == 50 and d["from"] == "a" and d["to"] == "b"


def test_identical_snapshots_have_no_changes():
    d = diff_commits(_commit("a", 100), _commit("a", 100))
    assert d["object_count_delta"] == 0 and d["ontology_changed"] is False and d["slice_changes"] == {}


def test_ontology_change_detected():
    d = diff_commits(_commit("a", 100, onto="v1"), _commit("b", 100, onto="v2"))
    assert d["ontology_changed"] is True


def test_slice_spec_change_reported():
    a = _commit("a", 100, spec={"name": "s", "min_conf": 0.5})
    b = _commit("b", 100, spec={"name": "s", "min_conf": 0.8})
    d = diff_commits(a, b)
    assert d["slice_changes"]["min_conf"] == {"from": 0.5, "to": 0.8}
