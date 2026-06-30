"""Milestone F: ground-plane labeling review. A horizontal plane with enough ground is ok; a tilted normal,
too few points / too little ground, or a degenerate plane each flag for manual labeling."""

from __future__ import annotations

from services.lidar.clean.ground_qa import ground_plane_status

_FLAT = [0.0, 0.0, 1.0, 0.0]


def test_good_ground_is_ok():
    res = ground_plane_status(_FLAT, ground_frac=0.4, n_points=5000)
    assert res["status"] == "ok" and res["needs_review"] is False and res["verticality"] == 1.0


def test_tilted_plane_flags_review():
    res = ground_plane_status([0.6, 0.0, 0.6, 0.0], ground_frac=0.4, n_points=5000)   # verticality ~0.707
    assert res["status"] == "tilted" and res["needs_review"] is True


def test_too_little_ground_is_sparse():
    res = ground_plane_status(_FLAT, ground_frac=0.05, n_points=5000)
    assert res["status"] == "sparse" and res["needs_review"] is True


def test_too_few_points_is_sparse():
    res = ground_plane_status(_FLAT, ground_frac=0.4, n_points=100)
    assert res["status"] == "sparse" and res["needs_review"] is True


def test_degenerate_plane_is_absent():
    res = ground_plane_status([0.0, 0.0, 0.0, 0.0], ground_frac=0.0, n_points=0)
    assert res["status"] == "absent" and res["needs_review"] is True
