"""Failure-driven mining had both ends built and no middle.

`Evaluation.failure_clusters` has a documented shape and was written `{}` unconditionally at the one place
an evaluation is recorded, so `services/sievyx/failure_mining.py:mine_from_failures` had no input and, in
consequence, no caller. Everything needed to fill it already existed: `EvalPatch` records every false
negative against a sealed gold set, a false negative carries the `object_id` of the ground truth the model
failed to find, and that object has a DINOv3 vector.

These tests cover what the grouping is allowed to claim, which matters more than that it runs. A cluster
that names a class its members mostly are not, or an empty result that cannot say why it is empty, both put
a confident wrong sentence in front of whoever reads the failure map.
"""

import numpy as np

from services.verdyx.failure_clusters import (
    MIN_CLASS_PURITY,
    MIN_CLUSTER_SIZE,
    centroids_from_clusters,
)


def test_the_cluster_floor_suits_a_small_gold_set():
    # The live corpus holds 108 false negatives across every evaluation ever run, and 11 to 47 per eval.
    # HDBSCAN's default minimum of 15 would return nothing but noise at that size.
    assert 2 <= MIN_CLUSTER_SIZE < 15


def test_a_class_is_only_named_when_the_cluster_is_mostly_that_class():
    assert MIN_CLASS_PURITY >= 0.5, "below a majority the group is held together by appearance, not label"


def test_centroids_are_recomputed_from_members_not_stored():
    """Storing a 768-dimensional vector per cluster would be a copy that can go stale against its source."""
    payload = {"clusters": {
        "0": {"member_object_ids": ["a", "b"], "size": 2},
        "1": {"member_object_ids": ["c"], "size": 1},
    }}
    vecs = {"a": [1.0, 0.0], "b": [3.0, 0.0], "c": [0.0, 4.0]}
    centroids, ids = centroids_from_clusters(payload, vecs)

    assert ids == ["0", "1"]
    assert np.allclose(centroids[0], [2.0, 0.0])     # the mean of a and b
    assert np.allclose(centroids[1], [0.0, 4.0])


def test_a_cluster_whose_members_have_no_vectors_is_dropped_not_zeroed():
    """A missing embedding must not become an origin centroid, which would attract every mined candidate."""
    payload = {"clusters": {"0": {"member_object_ids": ["gone"], "size": 1}}}
    centroids, ids = centroids_from_clusters(payload, {})
    assert centroids == [] and ids == []


def test_cluster_ids_stay_aligned_with_centroids():
    """`mine_from_failures` returns a `failure_mode` index, and it has to map back to a named cluster."""
    payload = {"clusters": {
        "0": {"member_object_ids": ["missing"], "size": 1},
        "7": {"member_object_ids": ["x"], "size": 1},
    }}
    centroids, ids = centroids_from_clusters(payload, {"x": [1.0, 1.0]})
    assert len(centroids) == len(ids) == 1
    assert ids[0] == "7", "the surviving cluster keeps its own id, not its position"


def test_the_miner_consumes_what_this_produces():
    """The seam that was never joined: centroids out of here go straight into mine_from_failures."""
    from services.sievyx.failure_mining import mine_from_failures

    payload = {"clusters": {"0": {"member_object_ids": ["a", "b"], "size": 2}}}
    centroids, ids = centroids_from_clusters(payload, {"a": [1.0, 0.0], "b": [1.0, 0.0]})

    pool = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    out = mine_from_failures(centroids, pool, ["near", "far", "close"], k=3)

    assert [o["id"] for o in out][:2] == ["near", "close"], "the pool ranks by likeness to the failure"
    assert ids[out[0]["failure_mode"]] == "0"
