"""Two consolidations: shared clustering mechanism, and one demand signal taking precedence over the other.

Three modules clustered embeddings with three implementations, and two signals both named classes as needing
labels while disagreeing about which. Neither was fixed by collapsing everything into one function: the
clustering call sites ask genuinely different questions, and one of the two demand signals is right in a
situation where the other has nothing to say.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.clustering import (
    connected_components,
    greedy_cosine,
    hdbscan_labels,
    normed,
)


# --- clustering: shared mechanism, distinct linkage ----------------------------------------------


def test_greedy_keeps_every_member_close_to_its_centroid():
    """The property the ontology steward needs before proposing "these crops are one new class"."""
    a = normed([1.0, 0.0, 0.0])
    b = normed([0.98, 0.2, 0.0])
    far = normed([0.0, 1.0, 0.0])
    clusters = greedy_cosine([("a", a), ("b", b), ("c", far)], sim_thresh=0.9)
    sizes = sorted(len(c["members"]) for c in clusters)
    assert sizes == [1, 2]


def test_greedy_refuses_to_chain_where_components_would():
    """The distinction that stops the two linkages being merged.

    A, B, C each similar to the next but A and C unlike each other. Components make one cluster of three,
    which is right for discovering a rare group strung out in embedding space and wrong for minting a class:
    the proposal would be for a class that does not exist.
    """
    a = normed([1.0, 0.0, 0.0])
    b = normed([0.75, 0.66, 0.0])
    c = normed([0.0, 1.0, 0.0])
    thr = 0.6

    greedy = greedy_cosine([("a", a), ("b", b), ("c", c)], sim_thresh=thr)
    X = np.stack([a, b, c])
    comps = connected_components(X @ X.T, thr)

    assert len(comps) == 1 and len(comps[0]) == 3, "components chain, by design"
    assert len(greedy) > 1, "greedy must not chain A to C through B"


def test_components_are_order_independent():
    """Greedy is not, which is why it is the wrong tool for discovery."""
    rng = np.random.default_rng(4)
    X = np.stack([normed(v) for v in rng.normal(size=(12, 5))])
    sim = X @ X.T
    first = connected_components(sim, 0.2)
    order = rng.permutation(12)
    permuted = connected_components(sim[np.ix_(order, order)], 0.2)
    assert sorted(len(c) for c in first) == sorted(len(c) for c in permuted)


def test_a_missing_hdbscan_returns_none_rather_than_silently_substituting():
    """Two runs of the same analysis clustering differently, with nothing in the output saying so, is worse
    than one run that declines."""
    import core.clustering as cl

    saved = cl.hdbscan_available
    try:
        cl.hdbscan_available = lambda: False
        assert hdbscan_labels(np.zeros((20, 4), dtype=np.float32)) is None
    finally:
        cl.hdbscan_available = saved


def test_the_three_call_sites_use_the_shared_primitives():
    """The consolidation is real rather than a new file nobody imports."""
    import inspect

    from services.agent import ontology_steward
    from services.curation import projection
    from services.sievyx import discovery

    assert "greedy_cosine" in inspect.getsource(ontology_steward._cluster)
    assert "connected_components" in inspect.getsource(discovery._connected_components)
    assert "hdbscan_labels" in inspect.getsource(projection.cluster_labels)


def test_each_call_site_keeps_its_own_linkage():
    """Collapsing them into one clustering function would have been the wrong fix: promotion needs tight
    clusters, discovery needs chained ones."""
    import inspect

    from services.agent import ontology_steward
    from services.sievyx import discovery

    assert "connected_components" not in inspect.getsource(ontology_steward._cluster)
    assert "greedy_cosine" not in inspect.getsource(discovery._connected_components)


# --- demand signals: recall over share ------------------------------------------------------------


def _share_demand(name: str, cid: int) -> dict:
    return {"slice": name, "class_id": cid, "weight": 1.0, "safety_weight": 2.0,
            "protected": True, "delta": -0.001, "source": "share"}


def test_a_gate_recall_demand_displaces_the_share_signal():
    """The documented case: pedestrian at 514 training instances reads as healthy by share and sat at 0.02
    recall, which is what actually blocked promotion. Running both let the weaker signal draw budget."""
    from services.autolabel.ontology import get_ontology
    from services.flywheel.auto import _preferred_demands

    onto = get_ontology()
    ped = next(c for c in onto.classes if c.name == "pedestrian")
    gate = [{"slice": "pedestrian", "class_name": "pedestrian", "weight": 9.0,
             "safety_weight": 2.0, "deficit": 0.48, "kind": "floor_miss"}]
    share = [_share_demand("cattle", 1)]

    out = _preferred_demands(gate, share)
    assert [d["slice"] for d in out] == ["pedestrian"]
    assert out[0]["class_id"] == ped.id
    assert out[0]["source"] == "gate_recall"
    assert out[0]["protected"] is True


def test_share_still_runs_when_no_gate_verdict_exists():
    """A new domain pack or a fresh corpus has no blocked run to read, and share is the only signal there
    is. Deleting it would leave the flywheel with nothing to say until the first promotion failed."""
    from services.flywheel.auto import _preferred_demands

    share = [_share_demand("cattle", 1)]
    assert _preferred_demands([], share) == share


def test_a_gate_demand_naming_no_real_class_is_dropped_rather_than_guessed():
    """A demand pointing at a class the ontology cannot resolve would draw budget and produce no work
    order, which is worse than not drawing it."""
    from services.flywheel.auto import _preferred_demands

    share = [_share_demand("cattle", 1)]
    gate = [{"slice": "not_a_class", "class_name": "not_a_class", "weight": 9.0, "deficit": 0.5}]
    # nothing resolvable in the gate demands, so the cycle falls back rather than running on an empty list
    assert _preferred_demands(gate, share) == share


def test_the_recall_deficit_carries_through_as_the_collection_split_signal():
    """`delta` is what routes an empty class to collection instead of labeling. A gate demand with no delta
    would silently stop being routed."""
    from services.flywheel.auto import _preferred_demands

    gate = [{"slice": "cattle", "class_name": "cattle", "weight": 3.0, "deficit": 0.31}]
    out = _preferred_demands(gate, [])
    assert out[0]["delta"] == pytest.approx(-0.31)
