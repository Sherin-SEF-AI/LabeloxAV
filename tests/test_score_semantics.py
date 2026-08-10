"""A standing guard against scores that are not scores.

Four separate instances of one mistake turned up in a single audit, which makes it a pattern rather than bad
luck: a quantity is computed, put in a field named for something else, and then compared against quantities
from other sources as if they were commensurable.

  - `near_dup_inconsistent` stored frame similarity where error likelihood belonged. A candidate cannot
    exist below the 0.96 similarity gate, so all 45,313 sat between 0.986 and 1.0 and took 98.5% of the top
    thousand of a queue ranked across detectors.
  - `embedding_outlier` stored raw cosine distance, which runs to 2.0. The corpus held a candidate at 1.065,
    ranked above a detector reporting certainty.
  - the active-learning selector took a bare max() across those incommensurable scores.
  - a quality certificate would issue for an evaluation that scored nothing, with a valid signature over the
    emptiness.

Every one was individually invisible: nothing crashed, nothing warned, and each number looked exactly like a
measurement. The common defence is not more care, it is a contract that says what a field means and a test
that enforces it, so the next detector cannot join the queue on a different scale without failing here.

These are deliberately cheap structural checks over source, not behaviour tests. They exist to fail when
somebody adds a detector, which is the moment the mistake gets made.
"""

from __future__ import annotations

import inspect
import re

import pytest

from services.errordetect import (
    confident,
    consistency,
    critic_detector,
    embedding_outlier,
    near_dup,
    policy,
)

# Every module that emits a candidate onto the shared queue. Adding a detector means adding it here, which
# is the point: the list is the contract about who is being ranked against whom.
DETECTOR_MODULES = {
    "confident_learning": confident,
    "embedding_outlier": embedding_outlier,
    "near_dup_inconsistent": near_dup,
    "policy_violation": policy,
    "critic_flag": critic_detector,
    "consistency": consistency,
}


def _score_expressions(module) -> list[str]:
    """Every literal `"score": <expr>` in a detector module."""
    src = inspect.getsource(module)
    return re.findall(r'"score":\s*([^,\n]+)', src)


@pytest.mark.parametrize("name,module", sorted(DETECTOR_MODULES.items()))
def test_every_detector_bounds_its_score(name, module):
    """A score on this queue is ranked against `confident_learning`, which emits a probability.

    An unbounded quantity in that field is not a slightly-wrong number, it is a number that always wins.
    Each expression must show its bound: a round() of something already in [0, 1], a min()/max() clamp, or a
    helper whose own tests pin the range.
    """
    exprs = _score_expressions(module)
    assert exprs, f"{name} emits no score expression; the parametrisation is stale"

    # The contract is one function. A raw expression here is the thing this test exists to catch, however
    # obviously in-range it looks today: policy_violation and critic_flag emit hand-assigned constants that
    # are correct by luck, and a new rule written with a severity of 5.0 would outrank every calibrated
    # detector with nothing saying so.
    bounded_markers = ("as_suspicion(", "_suspicion(")
    for e in exprs:
        e = e.strip()
        assert any(m in e for m in bounded_markers), (
            f"{name} emits score `{e}` without routing through as_suspicion(). Every score on this queue "
            f"is ranked against confident_learning's probability, so an unbounded one always outranks it. "
            f"Use services.errordetect.score.as_suspicion, which is where that contract lives.")


def test_the_near_dup_score_is_not_the_similarity_that_gated_it():
    """The original defect, pinned so it cannot come back.

    Scoring a candidate with the value that had to clear a threshold for the candidate to exist produces a
    number bounded below by that threshold, which carries almost no information and dominates everything.
    """
    src = inspect.getsource(near_dup.detect_near_dup_inconsistent)
    assert '"score": _suspicion(' in src
    assert '"score": round(float(sim)' not in src, "the score must not be the gating similarity"
    # the similarity is still reported, as evidence rather than as the ranking
    assert '"similarity": round(float(sim), 4)' in src


def test_the_outlier_score_cannot_exceed_a_certainty():
    """Cosine distance runs to 2.0. Emitted raw it outranks a detector that is sure."""
    src = inspect.getsource(embedding_outlier.detect_embedding_outliers)
    assert "as_suspicion(d)" in src


def test_the_shared_clamp_handles_the_values_that_defeat_a_bound_check():
    """NaN is the one that would slip through a naive clamp: it compares false against every threshold, so
    a NaN score sorts to one end of the queue and never trips a range assertion."""
    from services.errordetect.score import as_suspicion

    assert as_suspicion(float("nan")) == 0.0
    assert as_suspicion(5.0) == 1.0
    assert as_suspicion(-3.0) == 0.0
    assert as_suspicion(None) == 0.0
    assert as_suspicion("x") == 0.0
    assert as_suspicion(0.4267) == 0.4267


def test_cross_detector_ranking_weights_by_measured_precision():
    """Ranking incommensurable scores against each other needs a common currency, and the only honest one
    is how often each detector turns out to be right."""
    from services.activelearn import selector

    src = inspect.getsource(selector.score_candidates)
    assert "_detector_weights" in src
    assert "UNMEASURED_DETECTOR_WEIGHT" in src
    assert "max(err_scores.get(str(oid), 0.0), float(sc))" not in src, "the bare max() must stay gone"


def test_a_detector_added_without_a_declared_weight_defaults_to_unproven_not_trusted():
    """The failure mode when this list goes stale: a new detector inherits full trust silently."""
    from services.activelearn.selector import UNMEASURED_DETECTOR_WEIGHT

    assert 0.0 < UNMEASURED_DETECTOR_WEIGHT < 1.0, (
        "an unjudged detector must neither be trusted like a measured one nor silenced entirely")


def test_the_queue_ranks_every_detector_against_every_other():
    """Why all of the above matters. If the queue were per-detector, incommensurable scales would be
    harmless; because it is one ranked list, they are not."""
    from services.errordetect import queue

    src = inspect.getsource(queue.list_candidates)
    assert "ErrorCandidate.score.desc()" in src


def test_a_certificate_refuses_an_evaluation_that_measured_nothing():
    """The fourth instance: a signature over an empty claim verifies exactly as well as one over a real
    claim, so the refusal has to happen before signing."""
    from services.export import certificate

    src = inspect.getsource(certificate.build_certificate)
    assert "no scored patches" in src
    idx_guard = src.index("no scored patches")
    idx_sign = src.index("hmac.new")
    assert idx_guard < idx_sign, "the refusal must precede signing"
