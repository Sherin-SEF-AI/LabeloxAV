"""Making a relabel proposal argue for itself.

The existing reasoning is one signal: SigLIP scores a crop against every class name and proposes a change
when another name beats the current one by a margin. Its own docstring records what that was worth. Against
302 human-verified objects, all 10 changes it would have applied unreviewed overruled the person, including
traffic_sign -> milestone at 0.985 and minivan -> mpv at 0.945. A softmax over name similarity peaks exactly
where two names are close, which is where it knows least.

Two signals are added, both from evidence this system already owns.

The Review trail says which swaps are real. Of 296 class corrections, e_auto <-> motorcycle is 195 of them,
66%. A proposal to make that swap repeats something humans have done 195 times; traffic_sign -> milestone is
something nobody here has ever done. A bare confidence threshold treats those identically.

And the VLM judge, asked whether the existing label is right rather than what the object is, can abstain.
Measured this week: the proposer model returned the same verdict for a 248x830 crop and a 32x29 one, while
the judge abstains on a quarter of crops and cites what it can see.

Track context is deliberately absent and the reason is in the data: 99% of objects have a track, but of the
10,204 tracks with three or more members only 751 carry a single class. 93% mix classes, so a track's
majority would launder a tracking failure into a labelling decision.
"""

from __future__ import annotations

import pytest

from services.agent.relabel_reasoning import (
    REVERSE,
    SEEN,
    UNSEEN,
    adjudicate,
    bar_for,
    combine,
    needs_adjudication,
    rationale,
    swap_prior,
)

# The real shape of the corpus trail: one dominant bidirectional pair, one one-way, one rarity.
PAIRS = {
    ("e_auto", "motorcycle"): 125,
    ("motorcycle", "e_auto"): 70,
    ("rider", "e_auto"): 33,
    ("cycle", "e_auto"): 2,
}


# ------------------------------------------------------------------------------- what the corpus has seen

def test_a_swap_humans_have_made_here_is_recognised():
    p = swap_prior("e_auto", "motorcycle", PAIRS)
    assert p["prior"] == SEEN
    assert "125 times" in p["detail"]


def test_a_swap_nobody_has_ever_made_is_marked_unseen():
    """traffic_sign -> milestone at 0.985 was one of the ten that overruled a human."""
    p = swap_prior("traffic_sign", "milestone", PAIRS)
    assert p["prior"] == UNSEEN
    assert "no human has ever" in p["detail"]


def test_a_pair_corrected_the_other_way_still_counts_as_confusable():
    """It says nothing about which way this instance goes, but everything about whether the two classes are
    genuinely distinguishable, which is the question the bar is asking."""
    assert swap_prior("motorcycle", "e_auto", PAIRS)["prior"] in (SEEN, REVERSE)
    assert swap_prior("rider", "e_auto", PAIRS)["prior"] == SEEN


def test_one_off_corrections_do_not_make_a_precedent():
    """cycle -> e_auto happened twice. Two people is not the corpus knowing something."""
    assert swap_prior("cycle", "e_auto", PAIRS)["prior"] == UNSEEN


def test_an_empty_trail_makes_everything_unseen():
    """A fresh deployment has no precedent for anything, and should demand more, not less."""
    assert swap_prior("e_auto", "motorcycle", {})["prior"] == UNSEEN


# ------------------------------------------------------------------------------- the bar

def test_an_unseen_swap_has_to_clear_more_than_a_familiar_one():
    seen = bar_for(SEEN, cross_l1=False, cross_conf=0.90, cross_margin=0.55)
    unseen = bar_for(UNSEEN, cross_l1=False, cross_conf=0.90, cross_margin=0.55)
    assert unseen["conf"] > seen["conf"]
    assert unseen["margin"] > seen["margin"]


def test_no_prior_ever_makes_the_bar_lower_than_the_old_default():
    """The point is to demand more where there is less reason to believe, never to wave things through
    because the corpus happens to have a precedent."""
    for prior in (SEEN, REVERSE, UNSEEN):
        b = bar_for(prior, cross_l1=False, cross_conf=0.90, cross_margin=0.55)
        assert b["conf"] >= 0.45 and b["margin"] >= 0.15


def test_crossing_a_superclass_keeps_the_hard_bar_whatever_the_precedent():
    """50 of the 73 wrong changes on human-verified objects crossed a superclass. A precedent for the swap
    does not make the crossing safer."""
    b = bar_for(SEEN, cross_l1=True, cross_conf=0.90, cross_margin=0.55)
    assert b["conf"] == 0.90 and b["margin"] == 0.55


# ------------------------------------------------------------------------------- when to spend the judge

def test_the_judge_is_asked_where_the_corpus_offers_no_precedent():
    assert needs_adjudication(UNSEEN, cross_l1=False) is True


def test_a_familiar_swap_does_not_pay_for_a_gpu_call():
    """Half a million objects. The judge is spent where the classifier is least trustworthy, not everywhere."""
    assert needs_adjudication(SEEN, cross_l1=False) is False


def test_a_superclass_crossing_without_a_precedent_is_adjudicated():
    assert needs_adjudication(REVERSE, cross_l1=True) is True


# ------------------------------------------------------------------------------- combining

BAR = {"conf": 0.45, "margin": 0.15, "why": "prior=seen"}


def test_a_disagreeing_judge_kills_a_confident_proposal():
    """Otherwise the GPU call is spent and the answer ignored, which is worse than not asking."""
    out = combine(0.99, 0.9, BAR, "disagree")
    assert out["keep"] is False and "disagreed" in out["reason"]


def test_an_abstaining_judge_also_kills_it():
    """The instinct is to treat abstention as neutral. It is not, here: being asked at all means the
    classifier was not trusted for this swap, so "I cannot tell either" leaves nothing holding it up."""
    out = combine(0.99, 0.9, BAR, "uncertain")
    assert out["keep"] is False and "could not tell" in out["reason"]


def test_a_judge_seeing_something_else_entirely_kills_it():
    assert combine(0.99, 0.9, BAR, "other")["keep"] is False


def test_an_agreeing_judge_and_a_cleared_bar_keeps_it():
    out = combine(0.80, 0.40, BAR, "agree")
    assert out["keep"] is True and "judge agreed" in out["reason"]


def test_a_familiar_swap_passes_on_the_classifier_alone():
    out = combine(0.60, 0.20, BAR, None)
    assert out["keep"] is True


def test_the_bar_still_applies_when_the_judge_agrees():
    """The judge agreeing is not a licence to ignore the threshold; both are evidence, not overrides."""
    assert combine(0.10, 0.01, BAR, "agree")["keep"] is False


@pytest.mark.parametrize("conf,margin", [(0.40, 0.90), (0.99, 0.05)])
def test_failing_either_half_of_the_bar_is_enough_to_stop_it(conf, margin):
    out = combine(conf, margin, BAR, None)
    assert out["keep"] is False and "below the bar" in out["reason"]


# ------------------------------------------------------------------------------- what a reviewer reads

def test_the_rationale_carries_the_argument_not_just_the_verdict():
    """The panel showed "e_auto -> motorcycle 16x" and nothing about why. Somebody ruling on sixty of these
    needs the argument."""
    prior = swap_prior("e_auto", "motorcycle", PAIRS)
    bar = bar_for(prior["prior"], cross_l1=False, cross_conf=0.9, cross_margin=0.55)
    decision = combine(0.7, 0.3, bar, None)
    text = rationale("e_auto", "motorcycle", prior, bar, None, decision, 0.7)
    assert "e_auto -> motorcycle" in text
    assert "125 times" in text
    assert decision["reason"] in text


def test_the_rationale_says_when_the_judge_was_not_consulted():
    prior = swap_prior("e_auto", "motorcycle", PAIRS)
    bar = bar_for(prior["prior"], cross_l1=False, cross_conf=0.9, cross_margin=0.55)
    text = rationale("e_auto", "motorcycle", prior, bar, None, combine(0.7, 0.3, bar, None), 0.7)
    assert "not asked" in text


# ------------------------------------------------------------------------------- adjudication

class _Judge:
    """Records what it was asked, answers from a script."""

    def __init__(self, verdict: str = "agree", reason: str = "three wheels and a canopy"):
        self.verdict = verdict
        self.reason = reason
        self.calls: list[dict] = []

    def chat_json(self, prompt, *, image_jpeg=None, temperature=0.0, schema=None, model=None):
        self.calls.append({"prompt": prompt, "schema": schema, "model": model})
        return {"verdict": self.verdict, "reason": self.reason}


def _crop():
    import numpy as np

    return np.full((80, 80, 3), 128, dtype=np.uint8)


async def test_the_judge_is_asked_about_the_proposal_not_the_existing_label():
    """The easiest thing here to get backwards, and getting it backwards inverts the system silently: every
    proposal the judge rejected would be kept and every one it endorsed dropped, while every number in the
    pipeline still looked reasonable. The confusion critic asks the opposite question, so the two must never
    share a prompt."""
    j = _Judge("agree")
    await adjudicate(_crop(), "traffic_sign", "milestone", PAIRS, conf=0.9, margin=0.5,
                     cross_l1=False, cross_conf=0.9, cross_margin=0.55, judge=j)
    prompt = j.calls[0]["prompt"]
    assert "proposes changing it to `milestone`" in prompt
    assert "`agree` if `milestone` is clearly the better label" in prompt


async def test_an_unseen_swap_is_put_to_the_judge():
    j = _Judge("agree")
    _, why = await adjudicate(_crop(), "traffic_sign", "milestone", PAIRS, conf=0.9, margin=0.5,
                              cross_l1=False, cross_conf=0.9, cross_margin=0.55, judge=j)
    assert len(j.calls) == 1
    assert why["prior"] == UNSEEN
    assert why["kept"] is True


async def test_a_familiar_swap_never_pays_for_the_judge():
    j = _Judge("agree")
    _, why = await adjudicate(_crop(), "e_auto", "motorcycle", PAIRS, conf=0.6, margin=0.2,
                              cross_l1=False, cross_conf=0.9, cross_margin=0.55, judge=j)
    assert j.calls == []
    assert why["judge_verdict"] is None and why["kept"] is True


async def test_the_judge_rejecting_an_unseen_swap_drops_it():
    """traffic_sign -> milestone at 0.985 was one of the ten that overruled a human."""
    j = _Judge("disagree", "this is a roadside sign with a symbol, not a distance marker")
    _, why = await adjudicate(_crop(), "traffic_sign", "milestone", PAIRS, conf=0.985, margin=0.9,
                              cross_l1=False, cross_conf=0.9, cross_margin=0.55, judge=j)
    assert why["kept"] is False
    assert "disagreed" in why["rationale"]
    assert "distance marker" in why["judge_reason"]


async def test_an_unreachable_judge_does_not_wave_the_proposal_through():
    """The whole reason it was asked is that the classifier alone was not trusted for this swap. Failing open
    would make a network blip the most permissive state in the system."""
    class _Dead:
        def chat_json(self, *a, **k):
            raise ConnectionError("no judge")

    _, why = await adjudicate(_crop(), "traffic_sign", "milestone", PAIRS, conf=0.99, margin=0.9,
                              cross_l1=False, cross_conf=0.9, cross_margin=0.55, judge=_Dead())
    assert why["kept"] is False
    assert why["judge_verdict"] == "uncertain"


async def test_the_judge_answer_is_schema_constrained():
    j = _Judge("agree")
    await adjudicate(_crop(), "traffic_sign", "milestone", PAIRS, conf=0.9, margin=0.5,
                     cross_l1=False, cross_conf=0.9, cross_margin=0.55, judge=j)
    schema = j.calls[0]["schema"]
    assert schema["properties"]["verdict"]["enum"] == ["agree", "disagree", "uncertain", "other"]
    assert "reason" in schema["required"]


async def test_every_proposal_carries_an_argument():
    j = _Judge("agree")
    _, why = await adjudicate(_crop(), "traffic_sign", "milestone", PAIRS, conf=0.9, margin=0.5,
                              cross_l1=False, cross_conf=0.9, cross_margin=0.55, judge=j)
    assert "traffic_sign -> milestone" in why["rationale"]
    assert why["prior_detail"] in why["rationale"]


async def test_the_second_stage_is_off_unless_asked_for():
    """It only ever removes proposals, and nothing has measured that the ones it removes are the wrong ones.

    On this corpus that cannot be tested: the classifier proposes 3 changes in 3,227 objects and 0 on the 332
    human-verified labels, because the relabel pass converged and the classifier agrees with what it made. So
    there is no population left on which to show the filter helps. Defaulting it on anyway is the mistake
    this module already made with auto_keep, where the fix was not a better threshold but admitting the
    evidence was not there.
    """
    import inspect

    from services.agent.relabel_agent import plan_relabel

    assert inspect.signature(plan_relabel).parameters["reason"].default is False
    assert inspect.signature(plan_relabel).parameters["auto_keep"].default is False
