"""Second-stage reasoning for a relabel proposal: how much evidence this particular swap needs, and whether
a model that actually looks at the image agrees.

The existing reasoning is one signal. SigLIP scores the crop against every class *name* and proposes a change
when some other name beats the current one by a margin. Its own docstring is honest about what that is worth:
measured against 302 human-verified objects, all 10 changes it would have applied unreviewed overruled the
person, including traffic_sign -> milestone at 0.985. A softmax over name similarity is highest exactly where
two names are close, which is where it is least informative.

Two things are added here, and one obvious third is deliberately left out.

**The corpus knows which swaps are real.** 296 class corrections exist in the Review trail, and they are not
spread evenly: e_auto <-> motorcycle alone is 195 of them, 66%. A proposal to make that swap is proposing
something humans have made 195 times. A proposal of traffic_sign -> milestone is proposing something nobody
has ever done here. Treating those as equally likely, which a bare confidence threshold does, is throwing
away the most specific evidence this system owns. So a swap the corpus has seen clears a normal bar, and a
swap it has never seen has to earn it.

**A model that looks at the image gets a vote.** SigLIP matches a crop against a name; the VLM judge is asked
the different question of whether the existing label is right, using the constrained verdict schema, and it
can abstain. That distinction was measured this week: the proposer model answered identically for a 248x830
crop and a 32x29 one, while the judge model abstains on a quarter of crops and cites what it can see.

**Track context is not used, because this corpus cannot support it.** The obvious enhancement is to let an
object's track-mates vote on its class, and 99% of objects have a track. But of the 10,204 tracks with at
least three members, only 751 carry a single class: 93% contain more than one. A track's majority class is
contaminated by whatever else the tracker absorbed, so voting on it would launder a tracking failure into a
labelling decision. That is the same finding that made ORACLYX refuse 40% of tracks outright.

None of this changes what happens to a proposal. Everything still routes to review, because the measurement
that put it there has not been redone. What this changes is which proposals reach a human at all, which is
the thing worth improving when review is the scarce resource.

**And it is off by default.** This stage only ever removes proposals, and nothing has yet measured that the
ones it removes are the wrong ones. On the current corpus that cannot even be tested: the classifier now
proposes 3 changes in 3,227 objects, and 0 on the 332 human-verified labels, because the corpus relabel pass
already converged and the classifier agrees with what it produced. The 73-changes-on-302-objects benchmark
this module was tuned against no longer reproduces, so there is no population left on which to show that the
filter helps. Turning a filter on before it has been shown to filter correctly is the mistake this module
already made once with auto_keep, and the fix was not a better threshold, it was admitting the evidence was
not there. Callers opt in with `reason=True`; the value is on the next model's output, not on this one's.
"""

from __future__ import annotations

from collections import Counter

from core.logging import get_logger

log = get_logger("agent.relabel_reasoning")

# How a proposed swap relates to what this corpus has actually corrected.
SEEN = "seen"            # humans have made this exact swap here
REVERSE = "reverse"      # humans have made the opposite swap, so the pair is genuinely confusable
UNSEEN = "unseen"        # nobody has ever made this swap in this corpus

# A pair needs at least this many corrections before it counts as something the corpus knows, rather than
# something one person did once. Same floor the confusion critic uses, for the same reason.
MIN_PAIR = 3

# Evidence bars per prior. The `seen` bar is the module's existing default; everything else is stricter than
# it, never looser, because the point is to demand more where there is less reason to believe, not to wave
# things through where the corpus happens to have a precedent.
BARS = {
    SEEN: {"conf": 0.45, "margin": 0.15},
    REVERSE: {"conf": 0.45, "margin": 0.15},
    UNSEEN: {"conf": 0.70, "margin": 0.35},
}

# Crossing a superclass on a swap nobody has ever made is the combination that produced every one of the bad
# auto-keeps. It is not merely harder, it requires the judge to actively agree.
CROSS_L1_UNSEEN_NEEDS_VLM = True


def swap_prior(from_name: str, to_name: str, pair_counts: dict[tuple[str, str], int], *,
               min_pair: int = MIN_PAIR) -> dict:
    """Whether this corpus has ever seen a human make this change.

    `pair_counts` is {(old, new): count} from the Review trail. The reverse direction counts too: a pair
    corrected both ways is a boundary humans disagree about, which is the strongest possible evidence that
    the two classes are genuinely confusable, even though it says nothing about which way this instance goes.
    """
    fwd = int(pair_counts.get((from_name, to_name), 0))
    rev = int(pair_counts.get((to_name, from_name), 0))
    if fwd >= min_pair:
        kind = SEEN
    elif rev >= min_pair or (fwd + rev) >= min_pair:
        kind = REVERSE
    else:
        kind = UNSEEN
    return {"prior": kind, "forward": fwd, "reverse": rev,
            "detail": (f"humans have made this exact change {fwd} times here" if kind == SEEN else
                       f"humans have corrected between these two classes {fwd + rev} times" if kind == REVERSE
                       else "no human has ever made this change in this corpus")}


def bar_for(prior: str, *, cross_l1: bool, cross_conf: float, cross_margin: float) -> dict:
    """The confidence and margin this swap has to clear.

    A cross-superclass change keeps the existing, much harder bar regardless of prior, because the measured
    failure was concentrated there: 50 of 73 wrong changes on human-verified objects crossed a superclass.
    """
    if cross_l1:
        return {"conf": cross_conf, "margin": cross_margin, "why": "crosses a superclass"}
    b = BARS.get(prior, BARS[UNSEEN])
    return {**b, "why": f"prior={prior}"}


def needs_adjudication(prior: str, *, cross_l1: bool) -> bool:
    """Whether a proposal must be put to the VLM judge before it is worth a human's attention.

    Not every proposal: the judge costs a GPU call per object and the corpus has half a million of them. It
    is spent where the classifier is least trustworthy, which is exactly where the corpus offers no
    precedent for the swap being proposed.
    """
    if prior == UNSEEN:
        return True
    return bool(cross_l1 and CROSS_L1_UNSEEN_NEEDS_VLM and prior != SEEN)


def combine(classifier_conf: float, margin: float, bar: dict, verdict: str | None) -> dict:
    """The final decision on one proposal, with the reason it was made.

    `verdict` is the judge's answer, or None when it was not asked. The rules, in the order they apply:

    A judge that disagrees with the proposal kills it. That is the whole reason for asking, and letting a
    strong classifier score override it would be spending the GPU call and then ignoring the answer.

    A judge that abstains kills it too, when it was asked. Being asked means the classifier alone was not
    trusted for this swap, so "I cannot tell either" leaves nothing holding it up. This is the opposite of
    the usual instinct to treat abstention as neutral, and it is right here because abstention is only
    reachable on the proposals that already needed help.

    Otherwise the classifier's own bar decides.
    """
    meets = classifier_conf >= bar["conf"] and margin >= bar["margin"]
    if verdict == "disagree":
        return {"keep": False, "reason": "the judge disagreed with the proposed class"}
    if verdict == "other":
        return {"keep": False, "reason": "the judge says the object is neither the current nor the proposed class"}
    if verdict == "uncertain":
        return {"keep": False, "reason": "the judge could not tell, and this swap needed its support"}
    if not meets:
        return {"keep": False,
                "reason": (f"below the bar for this swap (conf {classifier_conf:.2f} < {bar['conf']:.2f}"
                           if classifier_conf < bar["conf"] else
                           f"below the bar for this swap (margin {margin:.2f} < {bar['margin']:.2f}") + ")"}
    return {"keep": True,
            "reason": ("the judge agreed and the classifier cleared its bar" if verdict == "agree"
                       else "the classifier cleared the bar for a swap this corpus has seen before")}


def rationale(from_name: str, to_name: str, prior: dict, bar: dict, verdict: str | None,
              decision: dict, classifier_conf: float) -> str:
    """One sentence a reviewer can act on, instead of a class name and a number.

    The panel showed "e_auto -> motorcycle 16x" and nothing about why any of them were proposed. A reviewer
    deciding sixty of these needs the argument, not the count.
    """
    parts = [f"{from_name} -> {to_name} at {classifier_conf:.2f}", prior["detail"]]
    if verdict:
        parts.append(f"the judge said {verdict}")
    else:
        parts.append("the judge was not asked, the corpus has seen this swap")
    parts.append(decision["reason"])
    return "; ".join(parts)


async def corpus_pairs(db) -> dict[tuple[str, str], int]:
    """{(old_name, new_name): count} over every class correction in the Review trail.

    Read once per run rather than per object: it is a scan of the whole review table, and the answer changes
    on the timescale of a review session, not of a frame.
    """
    from sqlalchemy import select

    from db.models import Review
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    rows = (await db.execute(select(Review.before, Review.after))).all()
    out: Counter = Counter()
    for before, after in rows:
        b = (before or {}).get("class_id")
        a = (after or {}).get("class_id")
        if b is None or a is None or b == a:
            continue
        try:
            out[(onto.by_id(int(b)).name, onto.by_id(int(a)).name)] += 1
        except Exception:  # noqa: BLE001 - a class removed since the correction was made
            continue
    return dict(out)


# The verdict schema for adjudicating one proposed swap. Constrained for the same reason the critic's is: the
# whole worth of a categorical judgement is that it lands in a known set.
def adjudication_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["agree", "disagree", "uncertain", "other"]},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }


def _judge_prompt(from_name: str, to_name: str) -> str:
    """The question, phrased about the PROPOSAL rather than the existing label.

    This polarity is the easiest thing in the module to get backwards, and getting it backwards inverts the
    system silently: every proposal the judge rejected would be kept and every one it endorsed dropped, while
    every number in the pipeline still looked reasonable. The confusion critic asks the opposite question
    ("is the existing label right?"), so the two must never share a prompt.
    """
    return (
        f"This object is currently labelled `{from_name}`. A model proposes changing it to `{to_name}`.\n"
        f"Judge the proposal against the image.\n"
        f"Answer `agree` if `{to_name}` is clearly the better label, `disagree` if `{from_name}` was right, "
        f"`other` if you can see the object clearly and it is neither, and `uncertain` if the crop is too "
        f"blurry, too small or too occluded to tell. Prefer `uncertain` over guessing.\n"
        "In `reason`, cite what you can actually see that decides it."
    )


async def adjudicate(crop_bgr, from_name: str, to_name: str, pair_counts: dict, *,
                     conf: float, margin: float, cross_l1: bool,
                     cross_conf: float, cross_margin: float,
                     judge: object = None) -> tuple[str | None, dict]:
    """Second-stage decision on one proposal: the prior, the bar, the judge if it is worth asking, and why.

    Returns `(verdict, why)` where `why` carries `kept` and a sentence a reviewer can act on. The judge is
    only consulted where the corpus offers no precedent for the swap, because it costs a GPU call per object
    and there are half a million of them.
    """
    prior = swap_prior(from_name, to_name, pair_counts)
    bar = bar_for(prior["prior"], cross_l1=cross_l1, cross_conf=cross_conf, cross_margin=cross_margin)

    verdict: str | None = None
    judge_reason = ""
    if needs_adjudication(prior["prior"], cross_l1=cross_l1):
        try:
            import cv2

            from core.config import get_settings
            from services.llm.router import local_vlm_client

            ok, buf = cv2.imencode(".jpg", crop_bgr)
            if ok:
                cfg = get_settings().models.vlm
                # Duck-typed on chat_json rather than typed as VlmClient: that Protocol describes verify()'s
                # question ("what is this?"), and this asks a different one.
                vlm: object = judge or local_vlm_client()
                data = vlm.chat_json(  # type: ignore[attr-defined]
                    _judge_prompt(from_name, to_name), image_jpeg=buf.tobytes(),
                    schema=adjudication_schema(), model=cfg.judge_tag or cfg.ollama_tag)
                v = str((data or {}).get("verdict") or "")
                if v in ("agree", "disagree", "uncertain", "other"):
                    verdict = v
                    judge_reason = str((data or {}).get("reason") or "")[:300]
        except Exception as exc:  # noqa: BLE001
            # An unreachable judge is not a licence to keep a proposal it was asked about. The whole reason
            # it was asked is that the classifier alone was not trusted for this swap.
            log.warning("relabel_reasoning.judge_failed", error=str(exc)[:160])
            verdict = "uncertain"

    decision = combine(conf, margin, bar, verdict)
    return verdict, {
        "kept": decision["keep"],
        "prior": prior["prior"],
        "prior_detail": prior["detail"],
        "bar": bar,
        "judge_verdict": verdict,
        "judge_reason": judge_reason,
        "rationale": rationale(from_name, to_name, prior, bar, verdict, decision, conf),
    }
