"""Tier 2: asking the reasoner to settle a specific conflict, not to re-do the annotation.

The obvious way to use a VLM here is to show it the crop and ask whether the label is right. That performs
much worse than it sounds, for a reason worth stating plainly: an open question invites the model to
free-associate over everything in the image, and its answer is then a second unreliable opinion sitting
next to the first with no way to choose between them.

What is done instead is narrow. Tier 1 has already established *what is in dispute* ("the track says
e_rickshaw, the detector says autorickshaw"), so Tier 2 is handed that dispute, the two candidate classes,
and the evidence, and asked only to break the tie. A model given a binary question with the discriminating
evidence in front of it is markedly more reliable than the same model asked an open one.

Three further constraints:

- **The shortlist is the dispute, not the ontology.** Offering 191 classes to a model adjudicating between
  two invites it to pick a third, which resolves nothing and creates a new disagreement.
- **The adjudicator can decline.** "Neither of these, or I cannot tell" is a real answer and the most
  useful one it gives, because it is what routes the object to a human instead of guessing.
- **Its agreement across votes is carried into the verdict.** A 3-2 split is not the same answer as 5-0,
  and collapsing them loses exactly the signal that says whether a human is still needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.logging import get_logger
from services.autolabel.reasoner.evidence import load_priors
from services.autolabel.reasoner.verdict import ACCEPT, REVIEW, Verdict

log = get_logger("reasoner_adjudicate")

# Below this agreement across votes the adjudicator has not settled anything, and the object goes to a
# human. A bare majority from a model that was asked because two other signals disagreed is not a
# resolution; it is a third opinion.
#
# Set above two thirds deliberately, so a 3-2 split across five votes does not resolve. At 0.6 it did,
# which contradicted the whole reason votes are counted rather than taken once: if a bare majority were
# enough, one vote would do.
MIN_AGREEMENT = 0.67


@dataclass
class Adjudication:
    """What Tier 2 concluded, and how firmly."""

    resolved: bool
    class_name: str | None
    agreement: float
    votes: int
    # True when the adjudicator picked the class the detector already had, as opposed to the alternative.
    upheld: bool = False
    detail: str = ""

    def as_trace(self) -> dict:
        return {"resolved": self.resolved, "class_name": self.class_name,
                "agreement": round(self.agreement, 3), "votes": self.votes,
                "upheld": self.upheld, "detail": self.detail}


def dispute_shortlist(current: str, suggested: str | None, onto) -> list[str]:
    """The candidate classes for one adjudication.

    The dispute itself plus, when the two are a known confusable pair, the rest of that pair. Nothing
    else: a model adjudicating between two classes that is offered a hundred and ninety-one will
    occasionally pick a third, which resolves nothing and creates a fresh disagreement to adjudicate.
    """
    names = [current]
    if suggested and suggested != current and onto.has_name(suggested):
        names.append(suggested)

    pairs = (load_priors().get("confusable_pairs") or [])
    for pair in pairs:
        if current in pair:
            for other in pair:
                if other != current and onto.has_name(other) and other not in names:
                    names.append(other)
    # A hard cap. Beyond a handful this stops being an adjudication and becomes classification again.
    return names[:4]


def adjudicate(verifier, image_bgr: np.ndarray, bbox: tuple, verdict: Verdict,
               onto, current_class: str, votes: int | None = None) -> Adjudication:
    """Break one tie with the VLM, restricted to the classes actually in dispute.

    Takes the existing VlmVerifier rather than a new client: it already owns the crop margins, the vote
    plans, the ontology validation and the attribute schema, and a second path to the same model would
    drift from it.

    `current_class` is passed rather than recovered from the verdict's findings. Parsing it back out of a
    human-readable sentence would break the moment somebody improved the wording, and a tie-breaker that
    silently forgets which side the detector was on is worse than none.
    """
    shortlist = dispute_shortlist(current_class, verdict.suggested_class, onto)
    if len(shortlist) < 2:
        # Nothing to adjudicate between. Saying so is better than asking a model to confirm its own input,
        # which it will nearly always do.
        return Adjudication(False, None, 0.0, 0,
                            detail="no alternative class was in dispute")

    try:
        # The verifier's own shortlist logic is bypassed by passing the disputed class, and the narrowing
        # happens here because only the reasoner knows what the dispute is.
        result = _verify_restricted(verifier, image_bgr, bbox, shortlist, votes)
    except Exception as exc:  # noqa: BLE001
        # An unavailable VLM is not a verdict. The object falls through to a human, which is the correct
        # outcome when the tie-breaker could not be consulted.
        log.warning("reasoner.adjudication_failed", error=f"{type(exc).__name__}: {exc}")
        return Adjudication(False, None, 0.0, 0, detail=f"the adjudicator was unavailable ({type(exc).__name__})")

    if not result.class_name:
        return Adjudication(False, None, float(result.agreement or 0.0), int(result.votes or 0),
                            detail="the adjudicator did not name a class")

    agreement = float(result.agreement or 0.0)
    if agreement < MIN_AGREEMENT:
        return Adjudication(False, result.class_name, agreement, int(result.votes or 0),
                            detail=(f"the adjudicator split {agreement:.0%} on {result.class_name}, "
                                    "which does not settle a dispute two other signals already raised"))

    upheld = result.class_name == current_class
    return Adjudication(True, result.class_name, agreement, int(result.votes or 0), upheld=upheld,
                        detail=(f"the adjudicator chose {result.class_name} with {agreement:.0%} "
                                f"agreement across {result.votes} votes"))


def _verify_restricted(verifier, image_bgr: np.ndarray, bbox: tuple, shortlist: list[str],
                       votes: int | None):
    """Run the verifier with the shortlist pinned to the dispute.

    Implemented by temporarily overriding the verifier's shortlist rather than by duplicating its voting
    loop, so crop margins, temperatures, ontology validation and attribute handling stay in one place and
    cannot drift between the two call sites.
    """
    from collections import Counter

    from services.autolabel.paths.path_c_qwen3vl import VlmResult, crop_object

    settings = verifier.settings
    n = max(1, votes if votes is not None else settings.models.vlm.vote_count)
    schema = verifier._attr_schema()

    results = []
    for margin, temp in verifier._vote_plans(n):
        crop = crop_object(image_bgr, bbox, margin)
        results.append(verifier._validate(
            verifier.client.verify(crop, shortlist, schema, temperature=temp)))

    named = [r.class_name for r in results if r.class_name]
    if not named:
        return VlmResult(votes=n, agreement=0.0)
    majority, count = Counter(named).most_common(1)[0]
    winners = [r for r in results if r.class_name == majority]
    attrs: dict = {}
    for r in winners:
        for k, v in r.attrs.items():
            attrs.setdefault(k, v)
    return VlmResult(class_name=majority, attrs=attrs,
                     caption=next((r.caption for r in winners if r.caption), ""),
                     confident=any(r.confident for r in winners),
                     votes=n, agreement=round(count / n, 2),
                     provider=next((r.provider for r in winners if r.provider), ""))


def apply_adjudication(verdict: Verdict, adj: Adjudication, current_class: str) -> Verdict:
    """Fold Tier 2's answer back into the verdict.

    An upheld label is accepted: two signals disagreed and the adjudicator, given the specific dispute,
    sided with the detector. A corrected one goes to review rather than being auto-applied, because the
    system now holds three different opinions about this object and a machine picking among them without a
    human is precisely the failure that drove pedestrian recall from 0.73 to 0.004.
    """
    verdict.findings = list(verdict.findings)
    if not adj.resolved:
        verdict.decision = REVIEW
        verdict.reasons = [*verdict.reasons, adj.detail]
        return verdict

    if adj.upheld:
        verdict.decision = ACCEPT
        verdict.reasons = [*verdict.reasons, adj.detail]
        verdict.suggested_class = None
        return verdict

    verdict.decision = REVIEW
    verdict.suggested_class = adj.class_name
    verdict.reasons = [*verdict.reasons, adj.detail,
                       f"the adjudicator disagrees with the label {current_class!r}; a human decides"]
    return verdict

