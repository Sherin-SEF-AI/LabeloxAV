"""Turning evidence into a decision, and deciding what needs a more expensive opinion.

The combiner is the part where a reasoning layer usually goes wrong, in one of two ways. Either every check
gets a veto, and the system becomes as brittle as its worst rule, so one bad prior demotes a whole class.
Or the checks are averaged, and a decisive finding ("no path proposed this class") is diluted by five mild
agreements into nothing.

What is done here instead:

- **Evidence is summed with weights, then compared against the detector's own confidence.** The detector is
  a witness, not the judge; it gets a vote proportional to how sure it is and no more.
- **Conflict is measured separately from the total.** A detection with 0.4 of support and 0.4 of opposition
  nets to zero, and that is a completely different situation from one with no evidence at all: the first
  needs adjudication, the second needs a human. Averaging cannot tell them apart, which is why `conflict`
  exists alongside `score`.
- **A single near-decisive finding can stand alone.** A boat on a road does not need a second opinion.

The output is a verdict *and* a reason. Every one of these decisions ends up in front of a person
eventually, either in rapid review or in an audit, and a decision nobody can read is one nobody can trust
or improve.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.logging import get_logger
from services.autolabel.reasoner.evidence import EvidenceContext, Finding, collect

log = get_logger("reasoner_verdict")

# What the reasoner concluded, before the gate turns it into a state.
ACCEPT = "accept"          # the evidence supports the label; auto-accept may proceed
REVIEW = "review"          # a human should look, with the evidence attached
ADJUDICATE = "adjudicate"  # the evidence conflicts; worth a Tier 2 opinion
REJECT = "reject"          # near-decisive evidence against; do not auto-accept, propose the alternative
ABSTAIN = "abstain"        # nothing could be assessed; the gate decides on its own terms

# Decisions that permit the gate to auto-accept. Abstaining is included, and that inclusion is the whole
# point of having it: a reasoner that demotes what it could not assess would put two thirds of a corpus
# without depth priors into the review queue and make the queue worthless, which is a worse outcome than
# the confident wrong labels it was built to catch.
PERMITS_AUTO_ACCEPT = frozenset({ACCEPT, ABSTAIN})

# A finding at or beyond this magnitude can decide on its own. Set where it is because the only checks
# that reach it are the ones with no plausible false positive: a class that cannot appear on a road, and a
# track that mostly says something else.
DECISIVE = 0.9

# Below this much total support, there is nothing to be confident about even without opposition.
SUPPORT_FLOOR = 0.25

# Opposition beyond this, when there is also real support, is a conflict rather than a refutation. This is
# the band Tier 2 exists for.
CONFLICT_FLOOR = 0.3

# And the weaker side must be at least this fraction of the stronger one for it to be a genuine conflict.
#
# Without this the check misfires in the most expensive direction. A detection that physics, geometry and
# the horizon all call impossible still collects a little support (the detector's own confidence, a mild
# "ordinary on this road type"), and taking the raw minimum of the two sides reads that as conflict and
# spends a Tier 2 call adjudicating something already refuted three ways. A refutation with noise in it is
# not a disagreement.
CONFLICT_BALANCE = 0.35

# Opposition this much larger than support is a refutation rather than a doubt. Reported as such so the
# reviewer sees "we think this is wrong" instead of "we are unsure", which are different things to be
# handed and lead to different actions.
REFUTATION_RATIO = 3.0
REFUTATION_FLOOR = 1.0


@dataclass
class Verdict:
    decision: str
    score: float                 # net evidence, positive supporting the label
    conflict: float              # how much the evidence disagrees with itself
    confidence: float            # the detector's own, carried through for the trace
    reasons: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    suggested_class: str | None = None
    # What Tier 2 should be asked, when it is asked. A narrow question with the conflict in it produces a
    # far more reliable answer than "is this right?".
    adjudication_question: str | None = None

    def as_trace(self) -> dict:
        """The record written onto provenance.

        Deliberately complete rather than summarised. This is the only artefact that can later answer
        "why was this label accepted", and a trace that kept only the total would make every check's
        contribution unmeasurable, which is exactly what `attribution` needs.
        """
        return {
            "decision": self.decision,
            "score": round(self.score, 4),
            "conflict": round(self.conflict, 4),
            "detector_conf": round(self.confidence, 4),
            "suggested_class": self.suggested_class,
            "question": self.adjudication_question,
            "findings": [{"check": f.check, "weight": round(f.weight, 3), "detail": f.detail,
                          "suggests": f.suggests_class} for f in self.findings],
        }


def combine(findings: list[Finding], confidence: float) -> Verdict:
    """Weigh the evidence against the detector's confidence and decide."""
    if not findings:
        # Nothing could be assessed: no depth prior, no scene, no track, no reviewed neighbours, one path.
        # This is not the same as having looked and found nothing, and treating it the same was a real
        # defect: on a real session it routed 67% of objects to review, because most objects genuinely
        # have no context to reason from. The reasoner abstains and the gate decides on confidence as it
        # did before, which is the honest position when there is nothing to add.
        return Verdict(ABSTAIN, 0.0, 0.0, confidence,
                       reasons=["no check could be applied to this detection"], findings=[])

    support = sum(f.weight for f in findings if f.weight > 0)
    against = -sum(f.weight for f in findings if f.weight < 0)

    # The detector votes for its own label, scaled by how sure it is, and centred so that a 0.5 detection
    # contributes nothing either way. A detector at 0.95 is worth about as much as one strong check, which
    # is roughly right: it is one more opinion, not the arbiter.
    detector_vote = (confidence - 0.5) * 1.2
    score = support - against + detector_vote

    # Conflict is the smaller of the two sides, but only when the two sides are actually comparable.
    # Two strong opposing bodies of evidence conflict; overwhelming opposition with a little noise on the
    # other side does not, and treating it as conflict spends a Tier 2 call on a settled question.
    for_side = support + max(0.0, detector_vote)
    smaller, larger = min(for_side, against), max(for_side, against)
    balanced = larger > 0 and (smaller / larger) >= CONFLICT_BALANCE
    conflict = smaller if balanced else 0.0

    decisive = [f for f in findings if f.weight <= -DECISIVE]
    suggested = _suggested_class(findings)

    if decisive:
        return Verdict(REJECT, score, conflict, confidence,
                       reasons=[f.detail for f in decisive], findings=findings,
                       suggested_class=suggested,
                       adjudication_question=None)

    # Overwhelming opposition from several independent checks is a refutation, even though no single one
    # of them was decisive on its own. Three checks agreeing that a box is impossible is a stronger
    # statement than any of them made separately.
    if against >= REFUTATION_FLOOR and against >= for_side * REFUTATION_RATIO:
        return Verdict(REJECT, score, conflict, confidence,
                       reasons=[f.detail for f in findings if f.weight < 0],
                       findings=findings, suggested_class=suggested)

    if conflict >= CONFLICT_FLOOR:
        question = _question(findings, suggested)
        return Verdict(ADJUDICATE, score, conflict, confidence,
                       reasons=[f.detail for f in findings if f.weight < 0],
                       findings=findings, suggested_class=suggested,
                       adjudication_question=question)

    if against > support + max(0.0, detector_vote):
        return Verdict(REVIEW, score, conflict, confidence,
                       reasons=[f.detail for f in findings if f.weight < 0],
                       findings=findings, suggested_class=suggested)

    if support + max(0.0, detector_vote) < SUPPORT_FLOOR:
        # Checks ran and found nothing either way. Distinct from the abstain above, where none could run:
        # here the evidence was available and was uninformative, which is a mild reason to look rather
        # than a reason to demote, so it abstains too but says which of the two happened.
        return Verdict(ABSTAIN, score, conflict, confidence,
                       reasons=["the checks that ran found nothing either way"],
                       findings=findings, suggested_class=suggested)

    return Verdict(ACCEPT, score, conflict, confidence,
                   reasons=[f.detail for f in findings if f.weight > 0][:3],
                   findings=findings, suggested_class=None)


def _suggested_class(findings: list[Finding]) -> str | None:
    """The alternative the evidence points at, if the checks agree on one.

    Weighted by how strongly each check argued, so a temporal majority outvotes a single mixed-neighbour
    hint rather than the last check written winning.
    """
    votes: dict[str, float] = {}
    for f in findings:
        if f.suggests_class and f.weight < 0:
            votes[f.suggests_class] = votes.get(f.suggests_class, 0.0) + abs(f.weight)
    if not votes:
        return None
    return max(votes.items(), key=lambda kv: kv[1])[0]


def _question(findings: list[Finding], suggested: str | None) -> str:
    """The narrow question Tier 2 is asked.

    Narrow on purpose. A VLM asked "is this label right?" free-associates; the same VLM asked "the track
    says e_rickshaw and the detector says autorickshaw, which is it?" with the crop in front of it is
    markedly more reliable, because the question carries the discriminating evidence.
    """
    against = [f for f in findings if f.weight < 0]
    if not against:
        return "does the crop match the labelled class?"
    reason = max(against, key=lambda f: abs(f.weight)).detail
    if suggested:
        return f"the evidence suggests {suggested} rather than the labelled class, because {reason}."
    return f"the labelled class is contradicted because {reason}."


def reason_about(ctx: EvidenceContext, only: list[str] | None = None) -> Verdict:
    """The whole Tier 1 pass for one detection: gather, weigh, decide."""
    findings = collect(ctx, only=only)
    verdict = combine(findings, ctx.obj.conf)
    log.debug("reasoner.verdict", cls=ctx.obj.class_name, decision=verdict.decision,
              score=round(verdict.score, 3), findings=len(findings))
    return verdict
