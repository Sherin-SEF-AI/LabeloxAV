"""Measure the machine judge against human work that has already been done.

The judge can rule on 570,379 objects and its verdict is worth nothing until somebody knows how often it is
right. The obvious way to find out is to have a person adjudicate a fresh sample, which costs a person's
afternoon and is why it had not happened.

It does not need one. The corpus already contains hundreds of human rulings, they are simply not stored in a
shape anybody thought to read as ground truth:

  - a review that **changed an object's class** is a person saying the machine's label was wrong
  - a review that left the class alone and moved the object to `accepted` is a person saying it was right

That is a labelled evaluation set for the judge, built entirely from work already paid for. This module
assembles it, runs the judge against it, and reports sensitivity and specificity so `judged_precision` can
correct a machine-derived rate through them.

Three things decide whether the resulting numbers mean anything, and getting any of them wrong produces a
confident number that is silently false.

**Ask about the class the machine asserted, not the class the object carries now.** For every negative the
object's current class is the human's correction. Asking "is this a motorcycle?" about an object a human
already relabelled to motorcycle would have the judge agree, the negative would score as a positive, and the
measured specificity would approach zero for a judge that is behaving perfectly. The question has to be the
one the machine originally got wrong, which is `review.before.class_id`.

**One track-level correction is one human decision, not one per object.** `reclassify_track` propagates a
single judgement across every object in a track: this corpus has 164 objects in the negative set and 31
tracks behind them. Treating the 164 as independent would shrink the specificity interval by about 2.3x and
would be exactly the kind of number that looks like a measurement and is not one. The interval is computed
on independent decisions; the object count is reported beside it because it is what the judge actually ran
on and the cost is real.

**A judge that abstains is not scored as correct.** `unsure` is counted separately in both directions.
Folding abstentions into the wrong side would let a judge that declines every hard crop report excellent
sensitivity on the easy ones.

**A refinement inside a superclass is not the same disagreement as a cross-superclass error, and conflating
them measures the calibration set rather than the judge.** This was found by comparing two models. On the
strict reading qwen3-vl:8b scored sensitivity 0.50 against qwen2.5vl:7b's 0.76, which reads as a much worse
judge. It is not: 31 of its 34 rejections of human-accepted labels proposed another four-wheeler, saying
"this is an SUV, not a sedan", with reasons like higher ground clearance and roof rails. At superclass level
the two score 0.956 and 0.943.

The gap is real and it is in the ground truth. A reviewer clicking accept on `sedan` for a car is answering
a coarser question than "is this precisely a sedan", so a stronger model looks worse by disagreeing more
usefully. Both numbers are therefore reported: `sensitivity` stays strict, and `sensitivity_superclass`
counts a within-L1 refinement as agreement. The same L1 comparison `apply_vlm` already uses to decide
whether a VLM override is cheap or a big claim.
"""

from __future__ import annotations

import uuid as _uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, MachineVerdict, Object, OntologyClass, Review
from services.labelops.sampling import wilson_interval

log = get_logger("labelops.judge_calibration")

# States a human moved an object to that mean "the label was right".
ACCEPTED_STATES = ("accepted", "submitted")


def _is_refinement(onto, asked_class: str | None, proposed_class_id: int | None) -> bool:
    """Whether a rejection swapped one class for a sibling under the same L1.

    sedan to SUV is a refinement; rider to pole is an error. The same comparison `apply_vlm` uses to decide
    whether a VLM override is cheap or a big claim, so the two cannot drift apart.
    """
    if not asked_class or proposed_class_id is None:
        return False
    try:
        return onto.by_name(asked_class).l1 == onto.by_id(int(proposed_class_id)).l1
    except Exception:  # noqa: BLE001
        return False


@dataclass
class CalibrationItem:
    """One human ruling, in the form the judge can be asked about."""

    object_id: str
    asked_class: str        # what the machine asserted, and therefore what the judge is asked to confirm
    human_says_correct: bool
    # The unit of independent human judgement. A track-level reclassify covers many objects and is one
    # decision; grouping on this is what keeps the interval honest.
    decision_key: str
    detail: dict = field(default_factory=dict)


async def build_calibration_set(db: AsyncSession, *, limit: int | None = None) -> dict:
    """Assemble the judge's evaluation set from review history.

    Returns the items plus the counts that matter: how many objects, and how many independent human
    decisions stand behind them.
    """
    rows = (await db.execute(
        select(Review.object_id, Review.action, Review.before, Review.after, Review.ts_ns,
               Object.class_id, Object.state, Object.track_id, OntologyClass.name)
        .join(Object, Object.object_id == Review.object_id)
        .join(OntologyClass, OntologyClass.id == Object.class_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .where(Frame.img_uri.isnot(None))
        .order_by(Review.ts_ns))).all()

    # Class ids resolve to names once; the judge is asked in names.
    names = dict((await db.execute(select(OntologyClass.id, OntologyClass.name))).all())

    by_object: dict[str, dict] = {}
    for oid, action, before, after, ts_ns, _class_id, state, track_id, current_name in rows:
        b = (before or {}).get("class_id")
        a = (after or {}).get("class_id")
        rec = by_object.setdefault(str(oid), {
            "state": state, "track_id": str(track_id) if track_id else None,
            "current": current_name, "first_before": None, "changed": False,
            "action": action, "ts_ns": ts_ns,
        })
        if b is not None and rec["first_before"] is None:
            rec["first_before"] = int(b)
        if b is not None and a is not None and int(b) != int(a):
            rec["changed"] = True
            rec["action"] = action

    positives: list[CalibrationItem] = []
    negatives: list[CalibrationItem] = []
    skipped_unresolvable = 0

    for oid, rec in by_object.items():
        if rec["changed"]:
            asked_id = rec["first_before"]
            asked = names.get(asked_id)
            if not asked:
                # The machine's original class is unrecoverable, so there is no question to ask. Skipped and
                # counted rather than substituted with the current class, which would invert the label.
                skipped_unresolvable += 1
                continue
            # One track-level reclassify is one decision. Falling back to the object id when there is no
            # track keeps a genuinely per-object correction counted as its own decision.
            key = f"neg:{rec['track_id']}:{asked_id}" if rec["track_id"] else f"neg:obj:{oid}"
            negatives.append(CalibrationItem(oid, asked, False, key,
                                             {"corrected_to": rec["current"], "action": rec["action"]}))
        elif rec["state"] in ACCEPTED_STATES:
            key = f"pos:{rec['track_id']}" if rec["track_id"] else f"pos:obj:{oid}"
            positives.append(CalibrationItem(oid, rec["current"], True, key,
                                             {"action": rec["action"]}))

    if limit:
        positives, negatives = positives[:limit], negatives[:limit]

    out = {
        "positives": positives,
        "negatives": negatives,
        "n_objects": len(positives) + len(negatives),
        "n_decisions": len({i.decision_key for i in positives + negatives}),
        "n_positive_objects": len(positives),
        "n_negative_objects": len(negatives),
        "n_positive_decisions": len({i.decision_key for i in positives}),
        "n_negative_decisions": len({i.decision_key for i in negatives}),
        "skipped_unresolvable": skipped_unresolvable,
    }
    log.info("judge_calibration.set", **{k: v for k, v in out.items()
                                         if k not in ("positives", "negatives")})
    return out


async def calibrate_judge(db: AsyncSession, *, limit: int | None = None, client=None,
                          model_version: str | None = None, persist: bool = True) -> dict:
    """Run the judge over the retrospective set and report how often it agrees with the humans.

    Costs no human time: every ruling it scores against was made months ago for another purpose.
    """
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from core.config import get_settings
    from core.timebase import now_ns
    from db.models import MachineVerdict
    from services.labelops.vlm_review import (
        JUDGE,
        _alternatives,
        _ask,
        _load_crop,
        parse_judge_reply,
    )
    from services.llm.router import make_vlm_client

    settings = get_settings()
    from services.autolabel.ontology import get_ontology
    from services.labelops.vlm_review import _model_version_for

    onto = get_ontology()
    client = client or make_vlm_client(settings)
    provider = getattr(settings.models.vlm, "vision_provider", "ollama")
    model_version = model_version or _model_version_for(settings, provider)
    margin = settings.models.vlm.crop_margin

    cal = await build_calibration_set(db, limit=limit)
    items: list[CalibrationItem] = cal["positives"] + cal["negatives"]

    # Per decision, not per object: the confusion counts have to be over independent human judgements or the
    # interval is a fiction. Objects sharing a decision key vote, and the majority is that decision's verdict.
    per_decision: dict[str, dict] = {}
    judged = unreadable = 0

    for item in items:
        obj = await db.get(Object, _uuid.UUID(item.object_id))
        if obj is None:
            continue
        crop = await _load_crop(db, obj, margin)
        if crop is None or crop.size == 0:
            unreadable += 1
            continue
        try:
            asked_id = onto.by_name(item.asked_class).id
        except Exception:  # noqa: BLE001
            continue
        reply = _ask(client, crop, item.asked_class, _alternatives(onto, asked_id), model=model_version)
        parsed = parse_judge_reply(reply, onto)
        judged += 1

        d = per_decision.setdefault(item.decision_key, {
            "human_says_correct": item.human_says_correct,
            "correct": 0, "incorrect": 0, "unsure": 0, "objects": 0, "refinement": False})
        d[parsed["verdict"]] += 1
        d["objects"] += 1
        if parsed["verdict"] == "incorrect" and _is_refinement(onto, item.asked_class,
                                                              parsed["proposed_class_id"]):
            d["refinement"] = True

        if persist:
            await db.execute(pg_insert(MachineVerdict).values(
                object_id=obj.object_id, judge=JUDGE, provider=provider, model_version=model_version,
                verdict=parsed["verdict"], proposed_class_id=parsed["proposed_class_id"],
                confidence=parsed["confidence"], agreement=None,
                detail={"asked_class": item.asked_class, "reason": parsed["reason"],
                        "calibration": True, "human_says_correct": item.human_says_correct,
                        **item.detail},
                batch_id="judge-calibration", ts_ns=now_ns(),
            ).on_conflict_do_update(
                constraint="uq_machine_verdict_object_judge_batch",
                set_={"verdict": parsed["verdict"], "confidence": parsed["confidence"],
                      "proposed_class_id": parsed["proposed_class_id"],
                      # detail too: leaving it stale is how a row ends up with one batch's id and another
                      # batch's contents, which is what made the theft invisible.
                      "detail": {"asked_class": item.asked_class, "reason": parsed["reason"],
                                 "calibration": True,
                                 "human_says_correct": item.human_says_correct, **item.detail},
                      "ts_ns": now_ns()}))
            if judged % 10 == 0:
                await db.commit()

    if persist:
        await db.commit()

    tp = fp = tn = fn = abstain_pos = abstain_neg = 0
    refinements = 0
    for d in per_decision.values():
        # Majority of the objects under one human decision. A tie, or all abstentions, counts as an
        # abstention rather than being broken arbitrarily toward agreement.
        if d["correct"] > d["incorrect"]:
            verdict = "correct"
        elif d["incorrect"] > d["correct"]:
            verdict = "incorrect"
        else:
            verdict = "unsure"

        if verdict == "unsure":
            if d["human_says_correct"]:
                abstain_pos += 1
            else:
                abstain_neg += 1
        elif d["human_says_correct"]:
            if verdict == "correct":
                tp += 1
            else:
                fn += 1
                if d.get("refinement"):
                    refinements += 1
        else:
            if verdict == "incorrect":
                tn += 1
            else:
                fp += 1

    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    sens_ci = wilson_interval(tp, tp + fn) if (tp + fn) else None
    spec_ci = wilson_interval(tn, tn + fp) if (tn + fp) else None

    result = {
        "judge": JUDGE, "provider": provider, "model_version": model_version,
        "objects_judged": judged, "unreadable": unreadable,
        # The honest denominator. Reported beside the object count so the gap between them is visible
        # rather than being a footnote somebody has to know to ask about.
        "independent_decisions": len(per_decision),
        "sensitivity": round(sens, 4) if sens is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "sensitivity_interval": sens_ci,
        "specificity_interval": spec_ci,
        # Counting a within-superclass refinement as agreement. Reported beside the strict figure rather
        # than instead of it: the gap between the two is a property of the calibration set, not of the
        # judge, and collapsing to either number alone hides that.
        "sensitivity_superclass": (round((tp + refinements) / (tp + fn), 4) if (tp + fn) else None),
        "refinements_within_superclass": refinements,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                      "abstained_on_correct": abstain_pos, "abstained_on_wrong": abstain_neg},
        "usable": sens is not None and spec is not None and (sens + spec) > 1.0,
        "source": ("retrospective: built from review history, so it cost no new human time. Positives are "
                   "objects a person accepted without changing the class; negatives are objects a person "
                   "reclassified, asked about the class the machine originally asserted."),
    }
    log.info("judge_calibration.done", **{k: v for k, v in result.items()
                                          if k not in ("confusion", "source")})
    return result


async def stored_calibration(db: AsyncSession, *, model_version: str | None = None) -> dict | None:
    """Read back a calibration run's verdicts and recompute sensitivity and specificity.

    Cheap: no crops, no model calls. Exists so `judged_precision` can correct without re-judging 253 crops
    every time somebody asks for a number.

    Deliberately not `judge_agreement`, which reads human ground truth off the object's current state. That
    is right for objects a human accepted or rejected and silently wrong for the negatives here: an object a
    human reclassified ends up in state `accepted`, so judge_agreement would count every negative as a
    positive and report a specificity of zero for a judge behaving perfectly. The ground truth for this set
    lives in `detail.human_says_correct`, recorded when the item was assembled.
    """
    # Refuse to blend judges. Once a second model has been calibrated, an unscoped read averages two
    # different judges into a sensitivity that belongs to neither, and the result looks exactly like a
    # measurement. The caller has to say which judge, and judged_precision derives it from the batch.
    if not model_version:
        judges = [r[0] for r in (await db.execute(
            select(MachineVerdict.model_version)
            .where(MachineVerdict.batch_id == "judge-calibration")
            .distinct())).all()]
        if len(judges) > 1:
            log.warning("judge_calibration.ambiguous", judges=sorted(judges))
            return None
        model_version = judges[0] if judges else None
        if model_version is None:
            return None

    rows = (await db.execute(
        select(MachineVerdict.verdict, MachineVerdict.detail, MachineVerdict.object_id,
               MachineVerdict.proposed_class_id)
        .where(MachineVerdict.batch_id == "judge-calibration",
               MachineVerdict.model_version == model_version))).all()
    if not rows:
        return None

    # Regroup by the decision the item belonged to, for the same reason calibrate_judge does: one
    # track-level reclassify is one human judgement however many objects it touched.
    cal = await build_calibration_set(db)
    key_by_object = {i.object_id: i.decision_key for i in cal["positives"] + cal["negatives"]}

    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    per_decision: dict[str, dict] = {}
    for verdict, detail, oid, proposed in rows:
        key = key_by_object.get(str(oid))
        if key is None:
            continue
        d = per_decision.setdefault(key, {"human_says_correct": bool((detail or {}).get("human_says_correct")),
                                          "correct": 0, "incorrect": 0, "unsure": 0, "refinement": False})
        d[verdict] = d.get(verdict, 0) + 1
        if verdict == "incorrect" and _is_refinement(onto, (detail or {}).get("asked_class"), proposed):
            d["refinement"] = True

    tp = fp = tn = fn = refinements = 0
    for d in per_decision.values():
        if d["correct"] > d["incorrect"]:
            v = "correct"
        elif d["incorrect"] > d["correct"]:
            v = "incorrect"
        else:
            continue                     # an abstention or a tie is not scored either way
        if d["human_says_correct"]:
            tp += v == "correct"
            fn += v == "incorrect"
            if v == "incorrect" and d.get("refinement"):
                refinements += 1
        else:
            tn += v == "incorrect"
            fp += v == "correct"

    if not (tp + fn) or not (tn + fp):
        return None
    return {
        "sensitivity": round(tp / (tp + fn), 4),
        "specificity": round(tn / (tn + fp), 4),
        "sensitivity_interval": wilson_interval(tp, tp + fn),
        "specificity_interval": wilson_interval(tn, tn + fp),
        "sensitivity_superclass": round((tp + refinements) / (tp + fn), 4) if (tp + fn) else None,
        "refinements_within_superclass": refinements,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "independent_decisions": len(per_decision),
        "model_version": model_version,
        "source": "retrospective calibration against existing human rulings",
    }
