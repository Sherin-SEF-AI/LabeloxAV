"""A VLM judging existing labels in bulk, and an honest account of how much that judgement is worth.

The corpus has 253 human verdicts across 570,379 objects. Precision is unmeasurable, the gate cannot be
tuned against anything, and the error detector holds 298,529 candidates with one confirmed verdict. Human
review does not reach that scale and will not; a machine judge does.

The temptation is to run the judge and report its agreement rate as precision. That is wrong in a way that
is hard to see later, because the number looks like a measurement. A judge has its own error rate, so its
agreement rate is a blend of how good the labels are and how good the judge is, and nothing downstream can
separate them again.

So this module does three things, and the third is the one that matters:

  1. `prereview_batch` judges every object in a batch and records what it said, in `machine_verdict`, which
     is deliberately not the human `review` table.
  2. `judge_agreement` compares the judge against humans wherever both have ruled on the same object, which
     gives the judge's sensitivity and specificity.
  3. `judged_precision` reports the raw agreement rate AND the rate corrected for the judge's measured
     error, and refuses to correct when nobody has adjudicated enough to measure the judge.

The asking is deliberately not the autolabel prompt. Path C asks "what is this?", offering a shortlist and
taking the answer as a proposal. A judge is asked "the label says X, is that right?", which is a different
question with a different failure mode: a model asked to name something will always name something, while a
model asked to confirm can decline. `unsure` is kept as a first-class answer for exactly that reason, and is
never folded into either side, because a judge that abstains on the hard crops and is scored only on the
easy ones reports a precision that flatters itself.
"""

from __future__ import annotations

import json
import uuid as _uuid

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from core.timebase import now_ns
from db.models import MachineVerdict, Object, OntologyClass, Review

log = get_logger("labelops.vlm_review")

JUDGE = "vlm"

# The verdicts a judge may return. `unsure` counts as neither correct nor incorrect and is reported
# separately, so abstention is visible rather than silently improving the score.
VERDICTS = ("correct", "incorrect", "unsure")


def build_judge_prompt(given_class: str, alternatives: list[str]) -> str:
    """Ask whether the existing label is right, rather than asking what the object is.

    The difference is not cosmetic. A model asked to name an object always names one, so its answer carries
    no signal about whether it was sure. Asked to confirm a specific claim it can decline, and the declines
    are exactly the crops worth a person's time.

    The alternatives are offered so that "incorrect" can come with a proposal, which turns a rejected label
    into a correction instead of just a deletion.
    """
    from services.domain import active_pack

    preamble = active_pack().autolabel_profile.vlm_prompt_template
    return (
        f"{preamble}\n"
        f"This crop has been labelled: {given_class}.\n"
        "Decide whether that label is correct for the main object in the crop.\n"
        f"If it is wrong, name the correct class from this list where possible: {alternatives}.\n"
        "Answer 'unsure' when the crop is too small, blurred, occluded or ambiguous to judge. Do not guess: "
        "an unsure answer is more useful than a wrong confident one.\n"
        'Respond with strict JSON only, no prose: '
        '{"verdict": "correct"|"incorrect"|"unsure", "correct_class": "<class or null>", '
        '"confidence": <0.0-1.0>, "reason": "<short>"}'
    )


def parse_judge_reply(data: dict, onto) -> dict:
    """Normalise a judge reply, refusing anything that is not one of the three verdicts.

    A reply that does not parse becomes `unsure` rather than being dropped. Dropping it would silently
    shrink the denominator, which biases the rate upward: the crops a judge garbles are not a random subset.
    """
    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        verdict = "unsure"

    proposed = data.get("correct_class")
    proposed_id = None
    if verdict == "incorrect" and proposed and onto.has_name(str(proposed)):
        proposed_id = onto.by_name(str(proposed)).id

    try:
        conf = float(data.get("confidence"))
        conf = max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        conf = None

    return {"verdict": verdict, "proposed_class_id": proposed_id,
            "confidence": conf, "reason": str(data.get("reason", ""))[:400]}


def _alternatives(onto, class_id: int, limit: int = 12) -> list[str]:
    """Plausible confusions for this class: its L1 siblings first, then the fallback.

    Siblings rather than the whole ontology because the realistic errors are within-superclass (autorickshaw
    against e_auto against tempo), and a list of 180 classes makes the reply worse, not better.
    """
    c = onto.by_id(int(class_id))
    names = [k.name for k in onto.classes if k.l1 == c.l1 and k.name != c.name]
    names.append("object_fallback")
    return list(dict.fromkeys(names))[:limit]


async def _load_crop(db: AsyncSession, obj: Object, margin: float) -> np.ndarray | None:
    """The object's pixels, with context margin. None when the frame is unreadable, which is a skip and not
    a verdict: a judge that never saw the crop has no opinion to record."""
    import cv2

    from core.storage import get_object_store
    from db.models import Frame
    from services.autolabel.paths.path_c_qwen3vl import crop_object

    frame = await db.get(Frame, obj.frame_id)
    if frame is None or not frame.img_uri:
        return None
    try:
        raw = get_object_store().get_bytes(frame.img_uri)
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    except Exception as exc:  # noqa: BLE001
        log.warning("vlm_review.frame_unreadable", frame=str(obj.frame_id), error=str(exc))
        return None
    if img is None:
        return None
    # bbox is xyxy and Python-indexed from 0 here. The SQL side reads bbox[1]..bbox[4] because Postgres
    # arrays are 1-based, which is the same four numbers and an easy off-by-one to import by copying.
    x1, y1, x2, y2 = (float(v) for v in obj.bbox[:4])
    return crop_object(img, (x1, y1, x2, y2), margin)


async def prereview_batch(db: AsyncSession, batch_id: str, *, limit: int | None = None,
                          client=None, model_version: str | None = None,
                          skip_judged: bool = True) -> dict:
    """Judge every object in a flywheel batch and record what the judge said.

    Idempotent by construction: the verdict table is unique on (object, judge, model_version), so re-running
    the same judge updates in place rather than double-counting, and a different model version writes its own
    row so two judges stay comparable on the same crops. `skip_judged` additionally avoids paying for calls
    that would only overwrite an identical verdict, which matters when the batch is 300 crops and the judge
    is a metered API.
    """
    from services.autolabel.ontology import get_ontology
    from services.llm.router import make_vlm_client

    settings = get_settings()
    onto = get_ontology()
    client = client or make_vlm_client(settings)
    provider = getattr(settings.models.vlm, "vision_provider", "ollama")
    model_version = model_version or _model_version_for(settings, provider)

    q = (select(Object)
         .where(Object.provenance["flywheel"]["cycle_id"].astext == batch_id)
         .order_by(Object.object_id))
    if limit:
        q = q.limit(limit)
    objects = list((await db.execute(q)).scalars().all())

    already: set[_uuid.UUID] = set()
    if skip_judged and objects:
        already = set((await db.execute(
            select(MachineVerdict.object_id).where(
                MachineVerdict.judge == JUDGE,
                MachineVerdict.model_version == model_version,
                MachineVerdict.object_id.in_([o.object_id for o in objects])))).scalars().all())

    counts = dict.fromkeys(VERDICTS, 0)
    judged = skipped = unreadable = 0
    margin = settings.models.vlm.crop_margin

    for obj in objects:
        if obj.object_id in already:
            skipped += 1
            continue
        crop = await _load_crop(db, obj, margin)
        if crop is None or crop.size == 0:
            unreadable += 1
            continue

        given = onto.by_id(int(obj.class_id)).name
        alts = _alternatives(onto, int(obj.class_id))
        reply = _ask(client, crop, given, alts)
        parsed = parse_judge_reply(reply, onto)
        counts[parsed["verdict"]] += 1
        judged += 1

        # Upsert on the natural key so a re-run is idempotent rather than additive.
        await db.execute(pg_insert(MachineVerdict).values(
            object_id=obj.object_id, judge=JUDGE, provider=provider, model_version=model_version,
            verdict=parsed["verdict"], proposed_class_id=parsed["proposed_class_id"],
            confidence=parsed["confidence"], agreement=None,
            detail={"given_class": given, "reason": parsed["reason"], "alternatives": alts},
            batch_id=batch_id, ts_ns=now_ns(),
        ).on_conflict_do_update(
            constraint="uq_machine_verdict_object_judge",
            set_={"verdict": parsed["verdict"], "proposed_class_id": parsed["proposed_class_id"],
                  "confidence": parsed["confidence"], "provider": provider,
                  "detail": {"given_class": given, "reason": parsed["reason"], "alternatives": alts},
                  "batch_id": batch_id, "ts_ns": now_ns()}))

    await db.commit()
    out = {"batch_id": batch_id, "objects": len(objects), "judged": judged, "skipped": skipped,
           "unreadable": unreadable, "by_verdict": counts, "judge": JUDGE, "provider": provider,
           "model_version": model_version}
    log.info("vlm_review.prereview", **out)
    return out


def _model_version_for(settings, provider: str) -> str:
    """Which model actually served the verdict, recorded so two judges can be compared later.

    Not cosmetic: the verdict table's uniqueness is on (object, judge, model_version), so getting this wrong
    would make an upgraded model silently overwrite the old judge's opinions instead of standing beside them.
    """
    if provider == "anthropic":
        return settings.anthropic.vision_model
    if provider == "groq":
        return settings.groq.vision_model
    return settings.models.vlm.ollama_tag


def _chat_capable(client):
    """The first thing in this client that can be asked an arbitrary question, or None.

    Three client shapes reach here and all of them can serve a judging prompt, but none of them agrees on
    where the chat lives: OllamaVlmClient carries chat_json directly, the cloud verifiers wrap a client that
    has it, and RoutedVlmClient holds a cloud primary plus a local floor. Walking the shapes here keeps the
    judge from being restricted to whichever one was written first, and keeps the routing policy (cloud
    first, local floor) owned by the router rather than duplicated.
    """
    if hasattr(client, "chat_json"):
        return client
    inner = getattr(client, "client", None)
    if inner is not None and hasattr(inner, "chat_json"):
        return client
    for attr in ("_primary", "ollama"):
        nested = getattr(client, attr, None)
        if nested is not None:
            found = _chat_capable(nested)
            if found is not None:
                return found
    return None


def _ask(client, crop, given_class: str, alternatives: list[str]) -> dict:
    """One judging call.

    A judging prompt is not what VlmClient.verify was built for: verify() takes a shortlist and an attribute
    schema and answers "what is this?". A model asked that always names something, so the reply carries no
    signal about whether it was sure. Asking directly is the whole point, so any client that can carry an
    arbitrary prompt is asked the judging question.

    The fallback exists for a client that cannot, and derives the verdict from whether the returned name
    matches. It is a weaker signal and says so in the recorded reason, so a batch that mixed the two paths
    stays interpretable rather than looking uniform.
    """
    import cv2

    prompt = build_judge_prompt(given_class, alternatives)
    target = _chat_capable(client)
    if target is not None:
        ok, buf = cv2.imencode(".jpg", crop)
        if ok:
            chat = target if hasattr(target, "chat_json") else target.client
            try:
                return chat.chat_json(prompt, model=getattr(target, "model", None),
                                      image_jpeg=buf.tobytes(), temperature=0.0)
            except Exception as exc:  # noqa: BLE001
                log.info("vlm_review.judge_call_failed", error=str(exc)[:160])
                return {"verdict": "unsure", "reason": "judge call failed"}

    res = client.verify(crop, [given_class, *alternatives], {})
    if not res.class_name:
        return {"verdict": "unsure", "reason": "no class returned"}
    same = res.class_name == given_class
    return {"verdict": "correct" if same else "incorrect",
            "correct_class": None if same else res.class_name,
            "confidence": 1.0 if res.confident else 0.5,
            "reason": f"derived from a naming call ({res.provider or 'local'})"}


async def judge_agreement(db: AsyncSession, *, judge: str = JUDGE,
                          model_version: str | None = None, batch_id: str | None = None) -> dict:
    """Measure the judge against humans on the objects where both have ruled.

    This is the step that makes a machine-derived precision worth quoting. Without it the judge's agreement
    rate is just a number, because there is no way to tell a corpus with 15% bad labels judged perfectly
    from a corpus with 5% bad labels judged badly.

    Human ground truth is read from the object state a reviewer moved it to: accepted or submitted means the
    person agreed the label was right, rejected means they did not. Objects a human only reclassified count
    as incorrect, since reclassifying is disagreeing with the original label.
    """
    q = (select(MachineVerdict.verdict, Object.state, func.count(Object.object_id))
         .join(Object, Object.object_id == MachineVerdict.object_id)
         .join(Review, Review.object_id == Object.object_id)
         .where(MachineVerdict.judge == judge,
                Object.state.in_(("accepted", "submitted", "rejected")))
         .group_by(MachineVerdict.verdict, Object.state))
    if model_version:
        q = q.where(MachineVerdict.model_version == model_version)
    if batch_id:
        q = q.where(MachineVerdict.batch_id == batch_id)

    tp = fp = tn = fn = unsure_on_good = unsure_on_bad = 0
    for verdict, state, n in (await db.execute(q)).all():
        human_says_correct = state in ("accepted", "submitted")
        n = int(n)
        if verdict == "unsure":
            if human_says_correct:
                unsure_on_good += n
            else:
                unsure_on_bad += n
        elif verdict == "correct":
            if human_says_correct:
                tp += n
            else:
                fp += n
        else:
            if human_says_correct:
                fn += n
            else:
                tn += n

    # Computed over the crops the judge committed on. The abstentions are reported beside them rather than
    # hidden in the denominator, because "the judge is 96% accurate on the 40% it will answer" is a
    # different claim from "the judge is 96% accurate".
    sens = tp / (tp + fn) if (tp + fn) else None
    spec = tn / (tn + fp) if (tn + fp) else None
    decided = tp + fp + tn + fn
    abstained = unsure_on_good + unsure_on_bad

    return {
        "judge": judge, "model_version": model_version, "batch_id": batch_id,
        "compared_against_human": decided + abstained,
        "decided": decided, "abstained": abstained,
        "sensitivity": round(sens, 4) if sens is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
                      "unsure_on_good": unsure_on_good, "unsure_on_bad": unsure_on_bad},
        "usable": sens is not None and spec is not None and (sens + spec) > 1.0,
    }


async def judged_precision(db: AsyncSession, batch_id: str, *, confidence: float = 0.95,
                           model_version: str | None = None,
                           agreement_batch_id: str | None = None) -> dict:
    """Precision from machine verdicts, with the judge's own error corrected for where it can be measured.

    Returns both numbers on purpose. `raw` is what the judge said, which is what a naive implementation
    would have reported as precision; `corrected` is that rate inverted through the judge's measured
    sensitivity and specificity. When nobody has adjudicated enough for those to exist, `corrected` is null
    and `caveat` says why, rather than quietly falling back to the raw number.

    The judge is measured on this same batch by default, and that default is a statistical claim rather
    than a convenience. A judge's sensitivity and specificity are properties of the population it is
    judging, not constants: one that separates pedestrians from poles cleanly may be much weaker on
    autorickshaw against e_auto. Measuring it on everything it has ever judged and applying that to one
    batch silently assumes the batches are alike. Pass `agreement_batch_id` to measure on a different
    population when that assumption is the one you actually want, for instance a dedicated adjudication set
    covering a wider slice than the batch under test.
    """
    from services.labelops.sampling import rogan_gladen, wilson_interval

    q = (select(OntologyClass.name, MachineVerdict.verdict, func.count(MachineVerdict.verdict_id))
         .join(Object, Object.object_id == MachineVerdict.object_id)
         .join(OntologyClass, OntologyClass.id == Object.class_id)
         .where(MachineVerdict.batch_id == batch_id, MachineVerdict.judge == JUDGE)
         .group_by(OntologyClass.name, MachineVerdict.verdict))
    if model_version:
        q = q.where(MachineVerdict.model_version == model_version)

    per_class: dict[str, dict] = {}
    for name, verdict, n in (await db.execute(q)).all():
        d = per_class.setdefault(name, dict.fromkeys(VERDICTS, 0))
        d[verdict] += int(n)

    total_correct = sum(d["correct"] for d in per_class.values())
    total_decided = sum(d["correct"] + d["incorrect"] for d in per_class.values())
    total_unsure = sum(d["unsure"] for d in per_class.values())

    agreement = await judge_agreement(db, model_version=model_version,
                                      batch_id=agreement_batch_id or batch_id)
    raw = wilson_interval(total_correct, total_decided, confidence)

    corrected = caveat = None
    if agreement["usable"]:
        corrected = rogan_gladen(raw["p"] or 0.0, sensitivity=agreement["sensitivity"],
                                 specificity=agreement["specificity"])
        if corrected is None:
            caveat = ("the judge's sensitivity and specificity sum to 1, meaning its verdicts carry no "
                      "information, so no correction is possible")
    else:
        caveat = (f"no correction applied: only {agreement['decided']} objects have both a machine verdict "
                  f"and a human ruling, so the judge's own error rate is unmeasured. The raw figure is the "
                  f"judge's agreement rate, not the label precision.")

    return {
        "batch_id": batch_id,
        "judged": total_decided, "unsure": total_unsure,
        "raw": raw,
        "corrected": round(corrected, 4) if corrected is not None else None,
        "judge_agreement": agreement,
        "caveat": caveat,
        "per_class": {k: {**v, "raw": wilson_interval(v["correct"], v["correct"] + v["incorrect"], confidence)}
                      for k, v in sorted(per_class.items())},
    }


def verdicts_to_jsonl(rows: list[MachineVerdict]) -> str:
    """Machine verdicts as JSONL, so a batch can be handed to a human adjudicator or another system without
    going through the database."""
    return "\n".join(json.dumps({
        "object_id": str(r.object_id), "verdict": r.verdict, "judge": r.judge,
        "provider": r.provider, "model_version": r.model_version,
        "proposed_class_id": r.proposed_class_id, "confidence": r.confidence,
        "detail": r.detail, "batch_id": r.batch_id,
    }) for r in rows)
