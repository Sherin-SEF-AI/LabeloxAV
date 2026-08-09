"""A second opinion on the classes this corpus actually confuses, asked in a form that can be counted.

The Review trail holds 296 class corrections. 250 of them involve `e_auto` and 206 involve `motorcycle`, and
the single pair e_auto <-> motorcycle accounts for 195, which is 66% of every class correction ever made
here. Nothing else is close: the third-largest contributor is `rider` at 35.

Two properties of that pair decide the design.

It is bidirectional. 125 corrections went e_auto -> motorcycle and 70 went motorcycle -> e_auto. A
one-directional confusion is a bias and can be fixed by moving a threshold. A symmetric one is an ambiguous
boundary, which means the annotators themselves are not applying a stable rule, and a critic that must choose
between the two will inherit exactly that ambiguity while reporting it as a decision. So `uncertain` is a
first-class verdict here rather than a fallback. Forcing a binary choice on a genuinely ambiguous crop does
not produce information, it produces confident noise, and the corpus already has 195 examples of humans
disagreeing about the same distinction.

And the evidence is thin. 296 corrections is not many, so the confusion set is derived from the trail with a
minimum count and reports what backs each pair, rather than being a constant somebody typed. When the
corrections change, the set changes; nothing here needs editing for a new confusion to be covered.

The verdict is schema-constrained for the same reason Path C's class choice now is: the entire worth of a
categorical judgement is that it lands in a known set. An unparseable "probably not, though it could be" is
not a soft answer, it is a discarded call. Constraining it also means the sweep's counts are counts of
verdicts, not counts of successful parses.

Where the output goes matters as much as what it says. Confident disagreements become `error_candidate` rows,
which already carry a proposed fix, a status, and a decided_by/decided_at pair that makes them calibratable:
confirmed over confirmed-plus-dismissed is this critic's precision, measured the same way the other detectors
are. Uncertain verdicts are not queued as errors, because "the model and the label disagree about a boundary
humans also disagree about" is not a suspected error. They are returned as gold-set candidates, which is what
an ambiguous boundary actually needs: adjudication once, written down, rather than re-litigation per crop.
"""

from __future__ import annotations

import uuid
from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ErrorCandidate, Frame, Object, Review
from services.autolabel.ontology import get_ontology

log = get_logger("agent.confusion_critic")

KIND = "vlm_confusion"          # error_candidate.kind is String(24)

# A pair seen fewer times than this is not a confusion, it is an incident. With 296 corpus-wide corrections a
# single occurrence is noise, and admitting it to the candidate set would have the critic hunting a class
# boundary that one person crossed once.
MIN_PAIR_COUNT = 3

# `other` exists because the first good sweep produced this reason, filed against a suggestion of
# `motorcycle`: "The image shows a large billboard or sign with text and graphics, not a vehicle of any
# kind." The neighbourhood is built from classes this one gets confused with, so it can only offer vehicles
# and people, and a crop that is none of those has nowhere to go. Without an escape the schema converts "this
# label is nonsense" into "this label is slightly the wrong vehicle", which is a worse answer than the model
# actually gave. It is also the more valuable finding: a billboard labelled e_auto is the traffic-sign
# contamination again, in a different class.
VERDICTS = ("agree", "disagree", "uncertain", "other")

# Below this a crop cannot settle an e_auto/motorcycle distinction, and a judge asked anyway will still
# answer. The first sweep judged crops of 36x25 and 40x44 pixels and filed confident disagreements about
# them. The other crop-consuming paths here already draw this line (24px in the relabel agent, 16px in the
# failure embedder); a judgement between two similar vehicle classes needs more than a detection does.
MIN_CROP_PX = 40


def critic_schema(candidates: list[str]) -> dict:
    """The verdict schema a constrained backend enforces.

    `suggested_class` is an enum over the confusion neighbourhood rather than the ontology, because the
    question being asked is "is it this or one of the things it gets mistaken for", and offering 186 classes
    invites an answer to a question nobody asked.
    """
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": list(VERDICTS)},
            "suggested_class": {"type": "string", "enum": list(candidates)},
            "reason": {"type": "string"},
        },
        # `reason` is required, `suggested_class` is not. A queued candidate whose argument is missing is
        # one a reviewer can only agree or disagree with on vibes, and it is unusable for calibrating the
        # critic afterwards. The first sweep left it optional and got it back empty in 28 of 28 rows, which
        # is what an optional field means to a model being scored on nothing else. Requiring it also makes
        # the verdict cheaper to audit than to re-derive.
        #
        # `suggested_class` stays optional because it is meaningless on `agree` and misleading on
        # `uncertain`, so demanding it would force the model to name a replacement it does not believe in.
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }


async def confusion_pairs(db: AsyncSession, *, min_count: int = MIN_PAIR_COUNT) -> list[dict]:
    """Class pairs the Review trail says get mixed up, most confused first.

    Derived rather than declared, so the sweep follows the corpus. Directional counts are kept separate
    because the direction is the diagnosis: asymmetric is a bias, symmetric is an ambiguous boundary.
    """
    onto = get_ontology()
    rows = (await db.execute(select(Review.before, Review.after))).all()

    directed: Counter = Counter()
    for before, after in rows:
        b = (before or {}).get("class_id")
        a = (after or {}).get("class_id")
        if b is None or a is None or b == a:
            continue
        directed[(int(b), int(a))] += 1

    seen: set[tuple[int, int]] = set()
    out: list[dict] = []
    for (b, a), _n in directed.most_common():
        key = (min(b, a), max(b, a))
        if key in seen:
            continue
        forward = directed[(b, a)]
        reverse = directed[(a, b)]
        total = forward + reverse
        if total < min_count:
            continue
        try:
            names = (onto.by_id(key[0]).name, onto.by_id(key[1]).name)
        except Exception:  # noqa: BLE001 - a class deleted since the correction was made
            continue
        seen.add(key)
        out.append({
            "class_ids": list(key),
            "classes": list(names),
            "count": total,
            "forward": forward,
            "reverse": reverse,
            # A pair corrected both ways is a boundary humans do not agree on. Naming it here is what stops
            # a caller reading a symmetric pair as a model defect.
            "symmetric": bool(forward and reverse),
        })
    out.sort(key=lambda p: -p["count"])
    return out


async def confusion_neighbourhood(db: AsyncSession, focus: str, *,
                                  min_count: int = MIN_PAIR_COUNT) -> dict:
    """The classes `focus` is confused with, and how much evidence says so."""
    onto = get_ontology()
    if not onto.has_name(focus):
        return {"focus": focus, "candidates": [], "detail": f"{focus} is not in the ontology"}
    fid = onto.by_name(focus).id

    pairs = [p for p in await confusion_pairs(db, min_count=min_count) if fid in p["class_ids"]]
    others = []
    for p in pairs:
        other = p["class_ids"][0] if p["class_ids"][1] == fid else p["class_ids"][1]
        others.append({"class": onto.by_id(other).name, "count": p["count"], "symmetric": p["symmetric"]})

    total = sum(o["count"] for o in others)
    return {
        "focus": focus,
        # The focus class itself is always offered, because "the label is right" has to be sayable.
        "candidates": [focus] + [o["class"] for o in others],
        "pairs": others,
        "evidence": total,
        "detail": (f"{focus} is confused with {len(others)} classes across {total} corrections"
                   if others else
                   f"no class pair involving {focus} reaches {min_count} corrections, so there is nothing "
                   "here a critic could check against"),
    }


def _prompt(current: str, candidates: list[str]) -> str:
    """The judge's question, which is not the verifier's question.

    verify() asks "what is this?". This asks "the label says X, is that right?", and the difference matters:
    a model asked to name an object from scratch will often name it correctly and still not notice that the
    existing label disagrees, because it was never shown the claim it is meant to be checking.
    """
    return (
        f"This object is currently labelled `{current}`.\n"
        f"It is commonly confused with: {', '.join(c for c in candidates if c != current)}.\n"
        "Judge the existing label against the image.\n"
        "Answer `agree` if the label is right, `disagree` if one of the other classes is clearly right, "
        "`other` if you can see the object clearly and it is none of the listed classes, and "
        "`uncertain` if the crop is too blurry, too small or too occluded to tell. "
        "The difference between `other` and `uncertain` is whether you can see it: a clear photograph of a "
        "building is `other`, an unreadable smear is `uncertain`. Prefer `uncertain` over guessing: these "
        "classes are ones people disagree about, so a confident wrong answer costs more than an abstention. "
        "On `disagree`, give the class you believe is correct in `suggested_class`. "
        "In `reason`, cite what you can actually see in the image that decides it (wheel count, cabin, "
        "body shape, whether it is carrying passengers). A reason that does not refer to the image is not "
        "a reason."
    )


async def run_confusion_sweep(db: AsyncSession, *, focus: str = "e_auto", limit: int = 200,
                              min_conf: float | None = None, client: object = None,
                              min_count: int = MIN_PAIR_COUNT) -> dict:
    """Judge a sample of `focus`-labelled objects against the classes they get confused with.

    Reads crops through the same object store path everything else uses, and writes confident disagreements
    to `error_candidate` so they enter the existing triage queue rather than a private one.
    """
    import cv2

    from core.storage import get_object_store
    from services.autolabel.paths.path_c_qwen3vl import crop_object
    from services.llm.router import local_vlm_client
    from services.recall.backends import load_image_bgr

    onto = get_ontology()
    hood = await confusion_neighbourhood(db, focus, min_count=min_count)
    if len(hood["candidates"]) < 2:
        return {"focus": focus, "judged": 0, **hood}

    schema = critic_schema(hood["candidates"])
    fid = onto.by_name(focus).id

    q = (select(Object, Frame.img_uri)
         .join(Frame, Frame.frame_id == Object.frame_id)
         .where(Object.class_id == fid, Object.source != "human"))
    if min_conf is not None:
        q = q.where(Object.conf <= min_conf)
    # Random rather than ordered: a sweep over the first N by id measures whichever session was ingested
    # first, and this corpus was ingested one vehicle at a time.
    rows = (await db.execute(q.order_by(func.random()).limit(limit))).all()

    # Duck-typed on chat_json rather than typed as VlmClient: that Protocol describes verify()'s question
    # ("what is this?") and the judge asks a different one ("the label says X, is that right?").
    vlm: object = client or local_vlm_client()
    # The judge model, not the proposer. This is not a preference, it is the difference between a working
    # critic and a broken one, and the corpus had already recorded why.
    #
    # `local_vlm_client()` serves models.vlm.ollama_tag, which is qwen2.5vl:7b. The config note beside
    # judge_tag says that model "returned `unsure` zero times in 547 crops", and a first sweep with it
    # reproduced exactly that: 40 crops judged, 40 disagreements, 0 agreements, 0 abstentions, every `reason`
    # empty. Probing it directly, it returned the same verdict and the same suggested class for a 248x830
    # crop and a 32x29 one, which is a model answering from the prompt without conditioning on the image.
    # Constraining the schema did not cause it and could not have fixed it: the unconstrained call gave the
    # same answer.
    judge_model = get_settings().models.vlm.judge_tag or get_settings().models.vlm.ollama_tag
    store = get_object_store()
    counts: Counter = Counter()
    queued: list[str] = []
    uncertain: list[str] = []
    other: list[str] = []
    prompt = _prompt(focus, hood["candidates"])

    for obj, img_uri in rows:
        try:
            img = load_image_bgr(store, img_uri)
            crop = crop_object(img, tuple(obj.bbox), 0.15)
            if min(crop.shape[:2]) < MIN_CROP_PX:
                # Skipped rather than judged. The model will answer a 36x25 crop confidently, and that
                # answer is noise wearing a verdict's clothes.
                counts["too_small"] += 1
                continue
            ok, buf = cv2.imencode(".jpg", crop)
            if not ok:
                counts["unreadable"] += 1
                continue
            data = vlm.chat_json(  # type: ignore[attr-defined]
                prompt, image_jpeg=buf.tobytes(), schema=schema, model=judge_model)
        except Exception as exc:  # noqa: BLE001 - one bad crop must not end a sweep
            counts["error"] += 1
            log.warning("confusion_critic.crop_failed", object_id=str(obj.object_id), error=str(exc)[:160])
            continue

        verdict = str(data.get("verdict") or "")
        if verdict not in VERDICTS:
            # Only reachable on an unconstrained backend. Counted rather than coerced: silently folding it
            # into `uncertain` would inflate the abstention rate with parse failures.
            counts["unparsed"] += 1
            continue
        counts[verdict] += 1

        if verdict == "uncertain":
            uncertain.append(str(obj.object_id))
            continue

        if verdict == "other":
            # Filed with no proposed label, because the whole point is that the critic cannot name one from
            # this set. A reviewer still gets something to act on, and "labelled e_auto, is not a vehicle"
            # is a more useful row than a wrong vehicle would have been.
            db.add(ErrorCandidate(
                candidate_id=uuid.uuid4(), object_id=obj.object_id, kind=KIND, score=1.0,
                proposed_label=None,
                detail={"critic": "vlm_confusion", "current": focus, "suggested": None,
                        "reason": str(data.get("reason") or "")[:400],
                        "candidates": hood["candidates"],
                        "note": "none of the confusion classes fit; the label is wrong in some other way",
                        "provider": getattr(vlm, "provider_name", getattr(vlm, "model", "unknown")),
                        "judge_model": judge_model}))
            other.append(str(obj.object_id))
            continue

        if verdict != "disagree":
            continue

        suggested = str(data.get("suggested_class") or "")
        if not onto.has_name(suggested) or suggested == focus:
            # A disagreement that cannot name an alternative is not actionable, and filing it would put a row
            # in the queue that a reviewer can only dismiss.
            counts["disagree_unnamed"] += 1
            continue

        db.add(ErrorCandidate(
            candidate_id=uuid.uuid4(), object_id=obj.object_id, kind=KIND,
            score=1.0,
            proposed_label={"class_id": onto.by_name(suggested).id, "class_name": suggested},
            detail={"critic": "vlm_confusion", "current": focus, "suggested": suggested,
                    "reason": str(data.get("reason") or "")[:400],
                    "candidates": hood["candidates"],
                    "provider": getattr(vlm, "provider_name", getattr(vlm, "model", "unknown")),
                    "judge_model": judge_model},
        ))
        queued.append(str(obj.object_id))

    await db.commit()

    judged = sum(counts[v] for v in VERDICTS)
    log.info("confusion_critic.swept", focus=focus, judged=judged, queued=len(queued),
             uncertain=len(uncertain), other=len(other))
    return {
        "focus": focus,
        "candidates": hood["candidates"],
        "evidence": hood["evidence"],
        "judge_model": judge_model,
        "sampled": len(rows),
        "judged": judged,
        "counts": dict(counts),
        "queued_as_errors": len(queued) + len(other),
        "error_candidate_ids": queued,
        # Wrong, and not in a way this confusion set can describe. Worth separating: a class boundary problem
        # is fixed by adjudicating the boundary, and this is not that.
        "out_of_set": other,
        # Not errors. An ambiguous boundary needs adjudicating once and writing down, not re-litigating per
        # crop, which is exactly what a gold set is for.
        "gold_candidates": uncertain,
        "detail": (f"judged {judged} of {len(rows)} sampled {focus} crops against "
                   f"{len(hood['candidates'])} candidate classes: {counts['agree']} agree, "
                   f"{counts['disagree']} disagree, {counts['uncertain']} uncertain, "
                   f"{counts['other']} wrong but outside the set"),
    }
