"""A signed quality certificate for a dataset release: per-class precision and recall, with intervals.

346 dataset commits have shipped from this system carrying a datasheet that says what is in the release and
nothing about how good it is. The quality claim was a number in a conversation.

What a buyer purchases is not the point estimate, it is the interval. "Precision 0.87" reads identically
whether it came from 12 gold instances or 12,000, and the difference is the entire value of the claim. So
every rate here carries a Wilson interval and the gold count it was computed from, and a class with too few
gold instances to say anything about is reported as not measured rather than given a number that looks like
the others.

Per-class precision is the piece that did not exist on the auditable path. `evaluate_gold_patches` emits
per-class recall and per-class AP from a sealed, content-addressed gold set and refuses to score an
unregistered model, which is exactly the posture a certificate needs. But precision came only from the
ultralytics val pass, which is opaque, computed against auto-labels rather than gold, and not attributable
to a sealed set. Deriving it from the EvalPatch rows the gold evaluation already writes puts it on the same
footing as recall: same gold set, same run, same audit trail, and a number anybody can recount by hand from
the patch rows.

Signed with an HMAC over the canonical manifest, following services/govern/redaction_proof.py, so a
recipient can detect a certificate that has been edited after issue. Same key, same canonicalisation, same
constant-time verify.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import EvalPatch, GoldSet
from services.labelops.sampling import wilson_interval

log = get_logger("export.certificate")

# Below this many gold instances a per-class rate is too wide to act on. 10 positives puts a 90% rate at
# roughly +/-26% even by Wilson, which is not a quality claim, it is a rumour. Such classes are named as
# unmeasured instead, because a certificate whose weak numbers are indistinguishable from its strong ones is
# worse than one that admits the gap.
MIN_GOLD_PER_CLASS = 10

CERTIFICATE_VERSION = "1.0"


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def gold_membership_fingerprint(object_ids: list) -> str:
    """A hash of which objects the gold set contains, order-independent.

    Sorted before hashing so that two reads of the same sealed set agree regardless of how the database
    returned them; an unsorted hash would flag a reordering as a content change and make the check
    worthless through false alarms.
    """
    joined = ",".join(sorted(str(o) for o in object_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


async def per_class_precision_recall(db: AsyncSession, eval_id: str, *,
                                     confidence: float = 0.95) -> dict:
    """Per-class precision and recall with intervals, recounted from the EvalPatch rows of one evaluation.

    Precision is counted over predicted class (of everything the model called a rider, how much was one) and
    recall over ground-truth class (of the riders that were there, how many were found). Using one class
    axis for both is the classic error, and it silently reports a different quantity than the label suggests:
    a false positive belongs to the class the model *claimed*, a false negative to the class that was
    actually there.
    """
    rows = (await db.execute(
        select(EvalPatch.outcome, EvalPatch.gt_class_id, EvalPatch.pred_class_id,
               func.count(EvalPatch.patch_id))
        .where(EvalPatch.eval_id == eval_id)
        .group_by(EvalPatch.outcome, EvalPatch.gt_class_id, EvalPatch.pred_class_id))).all()

    tally: dict[int, dict[str, int]] = {}

    def _slot(cid: int | None) -> dict[str, int] | None:
        if cid is None:
            return None
        return tally.setdefault(int(cid), {"tp": 0, "fp": 0, "fn": 0})

    for outcome, gt_cid, pred_cid, n in rows:
        n = int(n)
        if outcome == "tp":
            # A true positive is the same class on both axes by definition of the class-aware match, and it
            # counts toward the numerator of both rates.
            slot = _slot(pred_cid if pred_cid is not None else gt_cid)
            if slot is not None:
                slot["tp"] += n
        elif outcome == "fp":
            slot = _slot(pred_cid)      # charged to the class the model claimed
            if slot is not None:
                slot["fp"] += n
        elif outcome == "fn":
            slot = _slot(gt_cid)        # charged to the class that was actually present
            if slot is not None:
                slot["fn"] += n

    from services.autolabel.ontology import get_ontology

    onto = get_ontology()

    def _name(cid: int) -> str:
        try:
            return onto.by_id(int(cid)).name
        except Exception:  # noqa: BLE001
            return str(cid)

    out: dict[str, dict] = {}
    for cid, t in tally.items():
        predicted = t["tp"] + t["fp"]
        present = t["tp"] + t["fn"]
        measured = present >= MIN_GOLD_PER_CLASS
        out[_name(cid)] = {
            "gold_instances": present,
            "predicted_instances": predicted,
            "tp": t["tp"], "fp": t["fp"], "fn": t["fn"],
            "precision": wilson_interval(t["tp"], predicted, confidence) if predicted else None,
            "recall": wilson_interval(t["tp"], present, confidence) if present else None,
            "measured": measured,
            # Named, not implied. A reader scanning the table has to be able to tell a real 0.9 from one
            # resting on four instances without doing the arithmetic themselves.
            "note": None if measured else (
                f"not measured: only {present} gold instances, below the {MIN_GOLD_PER_CLASS} needed for "
                f"an interval narrow enough to act on"),
        }
    return dict(sorted(out.items()))


async def build_certificate(db: AsyncSession, *, commit_id: str, eval_id: str, gold_id: str,
                            model_version: str, key: str, confidence: float = 0.95,
                            overall: dict | None = None) -> dict:
    """Assemble and sign the certificate for a release.

    Refuses rather than issues when the gold set is missing, because a certificate that cannot name what it
    was measured against is exactly the artifact this exists to replace. The same reasoning as
    `evaluate_gold_patches` refusing to score an unregistered model: an unattributable number is worse than
    no number, since it will be quoted anyway.
    """
    gold = await db.get(GoldSet, gold_id)
    if gold is None:
        return {"error": "gold set not found", "gold_id": gold_id}
    gold_size = len(gold.object_ids or [])
    if not gold_size:
        return {"error": "gold set is empty, so nothing was measured", "gold_id": gold_id}

    per_class = await per_class_precision_recall(db, eval_id, confidence=confidence)
    if not per_class:
        # An eval_id with no patches is a typo, a deleted run, or an evaluation that never scored anything.
        # Without this the certificate issues anyway: zero classes, an overall precision of "not measured",
        # and a valid signature over the whole thing. That artifact is worse than no certificate, because it
        # is indistinguishable at a glance from one attesting a clean release and it verifies.
        return {"error": f"evaluation '{eval_id}' has no scored patches, so nothing was measured",
                "eval_id": str(eval_id), "gold_id": gold_id}

    measured = {k: v for k, v in per_class.items() if v["measured"]}
    unmeasured = sorted(k for k, v in per_class.items() if not v["measured"])

    total_tp = sum(v["tp"] for v in per_class.values())
    total_fp = sum(v["fp"] for v in per_class.values())
    total_fn = sum(v["fn"] for v in per_class.values())

    manifest = {
        "certificate_version": CERTIFICATE_VERSION,
        "commit_id": commit_id,
        "model_version": model_version,
        "gold_id": gold_id,
        # A hash of the sealed membership, recomputed here rather than trusted from the row. gold_id is
        # already content-addressed, so in a healthy system this is redundant, and that is the point: if a
        # gold set is ever resealed with different contents under the same id, a certificate issued before
        # and one issued after will disagree here and the discrepancy is discoverable. Without it, the only
        # evidence would be the id, which is exactly the thing that failed to change.
        "gold_fingerprint": gold_membership_fingerprint(gold.object_ids or []),
        "gold_objects": gold_size,
        "eval_id": str(eval_id),
        "confidence": confidence,
        "overall": {
            "precision": wilson_interval(total_tp, total_tp + total_fp, confidence),
            "recall": wilson_interval(total_tp, total_tp + total_fn, confidence),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        },
        "per_class": per_class,
        "classes_measured": sorted(measured),
        "classes_not_measured": unmeasured,
        # Stated on the face of the certificate rather than left to be inferred from the per-class table,
        # because "we support 42 classes" beside a certificate measuring 6 of them is the misreading this
        # is meant to prevent.
        "coverage_note": (f"{len(measured)} of {len(per_class)} classes have at least "
                          f"{MIN_GOLD_PER_CLASS} gold instances; the rest are reported as not measured"),
    }
    if overall:
        manifest["harness_overall"] = overall

    signature = hmac.new(key.encode("utf-8"), _canonical(manifest), hashlib.sha256).hexdigest()
    log.info("export.certificate", commit_id=commit_id, gold_id=gold_id,
             measured=len(measured), unmeasured=len(unmeasured))
    return {"manifest": manifest, "signature": signature}


def verify_certificate(manifest: dict, signature: str, key: str) -> bool:
    """Verify a certificate. Any edit to the manifest (a raised precision, a swapped gold id, a class moved
    out of the not-measured list) invalidates the signature. Constant-time comparison."""
    expected = hmac.new(key.encode("utf-8"), _canonical(manifest), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")


def render_certificate_markdown(cert: dict) -> str:
    """The certificate as a document a buyer reads, rather than a payload a system parses.

    Deliberately leads with what was not measured. A quality document that buries its gaps below its
    headline numbers is doing the opposite of its job.
    """
    m = cert["manifest"]
    lines = [
        f"# Label quality certificate: {m['commit_id']}",
        "",
        f"- Model: `{m['model_version']}`",
        f"- Gold set: `{m['gold_id']}` ({m['gold_objects']} objects)",
        f"- Confidence level: {int(m['confidence'] * 100)}%",
        f"- Certificate version: {m['certificate_version']}",
        f"- Signature: `{cert['signature'][:16]}...`",
        "",
        "## Scope of this measurement",
        "",
        m["coverage_note"] + ".",
        "",
    ]
    if m["classes_not_measured"]:
        lines += [
            "**Not measured**: " + ", ".join(f"`{c}`" for c in m["classes_not_measured"]) + ".",
            "",
            "These classes carry too few gold instances for an interval narrow enough to act on. No claim "
            "is made about them.",
            "",
        ]
    ov = m["overall"]
    lines += [
        "## Overall",
        "",
        f"- Precision: {_fmt(ov['precision'])}",
        f"- Recall: {_fmt(ov['recall'])}",
        f"- Counts: {ov['tp']} true positives, {ov['fp']} false positives, {ov['fn']} false negatives",
        "",
        "## Per class",
        "",
        "| Class | Gold | Precision | Recall |",
        "| --- | ---: | --- | --- |",
    ]
    for name, v in m["per_class"].items():
        if v["measured"]:
            lines.append(f"| `{name}` | {v['gold_instances']} | {_fmt(v['precision'])} | {_fmt(v['recall'])} |")
        else:
            lines.append(f"| `{name}` | {v['gold_instances']} | not measured | not measured |")
    lines += [
        "",
        "Intervals are Wilson score intervals, which hold their coverage at small sample sizes where the "
        "normal approximation does not and can report bounds outside [0, 1].",
        "",
    ]
    return "\n".join(lines)


def _fmt(ci: dict | None) -> str:
    if not ci or ci.get("p") is None:
        return "not measured"
    return f"{ci['p']:.3f} ({ci['lo']:.3f} to {ci['hi']:.3f}, n={ci['n']})"
