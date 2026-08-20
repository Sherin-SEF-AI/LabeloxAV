"""Milestone D: the unified DPDPA pre-sale compliance gate. Face, plate, and now human speech are ONE
fail-closed gate in the export path: a clip with any un-redacted detected face, plate, or personal speech
segment is refused, not warned. The gate is fail-closed by construction: a frame with no anonymization audit
is treated as un-redacted (refuse), never assumed clean.

The speech detector is a runtime seam (a VAD/speech model), so detection is not faked here; the gate logic
that refuses on un-redacted speech is real and tested.

Gate v2 adds the question the row-existence check could not ask: whether a blur covered anything. See
services/anonymize/coverage.py for the predicate and why it is annotation-relative rather than a count.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.logging import get_logger

log = get_logger("dpdpa")

# Postgres caps a statement at 65535 parameters; services/export/dataset.py chunks its IN-lists at the same
# size for the same reason.
_ID_CHUNK = 10_000

# A whole-slice refusal must not put 40,000 frame records into a 422 body. The count is always exact; the
# per-frame detail is a sample the operator can act on.
_MAX_BLOCKER_FRAMES = 25


def evaluate_dpdpa(export_frame_ids: set, audited_frame_ids: set, speech_segments: list[dict],
                   regime: str = "DPDPA", *, coverage_gaps: Sequence[dict] | None = None,
                   mode: str = "off") -> dict:
    """Fail-closed privacy verdict for the active legal regime (`regime`, from the pack; AV: DPDPA). Blocks if
    any frame in scope was never anonymized (no PiiAudit row means an un-redacted face or plate, refuse not
    warn) or any personal speech segment is not redacted. speech_segments: [{is_personal, redacted}]. Returns
    {pass, blockers, warnings, regime}.

    `coverage_gaps` and `mode` are keyword-only with inert defaults, so every existing three- and four-arg
    caller keeps its exact verdict and the v2 behaviour is something a caller opts into.
    """
    blockers = []
    warnings: list[dict] = []
    unaudited = export_frame_ids - audited_frame_ids
    if unaudited:
        blockers.append({"kind": "unredacted_visual_pii", "count": len(unaudited),
                         "detail": "frames with no face/plate anonymization audit"})
    unredacted_speech = [s for s in speech_segments
                         if s.get("is_personal", True) and not s.get("redacted", False)]
    if unredacted_speech:
        blockers.append({"kind": "unredacted_speech", "count": len(unredacted_speech),
                         "detail": "personal speech segments not masked"})

    if coverage_gaps and mode in ("advisory", "enforcing"):
        finding = {
            "kind": "unverified_blur_coverage",
            "count": len(coverage_gaps),
            "detail": "annotated people or vehicles with no redaction region covering them",
            "frames": list(coverage_gaps[:_MAX_BLOCKER_FRAMES]),
            "truncated": len(coverage_gaps) > _MAX_BLOCKER_FRAMES,
        }
        (blockers if mode == "enforcing" else warnings).append(finding)

    return {"pass": len(blockers) == 0, "blockers": blockers, "warnings": warnings, "regime": regime}


async def blur_coverage_gaps(db, frame_ids: Sequence, *, targets: Sequence[str] = ("face", "plate"),
                             ) -> list[dict]:
    """Frames whose own annotations demand a redaction the audit does not evidence.

    Two chunked queries for the whole export regardless of its size, never one per frame: export_dataset
    hands this whole slices, and a per-frame loop here would be an N+1 inside the gate that blocks delivery.

    Deliberately re-queries Object rather than reading the export's own records. An export delivers the
    IMAGE, so a pedestrian annotated on a frame in state "review" still has an unblurred face in the
    delivered JPEG even though that row is filtered out of the label set.
    """
    from sqlalchemy import select

    from db.models import Frame, Object, OntologyClass, PiiAudit
    from services.anonymize.backfill import ROI_SUFFIX
    from services.anonymize.coverage import REDACTION_L1, demands, summarize, unmet

    ids = [f if not isinstance(f, str) else f for f in frame_ids]
    if not ids:
        return []

    by_frame: dict[str, list] = {}
    dims: dict[str, tuple[int, int]] = {}
    audits: dict[str, tuple[list, str]] = {}
    for i in range(0, len(ids), _ID_CHUNK):
        chunk = ids[i:i + _ID_CHUNK]
        rows = (await db.execute(
            select(Object.frame_id, Object.bbox, OntologyClass.l0, OntologyClass.l1, OntologyClass.name,
                   Frame.width, Frame.height)
            .join(Frame, Frame.frame_id == Object.frame_id)
            .join(OntologyClass, OntologyClass.id == Object.class_id)
            .where(Object.frame_id.in_(chunk),
                   OntologyClass.l0 == "object",
                   OntologyClass.l1.in_(tuple(REDACTION_L1))))).all()
        for fid, bbox, l0, l1, name, w, h in rows:
            by_frame.setdefault(str(fid), []).append((bbox, l0, l1, name))
            dims[str(fid)] = (int(w or 0), int(h or 0))
        for fid, regions, method in (await db.execute(
                select(PiiAudit.frame_id, PiiAudit.regions, PiiAudit.method_version)
                .where(PiiAudit.frame_id.in_(chunk)))).all():
            audits[str(fid)] = (regions or [], method or "")

    gaps: list[dict] = []
    for fid, rows in by_frame.items():
        w, h = dims.get(fid, (0, 0))
        if w <= 0 or h <= 0:
            continue
        need = demands(rows, w, h, targets=targets)
        if not need:
            continue
        regions, method = audits.get(fid, ([], ""))
        missing = unmet(need, regions)
        if missing:
            # `examined` separates "a detector looked inside this person's box and found nothing" from
            # "nothing ever looked". The second is the 82.4% population and the one a re-check run fixes.
            gaps.append({"frame_id": fid, "examined": method.endswith(ROI_SUFFIX), **summarize(missing)})
    return gaps


async def dpdpa_export_gate(session_id, export_frame_ids: list) -> dict:
    """Load the anonymization audits and speech segments for the frames in scope and run the fail-closed
    DPDPA verdict. Called by the export path before any clip is written or delivered."""
    from sqlalchemy import select

    from core.config import get_settings
    from db.models import PiiAudit, SpeechSegment
    from db.session import get_sessionmaker
    from services.domain import legal_regime

    mode = get_settings().pii.coverage_gate
    fids = {str(f) for f in export_frame_ids}
    async with get_sessionmaker()() as db:
        audited = {str(f) for f in (await db.execute(
            select(PiiAudit.frame_id).where(PiiAudit.session_id == session_id))).scalars().all()}
        speech = [{"is_personal": s.is_personal, "redacted": s.redacted} for s in (await db.execute(
            select(SpeechSegment).where(SpeechSegment.session_id == session_id))).scalars().all()]
        gaps = await blur_coverage_gaps(db, list(export_frame_ids)) if mode != "off" else []
    verdict = evaluate_dpdpa(fids, audited, speech, regime=legal_regime(),
                             coverage_gaps=gaps, mode=mode)
    log.info("dpdpa.gate", session=str(session_id), frames=len(fids), passed=verdict["pass"],
             blockers=len(verdict["blockers"]), coverage_gaps=len(gaps), coverage_mode=mode)
    return verdict
