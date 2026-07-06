"""Governance persistence for M18: roll per-frame PiiAudit rows into a signed release redaction proof, read and
set consent/retention records, and trace the governance lineage of a subject."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from db.models import AuditDecision, ConsentRecord, PiiAudit, RedactionProof
from services.govern.redaction_proof import build_proof, redaction_coverage


async def build_release_proof(db: AsyncSession, release_commit: str, frame_ids: list[str],
                              method_version: str = "") -> dict:
    """Compute PII-gate coverage over a release's frames, build and sign the proof, and persist it. frame_ids
    is the release's frame set (resolved by the caller from the DatasetCommit); a frame with a PiiAudit row is
    covered."""
    fid_uuids = [uuid.UUID(f) if not isinstance(f, uuid.UUID) else f for f in frame_ids]
    audited = set()
    if fid_uuids:
        rows = (await db.execute(
            select(PiiAudit.frame_id).where(PiiAudit.frame_id.in_(fid_uuids)))).scalars().all()
        audited = {str(r) for r in rows}
    cov = redaction_coverage([str(f) for f in fid_uuids], audited)
    cfg = get_settings()
    proof = build_proof(release_commit, cov, cfg.phase4.govern.attestation_key, method_version,
                        cfg.phase4.govern.redaction_coverage_floor)
    row = RedactionProof(
        release_commit=release_commit, n_frames=cov["n_frames"], n_covered=cov["n_covered"],
        coverage=cov["coverage"], verdict=proof["verdict"], signature=proof["signature"],
        uncovered=cov["uncovered"][:500],
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return {"proof_id": str(row.proof_id), **proof}


async def set_consent(db: AsyncSession, session_id: uuid.UUID, consent_status: str,
                      legal_basis: str | None = None, retention_until=None) -> dict:
    """Upsert the consent/retention record for a session."""
    row = await db.get(ConsentRecord, session_id)
    if row is None:
        row = ConsentRecord(session_id=session_id)
        db.add(row)
    row.consent_status = consent_status
    row.legal_basis = legal_basis
    row.retention_until = retention_until
    await db.commit()
    return {"session_id": str(session_id), "consent_status": consent_status, "legal_basis": legal_basis,
            "retention_until": retention_until.isoformat() if retention_until else None}


async def governance_lineage(db: AsyncSession, subject: str, limit: int = 100) -> dict:
    """Trace the governance events recorded against a subject (a model version, release, or object id): the
    audit decisions and, when the subject is a release, its redaction proof. One call for buyer diligence."""
    audits = (await db.execute(
        select(AuditDecision).where(AuditDecision.subject == subject)
        .order_by(AuditDecision.created_at.desc()).limit(min(max(limit, 1), 500)))).scalars().all()
    proofs = (await db.execute(
        select(RedactionProof).where(RedactionProof.release_commit == subject)
        .order_by(RedactionProof.created_at.desc()).limit(20))).scalars().all()
    return {
        "subject": subject,
        "audit_decisions": [
            {"actor": a.actor, "decision": a.decision, "rationale": a.rationale,
             "created_at": a.created_at.isoformat() if a.created_at else None} for a in audits],
        "redaction_proofs": [
            {"proof_id": str(p.proof_id), "verdict": p.verdict, "coverage": p.coverage,
             "n_frames": p.n_frames, "created_at": p.created_at.isoformat() if p.created_at else None}
            for p in proofs],
    }
