"""Milestone I: attribute validation audit. The write paths now gate attributes through
Ontology.validate_attrs (create, single + bulk review, 3D edit), but two things leave invalid attributes in
the corpus anyway: labels written before validation existed, and a class change (relabel_track does not
re-validate) that makes a previously-valid attribute not-applicable to the new class. This scans existing
objects and reports every violation against its current class, so the bad labels surface instead of shipping
silently. The scan is pure over (class_id, attrs) records, so it is tested against the real ontology without
infra.
"""

from __future__ import annotations

from core.logging import get_logger

log = get_logger("attr_audit")


def audit_attrs(records: list[dict], onto) -> list[dict]:
    """records: [{object_id, class_id, attrs}]. Returns one entry per object whose attrs fail validate_attrs
    against its current class, worst (most errors) first."""
    violations = []
    for r in records:
        errors = onto.validate_attrs(r.get("attrs") or {}, r.get("class_id"))
        if errors:
            violations.append({"object_id": r["object_id"], "class_id": r.get("class_id"), "errors": errors})
    violations.sort(key=lambda v: len(v["errors"]), reverse=True)
    return violations


async def session_attr_audit(session_id) -> dict:
    """Scan a session's objects and report attribute violations against each object's current class."""
    from sqlalchemy import select

    from db.models import Frame, Object
    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    async with get_sessionmaker()() as db:
        rows = (await db.execute(
            select(Object.object_id, Object.class_id, Object.attrs)
            .join(Frame, Object.frame_id == Frame.frame_id)
            .where(Frame.session_id == session_id, Object.attrs != {}))).all()
    records = [{"object_id": str(oid), "class_id": cid, "attrs": attrs or {}} for oid, cid, attrs in rows]
    violations = audit_attrs(records, get_ontology())
    log.info("attr_audit.session", session=str(session_id), scanned=len(records), violations=len(violations))
    return {"session_id": str(session_id), "scanned": len(records), "violations": len(violations),
            "items": violations}
