"""Consent + retention enforcement (M18). DPDPA requires that data is processed only under a lawful basis and
purged when its retention window expires. This enforces both at the spine: an export is refused unless the
session's consent is granted, and a session past its retention deadline is flagged for purge rather than
silently kept. Pure over the consent record and a supplied 'now' (so retention expiry is deterministic and
testable); the DB reads/writes live in run.py."""

from __future__ import annotations

from datetime import datetime


def export_consent_gate(consent_status: str) -> dict:
    """May a session's data be exported/sold? Only an explicit 'granted' consent allows it; 'denied' and
    'unknown' both block, so the absence of a consent record fails closed rather than open."""
    allowed = consent_status == "granted"
    return {"allowed": allowed, "consent_status": consent_status,
            "reason": None if allowed else f"consent is '{consent_status}', not 'granted'"}


def retention_status(retention_until: datetime | None, now: datetime) -> dict:
    """Whether a session is within its retention window. A record with no deadline is retained indefinitely
    (no action); a record whose deadline has passed must be purged. Returns the required action, so a retention
    sweep acts on this rather than on ad-hoc policy."""
    if retention_until is None:
        return {"expired": False, "action": "retain", "retention_until": None}
    expired = now >= retention_until
    return {"expired": expired, "action": "purge" if expired else "retain",
            "retention_until": retention_until.isoformat()}
