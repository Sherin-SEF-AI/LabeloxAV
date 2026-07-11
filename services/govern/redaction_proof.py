"""PII redaction proof (M18). Gate A blurs faces and plates per frame and records a PiiAudit; this rolls those
per-frame audits up per release into a verifiable attestation. A release proof passes only when every frame in
the release carried a PII audit (coverage at the floor); the attestation is signed with an HMAC so a buyer can
verify PII redaction ran over the whole dataset and no unredacted frame slipped through. Pure over the audit
coverage, so the coverage gate, signing, and tamper detection are testable; the DB rollup lives in run.py. The
signing key comes from GovernSettings.attestation_key at the call site, never embedded."""

from __future__ import annotations

import hashlib
import hmac
import json


def redaction_coverage(frame_ids: list[str], audited_frame_ids: set[str]) -> dict:
    """Coverage of the PII gate over a release: which of the release's frames have a PiiAudit. Returns the
    counts, the coverage fraction, and the frame ids still missing an audit (the ones that would block the
    proof)."""
    total = len(frame_ids)
    covered = [f for f in frame_ids if f in audited_frame_ids]
    uncovered = [f for f in frame_ids if f not in audited_frame_ids]
    return {"n_frames": total, "n_covered": len(covered),
            "coverage": round(len(covered) / total, 6) if total else 0.0,
            "uncovered": uncovered}


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_proof(release_commit: str, coverage: dict, key: str, method_version: str = "",
                coverage_floor: float = 1.0) -> dict:
    """Build and sign a release redaction proof. verdict is 'pass' only when coverage meets the floor (every
    frame redacted by default); a shortfall yields 'fail' and the proof names the uncovered frames rather than
    attesting a clean release. The signature is an HMAC over the canonical manifest."""
    # Gate on exact integer counts, never the rounded coverage float: for a very large release a single
    # unredacted frame rounds to 1.0, which would otherwise sign a false clean proof. At the default floor of
    # 1.0 the proof passes only when every frame is covered; a lower floor compares the exact (unrounded)
    # fraction.
    n_frames, n_covered = coverage["n_frames"], coverage["n_covered"]
    if n_frames <= 0:
        covered_ok = False
    elif coverage_floor >= 1.0:
        covered_ok = n_covered == n_frames
    else:
        covered_ok = (n_covered / n_frames) >= coverage_floor
    verdict = "pass" if covered_ok else "fail"
    manifest = {
        "release_commit": release_commit,
        "n_frames": coverage["n_frames"],
        "n_covered": coverage["n_covered"],
        "coverage": coverage["coverage"],
        "method_version": method_version,
        "verdict": verdict,
    }
    signature = hmac.new(key.encode("utf-8"), _canonical(manifest), hashlib.sha256).hexdigest()
    return {"manifest": manifest, "signature": signature, "verdict": verdict,
            "uncovered": coverage["uncovered"]}


def verify_proof(manifest: dict, signature: str, key: str) -> bool:
    """Verify a redaction proof: any tamper with the manifest (a forged coverage or verdict) invalidates the
    signature. Constant-time comparison."""
    expected = hmac.new(key.encode("utf-8"), _canonical(manifest), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature or "")
