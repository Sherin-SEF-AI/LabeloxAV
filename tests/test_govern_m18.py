"""Governance M18 tests: redaction-proof coverage gate (an unredacted frame fails the release) with tamper
detection, consent gate failing closed, retention expiry, and the cloud cost ceilings (per-job cap and window
remaining)."""

from datetime import datetime, timedelta, timezone

from services.govern.consent import export_consent_gate, retention_status
from services.govern.cost import cost_gate, estimate_job_cost
from services.govern.redaction_proof import build_proof, redaction_coverage, verify_proof

KEY = "unit-test-attestation-key"


def test_redaction_proof_gate_and_tamper():
    frames = ["f1", "f2", "f3"]
    full = redaction_coverage(frames, {"f1", "f2", "f3"})
    p = build_proof("rel-1", full, KEY, method_version="pii-v3", coverage_floor=1.0)
    assert p["verdict"] == "pass" and verify_proof(p["manifest"], p["signature"], KEY)
    # one frame missing a PII audit fails the release and is named
    partial = redaction_coverage(frames, {"f1", "f2"})
    pf = build_proof("rel-1", partial, KEY, coverage_floor=1.0)
    assert pf["verdict"] == "fail" and pf["uncovered"] == ["f3"]
    # a forged coverage cannot verify against the honest signature
    forged = {**p["manifest"], "coverage": 0.5, "verdict": "fail"}
    assert verify_proof(forged, p["signature"], KEY) is False


def test_redaction_proof_gate_is_exact_not_rounded():
    # a single unredacted frame in a huge release rounds coverage to 1.0, but the proof must still FAIL:
    # the gate is on exact counts, not the rounded float, so a leaked frame cannot sign a clean attestation.
    n = 2_000_000
    cov = {"n_frames": n, "n_covered": n - 1, "coverage": round((n - 1) / n, 6), "uncovered": ["leaked-frame"]}
    assert cov["coverage"] == 1.0                       # the rounding trap
    p = build_proof("rel-huge", cov, KEY, coverage_floor=1.0)
    assert p["verdict"] == "fail"                        # exact-count gate catches the one leaked frame


def test_consent_gate_fails_closed():
    assert export_consent_gate("granted")["allowed"] is True
    assert export_consent_gate("denied")["allowed"] is False
    assert export_consent_gate("unknown")["allowed"] is False   # absent consent blocks, not permits


def test_retention_expiry():
    now = datetime(2026, 7, 6, tzinfo=timezone.utc)
    past = now - timedelta(days=1)
    future = now + timedelta(days=30)
    assert retention_status(past, now)["action"] == "purge"
    assert retention_status(future, now)["action"] == "retain"
    assert retention_status(None, now)["action"] == "retain"    # no deadline -> retained


def test_cost_ceilings():
    assert estimate_job_cost(2.0, 1.89) == 3.78
    # within both ceilings -> allowed
    ok = cost_gate(est_cost_usd=5.0, per_job_cap_usd=10.0, spent_usd=10.0, window_cap_usd=50.0)
    assert ok["allowed"] is True and ok["remaining_after"] == 35.0
    # over the hard per-job cap -> refused
    assert cost_gate(15.0, 10.0, 0.0, 50.0)["allowed"] is False
    # fits the per-job cap but not the remaining window -> refused
    tight = cost_gate(8.0, 10.0, 45.0, 50.0)
    assert tight["allowed"] is False and "remaining" in tight["reason"]
