"""FORGYX signed deployment packaging + rollout/rollback (M16). A model does not reach a vehicle as a bare
artifact: it is packaged with its manifest (lineage back to the release, the VERDYX verdict, the FORGYX
benchmark and thermal envelope), signed so a device can verify integrity and provenance before loading it, and
rolled out through canary to full with a rollback path to the prior deployment. This builds the manifest, signs
it with an HMAC over a canonical serialization, verifies a signature, and computes the rollout/rollback state
transitions. Pure and deterministic (a passed key), so signing, tamper detection, and the rollout state machine
are testable. The signing key is read from config/secret at the call site, never embedded."""

from __future__ import annotations

import hashlib
import hmac
import json


def build_manifest(deployment_id: str, model_version: str, target: str, export_format: str,
                   artifact_uri: str, release_commit: str | None, verdict_ref: str | None,
                   benchmark_ref: str | None, thermal_envelope: dict) -> dict:
    """Assemble the deployment manifest: identity, lineage, and the gates that admitted it. This is the object
    that gets signed, so a device can prove what it is loading and where it came from."""
    return {
        "deployment_id": deployment_id,
        "model_version": model_version,
        "target": target,
        "export_format": export_format,
        "artifact_uri": artifact_uri,
        "release_commit": release_commit,
        "verdict_ref": verdict_ref,
        "benchmark_ref": benchmark_ref,
        "thermal_envelope": thermal_envelope,
    }


def _canonical(manifest: dict) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest(manifest: dict, key: str) -> str:
    """HMAC-SHA256 over the canonical manifest serialization. The key comes from config/secret at the call
    site."""
    return hmac.new(key.encode("utf-8"), _canonical(manifest), hashlib.sha256).hexdigest()


def verify_manifest(manifest: dict, signature: str, key: str) -> bool:
    """Verify a manifest against its signature. Any tamper with the manifest (a swapped artifact_uri, a forged
    verdict) invalidates the signature. Constant-time comparison."""
    expected = sign_manifest(manifest, key)
    return hmac.compare_digest(expected, signature or "")


# the rollout state machine: none -> canary -> full, with rollback reachable from canary or full
_ROLLOUT_TRANSITIONS = {
    "none": {"canary"},
    "canary": {"full", "rolled_back"},
    "full": {"rolled_back"},
    "rolled_back": set(),
}


def plan_rollout(current_state: str, action: str) -> dict:
    """The rollout/rollback transition for a deployment. action is the target state (canary|full|rolled_back).
    Returns the new state and whether the transition is allowed; an illegal transition (e.g. none -> full,
    skipping canary) is refused so a model cannot go to the whole fleet unproven."""
    allowed = _ROLLOUT_TRANSITIONS.get(current_state, set())
    ok = action in allowed
    return {"from": current_state, "action": action, "allowed": ok,
            "new_state": action if ok else current_state,
            "reason": None if ok else f"cannot go from {current_state} to {action}"}
