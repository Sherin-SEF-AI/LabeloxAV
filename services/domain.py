"""Engine-facing access to the active domain pack.

The single seam the engine uses to consult the pack it belongs to, without statically importing any concrete
pack (it goes through packs.registry, the sanctioned bridge). Governance, curation, and eval call these
instead of the `{"vru","animal"}` literals and hardcoded strata the audit found copy-pasted across the tree.

Every helper takes an optional pack_id and defaults to the configured active pack ('av'), so the legacy paths
resolve to the AV pack and stay byte-identical. Per-session pack routing (Session.pack_id) plugs in here later
by passing the session's pack_id.
"""

from __future__ import annotations

from packs.base import DomainPack
from packs.registry import default_pack_id, get_pack


def active_pack(pack_id: str | None = None) -> DomainPack:
    return get_pack(pack_id or default_pack_id())


def has_capability(name: str, pack_id: str | None = None) -> bool:
    """Whether the active pack declares a capability (e.g. 'anpr', 'static_camera'). Capability-gated features
    consult this so a domain that does not authorise a feature never runs it."""
    return name in active_pack(pack_id).manifest.capabilities


def safety_l1(pack_id: str | None = None) -> frozenset[str]:
    """The l1 superclasses that define a safety-critical class for the active pack (AV: {'vru','animal'})."""
    return active_pack(pack_id).autolabel_profile.gate_policy.safety_l1


def is_safety_class(class_name: str, pack_id: str | None = None) -> bool:
    return active_pack(pack_id).safety_policy.is_safety_class(class_name)


def critical_class_names(pack_id: str | None = None) -> frozenset[str]:
    return active_pack(pack_id).safety_policy.critical_class_names()


def critical_class_ids(onto, pack_id: str | None = None) -> set[int]:
    """The active pack's safety-critical classes resolved to ids against `onto`. Replaces the hardcoded
    (and 0-based-buggy) verdyx CRITICAL_CLASSES with names resolved through the ontology."""
    return {onto.by_name(n).id for n in critical_class_names(pack_id) if onto.has_name(n)}


def protected_slices(pack_id: str | None = None) -> tuple[str, ...]:
    return active_pack(pack_id).eval_strata.protected_slices


def strata_dimensions(pack_id: str | None = None) -> tuple[str, ...]:
    return tuple(d.name for d in active_pack(pack_id).eval_strata.dimensions)


def redaction_targets(pack_id: str | None = None):
    """The active pack's privacy redaction targets (AV: face, plate). The anonymizer runs a detector per
    image target; audio targets (speech) are enforced on the export/audio path."""
    return active_pack(pack_id).privacy.redaction_targets


def legal_regime(pack_id: str | None = None) -> str:
    """The active pack's privacy legal regime (AV: DPDPA)."""
    return active_pack(pack_id).privacy.legal_regime


def forge_targets(pack_id: str | None = None):
    """The active pack's edge deployment targets."""
    return active_pack(pack_id).forge_targets


def resolve_forge_target(name: str):
    """Find an edge target by name across all loaded packs (AV Jetson silicon, Sec CCTV silicon, ...), or None.
    FORGYX's per-target thermal/latency checks resolve a target's profile through this instead of a hardcoded
    AV registry, so a pack's own hardware envelopes are honoured."""
    from packs.registry import load_packs

    for pack in load_packs().values():
        for t in pack.forge_targets:
            if t.name == name:
                return t
    return None
