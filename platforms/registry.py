"""The platform registry: the single source of truth for the seven platforms the LabeloxAV data engine hosts.

LabeloxAV is the host. The annotation core and the six folded subsystems (SANYX, CALYX, SIEVYX, ORACLYX,
VERDYX, FORGYX) are Platforms: distinct navigable UIs behind a launcher, over one shared backend spine.

This registry is mirrored on the frontend (web/platforms/registry.ts). Both must be kept in sync; the API
exposes this one at GET /api/platforms so the two can be reconciled in a test. Nothing about the existing
routers moves; a platform simply groups the domain routers that already exist (backend_prefix is the new
canonical mount, legacy_prefixes are the existing ones that keep working). ``gate`` marks a platform that can
block a session's progression through the flywheel (SANYX health, CALYX calibration) or a model's promotion
(VERDYX eval, FORGYX benchmark).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Platform:
    id: str
    label: str
    role: str                                   # one-line description shown in the launcher
    backend_prefix: str                         # new canonical API prefix, e.g. /sanyx
    legacy_prefixes: tuple[str, ...] = ()        # existing domain routers this platform groups (kept working)
    gate: str | None = None                     # health | calibration | eval | benchmark | None
    order: int = 0                              # left-rail / launcher ordering
    flywheel_stage: int | None = None           # position in the closed loop (section 4), None = not a stage

    def as_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "role": self.role,
            "backend_prefix": self.backend_prefix, "legacy_prefixes": list(self.legacy_prefixes),
            "gate": self.gate, "order": self.order, "flywheel_stage": self.flywheel_stage,
        }


PLATFORMS: tuple[Platform, ...] = (
    Platform(
        id="labelox", label="Labelox", role="annotation core: three-path auto-label + human workspace",
        backend_prefix="/labelox", order=0, flywheel_stage=4,
        legacy_prefixes=("/autolabel", "/objects", "/objects3d", "/review", "/corrections", "/tracks",
                         "/multicam", "/lanes", "/segmentation", "/ocr", "/signs", "/triage", "/recall"),
    ),
    Platform(
        id="sanyx", label="SANYX", role="ingest QA: health score, quarantine bad sessions",
        backend_prefix="/sanyx", legacy_prefixes=("/inspector", "/adverse"), gate="health",
        order=1, flywheel_stage=1,
    ),
    Platform(
        id="calyx", label="CALYX", role="calibration monitor: extrinsic / intrinsic drift",
        backend_prefix="/calyx", legacy_prefixes=("/calibration",), gate="calibration",
        order=2, flywheel_stage=2,
    ),
    Platform(
        id="sievyx", label="SIEVYX", role="curation and mining: embed, rank, dedup, decide what to label",
        backend_prefix="/sievyx", order=3, flywheel_stage=3,
        legacy_prefixes=("/curation", "/search", "/intelligence", "/activelearn", "/discovery"),
    ),
    Platform(
        id="oraclyx", label="ORACLYX", role="offline-fusion pseudo-GT: auto-truth the majority, route disagreements",
        backend_prefix="/oraclyx", legacy_prefixes=("/lidar", "/lidarscene", "/hdmap", "/mapassist"),
        order=4, flywheel_stage=5,
    ),
    Platform(
        id="verdyx", label="VERDYX", role="slice evaluation: per-slice regression + champion/challenger verdict",
        backend_prefix="/verdyx", legacy_prefixes=("/training", "/govern", "/models"), gate="eval",
        order=5, flywheel_stage=7,
    ),
    Platform(
        id="forgyx", label="FORGYX", role="edge optimization: quantize, compile, benchmark, gate on latency and accuracy",
        backend_prefix="/forgyx", legacy_prefixes=("/export",), gate="benchmark",
        order=6, flywheel_stage=8,
    ),
)

_BY_ID = {p.id: p for p in PLATFORMS}


def platform_by_id(pid: str) -> Platform | None:
    return _BY_ID.get(pid)


def platform_for_route(path: str) -> Platform | None:
    """Resolve the platform that owns an API path, by its canonical or a legacy prefix."""
    for p in PLATFORMS:
        if path.startswith(p.backend_prefix) or any(path.startswith(lp) for lp in p.legacy_prefixes):
            return p
    return None


def gates() -> list[Platform]:
    """Platforms that can block progression, in flywheel order (the DAG consults these)."""
    return sorted((p for p in PLATFORMS if p.gate), key=lambda p: p.order)


def as_dicts() -> list[dict]:
    return [p.as_dict() for p in sorted(PLATFORMS, key=lambda p: p.order)]
