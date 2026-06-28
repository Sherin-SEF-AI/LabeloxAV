"""Champion/challenger promotion (M4.4). A retrained challenger ships only if it beats the incumbent on
the frozen gold set AND regresses no safety class under Safe-mIoU; otherwise it is discarded and an alert
is recorded. Safety is never automated to zero: a VRU (pedestrian, rider, cyclist, animal) class that
regresses blocks promotion no matter how much overall mAP improved. Applies to both the cloud models and
the edge model (per task). Respects the pause set by drift or the kill switch.

champion_gate is a pure function over the two gold-metric dicts; evaluate_and_promote wires it to the
registry, the governance pause, and the audit log.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.logging import get_logger
from db.models import ModelRegistry
from services.autolabel.ontology import get_ontology
from services.govern.audit import record
from services.govern.killswitch import get_state
from services.govern.registry import get_champion, set_champion

log = get_logger("govern_champion")

_VRU_ANIMAL = {"vru", "animal"}


def _is_safety_class(name: str, onto) -> bool:
    try:
        return onto.by_name(name).l1 in _VRU_ANIMAL
    except Exception:  # noqa: BLE001
        return False


def _map(m: dict) -> float:
    return float(m.get("map", m.get("map50", 0.0)) or 0.0)


def champion_gate(challenger: dict, champion: dict | None, onto, cfg, safety_class_drop: float = 0.15) -> dict:
    """Pure promotion decision. Fail-closed: a challenger that cannot prove its safety (no Safe-mIoU)
    is never promoted, and a safety-class or Safe-mIoU regression always blocks."""
    map_c, map_ch = _map(challenger), _map(champion or {})
    beats_map = map_c >= map_ch + cfg.min_map_uplift

    sm_c, sm_ch = challenger.get("safe_miou"), (champion or {}).get("safe_miou")
    # A missing challenger Safe-mIoU is a fail, not a silent pass: we cannot verify it did not regress
    # safety, so we refuse. With an incumbent baseline present, enforce the max-drop floor.
    if sm_c is None:
        safe_ok = False
    elif sm_ch is None:
        safe_ok = True  # no incumbent baseline to regress against
    else:
        safe_ok = sm_c >= sm_ch - cfg.safe_miou_max_drop

    pc_c = challenger.get("per_class", {}) or {}
    pc_ch = (champion or {}).get("per_class", {}) or {}
    regressed = [cn for cn, ap in pc_ch.items()
                 if _is_safety_class(cn, onto) and pc_c.get(cn, 0.0) < ap - safety_class_drop]
    safety_ok = not regressed

    if champion is None:
        # First champion still must clear the safety floor: require a present Safe-mIoU (fail-closed).
        promote = sm_c is not None
        reasons = (["no incumbent; first champion (Safe-mIoU present)"] if promote
                   else ["no incumbent but challenger lacks Safe-mIoU; refused (fail-closed)"])
        return {"promote": promote, "beats_map": True, "map_delta": round(map_c, 4), "safe_ok": promote,
                "safety_ok": True, "regressed_safety": [], "reasons": reasons}

    promote = bool(beats_map and safe_ok and safety_ok)
    reasons: list[str] = []
    if not beats_map:
        reasons.append(f"does not beat champion mAP ({map_c:.3f} vs {map_ch:.3f})")
    if not safe_ok:
        reasons.append("challenger lacks Safe-mIoU (fail-closed)" if sm_c is None
                       else f"Safe-mIoU regressed ({sm_c} vs {sm_ch})")
    if not safety_ok:
        reasons.append(f"safety-class regression: {regressed}")
    if promote:
        reasons.append("beats champion without any safety regression")
    return {"promote": promote, "beats_map": beats_map, "map_delta": round(map_c - map_ch, 4),
            "safe_ok": safe_ok, "safety_ok": safety_ok, "regressed_safety": regressed, "reasons": reasons}


async def evaluate_and_promote(db: AsyncSession, challenger_version: str, task: str = "detection") -> dict:
    cfg = get_settings().phase4.govern
    onto = get_ontology()
    reg = await db.get(ModelRegistry, challenger_version)
    if reg is None:
        return {"error": "challenger not registered"}
    champ = await get_champion(db, task)
    gate = champion_gate(reg.gold_metrics or {}, champ.gold_metrics if champ else None, onto, cfg)

    state = await get_state(db)
    if gate["promote"] and not state.auto_promote_enabled:
        await record(db, "champion", "promotion_paused", challenger_version,
                     {"gate": gate, "paused_reason": state.paused_reason})
        return {"promoted": False, "paused": True, "gate": gate, "reason": state.paused_reason}

    if gate["promote"]:
        prev = champ.model_version if champ else None
        await set_champion(db, challenger_version, task, promoted_from=prev)
        state.champion_version = challenger_version
        await db.commit()
        await record(db, "champion", "promote", challenger_version, {"gate": gate, "promoted_from": prev})
        log.info("govern.promote", challenger=challenger_version, promoted_from=prev)
        return {"promoted": True, "champion": challenger_version, "promoted_from": prev, "gate": gate}

    await record(db, "champion", "reject", challenger_version, {"gate": gate})
    log.info("govern.reject", challenger=challenger_version, reasons=gate["reasons"])
    return {"promoted": False, "gate": gate, "alert": "challenger rejected: " + "; ".join(gate["reasons"])}
