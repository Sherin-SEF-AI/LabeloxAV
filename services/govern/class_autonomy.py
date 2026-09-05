"""The per-class autonomy ladder: how far the machine may go for each class, and why.

Three rungs. L0 propose-only: the machine writes into review and nothing else. L1 auto-accept band:
the gate may hold labels open under a measured, active ThresholdFit for the current champion. L2
settlement: an accepted lot's remainder may be closed. The rungs are evidence, not trust - the only
way up from 1 to 2 is a passed lot for the current epoch, and the ways down are automatic.

History rows with one active per class (the ThresholdFit idiom), so a step-down is a new row and the
whole ladder is readable in order. Two overrides a machine never crosses: a human-pinned level is not
auto-promoted past, and a cooldown (7 days after any step-down) must expire AND fresh lot evidence
must exist before re-promotion.

The default, when no row exists, is computed rather than seeded: L1 where the current champion has a
measured, active per-class threshold (that fit IS the evidence L1 requires), else L0. Seeding rows for
that at migration time would freeze a computed fact into data that then rots when the champion moves.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import ClassAutonomy

log = get_logger("govern.class_autonomy")

COOLDOWN_DAYS = 7


async def _active_row(db: AsyncSession, class_id: int) -> ClassAutonomy | None:
    return (await db.execute(select(ClassAutonomy).where(
        ClassAutonomy.class_id == class_id, ClassAutonomy.active.is_(True))
        .order_by(ClassAutonomy.created_at.desc()))).scalars().first()


async def _fitted_default(db: AsyncSession, class_id: int) -> tuple[int, dict]:
    """L1 where the champion carries a measured active threshold for the class, else L0."""
    from services.govern.registry import get_champion
    from services.oraclyx.threshold_fit import active_thresholds

    champ = await get_champion(db, "detection")
    if champ is None:
        return 0, {"reason": "no champion; nothing to fit a band to"}
    fitted = await active_thresholds(db, champ.model_version)
    if class_id in (fitted.get("by_class") or {}):
        return 1, {"reason": "champion carries a measured active threshold for this class",
                   "fit_id": fitted.get("fit_id"), "model_version": champ.model_version}
    return 0, {"reason": f"no measured active threshold for this class on champion "
                         f"{champ.model_version}"}


async def effective_level(db: AsyncSession, class_id: int) -> dict:
    """The level in force for a class, with its basis and any cooldown."""
    row = await _active_row(db, class_id)
    if row is not None:
        return {"class_id": class_id, "level": int(row.level), "basis": row.basis or {},
                "set_by": row.set_by, "pinned": bool(row.pinned),
                "cooldown_until": row.cooldown_until.isoformat() if row.cooldown_until else None,
                "explicit": True}
    level, basis = await _fitted_default(db, class_id)
    return {"class_id": class_id, "level": level, "basis": basis, "set_by": "computed_default",
            "pinned": False, "cooldown_until": None, "explicit": False}


async def set_level(db: AsyncSession, class_id: int, level: int, *, set_by: str,
                    basis: dict | None = None, pinned: bool = False,
                    cooldown: bool = False) -> dict:
    """Write a new active row (deactivating the old), unconditionally. The guards live in the callers
    that are machines; a person calling this IS the override, and the row records who."""
    if level not in (0, 1, 2):
        return {"error": f"level must be 0, 1 or 2, not {level}"}
    await db.execute(update(ClassAutonomy).where(
        ClassAutonomy.class_id == class_id, ClassAutonomy.active.is_(True)).values(active=False))
    row = ClassAutonomy(autonomy_id=uuid.uuid4(), class_id=class_id, level=level,
                        basis=basis or {}, set_by=set_by, pinned=pinned,
                        cooldown_until=(datetime.now(UTC) + timedelta(days=COOLDOWN_DAYS)
                                        if cooldown else None))
    db.add(row)
    await db.commit()
    log.info("class_autonomy.set", class_id=class_id, level=level, set_by=set_by, pinned=pinned)
    return {"class_id": class_id, "level": level, "set_by": set_by,
            "cooldown_until": row.cooldown_until.isoformat() if row.cooldown_until else None}


async def promote_to_settlement(db: AsyncSession, class_id: int, *, lot_id: str,
                                basis: dict) -> dict:
    """1 -> 2, on a passed lot. Refuses over a pin, an unexpired cooldown, or a level that is not 1."""
    cur = await effective_level(db, class_id)
    if cur["pinned"]:
        return {"error": f"level {cur['level']} is human-pinned; the machine does not promote past "
                         "a person's decision", "level": cur["level"]}
    if cur["cooldown_until"] and datetime.fromisoformat(cur["cooldown_until"]) > datetime.now(UTC):
        return {"error": f"cooldown until {cur['cooldown_until']}; a class that just stepped down "
                         "re-earns its rung with fresh evidence, not with patience",
                "level": cur["level"]}
    if cur["level"] != 1:
        return {"error": f"promotion to settlement starts from level 1, not {cur['level']}",
                "level": cur["level"]}
    return await set_level(db, class_id, 2, set_by=f"lot:{lot_id}", basis=basis)


async def step_down(db: AsyncSession, class_id: int, to_level: int, *, reason: str,
                    set_by: str) -> dict:
    """Automatic demotion, always with a 7-day cooldown. Never refuses: stepping DOWN is the safe
    direction and must work even over a pin (a pin caps promotion, not protection)."""
    cur = await effective_level(db, class_id)
    if to_level >= cur["level"]:
        return {"unchanged": True, "level": cur["level"],
                "detail": f"already at or below level {to_level}"}
    res = await set_level(db, class_id, to_level, set_by=set_by,
                          basis={"step_down_reason": reason, "from_level": cur["level"]},
                          cooldown=True)
    from services.govern.audit import record

    await record(db, "class_autonomy", "step_down", str(class_id),
                 {"from": cur["level"], "to": to_level, "reason": reason, "set_by": set_by})
    return res


async def on_champion_change(db: AsyncSession, new_champion: str) -> dict:
    """Every L2 class drops to L1 when the champion moves: the passed lot was evidence about the OLD
    epoch's labels. Already-settled lots stay settled - the old model's frozen output is unchanged by
    the new champion - but new settlement waits for a lot passed under the new epoch. No cooldown:
    this is bookkeeping, not a defect."""
    rows = (await db.execute(select(ClassAutonomy).where(
        ClassAutonomy.active.is_(True), ClassAutonomy.level == 2))).scalars().all()
    dropped = []
    for row in rows:
        await set_level(db, row.class_id, 1, set_by="champion_change",
                        basis={"reason": f"champion moved to {new_champion}; settlement re-earns "
                                         "its rung under the new epoch"})
        dropped.append(row.class_id)
    if dropped:
        log.info("class_autonomy.champion_change", dropped=dropped, champion=new_champion)
    return {"dropped_to_l1": dropped}
