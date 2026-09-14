"""Kill switch and governance state (M4.4). One action pauses auto-accept and auto-promotion and rolls
back to the last champion; release resumes. The singleton governance_state row is what the controller
reads each tick. Rollback restores the model the current champion was promoted from."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import GovernanceState
from services.govern.audit import record
from services.govern.registry import get_champion, set_champion

log = get_logger("govern_killswitch")


async def get_state(db: AsyncSession) -> GovernanceState:
    st = await db.get(GovernanceState, 1)
    if st is None:
        st = GovernanceState(id=1, loop_enabled=True, auto_accept_enabled=True, auto_promote_enabled=True)
        db.add(st)
        await db.flush()
    return st


async def rollback_to_last_champion(db: AsyncSession, task: str = "detection") -> dict:
    champ = await get_champion(db, task)
    if champ is None or not champ.promoted_from:
        return {"rolled_back": False, "reason": "no prior champion to roll back to"}
    prev = champ.promoted_from
    await set_champion(db, prev, task, promoted_from=None)
    st = await get_state(db)
    st.champion_version = prev
    await db.commit()
    return {"rolled_back": True, "from": champ.model_version, "to": prev}


async def engage(db: AsyncSession, reason: str, task: str = "detection") -> dict:
    """Pause the loop and roll back to the last champion."""
    st = await get_state(db)
    st.loop_enabled = False
    st.auto_accept_enabled = False
    st.auto_promote_enabled = False
    # Settlement stops with everything else, including its own nightly auto-revert: a person who pulled
    # the cord gets no surprise writes in either direction, only one-click proposals.
    st.settlement_enabled = False
    st.paused_reason = reason
    await db.commit()
    rollback = await rollback_to_last_champion(db, task)
    await record(db, "killswitch", "engage", None, {"reason": reason, "rollback": rollback})
    log.info("govern.killswitch_engaged", reason=reason, rollback=rollback)
    return {"engaged": True, "reason": reason, "rollback": rollback}


async def release(db: AsyncSession) -> dict:
    """Resume the loop. Deliberately NOT symmetric with engage.

    Releasing restores the loop and the auto-accept gate, and nothing else. `auto_promote_enabled`
    defaults to False under promote-by-approval (migration 0104) and is an explicit opt-in; a release
    that set it True would resurrect full autonomous promotion as a side effect of clearing an
    unrelated emergency, which is precisely the kind of silent re-arming a kill switch must not do.
    The same holds for `settlement_enabled` once it exists: anything that was an opt-in before the
    cord was pulled stays an opt-in after.
    """
    st = await get_state(db)
    st.loop_enabled = True
    st.auto_accept_enabled = True
    st.paused_reason = None
    await db.commit()
    await record(db, "killswitch", "release", None, {})
    return {"engaged": False}


_DRIFT_PAUSE_PREFIX = "drift breach"


async def pause_auto_promote(db: AsyncSession, reason: str) -> None:
    """Soft pause: stop auto-promotion (drift breach) without disabling the whole loop or rolling back."""
    st = await get_state(db)
    st.auto_promote_enabled = False
    st.paused_reason = reason
    await db.commit()


async def resume_auto_promote(db: AsyncSession) -> bool:
    """Clear a drift-induced soft pause once the breach clears - WITHOUT re-arming auto-promotion.

    This used to set auto_promote_enabled back to True, which was symmetric when the flag defaulted
    on. Under promote-by-approval (migration 0104) the flag's normal state is False, so the old resume
    GRANTED autonomy nobody opted into: a drift breach paused, the next clean scan "resumed", and the
    loop was silently promoting on its own - observed live when a single drift-scan click flipped the
    operator's chosen default. The rule is the release() rule: anything that was an opt-in before the
    pause stays an opt-in after. An operator who had opted in re-enables with one click; a notification
    tells them the breach cleared. Returns True if a drift pause was cleared."""
    st = await get_state(db)
    if not st.loop_enabled:
        return False
    if not (st.paused_reason or "").startswith(_DRIFT_PAUSE_PREFIX):
        return False  # paused by something other than drift; leave it to an operator
    st.paused_reason = None
    await db.commit()
    from services.notify import notify

    await notify(db, kind="drift_breach", severity="info",
                 title="drift breach cleared; auto-promotion stays off (opt-in)",
                 body="The metrics that breached are back inside their bounds. Promotion remains "
                      "propose-and-approve; re-enable unattended promotion on the govern page if "
                      "that is what you want.",
                 href="/govern", subject_type="drift", subject_id="soft-pause")
    return True


async def state_dict(db: AsyncSession) -> dict:
    st = await get_state(db)
    return {"loop_enabled": st.loop_enabled, "auto_accept_enabled": st.auto_accept_enabled,
            "auto_promote_enabled": st.auto_promote_enabled, "champion_version": st.champion_version,
            "paused_reason": st.paused_reason,
            "updated_at": st.updated_at.isoformat() if st.updated_at else None}
