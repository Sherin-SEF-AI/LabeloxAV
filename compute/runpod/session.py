"""The warm cloud-GPU session manager. Holds at most one RunPod pod across a work session and guarantees
it is torn down: on disconnect, on idle, on the max-session cap, on error, and on app shutdown.

Cost safety is enforced here, not left to a setting:
  - connect() requires the caller to acknowledge the hourly rate, reconnects to an existing warm pod
    instead of provisioning a second, and terminates the pod if recording the session ever fails.
  - refresh() (called by the status poll and the watchdog) recomputes the live cost and, if the idle
    window or the max-session cap is breached, terminates the pod immediately.
  - disconnect() terminates by default (full removal, billing stops); pause is opt-in and still bills the
    volume, so it is labeled as such.
  - find_orphans() surfaces a warm pod left running with no live session, so a forgotten pod cannot hide.

The DB row (cloud_session) is the source of truth, so state survives an app restart and a second worker
sees the same session. The orchestrator is injected so the guards can be tested without a billable pod.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta

from sqlalchemy import select

from compute.runpod import cost
from compute.runpod.orchestrator import Orchestrator, PodInfo, RunpodError, RunpodOrchestrator
from core.config import get_settings
from core.logging import get_logger
from db.models import CloudSession
from db.session import get_sessionmaker

log = get_logger("cloud_warm")

LIVE_STATES = ("provisioning", "connected", "running_job", "pausing", "terminating")
CONNECTED_STATES = ("connected", "running_job")
COLD_START_ESTIMATE_S = 90  # typical A100 on-demand boot + ssh ready; shown to the user on connect


def _utcnow() -> datetime:
    return datetime.now(UTC)


class WarmSessionManager:
    POD_NAME = "labeloxav-warm"

    def __init__(self, orchestrator: Orchestrator, cost_cfg: cost.CostConfig, cloud_settings,
                 sessionmaker) -> None:
        self._orch = orchestrator
        self._cfg = cost_cfg
        self._cloud = cloud_settings
        self._sm = sessionmaker

    # ---- helpers -------------------------------------------------------------------------------------
    def _image(self) -> str:
        if not self._cloud.image:
            raise RunpodError("cloud worker image is not configured (settings.cloud.image)")
        return self._cloud.image

    def _volume_id(self) -> str | None:
        return os.environ.get("LBX_RUNPOD_VOLUME_ID") or None

    async def _live(self, db) -> CloudSession | None:
        rows = (await db.execute(
            select(CloudSession).where(CloudSession.state.in_(LIVE_STATES))
            .order_by(CloudSession.created_at.desc()))).scalars().all()
        return rows[0] if rows else None

    async def _find_warm_pod(self) -> PodInfo | None:
        """A running warm pod (ours by name), for reconnect. Ignores ephemeral per-job pods."""
        try:
            pods = await asyncio.to_thread(self._orch.list_pods)
        except RunpodError:
            return None
        for p in pods:
            if p.name == self.POD_NAME and p.is_running:
                return p
        return None

    async def _safe_terminate(self, pod_id: str | None) -> None:
        """Best-effort teardown. Never raises: a teardown that fails must not strand the caller, and the
        watchdog / orphan check will catch a pod that somehow survived."""
        if not pod_id:
            return
        try:
            await asyncio.to_thread(self._orch.terminate, pod_id)
        except Exception as exc:  # noqa: BLE001
            log.error("warm.terminate_failed", pod_id=pod_id, error=str(exc))

    async def _safe_pause(self, pod_id: str | None) -> None:
        if not pod_id:
            return
        try:
            await asyncio.to_thread(self._orch.pause, pod_id)
        except Exception as exc:  # noqa: BLE001
            log.error("warm.pause_failed", pod_id=pod_id, error=str(exc))

    def _snapshot(self, sess: CloudSession | None, now: datetime, *, cold_start_s: int = 0) -> dict:
        if sess is None or sess.state == "disconnected":
            return {
                "state": "disconnected", "connected": False, "pod_id": None, "gpu_type": None,
                "uptime_s": 0, "gpu_seconds": 0.0, "est_cost": 0.0, "hourly_usd": self._cfg.hourly_usd,
                "idle_remaining_s": None, "session_remaining_s": None, "last_job_id": None,
                "cold_start_s": cold_start_s, "configured": getattr(self._orch, "configured", True),
            }
        return {
            "state": sess.state,
            "connected": sess.state in CONNECTED_STATES,
            "pod_id": sess.pod_id,
            "gpu_type": sess.gpu_type,
            "uptime_s": int(cost.gpu_seconds(sess.started_at, now)),
            "gpu_seconds": round(sess.gpu_seconds, 2),
            "est_cost": sess.est_cost,
            "hourly_usd": self._cfg.hourly_usd,
            "idle_remaining_s": cost.idle_remaining(sess.idle_since, now, self._cfg),
            "session_remaining_s": cost.session_remaining(sess.max_session_until, now),
            "last_job_id": str(sess.last_job_id) if sess.last_job_id else None,
            "cold_start_s": cold_start_s,
            "configured": getattr(self._orch, "configured", True),
        }

    # ---- connect / disconnect ------------------------------------------------------------------------
    async def connect(self, ack_hourly_usd: float, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        if abs(float(ack_hourly_usd) - self._cfg.hourly_usd) > 0.005:
            raise ValueError(
                f"acknowledged rate ${ack_hourly_usd}/hr does not match the current rate "
                f"${self._cfg.hourly_usd}/hr; reload and confirm the rate")
        async with self._sm() as db:
            live = await self._live(db)
            if live is not None:
                # already provisioning or connected: never fire a second pod
                return await self._refresh(db, live, now)

            adopt = await self._find_warm_pod()
            if adopt is not None:
                sess = CloudSession(
                    pod_id=adopt.id, mode="warm",
                    state="connected" if adopt.is_running else "provisioning",
                    gpu_type=adopt.gpu_type, started_at=now if adopt.is_running else None,
                    idle_since=now, gpu_seconds=0.0, est_cost=0.0,
                    max_session_until=now + timedelta(seconds=self._cfg.max_session_seconds))
                db.add(sess)
                await db.commit()
                log.info("warm.reconnect", pod_id=adopt.id)
                return await self._refresh(db, sess, now)

            pod = await asyncio.to_thread(
                self._orch.provision, self._cloud.warm_gpu_type_id, self._image(), self._volume_id())
            try:
                sess = CloudSession(
                    pod_id=pod.id, mode="warm", state="provisioning",
                    gpu_type=self._cloud.warm_gpu_type_id, started_at=None, idle_since=now,
                    gpu_seconds=0.0, est_cost=0.0,
                    max_session_until=now + timedelta(seconds=self._cfg.max_session_seconds))
                db.add(sess)
                await db.commit()
            except Exception:
                # provisioned but could not record it: tear the pod down so it cannot leak
                await self._safe_terminate(pod.id)
                raise
            log.info("warm.connect", pod_id=pod.id, gpu=self._cloud.warm_gpu_type_id)
            return self._snapshot(sess, now, cold_start_s=COLD_START_ESTIMATE_S)

    async def disconnect(self, now: datetime | None = None, *, terminate: bool = True) -> dict:
        now = now or _utcnow()
        async with self._sm() as db:
            sess = await self._live(db)
            if sess is None:
                return self._snapshot(None, now)
            sess.state = "terminating" if terminate else "pausing"
            await db.commit()
            # Teardown is best-effort: even if the runpod call errors, the session must finalize as
            # disconnected (never stuck "terminating"); a pod that somehow survives is caught by the
            # watchdog and the orphan check. This is the finally-guarantee that a forced error cannot leak.
            if sess.pod_id:
                if terminate:
                    await self._safe_terminate(sess.pod_id)
                else:
                    await self._safe_pause(sess.pod_id)
            # cost stops when the GPU stops: terminate -> billing stops; pause -> volume still bills, but the
            # GPU is down, so the session is over either way and the meter is finalized.
            sess.gpu_seconds = cost.gpu_seconds(sess.started_at, now)
            sess.est_cost = cost.est_cost(sess.gpu_seconds, self._cfg.hourly_usd)
            sess.state = "disconnected"
            sess.idle_since = None
            await db.commit()
            log.info("warm.disconnect", pod_id=sess.pod_id, terminate=terminate, est_cost=sess.est_cost)
            return self._snapshot(None, now)

    # ---- live refresh + guards -----------------------------------------------------------------------
    async def _refresh(self, db, sess: CloudSession, now: datetime) -> dict:
        """Poll the pod, advance the state machine, recompute cost, and enforce the guards. If a guard is
        breached or the pod is gone, terminate and end the session. Returns the snapshot."""
        if sess.pod_id:
            try:
                pod = await asyncio.to_thread(self._orch.status, sess.pod_id)
            except RunpodError as exc:
                log.error("warm.status_failed", pod_id=sess.pod_id, error=str(exc))
                pod = None
            if pod is not None:
                if pod.is_gone:
                    # terminated out from under us (or never came up): end the session cleanly
                    sess.gpu_seconds = cost.gpu_seconds(sess.started_at, now)
                    sess.est_cost = cost.est_cost(sess.gpu_seconds, self._cfg.hourly_usd)
                    sess.state = "disconnected"
                    sess.idle_since = None
                    await db.commit()
                    return self._snapshot(None, now)
                if pod.is_running and sess.started_at is None:
                    sess.started_at = now  # first observed running; the cost clock starts
                    if sess.state == "provisioning":
                        sess.state = "connected"
                if pod.gpu_type and not sess.gpu_type:
                    sess.gpu_type = pod.gpu_type

        sess.gpu_seconds = cost.gpu_seconds(sess.started_at, now)
        sess.est_cost = cost.est_cost(sess.gpu_seconds, self._cfg.hourly_usd)
        await db.commit()

        if sess.state in CONNECTED_STATES:
            breach = cost.guard_breach(now, sess.started_at, sess.idle_since,
                                       sess.max_session_until, self._cfg)
            if breach:
                log.info("warm.guard_terminate", pod_id=sess.pod_id, reason=breach)
                # terminate is the only safe action; reuse disconnect for the finally-guaranteed teardown
                return await self.disconnect(now, terminate=True)
        return self._snapshot(sess, now)

    async def refresh(self, now: datetime | None = None) -> dict:
        now = now or _utcnow()
        async with self._sm() as db:
            sess = await self._live(db)
            if sess is None:
                return self._snapshot(None, now)
            return await self._refresh(db, sess, now)

    async def status(self, now: datetime | None = None) -> dict:
        # status is a cheap pollable refresh (it also enforces the guards, so polling keeps the pod honest)
        return await self.refresh(now)

    # ---- idle hooks (used by job routing in M-CG.3) --------------------------------------------------
    async def job_started(self, job_id, now: datetime | None = None) -> None:
        """A job began running on the warm pod: clear idle so the idle window cannot trip mid-job."""
        now = now or _utcnow()
        async with self._sm() as db:
            sess = await self._live(db)
            if sess is None:
                return
            sess.idle_since = None
            sess.state = "running_job"
            sess.last_job_id = job_id
            await db.commit()

    async def job_finished(self, now: datetime | None = None) -> None:
        """A job finished: start the idle clock so genuine idleness eventually auto-terminates."""
        now = now or _utcnow()
        async with self._sm() as db:
            sess = await self._live(db)
            if sess is None:
                return
            sess.idle_since = now
            if sess.state == "running_job":
                sess.state = "connected"
            await db.commit()

    # ---- orphans -------------------------------------------------------------------------------------
    async def find_orphans(self, now: datetime | None = None) -> list[dict]:
        """Warm pods running with no live UI session: the dangerous 'left on by the button' case. Scoped to
        our warm pod name so the ephemeral per-job flow's pods are never touched."""
        now = now or _utcnow()
        async with self._sm() as db:
            live = await self._live(db)
            active_pod = live.pod_id if live else None
        try:
            pods = await asyncio.to_thread(self._orch.list_pods)
        except RunpodError:
            return []
        orphans = []
        for p in pods:
            if p.name == self.POD_NAME and p.is_running and p.id != active_pod:
                orphans.append({
                    "pod_id": p.id, "gpu_type": p.gpu_type, "uptime_s": p.uptime_s,
                    "est_cost": cost.est_cost(p.uptime_s, self._cfg.hourly_usd),
                })
        return orphans

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate a specific pod (used to kill an orphan from the UI)."""
        await self._safe_terminate(pod_id)


def get_manager() -> WarmSessionManager:
    """The process's warm-session manager: the real RunPod orchestrator plus the configured cost guards.
    Stateless (the cloud_session row is the source of truth), so a fresh instance per call is fine and
    picks up RUNPOD_API_KEY / config changes without a restart."""
    s = get_settings()
    return WarmSessionManager(
        RunpodOrchestrator(), cost.CostConfig.from_settings(s.cloud), s.cloud, get_sessionmaker())
