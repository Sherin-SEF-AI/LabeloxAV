"""Cost-safety tests for the warm cloud-GPU session, against a fake pod (no network, no billing). The whole
risk of a button that fires a GPU is a lingering pod, so these prove the pod is torn down on disconnect, on
idle, on the max-session cap, and on a forced error, that connect never fires a second pod, and that an
orphan is surfaced. Time is injected so the guards are deterministic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import delete, select

from compute.runpod.cost import CostConfig, est_cost
from compute.runpod.orchestrator import PodInfo
from compute.runpod.session import WarmSessionManager
from db.models import CloudSession
from db.session import get_sessionmaker

T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
RATE = 1.89


class FakeOrchestrator:
    """In-memory pods. Records terminate/pause calls so tests can assert teardown happened."""

    POD_NAME = "labeloxav-warm"

    def __init__(self, *, running_on_provision: bool = True):
        self.pods: dict[str, PodInfo] = {}
        self.terminated: list[str] = []
        self.paused: list[str] = []
        self.provisions = 0
        self.configured = True
        self._running_on_provision = running_on_provision
        self.terminate_raises = False

    def gpu_types(self):
        return [{"id": "NVIDIA A100 80GB PCIe", "displayName": "A100 80GB", "securePrice": RATE}]

    def provision(self, gpu_type_id, image, volume_id):
        self.provisions += 1
        pid = f"pod-{self.provisions}"
        status = "RUNNING" if self._running_on_provision else "CREATED"
        self.pods[pid] = PodInfo(id=pid, status=status, name=self.POD_NAME, gpu_type=gpu_type_id)
        return self.pods[pid]

    def seed_running(self, pid="pre-existing"):
        self.pods[pid] = PodInfo(id=pid, status="RUNNING", name=self.POD_NAME, gpu_type="A100 80GB", uptime_s=300)
        return pid

    def status(self, pod_id):
        return self.pods.get(pod_id, PodInfo(id=pod_id, status="GONE"))

    def terminate(self, pod_id):
        if self.terminate_raises:
            raise RuntimeError("simulated runpod terminate failure")
        self.terminated.append(pod_id)
        self.pods[pod_id] = PodInfo(id=pod_id, status="TERMINATED", name=self.POD_NAME)

    def pause(self, pod_id):
        self.paused.append(pod_id)
        self.pods[pod_id] = PodInfo(id=pod_id, status="EXITED", name=self.POD_NAME)

    def list_pods(self):
        return list(self.pods.values())


class _CloudCfg:
    image = "lbx/worker:test"
    warm_gpu_type_id = "NVIDIA A100 80GB PCIe"


def _mgr(orch, *, idle_s=900, max_s=14400):
    cfg = CostConfig(hourly_usd=RATE, idle_seconds=idle_s, max_session_seconds=max_s)
    return WarmSessionManager(orch, cfg, _CloudCfg(), get_sessionmaker())


async def _clean():
    async with get_sessionmaker()() as db:
        await db.execute(delete(CloudSession))
        await db.commit()


async def _count_live():
    async with get_sessionmaker()() as db:
        rows = (await db.execute(select(CloudSession).where(CloudSession.state != "disconnected"))).scalars().all()
        return len(rows)


@pytest.fixture(autouse=True)
async def _isolate():
    await _clean()
    yield
    await _clean()


async def test_connect_provisions_one_pod_and_meters_cost():
    orch = FakeOrchestrator()
    m = _mgr(orch, idle_s=10_000_000)   # idle disabled here so the one-hour meter can be read
    snap = await m.connect(RATE, now=T0)
    assert orch.provisions == 1
    assert snap["state"] == "provisioning"
    # first status poll flips it to connected and starts the cost clock
    s1 = await m.status(now=T0)
    assert s1["state"] == "connected" and s1["connected"] is True
    # one hour later the meter reflects one hour at the rate
    s2 = await m.status(now=T0 + timedelta(hours=1))
    assert s2["uptime_s"] == 3600
    assert s2["est_cost"] == pytest.approx(est_cost(3600, RATE), abs=1e-6)
    assert await _count_live() == 1


async def test_connect_is_idempotent_no_second_pod():
    orch = FakeOrchestrator()
    m = _mgr(orch)
    await m.connect(RATE, now=T0)
    await m.connect(RATE, now=T0 + timedelta(seconds=5))
    assert orch.provisions == 1            # never a second pod
    assert await _count_live() == 1


async def test_reconnects_to_existing_warm_pod():
    orch = FakeOrchestrator()
    orch.seed_running("ghost")              # a warm pod left from a prior app run
    m = _mgr(orch)
    snap = await m.connect(RATE, now=T0)
    assert orch.provisions == 0            # adopted the existing pod, did not provision
    assert snap["pod_id"] == "ghost"
    assert await _count_live() == 1


async def test_idle_window_auto_terminates():
    orch = FakeOrchestrator()
    m = _mgr(orch, idle_s=60)
    await m.connect(RATE, now=T0)
    await m.status(now=T0)                  # -> connected, idle clock from T0
    snap = await m.status(now=T0 + timedelta(seconds=61))  # 61s idle > 60s window
    assert snap["state"] == "disconnected"
    assert orch.terminated == ["pod-1"]
    assert await _count_live() == 0


async def test_max_session_cap_auto_terminates():
    orch = FakeOrchestrator()
    m = _mgr(orch, idle_s=10_000_000, max_s=3600)   # idle disabled so the cap is what trips
    await m.connect(RATE, now=T0)
    await m.status(now=T0)
    snap = await m.status(now=T0 + timedelta(seconds=3601))
    assert snap["state"] == "disconnected"
    assert orch.terminated == ["pod-1"]


async def test_disconnect_terminates():
    orch = FakeOrchestrator()
    m = _mgr(orch)
    await m.connect(RATE, now=T0)
    await m.status(now=T0)
    snap = await m.disconnect(now=T0 + timedelta(minutes=5))
    assert snap["state"] == "disconnected"
    assert orch.terminated == ["pod-1"] and orch.paused == []
    assert await _count_live() == 0


async def test_pause_does_not_terminate():
    orch = FakeOrchestrator()
    m = _mgr(orch)
    await m.connect(RATE, now=T0)
    await m.status(now=T0)
    await m.disconnect(now=T0 + timedelta(minutes=5), terminate=False)
    assert orch.paused == ["pod-1"] and orch.terminated == []
    assert await _count_live() == 0


async def test_provision_record_failure_tears_down_pod():
    """If recording the session fails after a pod is provisioned, the pod must be terminated, not leaked."""
    orch = FakeOrchestrator()

    class _BreakRecord:
        def __init__(self, real):
            self._real = real
            self.armed = True

        def __call__(self):
            sess = self._real()
            real_commit = sess.commit
            async def commit(*a, **k):
                if self.armed:
                    self.armed = False
                    raise RuntimeError("simulated DB failure recording the session")
                return await real_commit(*a, **k)
            sess.commit = commit
            return sess

    cfg = CostConfig(hourly_usd=RATE, idle_seconds=900, max_session_seconds=14400)
    m = WarmSessionManager(orch, cfg, _CloudCfg(), _BreakRecord(get_sessionmaker()))
    with pytest.raises(RuntimeError):
        await m.connect(RATE, now=T0)
    assert orch.provisions == 1
    assert orch.terminated == ["pod-1"]    # the leaked pod was torn down in the except path


async def test_teardown_guaranteed_even_if_terminate_errors():
    orch = FakeOrchestrator()
    orch.terminate_raises = True            # the runpod terminate call blows up
    m = _mgr(orch)
    await m.connect(RATE, now=T0)
    await m.status(now=T0)
    snap = await m.disconnect(now=T0 + timedelta(minutes=1))
    # the session is still finalized as disconnected (finally guarantee), never stuck "terminating"
    assert snap["state"] == "disconnected"
    assert await _count_live() == 0


async def test_find_orphans_surfaces_untracked_warm_pod():
    orch = FakeOrchestrator()
    orch.seed_running("orphan-1")           # a warm pod running with no live session
    m = _mgr(orch)
    orphans = await m.find_orphans(now=T0)
    assert len(orphans) == 1
    assert orphans[0]["pod_id"] == "orphan-1"
    assert orphans[0]["est_cost"] == pytest.approx(est_cost(300, RATE), abs=1e-6)


async def test_connect_requires_acknowledged_rate():
    orch = FakeOrchestrator()
    m = _mgr(orch)
    with pytest.raises(ValueError):
        await m.connect(0.50, now=T0)       # wrong rate -> refuse to fire the GPU
    assert orch.provisions == 0
