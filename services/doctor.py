"""Is this installation able to do its job? One list of checks, each with what to do when it fails.

A team running LabeloxAV on one GPU box meets its failures as symptoms: an upload that refuses, a job bar
that never moves, an editor SAM click that errors. Each has a cause an operator can fix in a minute (the
GPU worker is not running, Docker cannot see the GPU, a weight is missing, the schema is behind) and none
of those causes is visible from the symptom.

These checks name the cause. They run three ways: `python -m services.doctor` inside the API container
(scripts/doctor.sh does that after its host checks), `GET /api/system/checks` for an administrator, and the
system status page in the web app. Each result carries a status, what was found, and when it is not ok, the
command or setting that fixes it.

Statuses: ok; warn (works, but something is degraded or slow); fail (a core feature is broken); info
(a fact worth showing, not a judgement).
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from core.config import REPO_ROOT, get_settings
from core.logging import get_logger

log = get_logger("doctor")

OK, WARN, FAIL, INFO = "ok", "warn", "fail", "info"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""
    group: str = "services"


async def _timed(coro, timeout: float = 5.0):
    return await asyncio.wait_for(coro, timeout)


async def check_postgres() -> Check:
    from sqlalchemy import text

    from db.session import get_sessionmaker

    try:
        async with get_sessionmaker()() as db:
            version = (await _timed(db.execute(text("SHOW server_version")))).scalar_one()
        return Check("Database", OK, f"PostgreSQL {version}")
    except Exception as exc:  # noqa: BLE001
        return Check("Database", FAIL, f"cannot connect: {type(exc).__name__}: {exc}",
                     "docker compose ps postgres; docker compose logs postgres")


async def check_schema() -> Check:
    from alembic.config import Config
    from alembic.script import ScriptDirectory
    from sqlalchemy import text

    from db.session import get_sessionmaker

    try:
        head = ScriptDirectory.from_config(Config(str(REPO_ROOT / "alembic.ini"))).get_current_head()
        async with get_sessionmaker()() as db:
            current = (await _timed(db.execute(text("SELECT version_num FROM alembic_version")))).scalar()
    except Exception as exc:  # noqa: BLE001
        return Check("Database schema", FAIL, f"cannot read the schema version: {exc}",
                     "docker compose run --rm migrate")
    if current == head:
        return Check("Database schema", OK, f"up to date ({head})")
    return Check("Database schema", FAIL, f"at {current}, the code expects {head}",
                 "docker compose run --rm migrate")


async def check_redis() -> Check:
    import redis.asyncio as aredis

    r = aredis.Redis.from_url(get_settings().redis.url)
    try:
        await _timed(r.ping())
        return Check("Cache (Redis)", OK, "reachable")
    except Exception as exc:  # noqa: BLE001
        return Check("Cache (Redis)", FAIL, f"cannot connect: {type(exc).__name__}",
                     "docker compose ps redis; docker compose logs redis")
    finally:
        await r.aclose()


async def check_storage() -> Check:
    from core.storage import get_object_store

    try:
        await _timed(asyncio.to_thread(get_object_store().ensure_bucket))
        return Check("Object storage (MinIO)", OK, f"bucket '{get_settings().minio.bucket}' ready")
    except Exception as exc:  # noqa: BLE001
        return Check("Object storage (MinIO)", FAIL, f"cannot reach the bucket: {type(exc).__name__}",
                     "docker compose ps minio; check MINIO_ROOT_USER and MINIO_ROOT_PASSWORD in .env")


def _tcp(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def check_bus() -> Check:
    brokers = get_settings().redpanda.brokers
    host, _, port = brokers.split(",")[0].rpartition(":")
    up = await asyncio.to_thread(_tcp, host or "localhost", int(port or 9092))
    if up:
        return Check("Event bus (Redpanda)", OK, f"reachable at {brokers}")
    return Check("Event bus (Redpanda)", WARN,
                 f"not reachable at {brokers}: new frames are stored but not embedded, so similarity search "
                 "and active-learning diversity fall behind",
                 "docker compose ps redpanda; docker compose logs redpanda")


def check_ml_runtime() -> list[Check]:
    """torch, and whether it can see a GPU. Imported here, not at module scope, so the checks still run in
    an image without the ML libraries and can say that is the problem."""
    try:
        import torch
    except ImportError:
        return [Check("ML runtime", FAIL,
                      "this image has no torch: uploads that need plate blurring, auto-label, SAM and "
                      "embeddings cannot run",
                      "build the app image from the `gpu` target (the default in docker-compose.app.yml)",
                      group="models")]
    out = [Check("ML runtime", OK, f"torch {torch.__version__}, CUDA build {torch.version.cuda or 'none'}",
                 group="models")]
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        free_gb, total_gb = free / 2**30, total / 2**30
        status = OK if free_gb >= 4 else WARN
        out.append(Check("GPU", status, f"{name}, {free_gb:.1f} of {total_gb:.1f} GB free",
                         "" if status == OK else "another process holds most of the GPU memory; "
                                                  "nvidia-smi shows which", group="models"))
    else:
        out.append(Check("GPU", WARN, "no GPU visible: models run on the CPU, and auto-label is much slower",
                         "install the NVIDIA Container Toolkit, add docker-compose.gpu.yml to COMPOSE_FILE in "
                         ".env, then docker compose up -d", group="models"))
    return out


def _weight_present(name: str) -> bool:
    p = Path(name)
    if p.is_absolute():
        return p.exists()
    return (Path.cwd() / p).exists() or (REPO_ROOT / p).exists()


def check_weights() -> list[Check]:
    s = get_settings()
    out: list[Check] = []
    if s.pii.enabled:
        missing = [w for w in (s.pii.face_weights, s.pii.plate_weights) if not _weight_present(w)]
        out.append(Check("Privacy blur models", FAIL if missing else OK,
                         f"missing {', '.join(missing)}: uploads are refused until faces and plates can be "
                         "blurred" if missing else "face and plate detectors present",
                         "docker compose run --rm pii-models" if missing else "", group="models"))
    else:
        out.append(Check("Privacy blur models", WARN, "PII blurring is switched off (LBX_PII__ENABLED=false)",
                         "", group="models"))
    ov = s.models.openvocab
    auto = [s.models.yolo.weights, ov.detector_weights, ov.seg_weights]
    missing = [w for w in auto if not _weight_present(w)]
    out.append(Check("Auto-label models", WARN if missing else OK,
                     f"not on disk: {', '.join(missing)}; they download on first use, which needs internet"
                     if missing else f"{', '.join(auto)} present",
                     "python scripts/download_models.py" if missing else "", group="models"))
    return out


async def check_vlm() -> Check:
    vlm = get_settings().models.vlm
    if not vlm.enabled:
        return Check("Vision-language reviewer", INFO, "off (LBX_MODELS__VLM__ENABLED=false)", group="models")
    if vlm.backend != "ollama":
        return Check("Vision-language reviewer", INFO, f"backend {vlm.backend}", group="models")
    url = vlm.ollama_url.rstrip("/")

    def probe() -> list[str] | None:
        import urllib.request

        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=3) as r:  # noqa: S310 - configured URL
                return [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
        except Exception:  # noqa: BLE001
            return None

    models = await asyncio.to_thread(probe)
    if models is None:
        return Check("Vision-language reviewer", WARN,
                     f"Ollama not reachable at {url}: uncertain auto-labels go to people without a second "
                     "opinion",
                     "install Ollama on the host and let containers reach it (OLLAMA_HOST=0.0.0.0, or the "
                     "Docker bridge address, in its service environment), or set LBX_MODELS__VLM__ENABLED=false",
                     group="models")
    return Check("Vision-language reviewer", OK, f"Ollama at {url}, {len(models)} model(s) available",
                 group="models")


async def check_gpu_worker() -> list[Check]:
    """Whether anything will run queued jobs. The worker holds its advisory lock for as long as it lives, so
    the lock being held is the liveness signal, with no heartbeat table to keep."""
    from sqlalchemy import func, select, text

    from db.models import AutolabelJob
    from db.session import get_sessionmaker

    s = get_settings()
    key = s.training.advisory_lock_key
    out: list[Check] = []
    try:
        async with get_sessionmaker()() as db:
            held = (await db.execute(text(
                "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND granted "
                "AND ((classid::bigint << 32) | objid::bigint) = :k"), {"k": key})).scalar_one()
            pending = (await db.execute(
                select(func.count(), func.min(AutolabelJob.created_at))
                .where(AutolabelJob.status == "pending",
                       AutolabelJob.params["executor"].astext == "worker"))).one()
    except Exception as exc:  # noqa: BLE001
        return [Check("GPU worker", FAIL, f"cannot read job state: {exc}", group="jobs")]
    n_pending, oldest = int(pending[0]), pending[1]
    if s.jobs.autolabel_executor == "worker":
        if held:
            out.append(Check("GPU worker", OK, "running", group="jobs"))
        else:
            out.append(Check("GPU worker", FAIL,
                             "not running: auto-label and training jobs are queued but nothing runs them",
                             "docker compose up -d gpu-worker; docker compose logs gpu-worker", group="jobs"))
    else:
        out.append(Check("GPU worker", INFO,
                         "auto-label runs inside the API process (jobs.autolabel_executor=api)"
                         + ("; a worker is also running and drains training jobs" if held else ""),
                         group="jobs"))
    if n_pending:
        age_min = (time.time() - oldest.timestamp()) / 60 if oldest else 0.0
        out.append(Check("Job queue", WARN if age_min > 60 and not held else INFO,
                         f"{n_pending} auto-label job(s) waiting, oldest {age_min:.0f} min", group="jobs"))
    else:
        out.append(Check("Job queue", OK, "nothing waiting", group="jobs"))
    return out


async def run_checks() -> list[Check]:
    services = await asyncio.gather(check_postgres(), check_schema(), check_redis(), check_storage(),
                                    check_bus())
    out: list[Check] = list(services)
    out += await asyncio.to_thread(check_ml_runtime)
    out += await asyncio.to_thread(check_weights)
    out.append(await check_vlm())
    if out[0].status == OK:
        out += await check_gpu_worker()
    return out


def summarize(checks: list[Check]) -> dict:
    worst = FAIL if any(c.status == FAIL for c in checks) else WARN if any(c.status == WARN for c in checks) \
        else OK
    return {"status": worst, "checks": [asdict(c) for c in checks]}


_MARK = {OK: "ok  ", WARN: "warn", FAIL: "FAIL", INFO: "info"}


def main() -> int:
    import sys

    checks = asyncio.run(run_checks())
    if "--json" in sys.argv:
        print(json.dumps(summarize(checks), indent=1))
    else:
        width = max(len(c.name) for c in checks)
        for c in checks:
            print(f"  {_MARK[c.status]}  {c.name.ljust(width)}  {c.detail}")
            if c.fix and c.status in (WARN, FAIL):
                print(f"        {' ' * width}  fix: {c.fix}")
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
