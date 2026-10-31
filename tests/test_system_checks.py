"""The installation checks an administrator sees in the app, and that `scripts/doctor.sh` prints.

Each failing check must say how to fix it, because the reason this exists is that a symptom (an upload
refused, a job bar stuck at zero) never names its cause.
"""

from __future__ import annotations

import pytest

from services.doctor import FAIL, OK, WARN, Check, run_checks, summarize

pytestmark = pytest.mark.db


async def test_every_check_runs_and_every_problem_names_a_fix():
    checks = await run_checks()
    names = {c.name for c in checks}
    assert {"Database", "Database schema", "Cache (Redis)", "Object storage (MinIO)", "ML runtime",
            "GPU worker"} <= names
    for c in checks:
        assert c.status in ("ok", "warn", "fail", "info"), c
        if c.status in (WARN, FAIL) and c.name not in ("Job queue", "Privacy blur models"):
            assert c.fix, f"{c.name} reports {c.status} without saying how to fix it"


async def test_the_schema_this_suite_runs_on_is_current():
    checks = {c.name: c for c in await run_checks()}
    assert checks["Database schema"].status == OK, checks["Database schema"].detail


async def test_a_missing_worker_is_a_failure_only_when_jobs_depend_on_it(monkeypatch):
    from core.config import get_settings
    from services.doctor import check_gpu_worker

    api_mode = {c.name: c for c in await check_gpu_worker()}
    assert api_mode["GPU worker"].status in ("info", "ok")
    monkeypatch.setenv("LBX_JOBS__AUTOLABEL_EXECUTOR", "worker")
    get_settings.cache_clear()
    worker_mode = {c.name: c for c in await check_gpu_worker()}
    # No worker runs during the suite, so in worker mode nothing would run queued jobs.
    assert worker_mode["GPU worker"].status == FAIL
    assert "gpu-worker" in worker_mode["GPU worker"].fix


def test_the_overall_status_is_the_worst_one():
    assert summarize([Check("a", OK, ""), Check("b", WARN, "x", "y")])["status"] == WARN
    assert summarize([Check("a", WARN, "x", "y"), Check("b", FAIL, "x", "y")])["status"] == FAIL
    assert summarize([Check("a", OK, "")])["status"] == OK


def test_only_an_administrator_can_read_them():
    from _authutil import auth_headers
    from fastapi.testclient import TestClient

    from services.api.main import app

    with TestClient(app) as c:
        assert c.get("/api/system/checks", headers=auth_headers("annotator")).status_code == 403
        r = c.get("/api/system/checks", headers=auth_headers("admin"))
        assert r.status_code == 200
        body = r.json()
        assert body["status"] in ("ok", "warn", "fail") and body["checks"]
