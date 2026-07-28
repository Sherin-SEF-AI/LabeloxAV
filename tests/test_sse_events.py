"""Server-sent events replace polling for job and training progress.

There was no realtime anywhere: nine hardcoded setInterval loops each re-fetched a full snapshot every two or
three seconds whether or not anything had changed, with no pause when the tab was hidden and no backoff on
error. Ingest progress was scraped from a log file with a regex, which breaks as soon as the API and the
ingest script run in different containers.

Why the stream itself is tested against a real uvicorn process rather than TestClient: both
`fastapi.testclient.TestClient` and `httpx.ASGITransport` buffer a response until the ASGI app finishes, and
an SSE stream never finishes, so either harness hangs forever on an endpoint that works perfectly in
production. That is a property of the harness, not of the endpoint, and pretending otherwise would mean
either deleting the coverage or shipping a test that hangs CI.

Building this surfaced two real defects, both fixed and both guarded below:
  - `request.is_disconnected()` inside the generator deadlocked. It awaits the ASGI receive channel, which
    the BaseHTTPMiddleware-based auth layer already owns, so headers never flushed.
  - a `Depends(db_session)` on the route held a pooled connection for the entire life of a stream that never
    completes, i.e. one leaked connection per open browser tab.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from services.api.routers import events

pytestmark = pytest.mark.db


# ---------------- pure pieces (fast, no server) ----------------

def test_sse_frame_has_the_record_separator():
    # Without the trailing blank line an SSE client buffers the frame forever waiting for the record to end.
    frame = events._sse("jobs", {"a": 1})
    assert frame.startswith("event: jobs\n")
    assert frame.endswith("\n\n")
    assert json.loads(frame.split("data: ", 1)[1].strip()) == {"a": 1}


def test_sse_frame_serializes_values_json_cannot_handle_natively():
    # Job snapshots carry UUIDs and datetimes; a raw json.dumps would raise inside the generator and kill the
    # stream with no message to the client.
    import uuid
    from datetime import UTC, datetime

    frame = events._sse("jobs", {"id": uuid.uuid4(), "at": datetime.now(UTC)})
    assert "data: " in frame and frame.endswith("\n\n")


def test_the_route_does_not_take_a_request_scoped_session():
    # A request-scoped session lives until the response completes, and this response never completes, so
    # depending on one leaks a pooled connection per open tab.
    import inspect

    for fn in (events.job_events, events.training_job_events):
        params = inspect.signature(fn).parameters
        assert "db" not in params, f"{fn.__name__} must not hold a request-scoped session"


def test_the_stream_does_not_poll_is_disconnected():
    # It awaits the ASGI receive channel that the auth middleware already owns, which deadlocks before the
    # headers are ever flushed. Disconnects arrive as generator close instead.
    import ast
    import inspect

    # Walk the AST rather than grepping the text: the module docstring names the call verbatim in order to
    # explain why it is avoided, so a string match would flag the explanation itself.
    tree = ast.parse(inspect.getsource(events))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "is_disconnected"]
    assert calls == []


def test_poll_interval_is_sane():
    assert 0.5 <= events.POLL_INTERVAL_S <= 10.0


# ---------------- the live stream (real server) ----------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def live_server():
    """A real uvicorn process. See the module docstring for why an in-process harness cannot do this."""
    port = _free_port()
    env = {**os.environ, "LBX_POSTGRES__DB": os.environ.get("LBX_POSTGRES__DB", "labeloxav_test")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "services.api.main:app", "--port", str(port),
         "--log-level", "warning"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.skip("uvicorn failed to start")
            try:
                urllib.request.urlopen(f"{base}/api/health", timeout=2).read()
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.skip("uvicorn did not become ready")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _token() -> str:
    from _authutil import auth_headers

    return auth_headers("admin")["Authorization"].split()[1]


def _read_one_frame(url: str, token: str, timeout: float = 15.0) -> tuple[str, dict, dict]:
    """Open the stream, read exactly one frame, close. Bounded: the stream never ends on its own."""
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        headers = {k.lower(): v for k, v in r.headers.items()}
        buf = b""
        deadline = time.time() + timeout
        while b"\n\n" not in buf and time.time() < deadline:
            chunk = r.read(1)
            if not chunk:
                break
            buf += chunk
    raw = buf.decode().split("\n\n", 1)[0]
    event = next((ln[7:] for ln in raw.splitlines() if ln.startswith("event: ")), "")
    data = next((ln[6:] for ln in raw.splitlines() if ln.startswith("data: ")), "{}")
    return event, json.loads(data), headers


def test_job_stream_sends_a_snapshot_immediately(live_server):
    # A freshly opened page must render at once; emitting only on change would leave it blank for as long as
    # the system happened to be quiet, which is most of the time.
    event, data, headers = _read_one_frame(f"{live_server}/api/events/jobs", _token())
    assert event == "jobs"
    assert {"training", "import", "export", "autolabel"} <= set(data)
    assert headers["content-type"].startswith("text/event-stream")


def test_job_stream_disables_proxy_buffering(live_server):
    # Nginx buffers proxied responses by default, holding every frame until the buffer fills, which makes a
    # live stream arrive in bursts and look broken only in production.
    _, _, headers = _read_one_frame(f"{live_server}/api/events/jobs", _token())
    assert headers.get("x-accel-buffering") == "no"
    assert headers.get("cache-control") == "no-cache"


def test_job_stream_requires_authentication(live_server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{live_server}/api/events/jobs", timeout=10)
    assert exc.value.code == 401


def test_stream_accepts_a_token_in_the_query_string(live_server):
    # The browser EventSource API cannot set a request header, so this is the only way a stream can present a
    # credential. Without it the whole realtime path is unreachable from a browser.
    req = urllib.request.Request(f"{live_server}/api/events/jobs?token={_token()}")
    with urllib.request.urlopen(req, timeout=15) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        assert b"event: jobs" in r.read(64)


def test_a_query_token_does_not_authenticate_a_normal_data_route(live_server):
    # The query form is scoped to the event streams, which carry job progress and nothing sensitive, because
    # a URL reaches proxy and access logs in a way a header does not. It must not become a general
    # credential channel.
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{live_server}/api/ontology?token={_token()}", timeout=10)
    assert exc.value.code == 401


def test_a_forged_query_token_is_refused(live_server):
    # Same verification as everywhere else: signature, expiry, and the per-user revocation counter. A
    # different transport, not a weaker check.
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(f"{live_server}/api/events/jobs?token=lbx2.forged.forged", timeout=10)
    assert exc.value.code == 401


def test_training_stream_reports_a_real_job(live_server):
    import asyncio
    import uuid as uuidlib

    from db.models import TrainingJob
    from db.session import get_engine, get_sessionmaker

    async def _seed() -> str:
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
        async with get_sessionmaker()() as db:
            j = TrainingJob(purpose="sse-test", task_type="detection", compute_target="local",
                            status="done", stage="done", progress=1.0, config={},
                            counts={"epoch": 5}, metrics={"live": {"map50": 0.42}})
            db.add(j)
            await db.commit()
            return str(j.job_id)

    job_id = asyncio.run(_seed())
    event, data, _ = _read_one_frame(f"{live_server}/api/events/training/{job_id}", _token())
    assert event == "training"
    assert data["job_id"] == job_id and data["status"] == "done"
    assert data["metrics"]["live"]["map50"] == 0.42

    # cleanup so the row does not linger in the shared test database
    async def _drop():
        get_engine.cache_clear()
        get_sessionmaker.cache_clear()
        async with get_sessionmaker()() as db:
            row = await db.get(TrainingJob, uuidlib.UUID(job_id))
            if row:
                await db.delete(row)
                await db.commit()

    asyncio.run(_drop())


def test_training_stream_reports_a_missing_job_instead_of_hanging(live_server):
    import uuid as uuidlib

    event, data, _ = _read_one_frame(f"{live_server}/api/events/training/{uuidlib.uuid4()}", _token())
    assert event == "error" and "not found" in data["detail"]


def test_training_stream_rejects_a_malformed_id(live_server):
    # A bad id must not raise inside the generator, which would surface as a dead stream with no message.
    event, data, _ = _read_one_frame(f"{live_server}/api/events/training/not-a-uuid", _token())
    assert event == "error" and "invalid" in data["detail"]
