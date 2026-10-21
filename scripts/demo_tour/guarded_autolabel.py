"""Auto-label a session in small jobs, each one started only while the resource watcher says it is safe.

    .venv/bin/python scripts/demo_tour/guarded_autolabel.py --session <uuid> --batch 20

One auto-label job over a whole session is one long allocation the watcher cannot interrupt: the first
attempt on this machine loaded its models while a browser held most of memory, the kernel's out-of-memory
killer fired, and the job died having written nothing. So the work is cut into jobs of `--batch` unlabelled
frames. Before each job this waits on the watcher's pause flag (scripts/resource_watch.py), and after each
it records how much memory the machine had left, so a run that is heading for trouble stops at a batch
boundary instead of in the middle of one.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from resource_watch import _meminfo, wait_while_paused  # noqa: E402

API = "http://127.0.0.1:8000"
POLL_S = 5.0
JOB_TIMEOUT_S = 1800.0


def _call(method: str, path: str, token: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        API + path, method=method, data=None if body is None else json.dumps(body).encode(),
        headers={"authorization": f"Bearer {token}", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _token() -> str:
    req = urllib.request.Request(API + "/api/auth/dev-login", method="POST", data=b"{}",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())["token"]


def run(session_id: str, batch: int, max_batches: int, flag: Path) -> int:
    token = _token()
    for n in range(1, max_batches + 1):
        if not wait_while_paused(flag):
            print(f"batch {n}: resources stayed over their limits, stopping: {flag.read_text()}")
            return 2
        job = _call("POST", "/api/autolabel/start", token,
                    {"session_id": session_id, "limit": batch, "only_unlabelled": True})
        deadline = time.monotonic() + JOB_TIMEOUT_S
        while True:
            time.sleep(POLL_S)
            state = _call("GET", f"/api/autolabel/{job['job_id']}", token)
            if state.get("status") in ("done", "error", "failed", "cancelled") or time.monotonic() > deadline:
                break
        free = _meminfo().get("MemAvailable", 0.0)
        counts = state.get("counts") or {}
        print(f"batch {n}: {state.get('status')} counts={json.dumps(counts)} memory available {free:.0f} MB",
              flush=True)
        # The API ends a job over a fully labelled session with status "error" and this message, which is the
        # normal way for this loop to finish rather than a failure.
        if state.get("status") == "error" and "no frames" in (state.get("error") or ""):
            print("no unlabelled frames left")
            return 0
        if state.get("status") != "done":
            print(f"batch {n}: {state.get('error')}")
            return 1
        if not counts.get("frames"):
            print("no unlabelled frames left")
            return 0
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--max-batches", type=int, default=50)
    ap.add_argument("--flag", type=Path, default=Path(".scratch/watch/PAUSE.json"))
    args = ap.parse_args()
    return run(args.session, args.batch, args.max_batches, args.flag)


if __name__ == "__main__":
    sys.exit(main())
