"""Call every readable API route against the running system and report what answers.

Written because "everything works" is otherwise an assertion nobody has checked. The test suite covers
units and a few end-to-end paths; it does not answer whether all 352 GET routes still respond on a live
database with real data in it. A route that raises a 500 on the corpus as it actually is will never be
caught by a fixture.

Only GET is called. A blind sweep of 373 POST routes would train models, launch cloud jobs, delete
objects and spend money, so writes are listed as skipped rather than fired. Path parameters are filled
from the live database where the name is one this corpus can supply, and the rest are skipped by name so
the report says what was not covered instead of quietly reporting a smaller denominator.

A 4xx is not a failure here. A route that answers 404 for an id that does not exist, or 403 without a
role, is working. A 5xx is a failure, and so is a timeout or a body that does not parse.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from collections import Counter
from pathlib import Path

import httpx

API = "http://localhost:8000"

# How long one path-parameter lookup may take before it is abandoned.
FILL_TIMEOUT_S = 8.0

# Path parameter names this corpus can fill, mapped to the query that finds one.
FILLERS: dict[str, str] = {
    "session_id": "select session_id from session where origin='real' order by created_at desc limit 1",
    "session": "select session_id from session where origin='real' order by created_at desc limit 1",
    "frame_id": "select frame_id from frame order by created_at desc limit 1",
    "object_id": "select object_id from object order by created_at desc limit 1",
    "track_id": "select track_id from object where track_id is not null limit 1",
    "cloud_id": "select cloud_id from point_cloud order by created_at desc limit 1",
    "commit_id": "select commit_id from map_commit limit 1",
    "element_id": "select element_id from map_element limit 1",
    "job_id": "select job_id from training_job order by created_at desc limit 1",
    "run_id": "select run_id from agent_run order by created_at desc limit 1",
    "model_version": "select version from model_registry order by created_at desc limit 1",
    "user_id": "select user_id from \"user\" limit 1",
    "project_id": "select project_id from project limit 1",
    "dataset_id": "select dataset_id from dataset order by created_at desc limit 1",
    "batch_id": "select distinct batch_id from machine_verdict where batch_id is not null limit 1",
    "gold_id": "select gold_id from gold_set order by created_at desc limit 1",
    # Scoped to recent frames rather than the whole object table. Unscoped this is a scan of 600,000
    # rows on a JSON path with no index supporting it, and it hung the sweep before it had sent a single
    # request: the symptom was a process alive for half an hour with the API log showing only the web
    # UI's own polling.
    "cycle_id": "select provenance->'flywheel'->>'cycle_id' from object"
                " where provenance->'flywheel'->>'cycle_id' is not null"
                " and frame_id in (select frame_id from frame order by created_at desc limit 2000)"
                " limit 1",
}


async def fill() -> dict[str, str]:
    """One real id per fillable parameter name, or absent if the corpus has none."""
    from sqlalchemy import text

    from db.session import get_sessionmaker

    out: dict[str, str] = {}
    async with get_sessionmaker()() as db:
        for name, sql in FILLERS.items():
            # Each lookup is bounded. A filler that cannot answer quickly is one this corpus cannot
            # cheaply supply, and the routes needing it are better reported as skipped than allowed to
            # stall the whole sweep.
            try:
                res = await asyncio.wait_for(db.execute(text(sql)), timeout=FILL_TIMEOUT_S)
                v = res.scalar()
            except TimeoutError:
                print(f"  filler {name}: gave up after {FILL_TIMEOUT_S:.0f}s; routes needing it are skipped")
                await db.rollback()
                continue
            except Exception as exc:
                print(f"  filler {name}: {type(exc).__name__}")
                await db.rollback()
                continue
            if v is not None:
                out[name] = str(v)
    return out


def routes(spec: dict) -> list[tuple[str, dict]]:
    return sorted((p, v["get"]) for p, v in spec["paths"].items() if "get" in v)


def resolve(path: str, ids: dict[str, str]) -> tuple[str | None, str]:
    """The concrete URL to call, or None and the reason it cannot be built."""
    names = re.findall(r"\{([^}]+)\}", path)
    missing = [n for n in names if n not in ids]
    if missing:
        return None, f"no id for {', '.join(sorted(set(missing)))}"
    out = path
    for n in names:
        out = out.replace("{" + n + "}", ids[n])
    return out, ""


async def sweep(base: str, token: str | None, timeout: float, out_path: Path,
                concurrency: int = 6) -> int:
    ids = await fill()
    print(f"filled {len(ids)} path parameters from the corpus: {', '.join(sorted(ids))}\n")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(base_url=base, timeout=timeout, headers=headers) as c:
        spec = (await c.get("/openapi.json")).json()
        rs = routes(spec)
        results: list[dict] = []
        # Concurrent, because serially this does not finish. Several analytics routes take fifteen
        # seconds each against the real corpus, and 352 of them in a row exceeded any sensible timeout.
        # Six at a time is modest enough not to be a load test of the database by accident.
        sem = asyncio.Semaphore(concurrency)
        done = [0]

        async def call(path: str) -> None:
            url, why = resolve(path, ids)
            if url is None:
                results.append({"path": path, "status": None, "ms": 0, "outcome": "skipped", "note": why})
                return
            async with sem:
                t0 = time.time()
                try:
                    r = await c.get(url)
                    ms = int((time.time() - t0) * 1000)
                    # A 4xx is a working route answering about data that is not there, or a guard doing
                    # its job. Only a 5xx, a timeout or an unparseable body is this sweep's business.
                    if r.status_code >= 500:
                        outcome, note = "server error", (r.text or "")[:160].replace("\n", " ")
                    elif r.status_code >= 400:
                        outcome, note = "refused", f"HTTP {r.status_code}"
                    else:
                        outcome, note = "ok", ""
                        ct = r.headers.get("content-type", "")
                        if ct.startswith("application/json"):
                            try:
                                r.json()
                            except Exception:
                                outcome, note = "bad body", "200 with a body that is not the JSON it claims"
                    results.append({"path": path, "url": url, "status": r.status_code, "ms": ms,
                                    "outcome": outcome, "note": note})
                except Exception as exc:
                    results.append({"path": path, "url": url, "status": None,
                                    "ms": int((time.time() - t0) * 1000), "outcome": "error",
                                    "note": f"{type(exc).__name__}: {str(exc)[:120]}"})
            done[0] += 1
            if done[0] % 40 == 0:
                print(f"  {done[0]}/{len(rs)} ...")

        await asyncio.gather(*(call(p) for p, _op in rs))
        results.sort(key=lambda r: r["path"])

    writes = sum(len([m for m in v if m != "get"]) for v in spec["paths"].values())
    tally = Counter(r["outcome"] for r in results)
    print("\nGET routes swept:", len(results))
    for k in ("ok", "refused", "skipped", "bad body", "server error", "error"):
        if tally[k]:
            print(f"  {k:>13}  {tally[k]}")
    print(f"  {'not called':>13}  {writes}  (POST, PUT, PATCH, DELETE: a blind sweep would change data)")

    broken = [r for r in results if r["outcome"] in ("server error", "error", "bad body")]
    if broken:
        print(f"\n{len(broken)} routes failed:")
        for r in broken:
            print(f"  {r['path']}\n      {r['outcome']}: {r['note']}")
    else:
        print("\nNo route returned a server error, timed out, or sent a body it could not parse.")

    slow = sorted((r for r in results if r["outcome"] == "ok"), key=lambda r: -r["ms"])[:8]
    if slow:
        print("\nSlowest answering routes:")
        for r in slow:
            print(f"  {r['ms']:>6} ms  {r['path']}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"base": base, "results": results,
                                    "tally": dict(tally), "write_routes_not_called": writes}, indent=2))
    print(f"\nwrote {out_path}")
    return len(broken)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=API)
    ap.add_argument("--token", default=None)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--out", default=".scratch/verify/api_smoke.json")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    raise SystemExit(1 if asyncio.run(
        sweep(a.base, a.token, a.timeout, Path(a.out), concurrency=a.concurrency)) else 0)


if __name__ == "__main__":
    main()
