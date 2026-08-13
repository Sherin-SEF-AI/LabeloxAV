"""What the machine is doing right now: GPU, memory, CPU, disk, and this process's own share.

Everything in this system that can be slow is slow for one of a handful of reasons, and none of them were
visible from inside the app. An auto-label that will not start because a training job holds the GPU, an
export that is crawling because the disk is full, a machine that looks idle because the work is parked for
hardware that is not here: each of those was a question you answered by opening a terminal, and most people
using this do not have one.

Read at request time rather than sampled into a table. These numbers are worth nothing five minutes old, and
a metrics table is a second thing to keep correct; the stream that carries them polls, so the cost is one
read per tick shared by every connected client rather than one per viewer.

`nvidia-smi` rather than torch for the GPU. `core/gpu_slot.cuda_report` already answers "can the API see a
GPU", which is a question about this process and the right one for autolabel's refusal message. This is the
other question: what is on the card, including the processes that are not us. A training worker, an Ollama
model held resident, and this API are three different tenants of one 16 GB card, and the tenant you cannot
see is the one that breaks your run.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

from core.logging import get_logger

log = get_logger("resources")

# Long enough that a hung driver cannot stall a request, short enough to be invisible when healthy.
_SMI_TIMEOUT_S = 3
_SMI_QUERY = "index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw"
_SMI_PROC = "pid,process_name,used_memory"


def _smi(args: list[str]) -> list[list[str]]:
    """Run nvidia-smi and return parsed CSV rows. An empty list means no GPU, or no driver, or no answer."""
    try:
        out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def gpus() -> list[dict]:
    """Every visible GPU, with the processes holding memory on it.

    The processes matter more than the totals. "7.2 GB used" is not actionable; "llama-server is holding
    7.2 GB" is, and it is the difference between waiting and killing something.
    """
    rows = _smi(["--query-gpu=" + _SMI_QUERY])
    if not rows:
        return []
    procs_by_gpu: dict[int, list[dict]] = {}
    for p in _smi(["--query-compute-apps=" + _SMI_PROC]):
        if len(p) < 3:
            continue
        # nvidia-smi does not report which GPU a compute app is on in this query, so on a single-card box
        # they all belong to device 0. Reported honestly rather than guessed at on a multi-card host.
        procs_by_gpu.setdefault(0, []).append(
            {"pid": int(p[0]) if p[0].isdigit() else None, "name": p[1], "used_mb": _num(p[2])})

    out = []
    for r in rows:
        if len(r) < 7:
            continue
        idx = int(r[0]) if r[0].isdigit() else 0
        used, total = _num(r[2]), _num(r[3])
        out.append({
            "index": idx, "name": r[1],
            "memory_used_mb": used, "memory_total_mb": total,
            "memory_used_frac": round(used / total, 4) if used is not None and total else None,
            "utilization_pct": _num(r[4]), "temperature_c": _num(r[5]), "power_w": _num(r[6]),
            "processes": sorted(procs_by_gpu.get(idx, []), key=lambda p: -(p["used_mb"] or 0)),
        })
    return out


def host() -> dict:
    """CPU, memory and the disk the scratch space lives on."""
    import psutil

    vm = psutil.virtual_memory()
    # The scratch directory rather than root: it is where model weights, exports and frame caches land, and
    # it is the one that fills.
    scratch = os.path.abspath(".scratch")
    disk_path = scratch if os.path.isdir(scratch) else os.path.abspath(".")
    du = shutil.disk_usage(disk_path)
    load1, load5, load15 = os.getloadavg()
    return {
        "cpu_pct": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(logical=True),
        "load": {"1m": round(load1, 2), "5m": round(load5, 2), "15m": round(load15, 2)},
        "memory_used_mb": round((vm.total - vm.available) / 1e6),
        "memory_total_mb": round(vm.total / 1e6),
        "memory_used_frac": round(1 - vm.available / vm.total, 4),
        "disk": {"path": disk_path, "used_mb": round(du.used / 1e6),
                 "total_mb": round(du.total / 1e6),
                 "used_frac": round(du.used / du.total, 4) if du.total else None},
    }


def process() -> dict:
    """This API process's own share, so "the machine is busy" can be told from "we are busy"."""
    import psutil

    p = psutil.Process()
    with p.oneshot():
        return {
            "pid": p.pid,
            "rss_mb": round(p.memory_info().rss / 1e6),
            "cpu_pct": p.cpu_percent(interval=None),
            "threads": p.num_threads(),
            "uptime_s": round(time.time() - p.create_time()),
        }


def snapshot() -> dict:
    """Everything at once. Each part is independent: a missing GPU must not cost you the memory reading."""
    out: dict = {"ts": time.time()}
    for key, fn in (("gpus", gpus), ("host", host), ("process", process)):
        try:
            out[key] = fn()
        except Exception as exc:  # noqa: BLE001 -- a console that dies on one bad reading shows nothing
            log.warning("resources.read_failed", part=key, error=str(exc))
            out[key] = [] if key == "gpus" else {}
    return out
