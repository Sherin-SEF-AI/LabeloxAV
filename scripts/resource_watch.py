"""Watch CPU, memory, swap, GPU and disk, and raise a pause flag before the machine runs out.

    .venv/bin/python scripts/resource_watch.py --out .scratch/watch/resources.jsonl --interval 5
    .venv/bin/python scripts/resource_watch.py --once          # one sample to stdout, exit 0

Long recording and rendering runs share this workstation with a browser, a database and whatever else the
person at the keyboard is doing. The failure that matters is not a slow run, it is the out-of-memory killer
taking the database or the desktop with it. So this samples the host every few seconds, appends one JSON
object per sample, and maintains a flag file: while a resource is over its limit the flag exists and holds
the reason, and long jobs call `wait_while_paused` between batches instead of pressing on.

The flag is raised on one bad sample and cleared only after `CLEAR_SAMPLES` consecutive good ones at a
looser bound, because a run that resumes the instant memory dips back under the line just trips the line
again. The reason string names the resource and the number, so the journal says why work stopped rather
than leaving a gap.

A flag only helps work that reads it, and the work that nearly always causes the trouble does not: a model
job running inside the API, or a local vision model it calls, cannot be paused between batches from outside.
On this machine one of those took free memory from 15 GB to under 1 GB while the flag was up, swap was
already full, and the desktop froze hard enough to need a reset. So below `CRITICAL_MEM_MB` the watcher
stops asking and acts, once per episode: it asks Ollama to unload every resident model, which frees its
memory at once and costs only a reload on next use, and then sends SIGTERM to the processes listed in the
guard file. That file holds only processes the operator started for this work and can restart, one pid per
line; nothing else on the machine is ever signalled.

GPU memory is read through libcuda rather than nvidia-smi. The two are not interchangeable here: when the
driver's kernel module and its userspace libraries are different builds, which happens after an unattended
driver upgrade and until the next reboot, NVML fails and nvidia-smi prints nothing while CUDA itself keeps
working. A watcher that believed nvidia-smi would report no GPU on a machine that has one and is using it.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import shutil
import signal
import sys
import time
import urllib.request
from pathlib import Path

# Limits. Raise the flag at LIMIT, clear it at CLEAR, which is deliberately more generous.
MEM_FREE_MB = 1400.0
MEM_FREE_CLEAR_MB = 2600.0
SWAP_FREE_FRAC = 0.25
SWAP_FREE_CLEAR_FRAC = 0.40
# Exhausted swap is only dangerous once memory is also thin: a desktop that has been up for weeks parks
# idle browser tabs in swap and never gets them back, which is not a reason to stop work. The pairing
# threshold sits above the memory limit itself, so exhausted swap tightens the belt before memory alone does.
SWAP_WITH_MEM_FREE_MB = 2500.0
GPU_FREE_MB = 900.0
GPU_FREE_CLEAR_MB = 1800.0
DISK_FREE_GB = 12.0
DISK_FREE_CLEAR_GB = 18.0
LOAD_PER_CPU = 3.0
LOAD_PER_CPU_CLEAR = 2.0
CLEAR_SAMPLES = 3
# Below this the machine is minutes from thrashing, and waiting for work to notice a flag is too slow.
CRITICAL_MEM_MB = 900.0
OLLAMA = "http://127.0.0.1:11434"

# How often to pay for a CUDA context. Every sample would be wasteful; the GPU does not fill in seconds.
GPU_EVERY_S = 30.0


def _meminfo() -> dict[str, float]:
    """/proc/meminfo in MB. MemAvailable is the only honest answer to "can I allocate"."""
    out: dict[str, float] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            out[key] = float(parts[0]) / 1024.0
    return out


def _cpu_sample() -> tuple[float, float]:
    """(busy, total) jiffies from /proc/stat, for a percentage over an interval rather than since boot."""
    fields = [float(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    idle = fields[3] + (fields[4] if len(fields) > 4 else 0.0)
    total = sum(fields)
    return total - idle, total


def gpu_free_mb() -> tuple[float | None, float | None]:
    """Free and total device memory in MB, or (None, None) when there is no usable CUDA driver.

    Retains the primary context for the call and releases it again, so the watcher does not sit on device
    memory between samples.
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None, None
    dev = ctypes.c_int()
    ctx = ctypes.c_void_p()
    free = ctypes.c_size_t()
    total = ctypes.c_size_t()
    try:
        if lib.cuInit(0) != 0 or lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return None, None
        if lib.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != 0:
            return None, None
        try:
            if lib.cuCtxSetCurrent(ctx) != 0 or lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)) != 0:
                return None, None
            return free.value / 2**20, total.value / 2**20
        finally:
            lib.cuCtxSetCurrent(None)
            lib.cuDevicePrimaryCtxRelease_v2(dev)
    except (AttributeError, OSError):
        return None, None


def top_rss(n: int = 5) -> list[dict]:
    """The n processes holding the most resident memory, so a breach names its cause."""
    rows: list[dict] = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            statm = (proc / "statm").read_text().split()
            comm = (proc / "comm").read_text().strip()
        except (OSError, ValueError):
            continue
        rows.append({"pid": int(proc.name), "comm": comm, "rss_mb": round(int(statm[1]) * 4096 / 2**20, 1)})
    rows.sort(key=lambda r: r["rss_mb"], reverse=True)
    return rows[:n]


def unload_ollama() -> list[str]:
    """Ask a local Ollama to drop every resident model. Returns the names it was holding."""
    try:
        with urllib.request.urlopen(OLLAMA + "/api/ps", timeout=3) as resp:
            names = [m["name"] for m in json.loads(resp.read()).get("models", [])]
    except (OSError, ValueError):
        return []
    for name in names:
        body = json.dumps({"model": name, "keep_alive": 0}).encode()
        req = urllib.request.Request(OLLAMA + "/api/generate", data=body, method="POST",
                                     headers={"content-type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10).read()
        except OSError:
            pass
    return names


def stop_guarded(guard: Path | None) -> list[int]:
    """SIGTERM every live pid listed in the guard file, and nothing else."""
    if guard is None or not guard.exists():
        return []
    stopped = []
    for line in guard.read_text().split():
        try:
            pid = int(line)
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except (ValueError, ProcessLookupError, PermissionError):
            continue
    return stopped


class Watch:
    """One sampling loop with hysteresis on the pause flag."""

    def __init__(self, *, flag: Path, disk_path: Path, guard: Path | None = None) -> None:
        self.flag = flag
        self.disk_path = disk_path
        self.guard = guard
        self._acted = False
        self.cpus = os.cpu_count() or 1
        self._cpu_prev = _cpu_sample()
        self._gpu_at = 0.0
        self._gpu: tuple[float | None, float | None] = (None, None)
        self._good = 0

    def sample(self) -> dict:
        mem = _meminfo()
        busy, total = _cpu_sample()
        d_busy, d_total = busy - self._cpu_prev[0], total - self._cpu_prev[1]
        self._cpu_prev = (busy, total)
        now = time.time()
        if now - self._gpu_at >= GPU_EVERY_S:
            self._gpu = gpu_free_mb()
            self._gpu_at = now
        usage = shutil.disk_usage(self.disk_path)
        swap_total = mem.get("SwapTotal", 0.0)
        return {
            "ts": round(now, 3),
            "cpu_pct": round(100.0 * d_busy / d_total, 1) if d_total > 0 else None,
            "load1": os.getloadavg()[0],
            "cpus": self.cpus,
            "mem_free_mb": round(mem.get("MemAvailable", 0.0), 1),
            "mem_total_mb": round(mem.get("MemTotal", 0.0), 1),
            "swap_free_mb": round(mem.get("SwapFree", 0.0), 1),
            "swap_total_mb": round(swap_total, 1),
            "gpu_free_mb": None if self._gpu[0] is None else round(self._gpu[0], 1),
            "gpu_total_mb": None if self._gpu[1] is None else round(self._gpu[1], 1),
            "disk_free_gb": round(usage.free / 2**30, 2),
        }

    def breaches(self, s: dict, *, clearing: bool) -> list[str]:
        """Reasons this sample is over the line. `clearing` applies the looser bounds."""
        mem_lim = MEM_FREE_CLEAR_MB if clearing else MEM_FREE_MB
        gpu_lim = GPU_FREE_CLEAR_MB if clearing else GPU_FREE_MB
        disk_lim = DISK_FREE_CLEAR_GB if clearing else DISK_FREE_GB
        swap_lim = SWAP_FREE_CLEAR_FRAC if clearing else SWAP_FREE_FRAC
        load_lim = (LOAD_PER_CPU_CLEAR if clearing else LOAD_PER_CPU) * self.cpus
        out: list[str] = []
        if s["mem_free_mb"] < mem_lim:
            out.append(f"memory available {s['mem_free_mb']:.0f} MB under {mem_lim:.0f} MB")
        if (s["swap_total_mb"] > 0 and s["swap_free_mb"] / s["swap_total_mb"] < swap_lim
                and s["mem_free_mb"] < SWAP_WITH_MEM_FREE_MB):
            out.append(f"swap free {100 * s['swap_free_mb'] / s['swap_total_mb']:.0f}% under {100 * swap_lim:.0f}%"
                       f" with memory available {s['mem_free_mb']:.0f} MB")
        if s["gpu_free_mb"] is not None and s["gpu_free_mb"] < gpu_lim:
            out.append(f"gpu memory free {s['gpu_free_mb']:.0f} MB under {gpu_lim:.0f} MB")
        if s["disk_free_gb"] < disk_lim:
            out.append(f"disk free {s['disk_free_gb']:.1f} GB under {disk_lim:.0f} GB")
        if s["load1"] > load_lim:
            out.append(f"load {s['load1']:.1f} over {load_lim:.0f} on {self.cpus} cpus")
        return out

    def step(self, s: dict) -> dict:
        """Update the flag from one sample, and act if memory is critical. Returns the annotated sample."""
        if s["mem_free_mb"] < CRITICAL_MEM_MB:
            if not self._acted:
                s["acted"] = {"ollama_unloaded": unload_ollama(), "sigterm": stop_guarded(self.guard),
                              "top_rss": top_rss()}
                self._acted = True
        elif s["mem_free_mb"] > MEM_FREE_CLEAR_MB:
            self._acted = False
        paused = self.flag.exists()
        reasons = self.breaches(s, clearing=paused)
        if reasons:
            self._good = 0
            payload = {"at": s["ts"], "reasons": reasons, "top_rss": top_rss()}
            self.flag.write_text(json.dumps(payload, indent=1))
            s["paused"] = True
            s["reasons"] = reasons
        elif paused:
            self._good += 1
            s["paused"] = True
            s["reasons"] = [f"recovering {self._good}/{CLEAR_SAMPLES}"]
            if self._good >= CLEAR_SAMPLES:
                self.flag.unlink(missing_ok=True)
                self._good = 0
                s["paused"] = False
                s["reasons"] = []
        else:
            s["paused"] = False
            s["reasons"] = []
        return s


def wait_while_paused(flag: Path, *, poll_s: float = 5.0, timeout_s: float = 900.0) -> bool:
    """Block while the flag is up. False means it was still up when the timeout ran out.

    This is the half that callers use. A batch loop calls it between batches, so work stops at a boundary
    where stopping is safe rather than in the middle of a write.
    """
    deadline = time.monotonic() + timeout_s
    while flag.exists():
        if time.monotonic() > deadline:
            return False
        time.sleep(poll_s)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(".scratch/watch/resources.jsonl"))
    ap.add_argument("--flag", type=Path, default=None, help="pause flag path, default <out dir>/PAUSE.json")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--disk", type=Path, default=Path("."), help="filesystem to watch for free space")
    ap.add_argument("--once", action="store_true", help="print one sample and exit")
    ap.add_argument("--guard", type=Path, default=None,
                    help="file of pids (one per line) to SIGTERM when memory goes critical, default <out dir>/guard.pids")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    flag = args.flag or args.out.parent / "PAUSE.json"
    watch = Watch(flag=flag, disk_path=args.disk, guard=args.guard or args.out.parent / "guard.pids")

    if args.once:
        time.sleep(0.2)
        print(json.dumps(watch.step(watch.sample())))
        return 0

    with args.out.open("a", buffering=1) as fh:
        while True:
            time.sleep(args.interval)
            try:
                s = watch.step(watch.sample())
            except OSError as exc:
                s = {"ts": round(time.time(), 3), "error": str(exc)}
            fh.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    sys.exit(main())
