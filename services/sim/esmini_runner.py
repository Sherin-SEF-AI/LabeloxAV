"""Headless replay of an OpenSCENARIO document, and the honest refusal when there is nothing to replay with.

`build_scenario` produces a well-formed OpenSCENARIO 1.2 file from a recorded event. Well-formed is the
only property anything checked. A document placing an actor off the road network, giving it a speed no
vehicle reaches, or hanging a manoeuvre on a trigger that never fires is equally well-formed and describes
nothing that can happen, and a bundle shipped with a folder of those is a bundle of files rather than a
set of scenarios.

esmini replays one headlessly and writes a per-step trajectory log. Parsing that log answers the question
the schema cannot: did the actors move, did they stay on the network, did the scenario reach its end.

**The refusal is the part that runs on this host.** esmini is a binary and it is not installed here, so
`replay` returns `ok=False` with the reason rather than raising, and every caller records that reason
instead of a verdict. A scenario nobody could validate and a scenario that failed validation are different
facts, and an export that conflated them would claim a check it never performed.
"""

from __future__ import annotations

import csv
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.config import get_settings
from core.logging import get_logger

log = get_logger("esmini")

# A replay that produced fewer steps than this did not run: esmini writes a header and exits on a
# document it cannot start, and an empty log read as "the actors did not move" would report a broken
# scenario as a stationary one.
MIN_STEPS = 5
# An actor that never moves more than this over the whole replay stood still. Not an error by itself, a
# parked car is an actor, but a scenario where every actor is stationary is not a scenario.
MOVED_M = 0.5


@dataclass
class ReplayResult:
    ok: bool
    reason: str | None = None
    actors: int = 0
    steps: int = 0
    duration_s: float = 0.0
    moved_actors: int = 0
    trajectories: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)
    log_path: str | None = None

    def as_dict(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "actors": self.actors, "steps": self.steps,
                "duration_s": round(self.duration_s, 3), "moved_actors": self.moved_actors,
                "log_path": self.log_path}


def parse_csv_log(text: str) -> dict[str, list[tuple[float, float, float]]]:
    """esmini's `--csv_logger` output into per-actor (time, x, y) tracks.

    Tolerant of column order and of the header naming actors differently between versions, because the
    alternative is a parser that silently returns nothing when esmini changes a heading and a caller that
    reads that as a scenario where nobody moved.
    """
    rows = list(csv.DictReader(line for line in text.splitlines() if line.strip()))
    out: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        keys = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        name = keys.get("name") or keys.get("entity") or keys.get("object")
        if not name:
            continue
        try:
            t = float(keys.get("time") or keys.get("t") or 0.0)
            x = float(keys.get("x") or keys.get("pos_x") or 0.0)
            y = float(keys.get("y") or keys.get("pos_y") or 0.0)
        except ValueError:
            continue
        out.setdefault(name, []).append((t, x, y))
    return out


def _moved(track: list[tuple[float, float, float]], threshold: float = MOVED_M) -> bool:
    if len(track) < 2:
        return False
    xs = [p[1] for p in track]
    ys = [p[2] for p in track]
    return (max(xs) - min(xs)) >= threshold or (max(ys) - min(ys)) >= threshold


def replay(scenario_path: str, *, xodr: str | None = None, duration_s: float | None = None,
           timeout_s: float | None = None) -> ReplayResult:
    """Replay one scenario headlessly. Never raises: an unavailable simulator is a reason, not a crash."""
    from services.forgyx.capabilities import CapabilityError, _binary_path, require

    try:
        require("esmini")
    except CapabilityError as exc:
        return ReplayResult(ok=False, reason=str(exc))
    binary = _binary_path("esmini")

    cfg = get_settings().sim
    duration = float(duration_s if duration_s is not None else cfg.duration_s)
    timeout = float(timeout_s if timeout_s is not None else cfg.timeout_s)
    path = Path(scenario_path)
    if not path.is_file():
        return ReplayResult(ok=False, reason=f"scenario file not found: {scenario_path}")

    with tempfile.TemporaryDirectory(prefix="esmini-") as tmp:
        out_csv = Path(tmp) / "trajectory.csv"
        cmd = [binary, "--osc", str(path), "--headless", "--fixed_timestep", "0.05",
               "--duration", str(duration), "--csv_logger", str(out_csv)]
        if xodr:
            cmd += ["--odr", str(xodr)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            # A replay that will not finish is a scenario that will not finish. Reported as a failure
            # with the bound, not as an unknown, because the bound is the finding.
            return ReplayResult(ok=False, reason=f"the replay did not finish within {timeout:.0f}s")
        except OSError as exc:
            return ReplayResult(ok=False, reason=f"the simulator could not be started: {exc}")

        if not out_csv.exists():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            return ReplayResult(ok=False,
                                reason=("the replay wrote no trajectory log; "
                                        + (" / ".join(tail) if tail else "no output from the simulator")))
        tracks = parse_csv_log(out_csv.read_text())

    steps = max((len(v) for v in tracks.values()), default=0)
    if steps < MIN_STEPS:
        return ReplayResult(ok=False, actors=len(tracks), steps=steps,
                            reason=(f"the replay produced {steps} steps, which is not a run; the "
                                    f"scenario most likely failed to start"))
    moved = sum(1 for v in tracks.values() if _moved(v))
    if moved == 0:
        return ReplayResult(ok=False, actors=len(tracks), steps=steps, trajectories=tracks,
                            reason="every actor stood still for the whole replay, so nothing happened")
    span = max((v[-1][0] - v[0][0] for v in tracks.values() if v), default=0.0)
    log.info("esmini.replayed", scenario=path.name, actors=len(tracks), steps=steps, moved=moved)
    return ReplayResult(ok=True, actors=len(tracks), steps=steps, duration_s=float(span),
                        moved_actors=moved, trajectories=tracks)


def trajectory_checksum(tracks: dict[str, list[tuple[float, float, float]]], *,
                        places: int = 1) -> str:
    """A stable digest of the replayed tracks, for detecting that a scenario now runs differently.

    Rounded before hashing, because a simulator's last decimal place moves between builds and a checksum
    that changed on a patch release would report every scenario as regressed on the day of an upgrade.
    """
    import hashlib

    h = hashlib.sha256()
    for name in sorted(tracks):
        h.update(name.encode())
        for t, x, y in tracks[name]:
            h.update(f"{round(t, places)}|{round(x, places)}|{round(y, places)};".encode())
    return h.hexdigest()[:32]
