"""Classify a 3D track as moving, stopped, parked, turning, or braking using ego-motion-compensated velocity.

The cuboids live in the ego frame, which moves with the vehicle, so a parked object appears to slide backward
as the vehicle advances. Compensating with the CAN ego speed recovers the object's world-frame velocity: a
static object's ego-frame velocity is cancelled by the ego velocity, leaving near zero. Yaw rate gives turns;
a strong negative longitudinal acceleration of a moving object gives braking.
"""

from __future__ import annotations

import math

STOPPED_MPS = 0.5        # world speed below this is stationary
MOVING_MPS = 1.0         # world speed above this is moving
TURNING_RADPS = 0.25     # yaw rate above this (while moving) is a turn
BRAKING_MPS2 = 2.0       # longitudinal deceleration above this is braking


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def classify_track(samples: list[dict]) -> dict:
    """samples: [{ts_ns, center [x,y,z] ego metres, yaw, ego_speed m/s}] sorted by ts_ns. Returns the dynamic
    state plus the ego-compensated world speeds, so the decision is auditable."""
    s = sorted(samples, key=lambda x: x["ts_ns"])
    if len(s) < 2:
        return {"state": "stopped", "world_speeds": [], "note": "single observation"}

    world_speeds, yaw_rates, long_accels = [], [], []
    prev_fwd = None
    for a, b in zip(s[:-1], s[1:], strict=False):
        dt = (b["ts_ns"] - a["ts_ns"]) / 1e9
        if dt <= 0:
            continue
        vx = (b["center"][0] - a["center"][0]) / dt       # ego-frame velocity (x forward, y left)
        vy = (b["center"][1] - a["center"][1]) / dt
        ego = float(b.get("ego_speed") or 0.0)            # the vehicle's own forward speed
        # world velocity = ego-frame velocity plus the ego's velocity (forward in the ego frame)
        wfx, wfy = vx + ego, vy
        speed = math.hypot(wfx, wfy)
        world_speeds.append(speed)
        yaw_rates.append(_wrap(b["yaw"] - a["yaw"]) / dt)
        if prev_fwd is not None:
            long_accels.append((wfx - prev_fwd) / dt)
        prev_fwd = wfx

    if not world_speeds:
        return {"state": "stopped", "world_speeds": []}

    max_speed = max(world_speeds)
    if max_speed < STOPPED_MPS:
        state = "parked"                                   # static throughout the observed window
    else:
        tail = world_speeds[-2:]
        cur = sum(tail) / len(tail)
        tail_yaw = max((abs(r) for r in yaw_rates[-2:]), default=0.0)
        tail_acc = min(long_accels[-2:], default=0.0)
        if cur < STOPPED_MPS:
            state = "stopped"                              # was moving, now halted (e.g. at a signal)
        elif tail_yaw > TURNING_RADPS:
            state = "turning"
        elif tail_acc < -BRAKING_MPS2:
            state = "braking"
        else:
            state = "moving"

    return {"state": state, "world_speeds": [round(v, 2) for v in world_speeds],
            "max_world_speed": round(max_speed, 2)}
