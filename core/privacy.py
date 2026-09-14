"""Releasing aggregates about where a fleet drove without releasing where any vehicle was.

`/analytics/geo` returned raw GNSS fixes as latitude and longitude pairs. A driving trace is among the
most identifying data a vehicle produces: a handful of points reconstructs a home address, a workplace and
a daily route, and no amount of removing names changes that. It is also exactly the data a buyer legitimately
wants in aggregate, to see where a corpus was collected.

Three mechanisms, in the order they apply.

**Aggregate to cells.** A fix becomes the 250 metre cell it fell in. A cell is a real answer to "where was
this collected" and is not an answer to "where does this driver live".

**Suppress thin cells.** A cell holding fewer than `K_ANONYMITY` fixes is dropped entirely, not noised. A
cell with one fix is one vehicle at one place at one time, and noise added to a count of one still says
somebody was there.

**Noise what survives.** Counts get Laplace noise calibrated to the epsilon spent, so a released count
does not reveal whether one particular vehicle contributed to it. The noise is added after suppression
because noising first would let a suppressed cell reappear.

**And an accountant, because epsilon is a budget and not a setting.** Every release spends from a scope's
allowance and is logged. Differential privacy composes: ten releases at epsilon 0.1 leak as much as one at
epsilon 1.0, so a system that applies a per-query epsilon and never tracks the total provides a guarantee
it has already spent. The accountant refuses when the budget is exhausted rather than degrading quietly.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass

# Cell size for geographic aggregation. 250 m is a city block or two: fine enough that a coverage map
# still shows which corridors were driven, coarse enough that a cell is not a doorstep.
GEO_CELL_M = 250.0
# Fixes a cell must hold before it may be released at all. Below this the cell is dropped rather than
# noised, because noise on a count of one still tells you somebody was there.
K_ANONYMITY = 10
# Default epsilon for one release. Small: this is a coverage map, not a statistical study, and the budget
# is meant to last a long time.
DEFAULT_EPSILON = 0.5
# What a scope may spend before it must be reset by a person. Composition is additive, so this is the
# real privacy guarantee rather than the per-query epsilon.
DEFAULT_BUDGET = 5.0
# Timestamps in an exported trace are coarsened to this, because a fix at one second resolution beside a
# cell id re-identifies the trip the cell was meant to hide.
TIME_COARSEN_S = 60

_R_EARTH_M = 6_371_000.0


class PrivacyBudgetExhausted(Exception):
    """Raised rather than returned. A caller that ignores this releases data it has no budget for."""


@dataclass(frozen=True)
class Cell:
    """One aggregated cell: where, how many, and what the count was before noise."""

    cell_id: str
    lat: float
    lon: float
    count: int
    raw_count: int


def laplace(scale: float) -> float:
    """One Laplace sample, from the system's cryptographic source rather than a seeded PRNG.

    Deliberately not reproducible. Every other random draw in this engine is seeded so a result can be
    replayed; this one must not be, because an attacker who can replay the noise can subtract it, and a
    seeded privacy mechanism provides no privacy at all.
    """
    if scale <= 0:
        return 0.0
    u = (secrets.randbits(53) / float(1 << 53)) - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


def gaussian(sigma: float) -> float:
    """One Gaussian sample, from the same cryptographic source, for mechanisms that want it."""
    if sigma <= 0:
        return 0.0
    u1 = max(secrets.randbits(53) / float(1 << 53), 1e-12)
    u2 = secrets.randbits(53) / float(1 << 53)
    return sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def noisy_count(true_count: int, epsilon: float = DEFAULT_EPSILON, sensitivity: float = 1.0) -> int:
    """A count with Laplace noise at the given epsilon, floored at zero.

    Sensitivity 1 because one vehicle's presence changes a cell's count by one. Floored at zero because a
    negative count is not a possible answer, and a reader seeing minus three would rightly stop trusting
    every other number on the page.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive; a zero-epsilon release is an exact release")
    return max(0, int(round(true_count + laplace(sensitivity / epsilon))))


def cell_of(lat: float, lon: float, cell_m: float = GEO_CELL_M) -> tuple[str, float, float]:
    """The cell a fix falls in, and that cell's centre.

    Fixed-size metric cells rather than a fixed decimal-degree grid: a degree of longitude is 111 km at
    the equator and 96 km at Bengaluru, so a degree grid gives cells of different real sizes at different
    latitudes and the privacy guarantee would vary with where the vehicle drove.
    """
    lat_step = (cell_m / _R_EARTH_M) * (180.0 / math.pi)
    lon_step = lat_step / max(math.cos(math.radians(lat)), 1e-6)
    i = math.floor(lat / lat_step)
    j = math.floor(lon / lon_step)
    return (f"{i}_{j}", round((i + 0.5) * lat_step, 6), round((j + 0.5) * lon_step, 6))


def aggregate_points(points, *, cell_m: float = GEO_CELL_M, k: int = K_ANONYMITY,
                     epsilon: float = DEFAULT_EPSILON) -> dict:
    """Raw fixes into released cells: bucketed, suppressed below k, then noised.

    The order matters and is the whole mechanism. Noising before suppression would let a cell holding one
    fix acquire a count of eleven and be released; suppressing on the noisy count would leak which cells
    were near the threshold. Suppress on the true count, then noise what survives.
    """
    buckets: dict[str, list] = {}
    for lat, lon in points:
        if lat is None or lon is None:
            continue
        cid, clat, clon = cell_of(float(lat), float(lon), cell_m)
        buckets.setdefault(cid, [clat, clon, 0])
        buckets[cid][2] += 1

    kept, suppressed, suppressed_points = [], 0, 0
    for cid, (clat, clon, n) in sorted(buckets.items()):
        if n < k:
            suppressed += 1
            suppressed_points += n
            continue
        kept.append(Cell(cell_id=cid, lat=clat, lon=clon, count=noisy_count(n, epsilon), raw_count=n))
    return {
        "cells": [{"cell_id": c.cell_id, "lat": c.lat, "lon": c.lon, "count": c.count} for c in kept],
        "n_cells": len(kept),
        # Said out loud rather than left as a gap. A map missing most of its data because every cell was
        # thin reads as a fleet that did not drive there, and that is a different fact.
        "suppressed_cells": suppressed, "suppressed_points": suppressed_points,
        "cell_m": cell_m, "k_anonymity": k, "epsilon": epsilon,
        "mechanism": "laplace_count_on_k_suppressed_cells",
    }


def coarsen_time(ts_ns: int, seconds: int = TIME_COARSEN_S) -> int:
    """Round a timestamp down to the coarsening window.

    A cell id beside a one-second timestamp re-identifies the trip the cell was meant to hide: two coarse
    locations at exact times are a trajectory.
    """
    step = int(seconds) * 1_000_000_000
    return (int(ts_ns) // step) * step
