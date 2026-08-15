"""Train/val/test for an exported dataset, split where it does not leak.

Every export left here unsplit. Whoever received one split it themselves, and the obvious way to split a
folder of dashcam frames is per frame, which is the one way that is wrong: consecutive frames of the same
drive are near-duplicates, so a per-frame split puts the same vehicle, the same road and very nearly the
same pixels on both sides of the line. The model is then evaluated on what it memorised, and the score is
inflated by an amount nobody can measure after the fact. A dataset delivered without a split is a dataset
whose future numbers cannot be defended.

The training side already knew this: `services/training/dataset_builder.py` holds out whole sessions. This
is the same principle for the delivered artifact, and deliberately not the same code. That function orders
sessions gold-first because validation is the yardstick a gate reads, which is a training concern an export
does not have, and it takes whole sessions until a budget is met, which overshoots badly once there are two
budgets rather than one.

**A quota walk in hash order, not a shuffle and not plain bucketing.** Each group gets a fixed position
from `sha256(seed:group)`, and groups are filled into val, then test, then train until each split's frame
budget is met. The hash order is what a shuffle is not: it does not depend on which other groups exist, so
inserting a session leaves every group ahead of it in that order assigned exactly as before. The frame
budget is what plain per-group bucketing is not: this corpus has 377 sessions ranging from 1 to 1,928
frames, and independently bucketing groups that uneven put 42% of frames in a validation set that asked for
20%.

The two properties are in tension and cannot both be perfect. Accuracy wins here because the delivered
artifact is immutable: `splits.json` records the whole frame-to-split map, and the commit id seals it, so a
consumer who needs the exact split of a release reads it rather than recomputing it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from core.logging import get_logger
from services.export.records import ExportRecord

log = get_logger("export.splits")

SPLITS = ("train", "val", "test")

# What a group is. Session is the honest default for driving data: a session is one drive, and everything
# inside it is correlated in weather, light, vehicle and road.
#
# Track is deliberately absent. A frame holds many tracks, so grouping by track would put one frame's
# objects on both sides of the split, which means the image itself is in two splits at once. That is the
# leak this module exists to prevent, arriving through the door marked leakage control.
GROUP_KEYS = ("session", "frame")

# How far the realised fraction may drift from the requested one before it is worth saying so out loud.
# Uneven session sizes make some drift inevitable; a lot of drift means the request could not be honoured
# and the reader should know before they quote the number.
DRIFT_TOLERANCE = 0.10

# How far past its remaining budget a whole group may push a split before it is passed over for train.
# Some overshoot is unavoidable because groups are indivisible; unbounded overshoot is how a single large
# drive becomes the entire validation set.
OVERSHOOT_TOLERANCE = 0.25


def _group_of(rec: ExportRecord, group_by: str) -> str:
    return str(rec.session_id) if group_by == "session" else str(rec.frame_id)


def _bucket(seed: str, group: str) -> float:
    """A stable number in [0, 1) for this group, from the seed alone.

    sha256 rather than hash(): Python's hash is salted per process, so the same export would split
    differently on every run and no two machines would agree.
    """
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def assign_splits(records: Sequence[ExportRecord], *, val_frac: float = 0.0, test_frac: float = 0.0,
                  group_by: str = "session", seed: str = "") -> dict[str, str]:
    """Map each frame id to its split.

    Returns every frame as `train` when nothing is requested, so an export that asks for no split is
    untouched rather than quietly reorganised.
    """
    if group_by not in GROUP_KEYS:
        raise ValueError(f"group_by must be one of {GROUP_KEYS}, got {group_by!r}")
    if val_frac < 0 or test_frac < 0:
        raise ValueError("split fractions cannot be negative")
    if val_frac + test_frac >= 1.0:
        # Leaving no training data is a request that cannot be meant, and honouring it silently would
        # produce a dataset that trains on nothing.
        raise ValueError(f"val_frac + test_frac must leave a training set (got {val_frac} + {test_frac})")

    frames = {str(r.frame_id): _group_of(r, group_by) for r in records}
    if not val_frac and not test_frac:
        return dict.fromkeys(frames, "train")

    members: dict[str, list[str]] = {}
    for frame_id, group in frames.items():
        members.setdefault(group, []).append(frame_id)

    # Fixed position per group, independent of which other groups exist, so inserting a session cannot
    # reorder the ones ahead of it.
    order = sorted(members, key=lambda g: (_bucket(seed, g), g))

    total = len(frames)
    budget = {"val": val_frac * total, "test": test_frac * total}
    taken = {"val": 0, "test": 0}
    assigned: dict[str, str] = {}

    for group in order:
        n = len(members[group])
        # Whole groups only: splitting a group is the leak the grouping exists to prevent. So a group that
        # would blow past its budget is passed over rather than cut, because one 1,928-frame drive landing
        # in a validation set sized for 100 frames does not make a validation set, it makes a second
        # training set wearing the name. Passed-over groups fall to train, which can absorb them.
        target = "train"
        for split in ("val", "test"):
            room = budget[split] - taken[split]
            if room > 0 and n <= room * (1 + OVERSHOOT_TOLERANCE):
                target = split
                taken[split] += n
                break
        assigned[group] = target

    # A split that was asked for and came out empty is worse than one that came out large: the caller gets
    # a dataset whose val directory does not exist and no signal that anything went wrong. If every group
    # was too big to fit, take the smallest one rather than leaving it empty.
    for split, frac in (("val", val_frac), ("test", test_frac)):
        if frac > 0 and not any(v == split for v in assigned.values()):
            candidates = [g for g in order if assigned[g] == "train"]
            if candidates:
                assigned[min(candidates, key=lambda g: (len(members[g]), g))] = split

    out: dict[str, str] = {}
    for group, target in assigned.items():
        for frame_id in members[group]:
            out[frame_id] = target
    return out


def split_summary(records: Sequence[ExportRecord], assignment: dict[str, str], *,
                  group_by: str, seed: str, val_frac: float, test_frac: float) -> dict:
    """What the split actually came out as, which is not what was asked for.

    Requested and realised are both reported because hash bucketing over uneven groups cannot hit a
    fraction exactly, and a reader comparing two datasets needs the number that happened rather than the
    number that was intended.
    """
    frames_per: dict[str, set[str]] = {s: set() for s in SPLITS}
    objects_per = dict.fromkeys(SPLITS, 0)
    groups_per: dict[str, set[str]] = {s: set() for s in SPLITS}
    for r in records:
        s = assignment.get(str(r.frame_id), "train")
        frames_per[s].add(str(r.frame_id))
        groups_per[s].add(_group_of(r, group_by))
        objects_per[s] += 1

    n_frames = sum(len(v) for v in frames_per.values())
    realised = {s: (len(frames_per[s]) / n_frames if n_frames else 0.0) for s in SPLITS}
    requested = {"train": max(0.0, 1.0 - val_frac - test_frac), "val": val_frac, "test": test_frac}

    drift = {s: abs(realised[s] - requested[s]) for s in SPLITS}
    if any(d > DRIFT_TOLERANCE for d in drift.values()) and n_frames:
        log.warning("export.split_drift", requested=requested, realised=realised, group_by=group_by,
                    detail="uneven group sizes; the realised split is what the dataset carries")

    return {
        "group_by": group_by,
        "seed": seed,
        "requested": requested,
        "realised": realised,
        "frames": {s: len(frames_per[s]) for s in SPLITS},
        "objects": {s: objects_per[s] for s in SPLITS},
        "groups": {s: len(groups_per[s]) for s in SPLITS},
        # A group appearing in two splits is the failure this module exists to prevent, so it is asserted
        # in the artifact rather than left for a reader to check.
        "groups_shared_between_splits": sorted(
            (groups_per["train"] & groups_per["val"])
            | (groups_per["train"] & groups_per["test"])
            | (groups_per["val"] & groups_per["test"])),
    }
