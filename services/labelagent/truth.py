"""The answer key: what the machine said about an object, and what a person later ruled it was.

Every claim the label agent makes is graded against this, so the rules for what counts are strict and written
down here rather than left to a query:

- **Only a person's ruling is truth.** Rows written by machines into `review` are excluded: the track-relabel
  backfill (27,394 rows that copy one ruling onto a whole track), and the agent reviewers (`vlm-qa`,
  `relabel`, `error-detect`). Grading an agent against another agent's opinion measures agreement, not
  accuracy.
- **The person must have had the last word.** If the latest row on an object is a revert, a flag, or a
  machine write, the object's current class is not a ruling and the object is left out.
- **There must be a machine prediction to grade.** The object's first review row carries, in `before`, the
  label as the machine left it. Boxes a person drew (`create`, source `human`) have no prediction and are
  left out; so are objects whose history starts with a revert, because then the first recorded state is
  not the machine's.
- **"Not an object" is an answer.** A reject, or a reclassification to `undefined` or `background`, means the
  box should not exist. It is kept as the sentinel `NOT_OBJECT`, because catching those is half of what
  the agent is for: 125 human rejects, 81 of them on pedestrians, were all confident detections.
- **Single-object rulings and track rulings are kept apart.** A reclassification applied to a whole track
  (`reclassify_track`) is a person's decision, but they saw one frame of it; on the others it is an
  inference. Those items carry `tier="track"` and the headline numbers use `tier="direct"`.
- **A third tier, `keep`, measures the other failure.** People mostly open an object to fix it, so only 137
  of 2,442 direct items had a correct machine label: too few to see a 2% wrong-fix rate. Human-drawn objects
  a person accepted (the gold sets are drawn from these) are correct labels by construction, and showing them
  to the agent asks the question that matters most: does it leave a right answer alone? They have no machine
  prediction, so they count only toward the wrong-fix rate, never toward fixes. In production the agent never
  touches a human label at all; this is a measuring device, not a use.
- **Split by stretch of driving, not by object or even by session.** Objects close in time share lighting,
  camera and often the same physical object, so a split that puts neighbours on both sides lets memory or a
  probe "recognise" a test item. Sessions are not enough: DASHCAM-01's 158 ruled sessions are 60-second clips
  of one day's drive that overlap by about 7 seconds, so adjacent sessions contain the same hoardings. The
  unit is a 30-minute block of one vehicle's time, hashed, so it is stable as the corpus grows; memory and
  the probe add a time buffer on top (see memory.py).
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict, dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger

log = get_logger("labelagent.truth")

NOT_OBJECT = -1

# Names, not ids, so a renumbered ontology does not silently turn "not an object" into a class.
NOT_OBJECT_CLASS_NAMES = frozenset({"undefined", "background"})

# Rows in `review` that a machine wrote. Kept as an explicit list: a new agent reviewer has to be added
# here deliberately, and until it is, its rows are treated as human, which shows up as an implausibly
# good agent score rather than as silence.
MACHINE_REVIEWERS = frozenset({"track_relabel_backfill", "vlm-qa", "relabel", "error-detect"})

DIRECT_RULINGS = frozenset({"reclassify", "accept", "confirm", "reject", "adjust_geometry", "error_fix"})
TRACK_RULINGS = frozenset({"reclassify_track"})
HUMAN_RULINGS = DIRECT_RULINGS | TRACK_RULINGS

# Sources that mean the machine made the label. `imported` came from an external dataset: a valid crop to
# test a classifier on, but not our detector's prediction, so it is graded separately.
MACHINE_SOURCES = frozenset({"fused", "auto_accept", "interpolated", "propagated", "vlm_review",
                             "path_a", "path_b", "path_c"})

# Test-fixture vehicles (purged 2026-07-30, see the fixture-contamination note). Guarded again here because
# a fixture frame of flat colour would teach memory and the probe that "anything" is a sedan.
FIXTURE_VEHICLES = frozenset({"TIGOR-07", "EDIT-01", "TIGOR-T", "TIGOR-TEST", "RT-SRC", "RT-01"})

TEST_FRACTION = 0.30
SPLIT_BLOCK_NS = 30 * 60 * 1_000_000_000


@dataclass
class TruthItem:
    object_id: str
    frame_id: str
    session_id: str
    truth: int                      # a class id, or NOT_OBJECT
    pred: int                       # the machine's class id before any person touched it
    pred_source: str
    tier: str                       # "direct" | "track"
    split: str                      # "calib" | "test"
    bbox: list[float]               # the machine's box: what the agent will actually be shown
    pred_conf: float | None = None
    provenance: dict = field(default_factory=dict)
    action: str = ""                # the ruling that settled it
    vehicle_id: str = ""
    ts_ns: int = 0                  # the frame's time, for blocking neighbours that are the same moment

    @property
    def machine_made(self) -> bool:
        return self.pred_source in MACHINE_SOURCES

    @property
    def correct(self) -> bool:
        return self.pred == self.truth

    def as_dict(self) -> dict:
        return asdict(self)


def block_key(vehicle_id: str, ts_ns: int) -> str:
    """The stretch of driving an object belongs to: one vehicle's 30 minutes."""
    return f"{vehicle_id}:{int(ts_ns) // SPLIT_BLOCK_NS}"


def split_of(key: str, test_fraction: float = TEST_FRACTION) -> str:
    """A block's side of the split. A hash rather than a random draw, so it is the same on every run and a
    block never changes sides when new data arrives."""
    h = int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16)
    return "test" if (h % 1000) < int(test_fraction * 1000) else "calib"


def derive(reviews: list[dict], current: dict, *, not_object_ids: frozenset[int],
           session_id: str, frame_id: str, vehicle_id: str = "", ts_ns: int = 0) -> TruthItem | None:
    """One object's truth item from its review history, or None when the history does not make one.

    `reviews` are that object's rows, oldest first, each {action, reviewer, before, after, ts_ns}.
    `current` is the object row as it stands: {object_id, class_id, state, source}.
    """
    if not reviews:
        return None
    last = reviews[-1]
    if last["reviewer"] in MACHINE_REVIEWERS or last["action"] not in HUMAN_RULINGS:
        return None

    first = reviews[0]
    if first["action"] in ("create", "revert"):
        return None
    before = first.get("before") or {}
    if "class_id" not in before:
        return None
    pred_source = str(before.get("source") or current.get("source") or "")
    if pred_source == "human":
        return None
    bbox = before.get("bbox")
    if not bbox or len(bbox) != 4:
        return None

    rejected = last["action"] == "reject" or current.get("state") == "rejected"
    cls = int(current["class_id"])
    truth = NOT_OBJECT if (rejected or cls in not_object_ids) else cls
    pred = int(before["class_id"])
    if pred in not_object_ids:
        pred = NOT_OBJECT

    conf = before.get("conf")
    return TruthItem(
        object_id=str(current["object_id"]), frame_id=str(frame_id), session_id=str(session_id),
        truth=truth, pred=pred, pred_source=pred_source,
        tier="track" if last["action"] in TRACK_RULINGS else "direct",
        split=split_of(block_key(vehicle_id, ts_ns)),
        bbox=[float(v) for v in bbox],
        pred_conf=float(conf) if isinstance(conf, int | float) else None,
        provenance=dict(before.get("provenance") or {}),
        action=last["action"], vehicle_id=vehicle_id, ts_ns=int(ts_ns))


async def load_truth(db: AsyncSession) -> list[TruthItem]:
    """Every object in the corpus that a person has settled, as a truth item."""
    from db.models import Frame, Object, Review
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    not_object_ids = frozenset(c.id for c in onto.classes if c.name in NOT_OBJECT_CLASS_NAMES)

    # Objects with at least one human-written row. The machine rows are still fetched below, because
    # "the person had the last word" can only be judged against all of them.
    human_objs = (select(Review.object_id)
                  .where(Review.reviewer.notin_(MACHINE_REVIEWERS)).distinct().subquery())
    rows = (await db.execute(
        select(Review.object_id, Review.action, Review.reviewer, Review.before, Review.after, Review.ts_ns,
               Object.class_id, Object.state, Object.source, Object.frame_id,
               Frame.session_id, Frame.ts_ns.label("frame_ts"), DbSession.vehicle_id)
        .join(human_objs, human_objs.c.object_id == Review.object_id)
        .join(Object, Object.object_id == Review.object_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .join(DbSession, DbSession.session_id == Frame.session_id)
        .order_by(Review.object_id, Review.ts_ns, Review.review_id))).all()

    items: list[TruthItem] = []
    skipped_fixture = 0
    group: list = []

    def flush() -> None:
        nonlocal skipped_fixture
        if not group:
            return
        head = group[0]
        if head.vehicle_id in FIXTURE_VEHICLES:
            skipped_fixture += 1
            return
        item = derive(
            [{"action": r.action, "reviewer": r.reviewer, "before": r.before, "after": r.after,
              "ts_ns": r.ts_ns} for r in group],
            {"object_id": head.object_id, "class_id": head.class_id, "state": head.state,
             "source": head.source},
            not_object_ids=not_object_ids, session_id=str(head.session_id), frame_id=str(head.frame_id),
            vehicle_id=str(head.vehicle_id), ts_ns=int(head.frame_ts or 0))
        if item is not None:
            items.append(item)

    current: uuid.UUID | None = None
    for r in rows:
        if r.object_id != current:
            flush()
            group = []
            current = r.object_id
        group.append(r)
    flush()

    log.info("labelagent.truth_loaded", items=len(items), skipped_fixture=skipped_fixture)
    return items


async def load_keep_set(db: AsyncSession, *, exclude: set[str] | None = None) -> list[TruthItem]:
    """Correct labels to test restraint on: human-drawn objects a person accepted, as `tier="keep"` items
    whose prediction is the truth. `exclude` drops object ids already in the main set."""
    from db.models import Frame, Object
    from db.models import Session as DbSession
    from services.autolabel.ontology import get_ontology

    onto = get_ontology()
    not_object_ids = frozenset(c.id for c in onto.classes if c.name in NOT_OBJECT_CLASS_NAMES)
    rows = (await db.execute(
        select(Object.object_id, Object.class_id, Object.bbox, Object.frame_id, Frame.session_id,
               Frame.ts_ns, DbSession.vehicle_id)
        .join(Frame, Frame.frame_id == Object.frame_id)
        .join(DbSession, DbSession.session_id == Frame.session_id)
        .where(Object.source == "human", Object.state == "accepted"))).all()
    skip = exclude or set()
    out = []
    for r in rows:
        if (str(r.object_id) in skip or r.vehicle_id in FIXTURE_VEHICLES or r.class_id in not_object_ids
                or not r.bbox or len(r.bbox) != 4):
            continue
        out.append(TruthItem(
            object_id=str(r.object_id), frame_id=str(r.frame_id), session_id=str(r.session_id),
            truth=int(r.class_id), pred=int(r.class_id), pred_source="human", tier="keep",
            split=split_of(block_key(str(r.vehicle_id), int(r.ts_ns or 0))), bbox=[float(v) for v in r.bbox],
            action="accepted", vehicle_id=str(r.vehicle_id), ts_ns=int(r.ts_ns or 0)))
    return out


async def load_all(db: AsyncSession) -> list[TruthItem]:
    """The ruled items plus the keep set, with no object in both."""
    items = await load_truth(db)
    return items + await load_keep_set(db, exclude={i.object_id for i in items})


def summary(items: list[TruthItem]) -> dict:
    """Counts a reader needs before believing any score computed on the set."""
    out: dict = {"items": len(items), "by_tier": {}, "by_split": {}, "machine_made": 0,
                 "pred_correct": 0, "not_object": 0, "sessions": len({i.session_id for i in items}),
                 "blocks": len({block_key(i.vehicle_id, i.ts_ns) for i in items})}
    for i in items:
        out["by_tier"][i.tier] = out["by_tier"].get(i.tier, 0) + 1
        out["by_split"][i.split] = out["by_split"].get(i.split, 0) + 1
        out["machine_made"] += int(i.machine_made)
        out["pred_correct"] += int(i.correct)
        out["not_object"] += int(i.truth == NOT_OBJECT)
    return out
