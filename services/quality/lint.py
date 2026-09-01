"""Checking a frame at the moment it is saved, instead of in a corpus sweep nobody runs.

Everything needed for this already existed as a batch detector. `services/errordetect/policy.py::
check_object` is a pure function of one object, its siblings, the ontology and the frame size, returning
`[(rule, score, reason)]`, and it has exactly one caller: a corpus sweep that filters
`Object.source != "human"`. `services/agent/critic.py::critique_frame` is already per-frame, per-object
shaped and runs only inside the agent pipeline. There is no `POST /frames/{id}/lint` anywhere, no rule has
ever created an `Issue`, and so the editor's Issues panel, the one object-scoped surface an annotator
actually looks at, has been blind to every detector in the repo.

Two things make this different from pointing the sweep at one frame.

**A rule must be armed before it can fire.** The obvious relation and attribute rules all fire on 100% of
their scope today, because the counterpart data does not exist: `object_relationship` holds 98 rows in the
whole corpus and 56,410 riders have no `rider_of`, `occupant_count` is set on 0 objects. "Rider without a
mount" is not 56,410 findings, it is one fact about what nobody has annotated yet, and a linter that opens
it as 56,410 issues buries every real finding on its first run. So each rule declares a precondition over
the frame it is looking at, and stays dormant until the data it depends on is actually being collected
there. A dormant rule is reported as dormant, with its reason, rather than silently skipped.

**And a rule that fires on nearly everything is reported, not queued.** The same guard
`services/agent/reanalyze.py::_drop_systemic` applies at 80%, for the reason recorded there at length: a
check that fires more often on the objects that were fine is not a weak check, it is a harmful one.

**Measured over the corpus once it existed**, 150 frames sampled by hash, 2,878 objects:

    rule                        findings   % of objects   frames   systemic
    min_box_size                     302          10.5%       67          0
    self_intersecting_polygon         89           3.1%       64          0
    duplicate_box                     52           1.8%       20          0
    box_in_ego_mask                   46           1.6%       15          0
    degenerate_aspect                 16           0.6%       15          0
    rider_without_mount             DORMANT on 150 of 150 frames
    helmet_without_occupants        DORMANT on 150 of 150 frames
    trolley_without_towing          DORMANT on 150 of 150 frames

Every armed rule discriminates and none is systemic; every unarmed rule is dormant everywhere, which is
the arming doing exactly its job. Without it those three would have opened an issue on essentially every
rider, scooter and trolley in the sample.

`self_intersecting_polygon` is a genuinely new finding and it was checked a second way before being
believed: sampling 525 stored polygons directly and asking shapely, 23 are invalid (4.4%), and
`explain_validity` names the crossing coordinate on each. Nothing on the write path has ever looked.

The `source != "human"` filter is deliberately NOT inherited. The sweep excludes human work because it is
looking for machine mistakes; a linter that runs on save exists to check the edit that was just made.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.logging import get_logger
from db.models import Frame, Object, ObjectRelationship
from db.models import Session as DbSession

log = get_logger("lint")

# Same numbers as reanalyze.py, and the same reason. A rule objecting to this share of a frame's objects
# is describing the pipeline rather than the objects.
SYSTEMIC_FRACTION = 0.8
SYSTEMIC_MIN_OBJECTS = 5

# A mask whose area exceeds its box by this factor is not a tight outline of the same thing. Distinct from
# `mask_box_disagree`, which is an IoU computed at fusion time and never recomputed when a human edits a
# mask through PUT /objects/{id}/mask.
MASK_OVER_BOX = 1.35

# How much of a box must lie on the ego hood before it is worth saying so. Below the 0.5 that
# `cleanup_sweep` uses to delete: this reports rather than deletes, so it can afford to be more sensitive,
# and the interesting case is the half-on-the-bonnet reflection rather than the box that IS the bonnet.
EGO_FRACTION = 0.35


@dataclass
class Ctx:
    """Everything the rules can see about one frame."""
    frame: Frame
    objects: list[Object]
    onto: Any
    # Relations on this frame, by from_object_id. Empty is the normal case today.
    relations: dict[Any, list[ObjectRelationship]]
    n_confirmed_relations: int
    # Mask polygons by uri, loaded once for the frame. Empty for a frame with no masks.
    masks: dict[str, list[list[float]]]
    ego_mask: Any = None

    def name_of(self, o: Object) -> str:
        try:
            return self.onto.by_id(int(o.class_id)).name
        except Exception:  # noqa: BLE001
            return str(o.class_id)

    def l1_of(self, o: Object) -> str:
        try:
            return self.onto.by_id(int(o.class_id)).l1
        except Exception:  # noqa: BLE001
            return ""

    def attr_seen(self, key: str) -> bool:
        """Whether anybody has answered this attribute anywhere on this frame."""
        return any((o.attrs or {}).get(key) is not None for o in self.objects)


@dataclass
class Rule:
    name: str
    label: str
    severity: str                                   # high | medium | low
    score: float
    # Whether the data this rule depends on is being collected on this frame. Returns None when armed, or
    # the reason it is dormant. A rule with no precondition is always armed.
    armed: Callable[[Ctx], str | None] | None
    check: Callable[[Object, Ctx], str | None]      # the reason, or None for no finding


@dataclass
class Finding:
    object_id: str
    rule: str
    label: str
    severity: str
    score: float
    reason: str

    def as_dict(self) -> dict:
        return {"object_id": self.object_id, "rule": self.rule, "label": self.label,
                "severity": self.severity, "score": round(self.score, 3), "reason": self.reason}


# ---------------------------------------------------------------- geometry rules (always armed)

async def _load_masks(objects: list[Object]) -> dict[str, list[list[float]]]:
    """Every mask on the frame, fetched once and concurrently.

    A mask is a JSON blob in the object store behind `Object.mask_uri`, not a column, so the two geometry
    rules below need them loaded before the pure per-object checks can run. Serial GETs on a frame with
    sixteen masks is sixteen round trips on the editor's save path, which is the exact cost
    `_mask_polygons_bulk` was written to avoid on the read path.
    """
    import asyncio
    import json

    from core.storage import get_object_store

    uris = sorted({o.mask_uri for o in objects if o.mask_uri})
    if not uris:
        return {}
    store = get_object_store()
    sem = asyncio.Semaphore(16)

    def read(uri: str) -> list[list[float]]:
        try:
            return json.loads(store.get_bytes(uri)).get("polygons", [])
        except Exception:  # noqa: BLE001 - an unreadable mask is one absent check, not a failed lint
            return []

    async def one(uri: str):
        async with sem:
            return uri, await asyncio.to_thread(read, uri)

    return dict(await asyncio.gather(*(one(u) for u in uris)))


def _polygons(o: Object, ctx: Ctx) -> list[list[float]]:
    polys = ctx.masks.get(o.mask_uri or "", [])
    return [p for p in polys if isinstance(p, list) and len(p) >= 6]


def _self_intersecting(o: Object, ctx: Ctx) -> str | None:
    """A polygon that crosses itself. Nothing on the write path checks this today.

    `services/api/routers/objects.py::_write_mask` JSON-serialises whatever it is given, so a mask dragged
    through itself in the editor is stored and exported as-is, and every consumer that computes an area
    from it gets a number that is not the area of anything.
    """
    polys = _polygons(o, ctx)
    if not polys:
        return None
    try:
        from shapely.geometry import Polygon
    except Exception:  # noqa: BLE001 - shapely absent: report nothing rather than guess
        return None
    for p in polys:
        pts = [(p[i], p[i + 1]) for i in range(0, len(p) - 1, 2)]
        if len(pts) < 3:
            continue
        try:
            if not Polygon(pts).is_valid:
                return "the outline crosses itself, so its area is not the area of anything"
        except Exception:  # noqa: BLE001 - a polygon shapely cannot even build is itself the finding
            return "the outline could not be read as a polygon"
    return None


def _mask_exceeds_box(o: Object, ctx: Ctx) -> str | None:
    """A mask much larger than the box that is supposed to contain it.

    Different from `mask_box_disagree`, which is an IoU measured at fusion time and never recomputed. This
    one runs on the mask as it stands now, which is the point: a human editing a mask can push it outside
    the box, and until now nothing looked again.
    """
    polys = _polygons(o, ctx)
    if not polys:
        return None
    x1, y1, x2, y2 = (float(v) for v in o.bbox)
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    xs: list[float] = []
    ys: list[float] = []
    for p in polys:
        xs.extend(p[0::2])
        ys.extend(p[1::2])
    if not xs:
        return None
    hull = max(1.0, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    ratio = hull / box_area
    if ratio > MASK_OVER_BOX:
        return f"the outline covers {ratio:.1f}x the area of its box"
    return None


def _in_ego_mask(o: Object, ctx: Ctx) -> str | None:
    """A box sitting on the car's own bonnet.

    `services/agent/cleanup_sweep.py` deletes these in a batch run. Reporting is the right thing at save
    time: a person looking at the frame can tell a reflection from a real vehicle seen over the bonnet,
    and a linter that deleted their box while they were drawing it would be intolerable.
    """
    if ctx.ego_mask is None or not ctx.frame.width or not ctx.frame.height:
        return None
    frac = ctx.ego_mask.ego_fraction(tuple(float(v) for v in o.bbox),
                                     float(ctx.frame.width), float(ctx.frame.height))
    if frac >= EGO_FRACTION:
        return f"{frac * 100:.0f}% of this box is on the ego vehicle's own bonnet"
    return None


# ---------------------------------------------------------------- relation and attribute rules (armed)

def _relations_collected(ctx: Ctx) -> str | None:
    """Relation rules arm only where somebody is drawing relations.

    `object_relationship` holds 98 rows in the entire corpus and 56,410 riders have no `rider_of`. Firing
    on all of them is one fact about coverage stated 56,410 times.
    """
    if ctx.n_confirmed_relations == 0:
        return ("no relations have been drawn on this frame, so a missing one is a coverage gap "
                "rather than a mistake")
    return None


_TWO_WHEELERS = ("two_wheeler", "three_wheeler")


def _rider_without_mount(o: Object, ctx: Ctx) -> str | None:
    if ctx.l1_of(o) != "vru" or ctx.name_of(o) != "rider":
        return None
    for rel in ctx.relations.get(o.object_id, []):
        if rel.kind == "rider_of":
            return None
    # Only where there is actually something to be a rider of, so the finding names a real pairing.
    x1, y1, x2, y2 = (float(v) for v in o.bbox)
    for other in ctx.objects:
        if other.object_id == o.object_id or ctx.l1_of(other) not in _TWO_WHEELERS:
            continue
        ox1, oy1, ox2, oy2 = (float(v) for v in other.bbox)
        if not (ox2 < x1 or ox1 > x2 or oy2 < y1 or oy1 > y2):
            return f"this rider overlaps a {ctx.name_of(other)} and no rider_of links them"
    return None


def _occupants_collected(ctx: Ctx) -> str | None:
    if not ctx.attr_seen("occupant_count"):
        return "occupant_count is not being answered on this frame, so a helmet array has nothing to check"
    return None


def _helmet_without_occupants(o: Object, ctx: Ctx) -> str | None:
    attrs = o.attrs or {}
    helmet = attrs.get("helmet")
    if not isinstance(helmet, list):
        return None
    n = attrs.get("occupant_count")
    if not isinstance(n, int) or isinstance(n, bool):
        return "a helmet is recorded per rider and the occupant count is not set"
    if len(helmet) != n:
        return f"{len(helmet)} helmet values against {n} occupants"
    return None


def _towing_collected(ctx: Ctx) -> str | None:
    return _relations_collected(ctx)


def _trolley_without_towing(o: Object, ctx: Ctx) -> str | None:
    if ctx.name_of(o) != "tractor_trolley":
        return None
    for rel in ctx.relations.get(o.object_id, []):
        if rel.kind in ("towing", "towed_by"):
            return None
    return "a tractor trolley with nothing recorded as towing it"


RULES: list[Rule] = [
    Rule("self_intersecting_polygon", "Outline crosses itself", "high", 0.8, None, _self_intersecting),
    Rule("mask_exceeds_box", "Outline much larger than the box", "medium", 0.6, None, _mask_exceeds_box),
    Rule("box_in_ego_mask", "Box on the ego bonnet", "medium", 0.65, None, _in_ego_mask),
    Rule("rider_without_mount", "Rider with no rider_of", "medium", 0.6,
         _relations_collected, _rider_without_mount),
    Rule("helmet_without_occupants", "Helmet array with no occupant count", "medium", 0.6,
         _occupants_collected, _helmet_without_occupants),
    Rule("trolley_without_towing", "Trolley with nothing towing it", "low", 0.5,
         _towing_collected, _trolley_without_towing),
]

# The four rules that already existed as a corpus sweep, with their labels for the editor.
_POLICY_LABELS = {
    "min_box_size": ("Box below the minimum size", "medium"),
    "degenerate_aspect": ("Implausible box shape", "medium"),
    "attr_validity": ("Attribute the class cannot carry", "high"),
    "duplicate_box": ("Duplicate box", "medium"),
}


async def _context(db: AsyncSession, frame_id: UUID) -> Ctx | None:
    frame = await db.get(Frame, frame_id)
    if frame is None:
        return None
    from services.autolabel.ontology import get_ontology

    objs = list((await db.execute(select(Object).where(Object.frame_id == frame_id))).scalars().all())
    rels = list((await db.execute(
        select(ObjectRelationship).where(ObjectRelationship.frame_id == frame_id))).scalars().all())
    by_from: dict[Any, list[ObjectRelationship]] = {}
    for r in rels:
        by_from.setdefault(r.from_object_id, []).append(r)

    ego = None
    if frame.cam_id:
        sess = await db.get(DbSession, frame.session_id)
        if sess is not None and sess.vehicle_id:
            try:
                from services.autolabel.ego_mask import get_ego_mask

                ego = get_ego_mask(sess.vehicle_id, frame.cam_id)
            except Exception as exc:  # noqa: BLE001
                log.info("lint.no_ego_mask", frame=str(frame_id)[:8], reason=str(exc)[:120])

    return Ctx(frame=frame, objects=objs, onto=get_ontology(), relations=by_from,
               n_confirmed_relations=sum(1 for r in rels if r.status == "confirmed"),
               masks=await _load_masks(objs), ego_mask=ego)


def _split_systemic(findings: list[Finding], n_objects: int) -> tuple[list[Finding], dict[str, int]]:
    """Rules that fired on most of the frame are counted, not listed. Same rule as reanalyze.py."""
    if n_objects < SYSTEMIC_MIN_OBJECTS:
        return findings, {}
    per_rule: dict[str, int] = {}
    for f in findings:
        per_rule[f.rule] = per_rule.get(f.rule, 0) + 1
    systemic = {r: n for r, n in per_rule.items() if n / n_objects >= SYSTEMIC_FRACTION}
    if not systemic:
        return findings, {}
    return [f for f in findings if f.rule not in systemic], systemic


async def lint_frame(db: AsyncSession, frame_id: UUID) -> dict:
    """Every guideline this frame breaks, plus the rules that could not run and why.

    `dormant` is as much of the answer as `findings`. A rule that cannot run because nobody is collecting
    the data it needs is a fact about the corpus, and hiding it would make the linter look like it had
    checked something it never looked at.
    """
    ctx = await _context(db, frame_id)
    if ctx is None:
        return {"frame_id": str(frame_id), "findings": [], "dormant": [], "reason": "frame not found"}
    if not ctx.objects:
        return {"frame_id": str(frame_id), "n_objects": 0, "findings": [], "dormant": [], "systemic": {}}

    findings: list[Finding] = []
    dormant: list[dict] = []

    # The four rules that already existed, run without the sweep's source filter: a linter that runs on
    # save exists to check the edit that was just made, and the sweep skips exactly that.
    from services.errordetect.policy import check_object

    for o in ctx.objects:
        for rule, score, reason in check_object(o, ctx.objects, ctx.onto, ctx.frame.width, ctx.frame.height):
            label, severity = _POLICY_LABELS.get(rule, (rule.replace("_", " "), "medium"))
            findings.append(Finding(str(o.object_id), rule, label, severity, score, reason))

    for rule in RULES:
        why = rule.armed(ctx) if rule.armed else None
        if why is not None:
            dormant.append({"rule": rule.name, "label": rule.label, "reason": why})
            continue
        for o in ctx.objects:
            reason = rule.check(o, ctx)
            if reason:
                findings.append(Finding(str(o.object_id), rule.name, rule.label, rule.severity,
                                        rule.score, reason))

    kept, systemic = _split_systemic(findings, len(ctx.objects))
    order = {"high": 0, "medium": 1, "low": 2}
    kept.sort(key=lambda f: (order.get(f.severity, 3), -f.score))
    return {
        "frame_id": str(frame_id),
        "n_objects": len(ctx.objects),
        "findings": [f.as_dict() for f in kept],
        # Counted rather than listed, with the count, so somebody knows where to go and look.
        "systemic": systemic,
        "dormant": dormant,
    }


async def open_issues_for(db: AsyncSession, frame_id: UUID, findings: list[dict], *,
                          user_id: Any = None) -> dict:
    """Turn findings into `Issue` rows, which is the surface the editor already shows.

    No machine rule has ever created an Issue, so `web/components/labelops/IssuePanel.tsx` has been blind
    to every detector in the repo. Idempotent per (object, rule): re-linting a frame after a save must not
    stack a second copy of a finding nobody has dealt with.
    """
    from db.models import Issue, IssueComment

    existing = list((await db.execute(
        select(Issue).where(Issue.frame_id == frame_id, Issue.status == "open"))).scalars().all())
    seen = set()
    for i in existing:
        # The rule is carried in the opening comment, which is what a person reads. Keyed on the pair so a
        # different rule about the same object is a different thread.
        seen.add((str(i.object_id), i.kind))

    made = []
    for f in findings:
        kind = "bad_geometry" if f["rule"] in (
            "self_intersecting_polygon", "mask_exceeds_box", "min_box_size", "degenerate_aspect",
            "box_in_ego_mask") else "wrong_class" if f["rule"] == "duplicate_box" else "unclear"
        if (f["object_id"], kind) in seen:
            continue
        issue = Issue(frame_id=frame_id, object_id=UUID(f["object_id"]), kind=kind,
                      status="open", created_by=user_id)
        db.add(issue)
        await db.flush()
        db.add(IssueComment(issue_id=issue.issue_id, author_id=user_id,
                            body=f"{f['label']}: {f['reason']}"))
        seen.add((f["object_id"], kind))
        made.append(str(issue.issue_id))
    return {"opened": len(made), "issue_ids": made}
