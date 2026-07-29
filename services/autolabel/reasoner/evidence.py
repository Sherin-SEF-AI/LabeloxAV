"""Tier 1: deterministic evidence about one detection, gathered before it becomes a label.

The gate has only ever had one input worth the name: the detector's own confidence. That is the model
grading its own homework, and it cannot catch the failures that actually hurt this corpus, all of which are
*confident*: an autorickshaw called a delivery_rider_bike at 0.7, a pedestrian two pixels tall at ten
metres, a sedan in frame 40 that is an suv in frame 41, a boat on the Outer Ring Road.

Each collector below answers one question the detector cannot, from information the system already holds:

- **Physics** compares the box against the class's known real-world height and the frame's depth prior.
- **Geometry** compares the box against the class's aspect ratio, the horizon, and its own mask.
- **Temporal** asks whether the track agrees with itself across frames.
- **Scene** asks whether this class belongs on this kind of road at all.
- **Cross-model** treats the three paths' disagreement as a signal rather than averaging it away.
- **Corpus memory** asks what humans decided about the nearest-looking crops already reviewed.

Three properties hold throughout, and each exists because its absence produces a worse system:

- **Absent evidence is absent, never negative.** A class with no height prior yields nothing rather than
  evidence against, because "no prior" and "impossible" are different findings and conflating them would
  demote every class the moment it was added to the ontology.
- **Every finding carries its own weight and its own sentence.** A verdict has to be explainable to the
  reviewer it is routed to, and a weight nobody can read is a magic number.
- **Nothing here is a veto.** A single check that could kill a detection outright would make the whole
  system as brittle as its worst rule. The combiner weighs them; that is deliberate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from core.logging import get_logger
from core.schemas import BBox, UnifiedObject
from services.autolabel.ontology import Ontology

log = get_logger("reasoner_evidence")

_PRIORS_PATH = Path(__file__).resolve().parents[3] / "configs" / "class_priors.yaml"

# How far up a frame a road user can still legitimately appear. Measured, not assumed: see the note
# in check_elevation for what the assumed value cost.
ELEVATION_FRAC = 0.05


@lru_cache(maxsize=1)
def load_priors() -> dict:
    """The physical and contextual priors. Cached: they are read once per process and never change."""
    import yaml

    try:
        return yaml.safe_load(_PRIORS_PATH.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        # A missing priors file must not take the pipeline down. The reasoner degrades to the checks that
        # need no priors, and says so, rather than annotating with a silently weaker gate.
        log.warning("reasoner.priors_unavailable", path=str(_PRIORS_PATH), error=str(exc))
        return {}


@dataclass(frozen=True)
class Finding:
    """One piece of evidence, with its direction, its strength, and a sentence a human can read."""

    check: str
    # Positive supports the detection as labelled, negative argues against it. Magnitude is roughly
    # "how much would this alone move a reviewer", on a scale where 1.0 is decisive.
    weight: float
    detail: str
    # When a check has a specific alternative in mind ("this looks like an e_rickshaw"), it says so. That
    # is what lets Tier 2 ask a narrow question instead of an open one.
    suggests_class: str | None = None


@dataclass
class EvidenceContext:
    """Everything the collectors may read. Anything absent simply produces no finding from that check."""

    obj: UnifiedObject
    onto: Ontology
    frame_w: int
    frame_h: int
    others: list[UnifiedObject] = field(default_factory=list)
    # Frame.scene: weather, time_of_day, road_type, density. Populated by the scene model.
    scene: dict = field(default_factory=dict)
    # Metres per pixel at the object's image row, from the mono-depth prior, when available.
    depth_m: float | None = None
    focal_px: float | None = None
    camera_height_m: float = 1.3
    # Prior and next observations of the same track: (ts_ns, bbox, class_id).
    track_neighbours: list[tuple[int, BBox, int]] = field(default_factory=list)
    # Nearest human-reviewed crops in embedding space: (class_name, similarity, state).
    neighbours: list[tuple[str, float, str]] = field(default_factory=list)


# ---------------------------------------------------------------- physics

def check_physics(ctx: EvidenceContext) -> list[Finding]:
    """Does the box's size make sense for this class at this distance?

    A pinhole camera makes this checkable: an object of real height H at distance D projects to
    f*H/D pixels. A pedestrian at thirty metres is about forty pixels tall on a 1280-wide dashcam, and a
    four-pixel one is not a pedestrian whatever the detector says.
    """
    priors = load_priors()
    heights = (priors.get("heights_m") or {})
    band = heights.get(ctx.obj.class_name)
    if not band or not ctx.depth_m or not ctx.focal_px or ctx.depth_m <= 0:
        # No prior, or no depth. Nothing to say, and saying nothing is correct.
        return []

    px_h = max(1.0, ctx.obj.bbox.y2 - ctx.obj.bbox.y1)
    implied_m = (px_h * ctx.depth_m) / ctx.focal_px
    lo, hi = float(band[0]), float(band[1])

    if lo <= implied_m <= hi:
        return [Finding("physics", 0.35,
                        f"implied height {implied_m:.2f}m is consistent with {ctx.obj.class_name} "
                        f"({lo}-{hi}m) at {ctx.depth_m:.1f}m")]

    # Scaled by how far outside the band it falls, not a flat penalty. A pedestrian implied at 2.3m is a
    # tall person; one implied at 0.15m is a detection on a reflection.
    ratio = implied_m / lo if implied_m < lo else implied_m / hi
    off = abs(1.0 - ratio)
    weight = -min(0.9, 0.25 + off * 0.5)
    return [Finding("physics", weight,
                    f"implied height {implied_m:.2f}m at {ctx.depth_m:.1f}m is outside the "
                    f"{lo}-{hi}m range for {ctx.obj.class_name}")]


# ---------------------------------------------------------------- geometry

def check_aspect(ctx: EvidenceContext) -> list[Finding]:
    """Box shape against the class's usual proportions.

    Its own collector rather than part of a general "geometry", because a collector is the unit an operator
    can turn off and a finding is the unit the attribution grades, and those have to be the same thing or a
    rule measured as harmful cannot actually be switched off. This one currently measures below the base
    rate: an occluded or truncated vehicle has the wrong proportions and the right label, which is common
    enough to swamp the cases where a bad shape means a bad class.
    """
    priors = load_priors()
    bw = max(1.0, ctx.obj.bbox.x2 - ctx.obj.bbox.x1)
    bh = max(1.0, ctx.obj.bbox.y2 - ctx.obj.bbox.y1)
    ratio = bw / bh

    band = (priors.get("aspect_wh") or {}).get(ctx.obj.class_name)
    if not band:
        return []
    lo, hi = float(band[0]), float(band[1])
    if lo <= ratio <= hi:
        return [Finding("aspect", 0.2, f"aspect {ratio:.2f} is typical for {ctx.obj.class_name}")]
    off = abs(1.0 - (ratio / lo if ratio < lo else ratio / hi))
    return [Finding("aspect", -min(0.7, 0.2 + off * 0.4),
                    f"aspect {ratio:.2f} is outside the {lo}-{hi} range for {ctx.obj.class_name}")]


def check_mask_box(ctx: EvidenceContext) -> list[Finding]:
    """Whether the segment and the detector box cover the same thing.

    Computed at fusion and previously used only as a triage flag. When SAM's segment covers a third of the
    detector's box the two models are looking at different things and at least one is wrong, which is a
    strong argument in principle. Measured, it currently sits below the base rate, so it is doing less than
    it looks like it should.
    """
    if not ctx.obj.provenance.mask_box_disagree:
        return []
    return [Finding("mask_box", -0.45,
                    "the segment and the detector box cover different regions, so the two models are not "
                    "looking at the same thing")]


def check_elevation(ctx: EvidenceContext) -> list[Finding]:
    """Whether the box sits higher up the frame than a road user can be.

    The threshold is measured, not assumed, and the difference is the whole point of measuring. At a third
    of frame height, which is where somebody guessed the horizon was, this objected to 490 reviewed objects
    and was right 43% of the time against a base rate of 63%: it fired more often on the objects that were
    fine than on the ones that were wrong, so having it was worse than not. Distant vehicles and
    high-mounted cameras put perfectly good road users well above a third. At a twentieth the same rule is
    right 99% of the time.
    """
    priors = load_priors()
    if ctx.obj.class_name in set(priors.get("overhead_classes") or []) or ctx.frame_h <= 0:
        return []
    if ctx.obj.bbox.y2 >= ctx.frame_h * ELEVATION_FRAC:
        return []
    return [Finding("elevation", -0.5,
                    "the whole box sits at the very top of the frame, where a road user cannot be")]


# ---------------------------------------------------------------- temporal

def check_temporal(ctx: EvidenceContext) -> list[Finding]:
    """Does the track agree with itself?

    The strongest signal available and the one a per-frame detector structurally cannot use. A physical
    object does not change class between consecutive frames, does not teleport, and does not double in area
    in 40 milliseconds. Each of those is a different error with a different fix, so they are reported
    separately rather than as one "temporal" score.
    """
    if not ctx.track_neighbours:
        return []

    out: list[Finding] = []
    priors = load_priors()
    same_class = [c for _ts, _b, c in ctx.track_neighbours if c == ctx.obj.class_id]
    other_class = [c for _ts, _b, c in ctx.track_neighbours if c != ctx.obj.class_id]

    if other_class and len(other_class) > len(same_class):
        # The track mostly disagrees with this frame. Name the class it mostly says, so Tier 2 can ask
        # about that specific alternative rather than opening the question.
        from collections import Counter

        majority_id = Counter(other_class).most_common(1)[0][0]
        majority = ctx.onto.by_id(majority_id).name
        out.append(Finding("temporal", -0.6,
                           f"the track is mostly {majority} ({len(other_class)} of "
                           f"{len(ctx.track_neighbours)} neighbouring frames), not {ctx.obj.class_name}",
                           suggests_class=majority))
    elif same_class and not other_class:
        out.append(Finding("temporal", 0.4,
                           f"the track agrees on {ctx.obj.class_name} across "
                           f"{len(same_class)} neighbouring frames"))

    diag = max(1.0, ((ctx.obj.bbox.x2 - ctx.obj.bbox.x1) ** 2
                     + (ctx.obj.bbox.y2 - ctx.obj.bbox.y1) ** 2) ** 0.5)
    cx = (ctx.obj.bbox.x1 + ctx.obj.bbox.x2) / 2
    cy = (ctx.obj.bbox.y1 + ctx.obj.bbox.y2) / 2
    max_jump = float(priors.get("max_centroid_jump_ratio") or 1.5)
    max_area = float(priors.get("max_area_ratio_jump") or 2.2)
    area = max(1.0, ctx.obj.bbox.area)

    for _ts, nb, _c in ctx.track_neighbours:
        ncx, ncy = (nb.x1 + nb.x2) / 2, (nb.y1 + nb.y2) / 2
        jump = ((cx - ncx) ** 2 + (cy - ncy) ** 2) ** 0.5 / diag
        if jump > max_jump:
            out.append(Finding("temporal", -0.5,
                               f"the box moved {jump:.1f} box-diagonals between adjacent frames, which is "
                               "an association error rather than motion"))
            break

    for _ts, nb, _c in ctx.track_neighbours:
        r = max(area, nb.area) / max(1.0, min(area, nb.area))
        if r > max_area:
            out.append(Finding("temporal", -0.4,
                               f"the box area changed {r:.1f}x between adjacent frames, so it is "
                               "collapsing onto something else"))
            break
    return out


# ---------------------------------------------------------------- scene

def check_scene(ctx: EvidenceContext) -> list[Finding]:
    """Does this class belong in this scene at all?

    The check that catches the boat on the Outer Ring Road. Deliberately two-tiered: a class that cannot be
    on a road under any circumstances is near-decisive, while a class that is merely unusual for this road
    type is mild, because a cow on a national highway is a real and dangerous event this loop must be able
    to label rather than suppress.
    """
    priors = load_priors()
    out: list[Finding] = []

    if ctx.obj.class_name in set(priors.get("never_on_road") or []):
        out.append(Finding("scene", -0.95,
                           f"{ctx.obj.class_name} does not appear in road scenes; this is a detector "
                           "confusion rather than an unusual observation"))
        return out

    road_type = str((ctx.scene or {}).get("road_type") or "")
    if road_type:
        table = priors.get("implausible_on_road_type") or {}
        if ctx.obj.class_name in set(table.get(road_type) or []):
            out.append(Finding("scene", -0.3,
                               f"{ctx.obj.class_name} is unusual on a {road_type}, though not impossible"))
        # No finding when the class is merely not implausible here. That used to emit a small positive
        # weight, and measuring it showed why that was wrong: it fired for 1,350 of the reviewed objects and
        # was right 25% of the time, which is the base rate. It was not evidence, it was a constant added to
        # almost everything, and a constant that lifts every score lifts the wrong ones too. Saying a class
        # is not out of place here says nothing about whether this particular box is that class.
    return out


# ---------------------------------------------------------------- cross-model

def check_cross_model(ctx: EvidenceContext) -> list[Finding]:
    """What did the three paths actually say?

    Fusion already records each path's proposal and then reports a single fused confidence, which hides the
    most informative thing in the pipeline: whether the paths agreed. Two independent models naming the
    same class is far stronger than one model naming it loudly.
    """
    prov = ctx.obj.provenance
    out: list[Finding] = []

    proposals = list(prov.proposals or [])
    if len(proposals) >= 2:
        names = [p.class_name for p in proposals if getattr(p, "class_name", None)]
        distinct = set(names)
        # Unanimity is only agreement when the paths agree with the LABEL. Every path saying truck on an
        # object labelled bus is unanimous and is the strongest objection available, not corroboration;
        # reading it as agreement was a bug that would have endorsed exactly the confident relabelling
        # this layer exists to catch.
        if len(distinct) == 1 and ctx.obj.class_name in distinct:
            out.append(Finding("cross_model", 0.45,
                               f"{len(names)} independent paths agree on {ctx.obj.class_name}"))
        elif len(distinct) == 1:
            only = next(iter(distinct))
            out.append(Finding("cross_model", -0.6,
                               f"every path proposed {only}, not {ctx.obj.class_name}",
                               suggests_class=only))
        elif ctx.obj.class_name in distinct:
            rival = next((n for n in names if n != ctx.obj.class_name), None)
            out.append(Finding("cross_model", -0.4,
                               f"the paths disagree: {', '.join(sorted(distinct))}",
                               suggests_class=rival))
        else:
            out.append(Finding("cross_model", -0.55,
                               f"no path proposed {ctx.obj.class_name}; they said "
                               f"{', '.join(sorted(distinct))}"))
    elif not prov.agreement and proposals:
        out.append(Finding("cross_model", -0.2,
                           "only one path saw this object, so nothing corroborates it"))

    if prov.entropy is not None and prov.entropy > 0.9:
        # High entropy means the ensemble's votes spread across classes. A scalar confidence cannot
        # express that, which is why it is recorded separately at fusion.
        out.append(Finding("cross_model", -0.3,
                           f"the class vote is spread across several classes (entropy {prov.entropy:.2f})"))
    return out


# ---------------------------------------------------------------- corpus memory

def check_corpus_memory(ctx: EvidenceContext) -> list[Finding]:
    """What did humans decide about the nearest-looking crops?

    The corpus already holds thousands of human verdicts and the embeddings to find the relevant ones, and
    nothing consulted them at annotation time. If the ten visually nearest human-accepted crops are all
    riders and this says pedestrian, that is the corpus telling the detector it is making a mistake it has
    already been corrected on.
    """
    if not ctx.neighbours:
        return []

    accepted = [(name, sim) for name, sim, state in ctx.neighbours
                if state in ("accepted", "approved")]
    if len(accepted) < 3:
        # Too few reviewed neighbours to mean anything. Three is not a quorum either, but below it the
        # nearest neighbour is noise and reporting it as evidence would be worse than silence.
        return []

    from collections import Counter

    votes = Counter(name for name, _ in accepted)
    top, count = votes.most_common(1)[0]
    share = count / len(accepted)
    mean_sim = sum(s for _, s in accepted) / len(accepted)

    if top == ctx.obj.class_name and share >= 0.6:
        return [Finding("corpus_memory", 0.4 * mean_sim,
                        f"{count} of {len(accepted)} visually nearest reviewed crops are also "
                        f"{ctx.obj.class_name}")]
    if top != ctx.obj.class_name and share >= 0.7:
        return [Finding("corpus_memory", -0.55 * mean_sim,
                        f"{count} of {len(accepted)} visually nearest reviewed crops are {top}, "
                        f"not {ctx.obj.class_name}",
                        suggests_class=top)]
    return [Finding("corpus_memory", -0.15,
                    f"the nearest reviewed crops are mixed ({dict(votes.most_common(3))})")]


# ---------------------------------------------------------------- the pass

CHECKS = {
    "physics": check_physics,
    # One collector per rule. A collector is what an operator can disable and a finding name is what the
    # attribution grades; when those differ, a rule measured as harmful cannot be switched off, which is
    # what "geometry" was: three unrelated rules of very different strength under one switch.
    "aspect": check_aspect,
    "mask_box": check_mask_box,
    "elevation": check_elevation,
    "temporal": check_temporal,
    "scene": check_scene,
    "cross_model": check_cross_model,
    "corpus_memory": check_corpus_memory,
}


def collect(ctx: EvidenceContext, only: list[str] | None = None) -> list[Finding]:
    """Run every collector. A collector that raises is skipped and logged, never fatal.

    One broken check must not stop a session from annotating: the reasoner's job is to make labels better,
    and a reasoner that can halt the pipeline makes them worse than having none.
    """
    out: list[Finding] = []
    for name, fn in CHECKS.items():
        if only and name not in only:
            continue
        try:
            out.extend(fn(ctx))
        except Exception as exc:  # noqa: BLE001
            log.warning("reasoner.check_failed", check=name, error=f"{type(exc).__name__}: {exc}")
    return out
