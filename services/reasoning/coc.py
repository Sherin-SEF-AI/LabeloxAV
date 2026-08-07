"""Chain-of-Causation traces: why the ego did what it did, grounded in objects this corpus already holds.

Boxes and masks say what is in a frame. They do not say which of those things the driver was reacting to,
and that is the question a planning team is actually buying. NVIDIA's Alpamayo work published a schema for
saying it (arXiv 2511.00088): a closed-set driving decision, an open set of critical components that caused
it, and a natural-language trace linking the two. The schema is the useful part and it is public; this module
adopts it rather than inventing a private vocabulary, so a trace produced here is comparable with the 700K
public CoC traces instead of being a dialect only this product speaks.

**What this deliberately does not do.** The published model predicts ego trajectory, and its meta-actions are
statements about ego kinematics. That needs egomotion, and this corpus has `ego_speed` on 6 of 36,905 frames
and no pose at all. So `META_ACTIONS` is declared here because it belongs to the schema and a trace should be
able to carry one later, and `CocTrace.meta_action` stays None until something can measure it. Guessing an
ego meta-action from a forward camera is exactly the plausible-but-unfounded label this repo keeps finding,
and a wrong one is worse than an absent one because it reads as measured.

What is well founded is the other half: which agents and conditions made a scene difficult, cited by
object and track id. That is the half that needs no ego signal, and on Indian roads it is the half nobody
sells.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------------------------------------
# Driving decisions (Alpamayo Table 1). Closed sets: at most one of each channel, or None.
# ---------------------------------------------------------------------------------------------------------

LONGITUDINAL_DECISIONS: tuple[str, ...] = (
    "set_speed_tracking",
    "lead_obstacle_following",
    "speed_adaptation_road_events",
    "gap_searching",
    "acceleration_for_passing",
    "yield_agent_right_of_way",
    "stop_for_static_constraints",
)

LATERAL_DECISIONS: tuple[str, ...] = (
    "lane_keeping_centering",
    "merge_split",
    "out_of_lane_nudge",
    "in_lane_nudge",
    "lane_change",
    "pull_over_curb_approach",
    "turn",
    "lateral_maneuver_abort",
)

# ---------------------------------------------------------------------------------------------------------
# Meta-actions (Alpamayo Table 5): atomic instantaneous kinematic changes. Declared, not yet producible here.
# ---------------------------------------------------------------------------------------------------------

META_ACTIONS_LONGITUDINAL: tuple[str, ...] = (
    "gentle_accelerate", "strong_accelerate", "gentle_decelerate", "strong_decelerate",
    "maintain_speed", "stop", "go_straight", "reverse",
)

META_ACTIONS_LATERAL: tuple[str, ...] = (
    "steer_left", "steer_right", "sharp_steer_left", "sharp_steer_right",
    "reverse_left", "reverse_right",
)

META_ACTIONS: tuple[str, ...] = META_ACTIONS_LONGITUDINAL + META_ACTIONS_LATERAL

# ---------------------------------------------------------------------------------------------------------
# Critical components (Alpamayo Table 2). Open-ended by design: the category set is fixed so a trace can be
# queried, while the components themselves are free-form so a scene can say something the taxonomy lacks.
# ---------------------------------------------------------------------------------------------------------

CRITICAL_CATEGORIES: tuple[str, ...] = (
    "critical_object",
    "traffic_light",
    "yield_stop_control",
    "road_event",
    "lane_laneline",
    "routing_intent",
    "odd_constraint",
)

# The paper tags each component's uncertainty rather than forcing a confident claim, which is the right
# posture for a scene a model half-understands.
UNCERTAINTY: tuple[str, ...] = ("low", "high")

# Alpamayo Table 4. Carried as data because these are what a reviewer checks, and the review UI should show
# the same four questions the published pipeline audited against rather than a locally-invented rubric.
QA_CHECKS: tuple[tuple[str, str], ...] = (
    ("causal_coverage", "every factor that materially caused the decision is present"),
    ("causal_correctness", "each listed factor genuinely bears on the decision"),
    ("proximate_cause", "the trace names the immediate cause, not a distant precondition"),
    ("decision_minimality", "no decision is claimed beyond what the scene supports"),
)


class CocError(ValueError):
    """A trace that does not satisfy the schema."""


@dataclass(frozen=True)
class CriticalComponent:
    """One causal factor behind a decision.

    `object_id` and `track_id` are what separate this from a caption. A component that names "a cow" is a
    sentence; one that cites the object row for that cow is a label, and can be counted, exported, and
    checked against the box a human later corrects.
    """

    category: str
    description: str
    uncertainty: str = "low"
    object_id: str | None = None
    track_id: str | None = None

    def validate(self) -> None:
        if self.category not in CRITICAL_CATEGORIES:
            raise CocError(f"unknown critical category {self.category!r}")
        if self.uncertainty not in UNCERTAINTY:
            raise CocError(f"uncertainty must be one of {UNCERTAINTY}, got {self.uncertainty!r}")
        if not self.description.strip():
            raise CocError("a critical component with no description explains nothing")
        if self.category == "critical_object" and not (self.object_id or self.track_id):
            # The category asserts a specific agent caused the decision. Without a citation that assertion
            # cannot be checked, exported against the corpus, or corrected when the box is.
            raise CocError("a critical_object must cite an object_id or track_id")


@dataclass
class CocTrace:
    """One decision, the factors that caused it, and the reasoning that links them."""

    longitudinal: str | None = None
    lateral: str | None = None
    components: list[CriticalComponent] = field(default_factory=list)
    trace: str = ""
    # Declared by the schema, unmeasurable here: see the module docstring.
    meta_action: str | None = None

    def validate(self) -> None:
        if self.longitudinal is not None and self.longitudinal not in LONGITUDINAL_DECISIONS:
            raise CocError(f"unknown longitudinal decision {self.longitudinal!r}")
        if self.lateral is not None and self.lateral not in LATERAL_DECISIONS:
            raise CocError(f"unknown lateral decision {self.lateral!r}")
        if self.meta_action is not None and self.meta_action not in META_ACTIONS:
            raise CocError(f"unknown meta action {self.meta_action!r}")
        if self.longitudinal is None and self.lateral is None:
            # "None on both channels" is the schema's way of saying nothing was decided, which is not a
            # trace worth storing. Refusing keeps the plane free of rows that assert nothing.
            raise CocError("a trace must state at least one decision channel")
        for c in self.components:
            c.validate()
        if not self.components:
            raise CocError("a decision with no cause is an assertion, not a chain of causation")
        if not self.trace.strip():
            raise CocError("the composed trace is the part a reader reads; it cannot be empty")

    def cited_object_ids(self) -> list[str]:
        return [c.object_id for c in self.components if c.object_id]

    def cited_track_ids(self) -> list[str]:
        return [c.track_id for c in self.components if c.track_id]

    def to_dict(self) -> dict:
        return {
            "longitudinal": self.longitudinal,
            "lateral": self.lateral,
            "meta_action": self.meta_action,
            "trace": self.trace,
            "components": [
                {"category": c.category, "description": c.description, "uncertainty": c.uncertainty,
                 "object_id": c.object_id, "track_id": c.track_id}
                for c in self.components
            ],
        }


def from_dict(d: dict) -> CocTrace:
    """Build a trace from a model's JSON, validating it. Raises CocError on anything malformed.

    Deliberately strict about unknown decision names rather than mapping them to the nearest known one. A
    model that invents "slow_down" has not made a closed-set choice, and quietly rounding it to
    `gentle_decelerate` would turn a refusal into a label.
    """
    if not isinstance(d, dict):
        raise CocError(f"expected an object, got {type(d).__name__}")
    raw = d.get("components") or d.get("critical_components") or []
    if not isinstance(raw, list):
        raise CocError("components must be a list")
    components = []
    for c in raw:
        if not isinstance(c, dict):
            raise CocError("each component must be an object")
        components.append(CriticalComponent(
            category=str(c.get("category", "")).strip().lower(),
            description=str(c.get("description", "")).strip(),
            uncertainty=str(c.get("uncertainty", "low")).strip().lower(),
            object_id=(str(c["object_id"]) if c.get("object_id") else None),
            track_id=(str(c["track_id"]) if c.get("track_id") else None),
        ))
    t = CocTrace(
        longitudinal=(str(d["longitudinal"]).strip().lower() if d.get("longitudinal") else None),
        lateral=(str(d["lateral"]).strip().lower() if d.get("lateral") else None),
        components=components,
        trace=str(d.get("trace") or d.get("reasoning") or "").strip(),
        meta_action=(str(d["meta_action"]).strip().lower() if d.get("meta_action") else None),
    )
    t.validate()
    return t
