"""Ontology loader and validator (Principle 09: governed artifact, no inline label creation).

Loads ontology/labelox_in_v0.yaml, exposes class lookups, and validates that an object's
class_id and attrs conform. Reviewers and models pick from this; they never invent a label.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from core.config import get_settings

# Custom (annotator-added) classes live in a sidecar beside the governed YAML, in their own id block
# (>= CUSTOM_ID_BASE) so the frozen governed ids stay pristine. They default to india=true so the gate
# treats a brand-new class as rare and forces human review until it has been governed properly.
CUSTOM_ID_BASE = 200

# Thing vs stuff (panoptic split). THINGS are countable foreground objects (vehicles, people, animals,
# signs, poles, cones) that get one instance box each. STUFF is background: uncountable extended regions
# (sky, road and every surface, vegetation, barriers, walls, buildings) that belong to semantic
# segmentation and must never get an instance box. Boxing stuff is the "tree/barrier/sky is an object"
# error. Surfaces and ignore-regions are stuff by their l0; the rest of stuff is a curated name set, because
# infra/fixed mixes real things (pole, traffic_sign) with stuff (tree, barrier) under the same l0/l1. Extend
# STUFF_NAMES when a new uncountable class is added; a class absent here defaults to a thing.
STUFF_L0 = frozenset({"surface", "ignore"})
STUFF_NAMES = frozenset({
    # vegetation / foliage
    "tree", "vegetation", "fallen_tree",
    # barriers / fences / walls / railings
    "barrier", "crash_barrier", "median_barrier", "guardrail", "fence", "road_side_grill",
    "side_wall", "construction_barrier", "barricade_line", "temp_barricade", "sandbag",
    # buildings and large fixed structures
    "buildings", "shops", "foot_overbridge", "flyover_pillar", "fly_over", "bus_shelter",
    "bmtc_bus_shelter", "metro_bus_stop", "school_bus_stop", "temp_bus_stop", "toll_booth",
    "telephone_booth", "overhead_water_tank", "shrine", "hoarding", "metro_pillar",
    "festival_pandal", "roadside_shop",
    # amorphous ground clutter and lines
    "electric_line", "debris", "garbage_pile", "waterlogging", "excavation_pit",
})


@dataclass(frozen=True)
class OntologyClassDef:
    id: int
    name: str
    l0: str
    l1: str
    india: bool


@dataclass
class AttributeDef:
    name: str
    type: str
    values: list | None = None
    range: tuple[float, float] | None = None


@dataclass
class Ontology:
    version: str
    hierarchy_levels: int
    classes: list[OntologyClassDef]
    attributes: dict[str, AttributeDef] = field(default_factory=dict)
    # Per-subclass (l1) applicable-attribute allowlist. A subclass absent here means all attributes apply.
    attribute_scope: dict[str, list[str]] = field(default_factory=dict)

    _by_id: dict[int, OntologyClassDef] = field(default_factory=dict, repr=False)
    _by_name: dict[str, OntologyClassDef] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {c.id: c for c in self.classes}
        self._by_name = {c.name: c for c in self.classes}

    def by_id(self, class_id: int) -> OntologyClassDef:
        if class_id not in self._by_id:
            raise KeyError(f"class_id {class_id} not in ontology {self.version}")
        return self._by_id[class_id]

    def by_name(self, name: str) -> OntologyClassDef:
        if name not in self._by_name:
            raise KeyError(f"class name '{name}' not in ontology {self.version}")
        return self._by_name[name]

    def has_name(self, name: str) -> bool:
        return name in self._by_name

    def attrs_for_class(self, class_id: int) -> list[str] | None:
        """Attribute names applicable to a class (by its l1 subclass). None means all attributes apply."""
        try:
            l1 = self.by_id(class_id).l1
        except KeyError:
            return None
        return self.attribute_scope.get(l1)

    def concept_phrases(self, india_first: bool = True) -> list[str]:
        """Ontology names as open-vocab prompts for SAM 3.1 PCS. India/rare classes first."""
        ordered = sorted(self.classes, key=lambda c: (not c.india, c.id)) if india_first else self.classes
        return [c.name.replace("_", " ") for c in ordered]

    def fallback_ids(self) -> list[int]:
        return [c.id for c in self.classes if c.l1 == "fallback"]

    def is_fallback(self, class_id: int) -> bool:
        return self.by_id(class_id).l1 == "fallback"

    def is_stuff(self, class_id: int) -> bool:
        """True if the class is background stuff (semantic-seg only, never an instance box): any surface or
        ignore-region, plus the curated uncountable structures/vegetation/barriers in STUFF_NAMES."""
        c = self.by_id(class_id)
        return c.l0 in STUFF_L0 or c.name in STUFF_NAMES

    def is_thing(self, class_id: int) -> bool:
        """True if the class is a countable foreground object that legitimately gets one instance box."""
        return not self.is_stuff(class_id)

    def validate_attrs(self, attrs: dict, class_id: int | None = None) -> list[str]:
        """Return a list of validation errors; empty means valid. When class_id is given and its subclass
        declares an attribute scope, an attribute not in that scope is an error (not applicable to class)."""
        allowed = self.attrs_for_class(class_id) if class_id is not None else None
        errors: list[str] = []
        for key, val in attrs.items():
            if key not in self.attributes:
                errors.append(f"unknown attribute '{key}'")
                continue
            if allowed is not None and key not in allowed:
                errors.append(f"attribute '{key}' not applicable to class {class_id}")
                continue
            spec = self.attributes[key]
            if spec.type == "enum":
                if val not in (spec.values or []):
                    errors.append(f"attribute '{key}'={val!r} not in {spec.values}")
            elif spec.type == "float":
                if not isinstance(val, (int, float)):
                    errors.append(f"attribute '{key}' must be float")
                elif spec.range and not (spec.range[0] <= float(val) <= spec.range[1]):
                    errors.append(f"attribute '{key}'={val} out of range {spec.range}")
            elif spec.type == "int":
                if not isinstance(val, int) or isinstance(val, bool):
                    errors.append(f"attribute '{key}' must be int")
            elif spec.type == "bool":
                if not isinstance(val, bool):
                    errors.append(f"attribute '{key}' must be bool")
            elif spec.type == "bool_array":
                if not (isinstance(val, list) and all(isinstance(x, bool) for x in val)):
                    errors.append(f"attribute '{key}' must be a bool array")
        return errors


def load_ontology(path: str | Path | None = None) -> Ontology:
    p = Path(path) if path else get_settings().ontology_abspath()
    with open(p) as fh:
        data = yaml.safe_load(fh)

    classes = [
        OntologyClassDef(id=c["id"], name=c["name"], l0=c["l0"], l1=c["l1"], india=bool(c.get("india", False)))
        for c in data["classes"]
    ]

    attributes: dict[str, AttributeDef] = {}
    for name, spec in (data.get("attributes") or {}).items():
        rng = tuple(spec["range"]) if "range" in spec else None
        attributes[name] = AttributeDef(
            name=name, type=spec["type"], values=spec.get("values"), range=rng  # type: ignore[arg-type]
        )

    # Integrity checks: unique ids and names in the governed YAML.
    ids = [c.id for c in classes]
    names = [c.name for c in classes]
    if len(set(ids)) != len(ids):
        raise ValueError("ontology has duplicate class ids")
    if len(set(names)) != len(names):
        raise ValueError("ontology has duplicate class names")

    # Merge annotator-added custom classes (defensively skipping any id/name already governed, so a stale
    # sidecar can never break loading).
    seen_ids, seen_names = set(ids), set(names)
    for c in _read_custom(p):
        if c["id"] in seen_ids or c["name"] in seen_names:
            continue
        classes.append(OntologyClassDef(id=int(c["id"]), name=c["name"], l0=c.get("l0", "object"),
                                        l1=c.get("l1", "custom"), india=bool(c.get("india", True))))
        seen_ids.add(c["id"])
        seen_names.add(c["name"])

    scope = {k: list(v) for k, v in (data.get("attribute_scope") or {}).items()}
    return Ontology(
        version=data["version"],
        hierarchy_levels=int(data["hierarchy_levels"]),
        classes=classes,
        attributes=attributes,
        attribute_scope=scope,
    )


def _custom_path(ontology_path: Path | None = None) -> Path:
    base = Path(ontology_path) if ontology_path else get_settings().ontology_abspath()
    return base.parent / "custom_classes.json"


def _read_custom(ontology_path: Path | None = None) -> list[dict]:
    p = _custom_path(ontology_path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001 - a corrupt sidecar must never break the governed load
        return []


def normalize_class_name(name: str) -> str:
    # collapse runs of whitespace/hyphen to a single underscore (mirrors the web client), then drop any
    # remaining non-ascii-word characters, so the preview and the created name always agree.
    collapsed = re.sub(r"[\s\-]+", "_", name.strip().lower())
    return re.sub(r"[^a-z0-9_]", "", collapsed)


@lru_cache(maxsize=1)
def get_ontology() -> Ontology:
    return load_ontology()


def add_custom_class(name: str, l0: str = "object", l1: str = "custom", india: bool = True) -> dict:
    """Add an annotator-defined class to the sidecar and make it live (cache cleared). Idempotent: an
    existing name returns the existing class. Names are normalized to the ontology's snake_case style."""
    norm = normalize_class_name(name)
    if not norm:
        raise ValueError("class name must contain letters or digits")
    onto = get_ontology()
    if onto.has_name(norm):
        c = onto.by_name(norm)
        return {"id": c.id, "name": c.name, "l0": c.l0, "l1": c.l1, "india": c.india, "existed": True}

    new_id = max([c.id for c in onto.classes] + [CUSTOM_ID_BASE - 1]) + 1
    customs = _read_custom()
    customs.append({"id": new_id, "name": norm, "l0": l0, "l1": l1, "india": bool(india)})
    path = _custom_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(customs, indent=2, sort_keys=True))
    get_ontology.cache_clear()
    return {"id": new_id, "name": norm, "l0": l0, "l1": l1, "india": bool(india), "existed": False}
