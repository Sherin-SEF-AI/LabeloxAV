"""Ontology loader and validator (Principle 09: governed artifact, no inline label creation).

Loads ontology/labelox_in_v0.yaml, exposes class lookups, and validates that an object's
class_id and attrs conform. Reviewers and models pick from this; they never invent a label.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

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
    # buildings and large fixed structures.
    # `hoarding` is deliberately NOT here. A billboard has definite edges and is countable, which is the test
    # for a thing, and while it sat in this set `persist.py` discarded every one the autolabeller proposed.
    # Advertising then had nowhere to go and its contents were labelled `traffic_sign`, which is a large part
    # of why that class holds 48,322 objects against 518 hoardings and recalls at 0.163.
    "buildings", "shops", "foot_overbridge", "flyover_pillar", "fly_over", "bus_shelter",
    "bmtc_bus_shelter", "metro_bus_stop", "school_bus_stop", "temp_bus_stop", "toll_booth",
    "telephone_booth", "overhead_water_tank", "shrine", "metro_pillar",
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
    # What else this thing is called. Open-vocabulary detectors are prompted with words, not ids, and a
    # class named `autorickshaw` is invisible to a prompt that says "tuk-tuk". Aliases also stop the
    # duplicate-class failure that put 48,322 objects into traffic_sign: `handcart` and `thela` are
    # `push_cart`, and the answer is a synonym rather than a third class nobody reconciles.
    aliases: tuple[str, ...] = ()


@dataclass
class AttributeDef:
    name: str
    type: str
    values: list | None = None
    range: tuple[float, float] | None = None
    # Computed from another attribute at write time and refused on a direct write. `triple_riding` is not a
    # second opinion about `occupant_count`, it is a reading of it, and letting both be set independently
    # produces frames that say three people and not triple riding.
    derived_from: str | None = None


# How each derived attribute is computed from its source. Keyed by the derived attribute's name, matching
# `derived_from` in the YAML; a name here with no YAML entry is inert, a YAML entry with no deriver raises.
#
# Three is the legal threshold for a two-wheeler and the number the traffic code names, so the comparison is
# `>= 3` rather than "more than the seats".
_DERIVERS: dict[str, Callable[[Any], Any]] = {
    "triple_riding": lambda n: bool(isinstance(n, int) and not isinstance(n, bool) and n >= 3),
}


@dataclass
class Ontology:
    version: str
    hierarchy_levels: int
    classes: list[OntologyClassDef]
    attributes: dict[str, AttributeDef] = field(default_factory=dict)
    # Per-subclass (l1) applicable-attribute allowlist. A subclass absent here means all attributes apply.
    attribute_scope: dict[str, list[str]] = field(default_factory=dict)
    # Per-class extras, keyed by class NAME, unioned onto whatever the l1 scope allows.
    #
    # l1 is too coarse for the attributes that matter here. `heavy` holds bus, school_bus, truck, tractor,
    # tipper, ambulance, fire_truck, bullock_cart and harvester together, so a footboard-passenger attribute
    # scoped at l1 is offered on every truck, and an attribute offered is an attribute somebody sets. This
    # layer is additive: no class changes l1, no existing scope entry moves, and a class absent from it
    # behaves exactly as before.
    attribute_scope_class: dict[str, list[str]] = field(default_factory=dict)

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
        """Attribute names applicable to a class. None means all attributes apply.

        The l1 scope, plus any per-class extras. Union rather than override: a city_bus is still a heavy
        vehicle and still wants the heavy attributes, it just also wants two of its own.
        """
        try:
            c = self.by_id(class_id)
        except KeyError:
            return None
        base = self.attribute_scope.get(c.l1)
        extra = self.attribute_scope_class.get(c.name)
        if base is None:
            # The l1 is unscoped, which already means "everything applies", so extras add nothing.
            return None
        return base if not extra else [*base, *(a for a in extra if a not in base)]

    def derive_attrs(self, attrs: dict, class_id: int | None = None) -> dict:
        """Return `attrs` with every derived attribute recomputed from its source.

        Called after the merge on every write path, so the derived value is a fact about the stored attrs
        rather than a second thing to keep in step. A derived key whose source is absent or out of scope is
        removed: leaving a stale `triple_riding: true` behind after somebody corrected the occupant count to
        one is worse than not having the attribute.
        """
        out = dict(attrs)
        scope = self.attrs_for_class(class_id) if class_id is not None else None
        for name, spec in self.attributes.items():
            if not spec.derived_from:
                continue
            if scope is not None and name not in scope:
                out.pop(name, None)
                continue
            src = out.get(spec.derived_from)
            if src is None:
                out.pop(name, None)
                continue
            fn = _DERIVERS.get(name)
            if fn is None:
                # Declared in the YAML with no implementation here. Refusing to guess: an attribute that
                # silently derives nothing is a field consumers will read and trust.
                raise ValueError(f"attribute '{name}' is declared derived but has no deriver")
            out[name] = fn(src)
        return out

    def aliases_for(self, class_id: int) -> list[str]:
        """What else this class is called, the display name first. Never empty."""
        c = self.by_id(class_id)
        return [c.name, *c.aliases]

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
            if spec.derived_from:
                errors.append(f"attribute '{key}' is computed from '{spec.derived_from}' and cannot be set")
                continue
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
                elif spec.range and not (spec.range[0] <= val <= spec.range[1]):
                    # The float branch has always checked this and the int branch never did, so
                    # `occlusion_pct: 400` and `passenger_load: -3` both validated.
                    errors.append(f"attribute '{key}'={val} out of range {spec.range}")
            elif spec.type == "bool":
                if not isinstance(val, bool):
                    errors.append(f"attribute '{key}' must be bool")
            elif spec.type == "bool_array":
                if not (isinstance(val, list) and all(isinstance(x, bool) for x in val)):
                    errors.append(f"attribute '{key}' must be a bool array")
            elif spec.type == "multi_select":
                # A set of values from the vocabulary, not one. `script` is the case it exists for: a
                # Bengaluru signboard routinely carries Kannada and English together, and forcing one
                # records the wrong half.
                if not isinstance(val, list):
                    errors.append(f"attribute '{key}' must be a list")
                elif bad := [v for v in val if v not in (spec.values or [])]:
                    errors.append(f"attribute '{key}' has values not in {spec.values}: {bad}")
                elif len(set(val)) != len(val):
                    errors.append(f"attribute '{key}' has duplicate values")
            else:
                # An unimplemented type used to fall through here and accept anything at all, so adding a
                # type to the YAML silently disabled validation for every attribute using it.
                errors.append(f"attribute '{key}' has unsupported type '{spec.type}'")
        return errors


class _StrictLoader(yaml.SafeLoader):
    """A YAML loader that refuses duplicate mapping keys.

    `yaml.safe_load` takes the last of a repeated key and says nothing, which is how this file spent a
    commit with `aliases` written twice on seven class lines. It loaded, it validated, every test passed,
    and the file was wrong. A class tree edited by hand needs the parser to be the thing that notices.
    """


def _no_duplicate_keys(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"duplicate key '{key}' in ontology YAML at line {key_node.start_mark.line + 1}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicate_keys)


def load_ontology(path: str | Path | None = None) -> Ontology:
    p = Path(path) if path else get_settings().ontology_abspath()
    with open(p) as fh:
        data = yaml.load(fh, Loader=_StrictLoader)  # noqa: S506 - _StrictLoader subclasses SafeLoader

    classes = [
        OntologyClassDef(id=c["id"], name=c["name"], l0=c["l0"], l1=c["l1"],
                         india=bool(c.get("india", False)), aliases=tuple(c.get("aliases") or ()))
        for c in data["classes"]
    ]

    attributes: dict[str, AttributeDef] = {}
    for name, spec in (data.get("attributes") or {}).items():
        rng = tuple(spec["range"]) if "range" in spec else None
        attributes[name] = AttributeDef(
            name=name, type=spec["type"], values=spec.get("values"), range=rng,  # type: ignore[arg-type]
            derived_from=spec.get("derived_from"),
        )

    # Integrity checks: unique ids and names in the governed YAML.
    ids = [c.id for c in classes]
    names = [c.name for c in classes]
    if len(set(ids)) != len(ids):
        raise ValueError("ontology has duplicate class ids")
    if len(set(names)) != len(names):
        raise ValueError("ontology has duplicate class names")

    # An alias that collides with a real class name, or with another class's alias, is worse than no alias:
    # a prompt or an importer resolving "handcart" would get whichever class the loader happened to see
    # first. Checked at load so a bad edit fails on startup rather than in a labelling session.
    alias_owner: dict[str, str] = {}
    for c in classes:
        for a in c.aliases:
            if a in names:
                raise ValueError(f"alias '{a}' on '{c.name}' is already a class name")
            if a in alias_owner:
                raise ValueError(f"alias '{a}' claimed by both '{alias_owner[a]}' and '{c.name}'")
            alias_owner[a] = c.name

    for name, spec in attributes.items():
        if spec.derived_from and spec.derived_from not in attributes:
            raise ValueError(f"attribute '{name}' derives from unknown attribute '{spec.derived_from}'")

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
    scope_class = {k: list(v) for k, v in (data.get("attribute_scope_class") or {}).items()}
    unknown = sorted(set(scope_class) - seen_names)
    if unknown:
        raise ValueError(f"attribute_scope_class names unknown classes: {unknown}")

    # A misspelled attribute in a scope list does not fail, it hides: the editor stops offering the
    # attribute for that class and validate_attrs starts refusing it, and both look like the attribute was
    # deliberately scoped out.
    for label, table in (("attribute_scope", scope), ("attribute_scope_class", scope_class)):
        for key, names_in in table.items():
            bad = sorted(set(names_in) - set(attributes))
            if bad:
                raise ValueError(f"{label}['{key}'] names unknown attributes: {bad}")
    return Ontology(
        version=data["version"],
        hierarchy_levels=int(data["hierarchy_levels"]),
        classes=classes,
        attributes=attributes,
        attribute_scope=scope,
        attribute_scope_class=scope_class,
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


def get_ontology(pack_id: str = "av") -> Ontology:
    """The ontology for a domain pack. Defaults to the AV pack, whose ontology is the governed
    labelox_in_v0.yaml resolved from config, identical to the historical single-ontology behaviour (every
    existing no-arg caller resolves here). A non-AV pack resolves its ontology path through the pack
    registry. Cache is keyed by pack id, so multiple packs coexist in one process."""
    return _get_ontology_cached(pack_id or "av")


@cache
def _get_ontology_cached(pack_id: str) -> Ontology:
    if pack_id == "av":
        return load_ontology()  # byte-identical to the pre-pack behaviour: same YAML, same sidecar merge
    # The registry is the sanctioned bridge to a concrete pack; import it lazily so the AV path never
    # touches it and no static engine->pack edge exists.
    from packs.registry import get_pack

    return load_ontology(get_pack(pack_id).ontology.yaml_path)


# Preserve the historical `get_ontology.cache_clear()` call surface (add_custom_class, tests) now that the
# cache lives on the inner function.
get_ontology.cache_clear = _get_ontology_cached.cache_clear  # type: ignore[attr-defined]


# Strings that are always a bug rather than a class. A caller that stringifies an empty variable produces
# one of these, and they pass every emptiness check because they are not empty.
_RESERVED_CLASS_NAMES = frozenset({
    "undefined", "null", "none", "nan", "true", "false", "nil", "unknown_class", "object_object",
})


def add_custom_class(name: str, l0: str = "object", l1: str = "custom", india: bool = True) -> dict:
    """Add an annotator-defined class to the sidecar and make it live (cache cleared). Idempotent: an
    existing name returns the existing class. Names are normalized to the ontology's snake_case style."""
    norm = normalize_class_name(name)
    if not norm:
        raise ValueError("class name must contain letters or digits")
    if norm in _RESERVED_CLASS_NAMES:
        # Never a name somebody meant. These are what a client sends when a variable was empty and got
        # stringified on the way out, and the corpus already carries one: a class literally called
        # `undefined` at id 229, with 262 objects labelled into it before anything noticed. The client
        # guards emptiness, but a coerced placeholder is a non-empty string and walks straight through, so
        # the refusal belongs here - the sidecar is a durable artifact and the vocabulary every later label
        # is drawn from.
        raise ValueError(f"'{norm}' is a placeholder, not a class name; it usually means the caller sent an "
                         f"empty value that was stringified")
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
    _get_ontology_cached.cache_clear()
    return {"id": new_id, "name": norm, "l0": l0, "l1": l1, "india": bool(india), "existed": False}
