"""Making the shortlist a constraint the model obeys instead of a request it can decline.

Path C offers the VLM a per-object shortlist: the object's current class, the cross-superclass anchors, its
L1 siblings, and object_fallback, capped at 20 names. That shortlist went into the prompt as English, and
what came back was checked afterwards:

    if res.class_name and not self.onto.has_name(res.class_name):
        res.class_name = None

So an out-of-vocabulary answer was not repaired, it was thrown away, and with it the crop, the prefill and
the decode. Path C runs only on the uncertain subset, which is the expensive subset by construction, so the
discarded work is the costly work. The corpus has already paid for this once in a different currency: 69% of
traffic-sign detections were not signs, including a bus, produced through exactly this
shortlist-as-suggestion path.

Both local backends can enforce it instead. Ollama takes a JSON schema in `format`, and llama-server takes
one in `response_format` and compiles it to GBNF itself. That server-side compilation is the reason there is
no hand-written GBNF here: emitting a grammar by hand would be a second encoding of the same schema to keep
in sync with the first, and the sync failure would be silent.

The second thing this buys is prefix reuse, and it only works because of where the constraint lives. The
shortlist varies per object, so while it sat in the prompt every object was a fresh prefix past the pack
preamble and `--cache-reuse` had almost nothing to hold. A schema is a sampler constraint rather than KV
state, so it can vary per object at no cache cost at all. Moving the shortlist out of the prompt and into the
schema therefore makes the prompt byte-identical across every object in a sweep, which is what makes the
cache worth enabling. Tighter constraints and better caching are usually a trade; here they are the same
change.
"""

from __future__ import annotations

import json

# Ontology attribute types, mapped to JSON Schema. `bool_array` is a per-instance list (helmet, one entry per
# rider), and `enum` carries its own values, so both need more than a name lookup.
_JSON_TYPE = {
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "str": "string",
    "string": "string",
    "enum": "string",
}


def _attr_property(spec: dict) -> dict:
    """One ontology attribute as a JSON Schema property."""
    kind = str(spec.get("type", "string"))
    values = spec.get("values")

    if kind == "bool_array":
        return {"type": "array", "items": {"type": "boolean"}}
    if values:
        # An enum's values are typed by the values themselves. occlusion is [0, 25, 50, 75, 100], which is
        # integers under a type field that says "enum", and quoting those would make every occlusion value
        # fail the ontology's own validator downstream.
        return {"enum": list(values)}
    return {"type": _JSON_TYPE.get(kind, "string")}


def verdict_schema(shortlist: list[str], attr_schema: dict) -> dict:
    """The JSON Schema a constrained backend enforces for one object.

    `class` is an enum over the shortlist, which is the whole point: the model becomes unable to name a class
    that is not on offer, rather than being asked not to.

    `additionalProperties` is false on the attribute object so an invented attribute cannot ride along. The
    existing validator drops those afterwards anyway, but dropping them after generation means they were
    still sampled, and tokens spent on a field that will be discarded are tokens not spent on the answer.
    """
    props = {name: _attr_property(spec) for name, spec in (attr_schema or {}).items()}
    return {
        "type": "object",
        "properties": {
            "class": {"type": "string", "enum": list(shortlist)},
            "attributes": {"type": "object", "properties": props, "additionalProperties": False},
            "caption": {"type": "string"},
            "confident": {"type": "boolean"},
        },
        # Deliberately not requiring `attributes` or `caption`. An attribute that does not apply should be
        # absent, and forcing the field would make the model invent one to satisfy the schema.
        "required": ["class", "confident"],
        "additionalProperties": False,
    }


def static_prompt(attr_schema: dict) -> str:
    """The prompt for a backend that enforces the schema.

    Carries no shortlist, so it is byte-identical for every object sharing an ontology and a pack, which is
    what lets a server prefix-cache it once per slot instead of once per object. The attribute schema stays
    in the text because naming the fields improves the values chosen within them; the schema constrains their
    shape but cannot explain what `occlusion` means.
    """
    from services.domain import active_pack

    template = active_pack().autolabel_profile.vlm_prompt_template
    return (
        f"{template}\n"
        f"Attribute schema (return only those that apply): {json.dumps(attr_schema, sort_keys=True)}.\n"
        "Choose exactly one class from the permitted set. "
        'Respond with strict JSON only, no prose: '
        '{"class": "<permitted class>", "attributes": {<name>: <value>}, '
        '"caption": "<short description>", "confident": <true|false>}.'
    )


def permitted_classes(schema: dict) -> list[str]:
    """The classes a schema allows, for logging and for tests that must not reach into the shape by hand."""
    return list(((schema.get("properties") or {}).get("class") or {}).get("enum") or [])
