"""The per-project labeling interface, and the validation that makes it mean something.

A project declares what may be labeled and with what controls. Everything written to `annotation` is checked
against BOTH the structural rules of its kind (a bbox needs four ordered numbers; a span needs start < end)
and the project's config (this label exists here; this field is one of the allowed enum values).

Validating on the way in is the whole point. A JSONB payload column is what lets one table hold boxes, text
spans, audio regions and preference rankings, and without a gate at the door that same flexibility silently
accumulates malformed rows that only surface much later, in an export, as a corrupt dataset.

The field-type vocabulary is deliberately the one the AV ontology already uses (`AttributeDef` in
services/autolabel/ontology.py): enum, float, int, bool, plus text. One notion of "a typed label attribute"
across the whole product rather than two that drift.

Config shape:
    {
      "media": "audio",
      "labels": [{"name": "speech", "color": "#4c8dff", "kinds": ["region"]}, ...],
      "fields": [{"name": "speaker", "type": "enum", "values": ["a", "b"], "required": false}, ...],
      "allow_kinds": ["region", "classification"]      # optional whitelist, else derived from labels
    }
"""

from __future__ import annotations

from typing import Any

# Every annotation shape the project spine understands, and what its payload must contain.
KINDS: dict[str, str] = {
    "bbox": "spatial box on an image or document page",
    "polygon": "closed spatial polygon",
    "polyline": "open spatial polyline",
    "keypoints": "ordered points with visibility",
    "mask": "raster or RLE mask reference",
    "span": "character range in a text asset",
    "relation": "a directed link between two annotations",
    "region": "time interval on audio, video or a time series",
    "classification": "a whole-asset label or choice set",
    "transcription": "text content for an asset or a region of it",
    "preference": "which of N candidate responses a human preferred",
    "rubric": "scored criteria for one response",
    "ranking": "a full ordering over candidate responses",
}

FIELD_TYPES = ("enum", "float", "int", "bool", "text")


class ConfigError(ValueError):
    """The project's label_config is malformed."""


class PayloadError(ValueError):
    """An annotation does not satisfy its kind or the project's config."""


def validate_config(config: dict | None) -> dict:
    """Check a project label_config and return it normalized. An empty config is legal and means the project
    inherits the AV ontology (the existing default), so adopting this layer is opt-in."""
    if not config:
        return {}
    if not isinstance(config, dict):
        raise ConfigError("label_config must be an object")

    labels = config.get("labels") or []
    if not isinstance(labels, list):
        raise ConfigError("labels must be a list")
    seen: set[str] = set()
    norm_labels = []
    for entry in labels:
        if isinstance(entry, str):
            entry = {"name": entry}
        if not isinstance(entry, dict) or not str(entry.get("name", "")).strip():
            raise ConfigError("each label needs a name")
        name = str(entry["name"]).strip()
        if name in seen:
            raise ConfigError(f"duplicate label {name!r}")
        seen.add(name)
        kinds = entry.get("kinds") or []
        for k in kinds:
            if k not in KINDS:
                raise ConfigError(f"label {name!r} allows unknown kind {k!r}")
        norm_labels.append({"name": name, "color": entry.get("color"), "kinds": list(kinds)})

    fields = config.get("fields") or []
    if not isinstance(fields, list):
        raise ConfigError("fields must be a list")
    norm_fields = []
    for f in fields:
        if not isinstance(f, dict) or not str(f.get("name", "")).strip():
            raise ConfigError("each field needs a name")
        ftype = f.get("type", "text")
        if ftype not in FIELD_TYPES:
            raise ConfigError(f"field {f['name']!r} has unknown type {ftype!r} (allowed: {FIELD_TYPES})")
        if ftype == "enum" and not f.get("values"):
            raise ConfigError(f"enum field {f['name']!r} needs values")
        norm_fields.append({"name": str(f["name"]).strip(), "type": ftype,
                            "values": list(f.get("values") or []),
                            "required": bool(f.get("required", False))})

    allow = config.get("allow_kinds") or []
    for k in allow:
        if k not in KINDS:
            raise ConfigError(f"allow_kinds contains unknown kind {k!r}")

    out = {"labels": norm_labels, "fields": norm_fields, "allow_kinds": list(allow)}
    if config.get("media"):
        out["media"] = config["media"]
    return out


def _nums(v: Any, n: int, what: str) -> list[float]:
    if not isinstance(v, list | tuple) or len(v) != n:
        raise PayloadError(f"{what} must be a list of {n} numbers")
    try:
        return [float(x) for x in v]
    except (TypeError, ValueError):
        raise PayloadError(f"{what} must be numeric") from None


def validate_payload(kind: str, payload: dict | None) -> dict:
    """Structural validation for one annotation kind. Returns the normalized payload."""
    if kind not in KINDS:
        raise PayloadError(f"unknown kind {kind!r} (allowed: {sorted(KINDS)})")
    p = dict(payload or {})

    if kind == "bbox":
        box = _nums(p.get("bbox"), 4, "bbox")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise PayloadError("bbox must be [x1,y1,x2,y2] with x2 > x1 and y2 > y1")
        p["bbox"] = box

    elif kind in ("polygon", "polyline"):
        pts = p.get("points")
        if not isinstance(pts, list) or len(pts) < (3 if kind == "polygon" else 2):
            raise PayloadError(f"{kind} needs at least {3 if kind == 'polygon' else 2} points")
        p["points"] = [_nums(pt, 2, "point") for pt in pts]

    elif kind == "keypoints":
        pts = p.get("points")
        if not isinstance(pts, list) or not pts:
            raise PayloadError("keypoints needs points")
        norm = []
        for pt in pts:
            if not isinstance(pt, list | tuple) or len(pt) != 3:
                raise PayloadError("each keypoint must be [x, y, visibility]")
            norm.append([float(pt[0]), float(pt[1]), int(pt[2])])
        p["points"] = norm

    elif kind == "mask":
        if not p.get("uri") and not p.get("rle") and not p.get("polygons"):
            raise PayloadError("mask needs one of uri, rle or polygons")

    elif kind == "span":
        try:
            start, end = int(p["start"]), int(p["end"])
        except (KeyError, TypeError, ValueError):
            raise PayloadError("span needs integer start and end") from None
        if start < 0 or end <= start:
            raise PayloadError("span needs 0 <= start < end")
        p["start"], p["end"] = start, end

    elif kind == "relation":
        if not p.get("from_annotation_id") or not p.get("to_annotation_id"):
            raise PayloadError("relation needs from_annotation_id and to_annotation_id")
        if p["from_annotation_id"] == p["to_annotation_id"]:
            raise PayloadError("a relation cannot point at itself")

    elif kind == "region":
        try:
            t0, t1 = float(p["t_start"]), float(p["t_end"])
        except (KeyError, TypeError, ValueError):
            raise PayloadError("region needs numeric t_start and t_end (seconds)") from None
        if t0 < 0 or t1 <= t0:
            raise PayloadError("region needs 0 <= t_start < t_end")
        p["t_start"], p["t_end"] = t0, t1
        if p.get("channel") is not None:
            p["channel"] = str(p["channel"])

    elif kind == "classification":
        if "choices" in p:
            if not isinstance(p["choices"], list):
                raise PayloadError("choices must be a list")
            p["choices"] = [str(c) for c in p["choices"]]

    elif kind == "transcription":
        if not str(p.get("text", "")).strip():
            raise PayloadError("transcription needs text")

    elif kind == "preference":
        cands = p.get("candidates")
        if not isinstance(cands, list) or len(cands) < 2:
            raise PayloadError("preference needs at least 2 candidates")
        if "chosen" not in p:
            raise PayloadError("preference needs a chosen index")
        chosen = int(p["chosen"])
        if not 0 <= chosen < len(cands):
            raise PayloadError("chosen must index into candidates")
        p["chosen"] = chosen

    elif kind == "rubric":
        scores = p.get("scores")
        if not isinstance(scores, dict) or not scores:
            raise PayloadError("rubric needs a scores object")
        p["scores"] = {str(k): float(v) for k, v in scores.items()}

    elif kind == "ranking":
        order = p.get("order")
        if not isinstance(order, list) or len(order) < 2:
            raise PayloadError("ranking needs an order of at least 2 items")
        if len(set(map(str, order))) != len(order):
            raise PayloadError("ranking order must not repeat an item")

    return p


def validate_fields(config: dict | None, fields: dict | None) -> dict:
    """Check typed per-annotation fields against the project's declared field schema."""
    schema = {f["name"]: f for f in (validate_config(config).get("fields") or [])}
    vals = dict(fields or {})
    for name, f in schema.items():
        if f["required"] and name not in vals:
            raise PayloadError(f"field {name!r} is required")
    for name, v in list(vals.items()):
        f = schema.get(name)
        if f is None:
            continue          # unknown fields pass through; the config is a guide, not a straitjacket
        t = f["type"]
        try:
            if t == "enum":
                if str(v) not in [str(x) for x in f["values"]]:
                    raise PayloadError(f"field {name!r} must be one of {f['values']}")
            elif t == "float":
                vals[name] = float(v)
            elif t == "int":
                vals[name] = int(v)
            elif t == "bool":
                vals[name] = bool(v)
            else:
                vals[name] = str(v)
        except (TypeError, ValueError):
            raise PayloadError(f"field {name!r} is not a valid {t}") from None
    return vals


def check_label_allowed(config: dict | None, label: str | None, kind: str) -> None:
    """A label must exist in the config, and if it restricts kinds, this kind must be one of them. A project
    with no configured labels accepts anything, which is what keeps the AV ontology projects working."""
    cfg = validate_config(config)
    labels = cfg.get("labels") or []
    if not labels:
        return
    allow = cfg.get("allow_kinds") or []
    if allow and kind not in allow:
        raise PayloadError(f"kind {kind!r} is not allowed in this project (allowed: {allow})")
    if label is None:
        return
    entry = next((x for x in labels if x["name"] == label), None)
    if entry is None:
        raise PayloadError(f"label {label!r} is not declared in this project")
    if entry["kinds"] and kind not in entry["kinds"]:
        raise PayloadError(f"label {label!r} cannot be used as {kind!r} (allowed: {entry['kinds']})")
