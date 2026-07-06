"""Byte-stable reproducibility (M19). A dataset a buyer paid for must be reproducible: the same release built
twice must produce the same bytes, or lineage and audit are meaningless. This gives a canonical serialization
that quantizes floats to a fixed precision (so harmless floating-point jitter does not change the hash),
content-hashes it, and compares two build outputs for byte-stable equality, naming the first field that
diverged when they differ. Pure and deterministic, so reproducibility is itself testable."""

from __future__ import annotations

import hashlib
import json

FLOAT_PRECISION = 6


def _quantize(obj):
    """Recursively round floats to FLOAT_PRECISION so floating-point jitter does not break byte-stability, and
    sort nothing here (json.dumps with sort_keys handles ordering)."""
    if isinstance(obj, float):
        return round(obj, FLOAT_PRECISION)
    if isinstance(obj, dict):
        return {k: _quantize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_quantize(v) for v in obj]
    return obj


def canonical_bytes(obj) -> bytes:
    """A canonical, deterministic serialization: quantized floats, sorted keys, no insignificant whitespace."""
    return json.dumps(_quantize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def content_hash(obj) -> str:
    """The SHA-256 content hash of the canonical serialization. The byte-stable identity of a build output."""
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def _first_divergence(a, b, path="") -> str | None:
    """The path of the first field where two structures diverge, for a readable reproducibility failure."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            d = _first_divergence(a.get(k), b.get(k), f"{path}.{k}" if path else str(k))
            if d is not None:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}[len {len(a)} != {len(b)}]"
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_divergence(x, y, f"{path}[{i}]")
            if d is not None:
                return d
        return None
    return None if _quantize(a) == _quantize(b) else (path or "<root>")


def check_reproducible(run_a, run_b) -> dict:
    """Compare two build outputs for byte-stable reproducibility. Returns whether the content hashes match and,
    when they do not, the path of the first diverging field."""
    ha, hb = content_hash(run_a), content_hash(run_b)
    reproducible = ha == hb
    return {"reproducible": reproducible, "hash_a": ha, "hash_b": hb,
            "first_divergence": None if reproducible else _first_divergence(run_a, run_b)}
