"""Put a model this engine did not train into the registry, so the gold harness can score it.

The registry has always accepted a `weights_uri`, but nothing could ever produce one except this engine's own
training job: there is no upload route in any router, so a model trained elsewhere had no way in. That made
the one genuinely useful thing about an external model, an independent number on sealed gold, unreachable.

The interesting part is not the upload. It is the vocabulary check.

`services/training/gold.py:_materialize_aligned` aligns the gold set to the model's own class order and drops
every gold object whose ontology class the model does not know, which is correct: you cannot score a model on
a class it was never taught. But the drop is silent, and the val pass afterwards reports a single confident
mAP50 with no indication of how much of the gold set it actually covered. A detector that knows four of the
ontology's names scores just as cleanly as one that knows forty.

DashLab's released `detector_9class` is the case in point. Four of its nine names exist in this ontology;
`person`, `bicycle`, `car`, `traffic_light` and `stop_sign` do not, because this ontology spells them
`pedestrian`, `cyclist` and so on. Registered naively it would have produced a number about motorcycles,
buses, trucks and autorickshaws, presented as a number about the model.

So this refuses by default when any class is unmapped, and records the coverage it measured on the row, so a
later reader can see what the number was computed over instead of inferring it was everything.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
import tempfile
import urllib.request
from pathlib import Path

from core.logging import get_logger
from core.storage import get_object_store
from db.session import get_sessionmaker
from services.autolabel.ontology import get_ontology
from services.govern.registry import register

log = get_logger("register_external_model")


def fetch(source: str, dest: Path) -> Path:
    """A local path or an http(s) URL, resolved to a local file."""
    if source.startswith(("http://", "https://")):
        urllib.request.urlopen(source)  # noqa: S310 - operator-supplied release URL
        with urllib.request.urlopen(source) as r, dest.open("wb") as fh:  # noqa: S310
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        return dest
    p = Path(source).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"no such file: {p}")
    return p


def verify(path: Path, expected_sha256: str | None) -> str:
    """The digest, checked against the publisher's if one was given.

    Worth the argument: these weights arrive over the network from a release page, and a model that is not
    the one whose checksum was published is not the model whose numbers get recorded against this name.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    got = h.hexdigest()
    if expected_sha256 and got.lower() != expected_sha256.lower():
        raise SystemExit(f"checksum mismatch\n  expected {expected_sha256}\n  got      {got}")
    return got


def class_names(path: Path) -> list[str]:
    from ultralytics import YOLO

    names = YOLO(str(path)).names
    return [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)


def vocabulary_coverage(names: list[str]) -> dict:
    """Which of the model's classes this ontology can score, and which the gold aligner would drop.

    Mirrors the membership test in `_materialize_aligned` exactly (an ontology name, matched literally), so
    what this reports is what that will do rather than an approximation of it.
    """
    have = {c.name for c in get_ontology().classes}
    matched = [n for n in names if n in have]
    dropped = [n for n in names if n not in have]
    return {"model_classes": len(names), "scorable": matched, "unscorable": dropped,
            "coverage": round(len(matched) / len(names), 4) if names else 0.0}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", help="local .pt path or an https URL to one")
    ap.add_argument("--model-version", required=True, help="registry key, e.g. dashlab-det9class-v0.1.0")
    ap.add_argument("--task", default="detection")
    ap.add_argument("--sha256", default=None, help="publisher's digest, verified before anything is stored")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--allow-partial-vocabulary", action="store_true",
                    help="register even though some classes cannot be scored; the gold number will then "
                         "describe only the classes listed as scorable")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        local = fetch(args.source, Path(td) / "weights.pt")
        digest = verify(local, args.sha256)
        names = class_names(local)
        cov = vocabulary_coverage(names)

        print(f"  model      {args.model_version}")
        print(f"  sha256     {digest}")
        print(f"  classes    {cov['model_classes']}  ->  coverage {cov['coverage']:.0%}")
        print(f"  scorable   {', '.join(cov['scorable']) or '(none)'}")
        print(f"  unscorable {', '.join(cov['unscorable']) or '(none)'}")

        if cov["unscorable"] and not args.allow_partial_vocabulary:
            print("\n  refusing: the gold aligner drops every object of an unscorable class without saying so,\n"
                  "  so the resulting mAP would describe only the scorable ones while looking like a number\n"
                  "  about the whole model. Map these names into the ontology, or pass\n"
                  "  --allow-partial-vocabulary to record the gap on the row and proceed.", file=sys.stderr)
            return 2
        if not cov["scorable"]:
            print("\n  refusing: no class of this model exists in the ontology, so there is nothing to score.",
                  file=sys.stderr)
            return 2

        store = get_object_store()
        store.ensure_bucket()
        weights_uri = store.put_file(f"models/{args.model_version}/best.pt", local,
                                     "application/octet-stream")

    note = args.notes or f"external model, sha256={digest}"
    if cov["unscorable"]:
        note += (f"; scored on {len(cov['scorable'])}/{cov['model_classes']} classes, "
                 f"unscorable: {','.join(cov['unscorable'])}")

    async with get_sessionmaker()() as db:
        # gold_metrics stays empty: this model has not been scored yet, and seeding it with the publisher's
        # own numbers would put a figure this engine did not measure in front of the promotion gate.
        row = await register(db, args.model_version, args.task, {}, None, weights_uri, note)

    print(f"\n  registered  {weights_uri}")
    print(f"  next        evaluate it on sealed gold, then the gate decides: {row.get('model_version')}")
    log.info("external_model.registered", model_version=args.model_version, coverage=cov["coverage"],
             unscorable=cov["unscorable"], sha256=digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
