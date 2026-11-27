"""Score classifier stages against the human-ruling answer key.

    .venv/bin/python scripts/label_benchmark.py                       # database-only stages, test split
    .venv/bin/python scripts/label_benchmark.py --stages all --save   # every stage, stored in MinIO

Stages that load a model (classify_crop) do not start during office hours unless told to, the same rule
auto-labelling follows. The report goes to stdout as a table, and with --save to
`labelagent/benchmarks/<utc>.json` and `labelagent/benchmarks/latest.json` in the object store.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_STAGES = ["detector", "vlm_stored", "reasoner"]
# Stages that learn nothing from the calibration split: every item is unseen to them, so they are also
# scored on the whole set, which is what gives the wrong-fix bound enough right labels to mean something.
UNTRAINED = {"detector", "vlm_stored", "reasoner", "classify_crop"}
MODEL_STAGES = ["classify_crop"]
CROP_FLOORS = (0.0, 0.5, 0.8)


def _row(name: str, rep: dict, cov: float | None) -> str:
    h, p = rep["headline"], rep["pooled"]

    def f(v):
        return "   -  " if v is None else f"{v * 100:5.1f}%"

    return (f"{name:<22} {h['n']:>5}  fix {f(h['fix_rate'])}  broke {f(h['wrong_fix_rate'])}"
            f" (<= {f(h['wrong_fix_upper95'])}, n={h['n_right_labels']})"
            f"  pooled-broke {f(p['wrong_fix_rate'])} (<= {f(p['wrong_fix_upper95'])}, n={p['n_right_labels']})"
            f"  chg-prec {f(h['change_precision'])}  notobj {f(h['not_object_catch'])}"
            f" / fa {f(h['not_object_false_alarm'])}  cov {f(cov)}")


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default=",".join(DB_STAGES),
                    help=f"comma list from {DB_STAGES + MODEL_STAGES}, or 'all'")
    ap.add_argument("--split", default="test", choices=["test", "calib", "all"])
    ap.add_argument("--save", action="store_true", help="store the report in the object store")
    ap.add_argument("--out", help="also write the report JSON to this local path")
    ap.add_argument("--allow-office-hours", action="store_true",
                    help="run model stages even inside the quiet window")
    args = ap.parse_args()

    stages = DB_STAGES + MODEL_STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = set(stages) - set(DB_STAGES + MODEL_STAGES)
    if unknown:
        ap.error(f"unknown stages: {sorted(unknown)}")

    from services.tools.registry import in_quiet_hours
    if any(s in MODEL_STAGES for s in stages) and in_quiet_hours() and not args.allow_office_hours:
        print("model stages wait for office hours to end (10:00-18:00 IST); "
              "rerun later or pass --allow-office-hours", file=sys.stderr)
        return 2

    from db.session import get_sessionmaker
    from services.autolabel.ontology import get_ontology
    from services.labelagent import benchmark as B
    from services.labelagent.truth import load_all, summary

    onto = get_ontology()
    report: dict = {"created": datetime.now(UTC).isoformat(timespec="seconds"), "split": args.split, "stages": {}}
    async with get_sessionmaker()() as db:
        items = await load_all(db)
        report["truth"] = summary(items)
        report["truth_test"] = summary([i for i in items if i.split == "test"])
        print(f"truth: {report['truth']}")
        print(f"test split: {report['truth_test']}\n")

        for s in stages:
            t0 = time.perf_counter()
            if s == "detector":
                runs = {"detector": B.stage_detector(items)}
            elif s == "vlm_stored":
                runs = {"vlm_stored": await B.stage_vlm_stored(db, items)}
            elif s == "reasoner":
                runs = {"reasoner": await B.stage_reasoner(db, items)}
            elif s == "classify_crop":
                raw = await B.stage_classify_crop(db, items)
                runs = {f"classify_crop@{fl:g}": B.with_floor(raw, fl) for fl in CROP_FLOORS}
            dt = time.perf_counter() - t0
            for name, preds in runs.items():
                splits = [args.split] + (["all"] if s in UNTRAINED and args.split != "all" else [])
                entry = {"seconds": round(dt, 1)}
                for sp in splits:
                    rep = B.score(items, preds, onto, split=sp)
                    rep["coverage"] = B.coverage([p for p, i in zip(preds, items, strict=True)
                                                  if sp == "all" or i.split == sp])
                    entry[sp] = rep
                    print(_row(f"{name} [{sp}]", rep, rep["coverage"]))
                report["stages"][name] = entry

    blob = json.dumps(report, indent=1).encode()
    if args.out:
        Path(args.out).write_bytes(blob)
        print(f"\nwrote {args.out}")
    if args.save:
        from core.storage import get_object_store
        store = get_object_store()
        key = f"labelagent/benchmarks/{report['created'].replace(':', '')}.json"
        store.put_bytes(key, blob, "application/json")
        store.put_bytes("labelagent/benchmarks/latest.json", blob, "application/json")
        print(f"\nsaved {store.uri(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
