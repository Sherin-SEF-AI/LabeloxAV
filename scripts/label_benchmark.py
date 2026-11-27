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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB_STAGES = ["detector", "vlm_stored", "reasoner", "memory", "probe", "zs_all_stored", "family_zs", "agent"]
# Read stored SigLIP 2 vectors and run only the text encoder; inside office hours that runs on the CPU.
TEXT_STAGES = {"zs_all_stored", "family_zs", "agent"}
MODEL_STAGES = ["classify_crop", "agent_vlm"]


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
    ap.add_argument("--vlm-max", type=int, default=None, help="cap the questions the VLM is asked")
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

    if any(s in TEXT_STAGES for s in stages) and in_quiet_hours():
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ""      # keep the GPU free during office hours

    from db.session import get_sessionmaker
    from services.labelagent.report import build_report, save_report

    async with get_sessionmaker()() as db:
        from services.labelagent.agent import _stamp

        report = await build_report(db, stages, split=args.split, vlm_max=args.vlm_max,
                                    on_row=lambda name, sp, rep: print(_row(f"{name} [{sp}]", rep, rep["coverage"])))
        report["rulings_at"] = (await _stamp(db))[0]
    print(f"\ntruth: {report['truth']}\ntest split: {report['truth_test']}")

    blob = json.dumps(report, indent=1).encode()
    if args.out:
        Path(args.out).write_bytes(blob)
        print(f"\nwrote {args.out}")
    if args.save:
        print(f"\nsaved {save_report(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
