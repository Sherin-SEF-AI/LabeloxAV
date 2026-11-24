"""Fail on a dependency advisory nobody has looked at yet.

    python scripts/audit_deps.py            check installed packages against the baseline
    python scripts/audit_deps.py --update   rewrite the baseline to the advisories present now

`pip-audit --strict` was the CI step, and it could never pass: it tries to look the project itself up on
PyPI, fails, and was therefore run with `|| true`. A step that cannot fail reports nothing, so the pins
went stale unobserved: on 2026-11-24 the installed set carried 159 advisories across 21 packages.

Fixing all of those is a set of upgrades that each need their own test run, not something a lint job can
demand in one go. What a lint job can do is stop the number growing. configs/pip_audit_baseline.txt lists
the advisories known today; this fails on any advisory that is not on that list, which is what a new pin
or a newly published advisory looks like, and passes otherwise. It also names baseline entries that no
longer apply, so the list shrinks as packages are upgraded instead of becoming a permanent excuse.

Exit 0: nothing new. Exit 1: new advisories. Exit 2: pip-audit could not run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "pip_audit_baseline.txt"


def current() -> dict[str, set[str]]:
    """{package: {advisory ids}} for everything installed, the project itself excepted."""
    proc = subprocess.run([sys.executable, "-m", "pip_audit", "--skip-editable", "--progress-spinner", "off",
                           "-f", "json"], capture_output=True, text=True)
    if not proc.stdout.strip():
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(2)
    data = json.loads(proc.stdout)
    deps = data["dependencies"] if isinstance(data, dict) else data
    out: dict[str, set[str]] = {}
    for dep in deps:
        ids = {v["id"] for v in dep.get("vulns", [])}
        if ids:
            out[dep["name"].lower()] = ids
    return out


def read_baseline() -> set[tuple[str, str]]:
    if not BASELINE.exists():
        return set()
    known = set()
    for line in BASELINE.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            pkg, _, vid = line.partition(" ")
            known.add((pkg.lower(), vid.strip()))
    return known


def write_baseline(found: dict[str, set[str]]) -> None:
    lines = ["# Dependency advisories known and not yet fixed. One `package advisory-id` per line.",
             "# scripts/audit_deps.py fails CI on any advisory that is not listed here.",
             "# Remove a line by upgrading the package, then run: python scripts/audit_deps.py --update",
             ""]
    for pkg in sorted(found):
        lines.extend(f"{pkg} {vid}" for vid in sorted(found[pkg]))
    BASELINE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE.write_text("\n".join(lines) + "\n")


def compare(found: dict[str, set[str]], known: set[tuple[str, str]]) -> tuple[list, list]:
    """(new advisories, baseline entries that no longer apply), both sorted."""
    have = {(pkg, vid) for pkg, ids in found.items() for vid in ids}
    return sorted(have - known), sorted(known - have)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update", action="store_true", help="rewrite the baseline to what is present now")
    args = ap.parse_args()

    found = current()
    if args.update:
        write_baseline(found)
        print(f"baseline rewritten: {sum(map(len, found.values()))} advisories in {len(found)} packages")
        return 0

    new, gone = compare(found, read_baseline())
    total = sum(map(len, found.values()))
    print(f"{total} advisories in {len(found)} installed packages; {len(new)} not in the baseline")
    if gone:
        # A different install (CI has no ML extra) legitimately lacks some packages, so this is a note.
        print(f"note: {len(gone)} baseline entries do not apply to this install "
              "(upgraded, or the package is not installed here)")
    if new:
        print("\nNEW advisories (upgrade the package, or add the line to configs/pip_audit_baseline.txt "
              "with a reason):", file=sys.stderr)
        for pkg, vid in new:
            print(f"  {pkg} {vid}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
