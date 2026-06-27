"""Inspect an extracted IDD-Detection directory before converting: confirm the layout, count
images/annotations, list the class names IDD actually uses (so you can tune IDD_TO_ONTOLOGY in
idd_to_yolo.py), and show one sample annotation.

    python scripts/inspect_idd.py --idd-root /data/IDD_Detection
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from services.autolabel.ontology import get_ontology


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idd-root", required=True)
    args = ap.parse_args()
    root = Path(args.idd_root)
    onto = get_ontology()

    print(f"== inspecting {root} ==")
    for sub in ("JPEGImages", "Annotations"):
        d = root / sub
        n = sum(1 for _ in d.rglob("*")) if d.exists() else 0
        print(f"  {sub}/: {'present' if d.exists() else 'MISSING'} ({n} entries)")
    for split in ("train.txt", "val.txt", "test.txt"):
        f = root / split
        n = len(f.read_text().splitlines()) if f.exists() else 0
        print(f"  {split}: {'present' if f.exists() else 'absent'} ({n} lines)")

    xmls = list((root / "Annotations").rglob("*.xml")) if (root / "Annotations").exists() else []
    print(f"\n  total annotation XMLs: {len(xmls)}")
    if not xmls:
        print("  no XMLs found — is this the IDD_Detection root? (expects Annotations/<scene>/*.xml)")
        return

    classes: Counter = Counter()
    for x in xmls[:5000]:
        try:
            for obj in ET.parse(x).getroot().findall("object"):
                nm = obj.findtext("name", "").strip()
                if nm:
                    classes[nm] += 1
        except Exception:
            continue

    print(f"\n  IDD class names (sampled) -> mapped ontology name:")
    from scripts.idd_to_yolo import _map_name

    for name, cnt in classes.most_common():
        mapped = _map_name(name, onto)
        flag = "" if (mapped and mapped != "object_fallback") else "  <-- maps to fallback, consider editing IDD_TO_ONTOLOGY"
        print(f"    {name:24} ({cnt:6})  -> {mapped}{flag}")

    sample = next((x for x in xmls), None)
    if sample:
        print(f"\n  sample annotation: {sample}")
        print("  " + "\n  ".join(sample.read_text().splitlines()[:14]))


if __name__ == "__main__":
    main()
