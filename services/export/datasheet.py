"""Templated buyer-diligence documents: dataset datasheets, model cards, and weekly quality reports.

Pure renderers (data in, Markdown out) so they are unit-testable; the Documentation Agent harvests the
inputs from the analytics + governance surfaces and stores the rendered artifact. These are the artifacts
buyer diligence asks for, and they should never be hand-written twice.
"""

from __future__ import annotations


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{round(float(x) * 100, 1)}%"


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_datasheet(*, title: str, size: dict, coverage: dict, class_dist: list[dict],
                     scene: dict, geo: dict, quality: dict | None = None) -> str:
    lines = [f"# Dataset datasheet: {title}", ""]
    lines += ["## Composition", "",
              _table(["metric", "value"], [
                  ["sessions", size.get("sessions", "n/a")],
                  ["frames", size.get("frames", "n/a")],
                  ["objects", size.get("objects", "n/a")],
                  ["human-verified", size.get("human_labeled", size.get("human_touched", "n/a"))],
                  ["ontology version", size.get("ontology_version", "n/a")],
              ]), ""]
    top = sorted(class_dist, key=lambda c: c.get("count", 0), reverse=True)[:12]
    lines += ["## Class distribution (top 12)", "",
              _table(["class", "l1", "count"], [[c.get("name"), c.get("l1"), c.get("count")] for c in top]), ""]
    lines += ["## Scene coverage", ""]
    for axis, dist in (scene or {}).items():
        if isinstance(dist, dict):
            lines.append(f"- **{axis}**: " + ", ".join(f"{k} {v}" for k, v in dist.items()))
    lines += ["", "## Geographic coverage", "",
              ", ".join(f"{k} {v}" for k, v in (geo or {}).items()) or "not tagged", ""]
    gaps = coverage.get("gaps", []) if coverage else []
    lines += ["## Known gaps (collection guidance)", ""]
    lines += ([f"- {g}" for g in gaps] or ["- none flagged"]) + [""]
    if quality and quality.get("metrics"):
        m = quality["metrics"]
        lines += ["## Measured quality (gold set)", "",
                  _table(["metric", "value"], [
                      ["mAP50", m.get("map50")], ["mAP", m.get("map")],
                      ["safe-mIoU", m.get("safe_miou")], ["objects", quality.get("n_objects")],
                  ]), ""]
    lines += ["## Provenance", "",
              "Every label is provenance-stamped (model path or human reviewer) and reversible. Faces and "
              "license plates are blurred at ingest under DPDPA before any frame reaches storage.", ""]
    return "\n".join(lines)


def render_model_card(*, model: dict, dataset_commit: dict | None = None) -> str:
    gm = model.get("gold_metrics") or {}
    lines = [f"# Model card: {model.get('model_version')}", "",
             f"- **Task**: {model.get('task')}",
             f"- **Champion**: {'yes' if model.get('is_champion') else 'no'}",
             f"- **Promoted from**: {model.get('promoted_from') or 'initial'}",
             f"- **Weights**: {model.get('weights_uri')}",
             f"- **Training data commit**: {model.get('dataset_commit') or 'n/a'}", "",
             "## Gold-set metrics", "",
             _table(["metric", "value"], [
                 ["mAP50", gm.get("map50")], ["mAP", gm.get("map")],
                 ["safe-mIoU", gm.get("safe_miou")],
                 ["precision", gm.get("precision")], ["recall", gm.get("recall")],
             ]), ""]
    per = gm.get("per_class_pr") or {}
    if per:
        rows = [[k, _pct(v.get("precision")), _pct(v.get("recall")), v.get("ap50")] for k, v in list(per.items())[:12]]
        lines += ["## Per-class (top 12)", "", _table(["class", "precision", "recall", "ap50"], rows), ""]
    if dataset_commit:
        lines += ["## Training data", "",
                  f"- commit {dataset_commit.get('commit_id')}, {dataset_commit.get('object_count')} objects, "
                  f"ontology {dataset_commit.get('ontology_version')}", ""]
    lines += ["## Intended use and limitations", "",
              "Trained for Indian-road perception (dense mixed traffic, autorickshaws, two-wheelers with "
              "riders). Safety classes (VRU, animal) are gated to a higher auto-accept bar and monitored for "
              "recall regression. Not validated outside the covered cities and scene mix above.", ""]
    return "\n".join(lines)


def render_weekly_report(*, precision: dict, drift: list[dict], promotions: list[dict],
                         coverage_gaps: list[str]) -> str:
    lines = ["# Weekly quality report", "",
             "## Auto-accept precision", "",
             f"- measured precision: {_pct(precision.get('precision'))} "
             f"over {precision.get('reviewed', 0)} reviewed controls ({precision.get('pending', 0)} pending)", "",
             "## Drift events", ""]
    breached = [d for d in drift if d.get("breach")]
    lines += ([f"- {d['metric']} = {d['value']} (breach)" for d in breached] or ["- none breached this week"]) + [""]
    lines += ["## Model promotions", ""]
    lines += ([f"- {p.get('subject')}: {p.get('decision')}" for p in promotions] or ["- none this week"]) + [""]
    lines += ["## Open collection gaps", ""]
    lines += ([f"- {g}" for g in coverage_gaps[:8]] or ["- none flagged"]) + [""]
    return "\n".join(lines)
