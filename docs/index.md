# LabeloxAV

<p align="center">
  <img src="logo.png" alt="LabeloxAV" width="380">
</p>

**A data engine for autonomous driving, built for Indian roads.**

LabeloxAV takes raw fleet footage, machine-labels it through a three-path fusion pipeline, routes every label
through a confidence gate to human review, mines the rare and risky moments, and retrains its own models in a
closed loop. One ontology, 178 governed classes, tuned for what global datasets never saw: autorickshaws,
cattle on the carriageway, overloaded two-wheelers, hand carts, potholes.

[Get started](getting-started.md){ .md-button .md-button--primary }
[REST API](api/rest.md){ .md-button }
[Engineering log](ENGINEERING_LOG.md){ .md-button }

## What actually runs

| Path | Role | Model in this build |
| --- | --- | --- |
| `path_a_detect` | Closed-set detector | `yolo11l.pt` (target YOLO26; weights swap by config) |
| `path_b_openvocab` | Open-vocabulary + segmentation | YOLO-World + `sam_b.pt` |
| `path_c_vlm` | VLM verifier | `qwen2.5vl:7b` via Ollama |

Fused proposals are calibrated with isotonic regression, fit against a judge whose own sensitivity and
specificity are measured and corrected for. Calibrated confidence then routes each object to `auto_accept`,
`review` or `annotate`.

!!! warning "What the gate's thresholds mean"
    `auto_accept` sits at 0.45, safety classes at 0.47, on a calibrated scale topping out near 0.48. Those
    are **configured constants, not measured precision floors.** A per-class fitted operating point replaces
    them where one exists, and the gate logs which it used. The realized precision of the auto-accepted
    subset has not been measured against human verdicts. See [Measurement](MEASUREMENT.md).

## Honest numbers

Measured 2026-08-27 against the live corpus.

| | |
| --- | --- |
| Objects / human-verified | 578,399 / 1,577 (0.27%) |
| Sessions | 377, of which 98.9% are one city |
| Per-class label precision | `motorcycle` 0.87, `pedestrian` 0.87, `sedan` 0.24, `traffic_signal` 0.05, `object_fallback` 0.00 |
| Auto-accepted subset | 0.93 strict, machine-judged, **not** measured against humans |
| Blind recall audit | seeded, unscored - recall is against labels somebody already found |

!!! note "The corpus shrank by 19% on 2026-08-27, deliberately"
    A gap-filling pass had interpolated between track endpoints that were not the same object; the 137,904
    objects it produced judged at 0.209 against 0.603 for real detections. Reverting it moved 11 of 13
    measured classes up. Every number above is post-revert.

Every number above is produced by a script in this repo and every gap is stated rather than omitted. The
per-class table comes from `scripts/run_class_precision.py`, which judges a hash-stable random sample per
class and reports both the raw Wilson interval and a Rogan-Gladen correction through the judge's own
measured error.

## Design commitments

**Everything corpus-wide is reversible.** Any sweep that touches labels records an `AgentRun` and can be
undone with one call. A path that cannot be undone says so in its docstring.

**The domain is a pack, not the engine.** `packs/av` and `packs/sec` serve autonomous driving and physical
security from one engine. `.importlinter` forbids `core`, `services` and `db` from importing either.

**Machine opinion is not human opinion.** A VLM verdict is written to `machine_verdict`, never to `review`.
Precision sampling, corpus precision and annotator scorecards all read `review`, and mixing the two would
corrupt all three invisibly.

**Absence is reported, not defaulted.** A quantity that cannot be measured honestly prints the reason
instead of a number. Every export ships a coverage datasheet built on that rule.
