# Measurement: how evaluation works, and why the prediction plane exists

This is the document that would have caught the original defect. Read it before touching anything that produces
or consumes a model metric.

## The defect it fixes

Evaluation used to draw "predictions" from live corpus rows: objects on a gold frame whose `source` was one of
the machine sources (`fused`, `auto_accept`, ...). But human review mutates that row in place: confirming a
detection sets `Object.source = "human"`. So every detection the model got right and a human confirmed was
removed from the prediction population by the act of confirming it. What remained as a "prediction" was the
residue a human rejected or never touched. The published precision 0.034 / recall 0.018 on a 400-object gold
slice was an artifact of destructive provenance, not a model result.

The fix is structural, not a patched query: predictions live in their own append-only plane that review never
writes to.

## The two planes

- **Object** is the human-owned annotation. Review writes here (`source = "human"`, edits to class/box/attrs).
- **Prediction** is the model-owned inference output. The inference writer writes here, once per run, verbatim.

They never share a row. This is the invariant. It is stated in `db/models.py::Prediction` and enforced in
review: no code path outside `services/verdyx/inference_run.py` may `UPDATE` or `DELETE` a `Prediction`.

## End to end

1. **Gold sealing** (`services/training/gold.py`). A gold set is sealed from the fleet's own human-verified
   frames (`Object.source == "human" AND state IN (...)`). Sealing freezes the exact object-id list
   (content-addressed `gold_id`) so the yardstick cannot drift. The gold objects are the ground truth.

2. **Inference run** (`services/verdyx/inference_run.py`, CLI `lbx-infer`). A registered model scores a set of
   frames once. Every detection is written as a `Prediction` under one `InferenceRun`. Two deliberate choices:
   - inference runs at a **low conf floor (0.001)**, not the auto-accept threshold, so the full score
     distribution is captured and a PR curve is computable. Gating happens at scoring time, not inference time.
   - a run is keyed by `(model_version, gold_id, code_sha, params)` and de-duplicated, so the same evaluation
     is reproducible rather than recomputed against changing state. `code_sha` (`core/version.py`) ties a run to
     the exact tree that produced it.

3. **Scoring** (`services/analytics/evaluation.py::evaluate_gold_patches`). Takes a `run_id`. Ground truth is
   the sealed gold objects; predictions are the `Prediction` rows of that run, nothing from `Object`. Two
   matches run per frame, for two different questions:
   - a **class-agnostic** greedy IoU match populates the confusion patches (an off-diagonal cell exists only
     when a prediction of one class lands on a gold box of another, so cross-class pairing must be allowed);
   - a **class-aware** match (`core/accel/matching.match_detections`) is what AP is defined on (a true positive
     requires the same class), at each of the ten IoU thresholds 0.5..0.95.
   The metric is our own auditable 101-point interpolated AP (`core/accel/ap.py`), not an opaque val pass.
   Every outcome is persisted as an `EvalPatch` (tp/fp/fn) pointing at its `Prediction` or gold `Object`, so a
   confusion cell opens to the real crops (`GET /api/predictions/{id}/crop`, `GET /api/objects/{id}/crop`).

4. **Provenance report** (`gold_provenance_report`). Splits the gold set into detections a human confirmed
   (recoverable from `Review.before.source`) versus boxes a human drew from scratch. Only the second population,
   plus classes never detected, is a genuine model miss. Conflating the two was half the original confusion.
   Historical objects reviewed before the fix have no recoverable source and are reported as `unknown`.

## The two harnesses, and why both exist

There are two evaluation paths, and they cross-check each other:

- the **val-pass** (`services/govern/gold_eval.py`), the ultralytics `model.val()` over a class-aligned gold
  split, is the long-standing common-yardstick comparison the promotion gate has always used;
- the **prediction plane** (above) is the auditable, inspectable path.

`gold_eval` runs both on the same gold set and model and reconciles them: if the val-pass mAP50 and the
prediction-plane AP50 differ by more than `govern.harness_reconcile_epsilon`, the metrics are flagged
`harness_divergent` and the promotion gate refuses to promote. Two harnesses that disagree is a signal, not
noise.

## The denominator, and why gold recall is an overestimate

Everything above makes the **numerator** honest: predictions live in their own append-only plane, so a
detection a human confirmed is no longer erased from the population it was scored in.

None of it touches the **denominator**. Recall is measured against the sealed gold set, and that set was
built by people reviewing machine proposals. Confirming a machine box costs one click; drawing a box the
machine missed costs half a minute of looking at apparently empty road. The two are not equally likely, so
the gold set leans toward what the model already sees, and gold recall is an overestimate by an amount
nothing in the two harnesses above can detect. Both harnesses share the bias, so their agreement is not
evidence against it.

A **blind audit** (`services/verdyx/blind_audit.py`, `core/accel/recapture.py`) is the fix. A sample of
frames is served to an annotator with every prediction and every existing label withheld, and they label
from scratch. The model and that annotator are then two observers of one population, and Lincoln-Petersen
with Chapman's small-sample correction estimates, in closed form, how many objects **neither** found:

    N_hat = (n1 + 1)(n2 + 1) / (m2 + 1) - 1
    var   = (n1 + 1)(n2 + 1)(n1 - m2)(n2 - m2) / [(m2 + 1)^2 (m2 + 2)]

Four things about it are load-bearing:

- **The blindness is server-side.** `services/api/routers/objects.py` withholds the boxes in the fetch
  handler, and scopes `n_objects` on the frame route to the auditor's own work. Hiding in the editor would
  still ship the predictions to the browser, and an object count is nearly as informative as the objects.
  The same rule (`blind_audit.active_audit_id`) decides both what the auditor may see and what their new
  boxes are stamped with, because if the two ever disagreed the audit would silently collect nothing and
  report the model's recall as perfect.
- **Frames are stratified by prediction density** and sampled evenly across strata, not proportionally.
  Capture probability is not constant: an empty highway and a crowded junction do not share a detection
  rate. Pooling is a sum of per-stratum populations with summed variance, never an estimate over collapsed
  counts.
- **The pooled row is class-agnostic and the per-class rows are class-aware**, so they deliberately do not
  sum to each other. The first asks how many objects exist; the second asks how many of class *c* each
  observer correctly identified, which is what compares to per-class gold recall.
- **Independence is assumed and is not fully met.** A small, occluded, badly lit object is harder for both
  observers, so the captures correlate positively, `m2` runs high, and `N_hat` is biased **down**. Every
  number is a lower bound on what was missed and an upper bound on recall. It is reported that way and must
  never be presented as exact.

Nothing is stored in a log line: `blind_audit`, `blind_audit_frame` and `recapture_estimate` (migration
0095) hold the counts per frame and the estimate per stratum and per class, keyed on `run_id` and
`gold_id`, and it renders in the Analytics page under the gold comparison. A slice that could not be
estimated is stored as a row with `measured = false` and a reason rather than omitted, because a missing
row reads as "not computed" and this needs to say "computed, and the answer is that we cannot tell".

## What the promotion gate reads

`services/govern/champion.py::champion_gate` is a pure decision over a metrics dict. It fails closed:

- a `reconstructed` run (predictions backfilled from review history, no real confidence) is refused outright:
  no PR curve, no AP, cannot gate a promotion;
- a `harness_divergent` flag is refused outright;
- otherwise the usual floors apply: beat champion mAP, no Safe-mIoU regression, safety-class AP and recall
  floors (VRU and animal held tighter than the default);
- and the recapture condition, which asks whether the denominator the floors above are computed against has
  been checked at all. Three outcomes, and the middle one is the point of storing unmeasurable slices:
  - **measured**: if gold recall exceeds the audit estimate by more than `govern.recapture_max_overstatement`
    (0.15), promotion is refused. At that gap the gold denominator is missing enough that none of the floors
    above means what its name says. On by default, because it can only fire once an audit exists.
  - **unmeasured**: an audit ran and could not conclude (no object found by both observers, so the
    population is unbounded above). Always refused. "We looked and cannot tell" is not "we looked and it
    was fine".
  - **unchecked**: no audit for this run. Stated in the gate's reasons on every promotion, and refused only
    when `govern.blind_audit_required` is set. It ships **off**, deliberately: no blind audit exists yet,
    one cannot exist until somebody labels a couple of hundred frames, and a gate that starts red on every
    promotion gets switched off within a week. Flip it once the first audit is scored.

## Reconstructed runs

`scripts/backfill_prediction_from_review.py` reconstructs `Prediction` rows for objects reviewed before the
plane existed, from `Review.before` (class + box; conf was never captured, so it is null). These live under a
synthetic run with `params = {"reconstructed": true}`. Because conf is null, the eval refuses AP/PR for them and
returns only fixed-threshold precision/recall with a caveat, and the gate refuses to promote on them. They exist
so historical frames are not silently empty, never to be mistaken for a real inference.

## The real numbers (current champion, 400-object gold slice)

Through the prediction plane: AP@50 0.083, AP@50:95 0.068; at a 0.25 confidence operating point precision 0.164,
recall 0.146; safety-class recall pedestrian 0.083, rider 0.545, motorcycle 0.636, cycle 1.00, cattle 0.333. Of
the 400 gold objects, 351 were reviewed before provenance capture and their origin is unrecoverable. These are
weak and reported unrounded, because a number you trust is worth more than a good number you do not. See the
README honest-status section for the correction of the previously-published 0.034 / 0.018.

## The ten algorithms, and what each one measured

Added 2026-08-19/20. Each is a `core/accel` primitive with a NumPy reference and a torch path, a service
that consumes it, and a fixture test whose expected value was derived by hand. The numbers below are from
the champion (`mr-idd-yolo11l-local-aa408c72b0`) on its gold run, and several are worse than the ones they
replace, which is the point.

| # | What it measures | Result on this corpus |
| --- | --- | --- |
| 1 | Capture-recapture recall (`core/accel/recapture.py`) | Audit `445bdc59` seeded, 60 frames, **pending a human** |
| 2 | Neyman-Pearson thresholds (`core/accel/np_threshold.py`) | 5 of 12 classes fitted, all **0.42-0.49 above** the configured 0.45; 7 including every VRU admit no threshold |
| 3 | Relationship-aware NMS (`core/accel/rel_nms.py`) | Matrix over 170 frames, 347 human objects, 203 cells; motorcycle-rider 0.700 (kept apart) vs motorcycle-scooter 0.256 (merged) |
| 4 | Confusion-clique AL (`core/accel/clique_margin.py`) | Built; posteriors all at Beta(1,1), so allocation is uniform and says so |
| 5 | Ego propagation (`core/accel/ego_homography.py`) | **Refuses on every session**: GNSS on 3 frames of 41,752, no attitude, no pose table |
| 6 | Tube consistency (`core/accel/tube_score.py`) | Built; needs a tracker run with `track_id` to fit against |
| 7 | Density calibration (`services/oraclyx/density_calibration.py`) | Worst-bucket ECE 0.00277 -> 0.00241; **33,600 predictions got `conf_calibrated`**, first ever written |
| 8 | Reflection twins (`core/accel/reflection_twin.py`) | Built; hood-mask estimator rewritten to streaming Welford |
| 9 | Hierarchical AP (`core/accel/hier_ap.py`) | leaf AP50 **0.072**, l1 **0.143**: half the apparent failure is naming, not finding |
| 10 | Redact-then-verify text (`services/anonymize/text_regions.py`) | Built; needs `LBX_PII__TEXT_WEIGHTS` set and `make pii-models` |

Three of these interlock and must be read together. Item 2 says most classes cannot be auto-accepted at
any threshold; item 9 says half the model's apparent failure is naming rather than finding; item 1 has not
run. All three are computed against a gold denominator of 302 objects on 157 frames, while the model
emits 26 detections per frame. Until audit `445bdc59` is labelled, "the model is bad" and "the gold set
never recorded most of what is there" fit the evidence equally well, and the audit is the only thing that
separates them.
