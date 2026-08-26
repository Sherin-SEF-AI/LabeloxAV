# Workflows

End-to-end recipes. Each names the screens and the API routes, so you can drive it from the UI or from a
script.

## Ingest to dataset

The core loop, start to finish.

**1. Bring a drive in.** *File → Import* in the app, or `POST /api/ingest`. A dashcam MP4, an MCAP bag, or a
folder of frames. Ingest extracts frames at `ingest.target_fps` (3 fps by default, with smart extraction
that drops near-duplicates), runs the PII pass, and writes a `Session`.

**2. Let SANYX judge it.** `/sanyx` scores the session's health. A session that fails is quarantined and
**will not auto-label** - that gate is deliberate, because labelling footage with a dead camera or a
saturated exposure spends budget on frames nobody can use.

**3. Auto-label.** *Run → Auto-label this whole drive*, or `POST /api/autolabel/start`. Three paths run and
fuse: a closed-set detector, an open-vocabulary detector with SAM, and a VLM verifier on the uncertain
subset only. It takes the GPU slot, so it will not run beside a training job.

!!! warning "It runs batch by batch"
    Auto-label holds the GPU advisory lock for one session at a time and releases between drives. A
    training job queued behind it waits one drive, not the whole corpus.

**4. Review what the gate did not accept.** `/review/queue` ranks by uncertainty times class rarity.
`/review/rapid` is the keyboard-only flow: `A` accept, `X` reject, `J`/`K` move. `/review/grid` shows crops
of one class at a time, which is the fastest way to catch a systematic error.

**5. Seal and export.** `/datasets`, or `POST /api/export`. Choose a slice - by state, class, region,
context, rarity band, or track-event presence. The export writes COCO, Parquet, YOLO, and a
**coverage datasheet** stating what the release does not know.

## Audit label quality

Use this when you inherit a corpus, or before you trust a number.

```bash
.venv/bin/python -m scripts.run_class_precision --n 80
```

Samples each class above 10,000 objects, asks a calibrated VLM judge whether each label is right, and
reports precision two ways: the raw Wilson interval, and a Rogan-Gladen correction through the judge's own
measured sensitivity and specificity. Where the judge is not calibrated it prints the caveat rather than
correcting.

The sample is hash-ordered by object id, so re-running after a fix compares the same crops and the
difference is the fix rather than the draw.

Read the `wrong kind` column, not just precision: a `sedan` that should be an `suv` is taxonomy drift, and a
pole labelled `traffic_signal` is a different failure entirely.

## Fix a class that is systematically wrong

When the audit condemns a class:

1. **Look at the crops first.** Any object set can be rendered as a contact sheet; do not act on an
   aggregate you have not eyeballed.
2. **Decide which kind of wrong it is.** Contaminated (the label names the wrong kind of thing) wants a
   per-object relabel pass. Absorbing (one class swallowing its siblings) is a taxonomy fault and wants an
   ontology merge or an attribute, not a relabel. Junk wants deletion.
3. **Run it reversibly.** Every corpus-wide write records an `AgentRun`; `POST /api/agent/runs/{id}/revert`
   undoes it. Corrections land in `review`, never `accepted` - a machine correcting a machine is not
   verification.

## Correct one object everywhere it appears

Change the class in the frame editor and the correction propagates along the track, so a fix on frame 1
reaches all 93. Under the hood that is `POST /api/tracks/{id}/relabel`, which respects the review state
machine, refuses implausible class moves, and is undoable as one batch.

`/annotate/timeline/[trackId]` shows the track as a strip; `/track/[id]` shows the crops with class flips
highlighted.

## Find frames worth labelling

- `/search` - natural language plus visual similarity, blended with a rarity term so the results are not
  thirty near-identical sedans.
- `/discovery` - novelty queue: frames unlike anything already labelled.
- `/curation` - frame-level active learning over the whole corpus.
- `/sievyx/longtail` - ODD gaps: what your corpus does *not* contain.

## Train and promote

`/training` starts a run. `/verdyx` compares champion against challenger **per slice**, not in aggregate,
and `/verdyx/safety` checks recall on the safety-critical set. A regression on a slice that matters blocks
promotion even when the headline metric improved. `/govern` holds the loop control and the championship.

## Ship to a device

`/forgyx` quantizes and compiles, then benchmarks latency against accuracy on a Pareto front.
`/forgyx/deploy` handles the thermal envelope and rollout. The benchmark gate can block a model that is
accurate and too slow.
