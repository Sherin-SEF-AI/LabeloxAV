# What this is for

LabeloxAV is a data engine, not a labelling tool. The distinction matters for how you use it: a labelling
tool's job is to get boxes drawn, and this system's job is to decide *which* boxes are worth a person's
attention, then prove whether the ones it drew itself were right.

If you are here to draw boxes on a hundred images, this is heavier than you need.

## Use cases it is built for

### Turning fleet footage into a training set

The core loop. You ingest dashcam drives, the three-path pipeline labels them, the confidence gate routes
everything to `auto_accept`, `review`, or `annotate`, and a person works the review queue in uncertainty
order rather than frame order. You export a sealed dataset with a coverage datasheet that states what the
release does *not* know.

Start at [Ingest to dataset](workflows.md#ingest-to-dataset).

### Finding the moments that matter in a long drive

Most frames of a dashcam drive teach a model nothing. SIEVYX exists to find the ones that do: embedding
search, near-duplicate suppression, rare-class discovery, and behavioural scenario mining. You can ask for
"autorickshaw cutting in at a junction" in natural language and get ranked frames, blended with a rarity
term so the answer is not thirty near-identical sedans.

### Auditing labels you did not make

Point it at an existing corpus and ask how good the labels are. The judged per-class precision pass
(`scripts/run_class_precision.py`) samples each class, asks a calibrated VLM whether each label is right,
and reports precision with the judge's own error corrected out. On this corpus that surfaced a class at
0.02 precision that confidence scores said was fine.

See [Audit label quality](workflows.md#audit-label-quality).

### Closing the loop

Train on what you labelled, evaluate per slice rather than in aggregate, and let a champion/challenger gate
decide whether the new model ships. A regression on a slice that matters blocks promotion even when the
headline metric improved.

### Running a second domain on the same engine

The domain is a pack, not the engine. `packs/sec` serves physical security from the same spine: same
ingest, same review, same gate, different ontology and different safety definitions. If your domain is not
driving, this is the seam to read: [Domain packs](../PACK_INTERFACE.md).

## What it is not

- **Not a crowdsourcing platform.** There is no task marketplace, no worker payment, no consensus across
  five anonymous annotators. It assumes a small number of trusted reviewers.
- **Not real-time.** Nothing here runs on a vehicle. It is an offline engine.
- **Not a general-purpose image tool.** The ontology, the confusion cliques, the safety definitions and the
  scene attributes are all shaped around road scenes.

## How the pieces fit

Seven platforms, one backend, one flywheel. Each owns a stage and some can block the ones after them.

```
SANYX      ingest QA          can quarantine a bad session
CALYX      calibration        can block on rig drift
SIEVYX     curation           decides what gets labelled
LABELOX    annotation         auto-label + human review
ORACLYX    pseudo-GT          auto-truths the majority, routes disagreement
VERDYX     evaluation         champion/challenger verdict, can block promotion
FORGYX     edge optimization  quantize and benchmark, can block on latency
```

Full detail: [The seven platforms](platforms.md).
