# The autonomy program

One entry per work package of the feature program that started on 2026-09-05, in landing order.
Every number here comes from a real run over the real corpus or from the rule itself evaluated
exhaustively; where a number could not be produced honestly, its reason is printed instead.

The invariants every package keeps: `accepted` means a person ruled, forever, and the machine never
writes it; every autonomous write is one revertible chunked `AgentRun`; refuse with a reason rather
than guess, and an unmeasured quantity is never reported as zero; GPU work runs batch by batch behind
`gpu_slot`, `training_holds_gpu` and `VramGuard`; the killswitch gates all of it; every schema change
ships an Alembic migration whose `downgrade()` has been run.

---

## WP1. Sequential acceptance (SPRT) and the verdict-minute allocator

### The defect it fixes

Settlement (migration 0106) proved a class clean by a fixed draw: `sample_target(far)` crops, sized so
one defect still passes Wilson at the class's failure-rate bound. That is 110 crops at far 0.05, 283 at
0.02, 562 at 0.01. A fixed draw spends the same number of human verdicts on a class that is obviously
clean and on one that is obviously not, and the two live lots (motorcycle at far 0.01, traffic sign at
far 0.05) were waiting on 672 verdicts, roughly 67 human minutes, before either could say anything.

### The rule

`services/labelops/sampling.py::sprt_decision(defects, n, *, p0, p1, alpha=0.05, beta=0.10)` is Wald's
sequential probability ratio test on the defect rate. `p0 = far` is the rate the lot must reject,
`p1 = far / 2` the rate it must accept (`settlement.py::sprt_params`). The log-likelihood ratio moves
by `ln(p0/p1) = ln 2` on every defect and by `ln((1-p0)/(1-p1))` on every clean verdict; it stops at
`ln(beta/(1-alpha)) = -2.2513` (accept) or `ln((1-beta)/alpha) = 2.8904` (reject). The operating
characteristic is solved by bisection and reported per lot, and the expected remaining verdicts (Wald's
ASN from the current llr) is what the allocator reads.

Three things about how it is wired, because each one is where a sequential test goes wrong:

- **Accept is conjunctive.** A lot accepts only when the SPRT crosses its accept bound and
  `acceptance_decision` (the Wilson rule from 0106) also accepts at the same `(defects, n)`. Measured
  over every `(k, n)` up to the cap: the SPRT is the binding rule at 0 to 2 defects and Wilson is the
  binding rule at 3 or more, so neither implies the other and the conjunction is not decorative. Reject
  is the SPRT alone; it is the conservative direction and steps the class down exactly as before.
- **Verdicts arrive in draw order, and only whole increments count.** Triage served every batch in
  `(1-conf) * rarity * boost` order, which would have made "the verdicts so far" the hardest-first prefix
  and the early stop a biased one. For batches whose cycle id starts with `settle-`, triage now ranks by
  the crop's index in the lot's draw (`sample_object_ids`); the formula for every other batch is
  untouched and a test holds it there. `tally_lot` counts only increments judged to the completion floor
  (0.9), never a partial one, and `top_up_lot` refuses to draw while the latest increment is incomplete.
- **Increments are a prefix of one permutation.** The draw keeps the `settle-sample` salt and the
  `not in have` exclusion, so 25 more crops are the next 25 of the same md5 order. At
  `cap_n = sample_target(far)` the sample is identical to the old fixed draw and Wilson decides verbatim
  (`rule="wilson_at_cap"`), so the worst case of the sequential rule is exactly the old rule.

Migration `0107_settlement_sprt` adds `rule`, `cap_n`, `llr`, `sprt` and `increments` to
`settlement_lot`. `llr` is null until computed, never 0. Downgrade drops the five columns and a judging
lot then tallies as Wilson, the stricter rule, so nothing has to be moved; the round trip is exercised
on a seeded lot by `tests/test_migrations_roundtrip.py`.

### The spot check that never reached a person

`settle_lot` created `settlement_spot` rows on `settled` objects with no cycle id, and triage defaults
to `review,annotate`, so no spot was ever served and nothing wrote `human_verdict`. Spot objects are
now stamped `provenance.flywheel.cycle_id = spot-{lot8}`, the settlement notification links to
`/review/grid?flywheel=spot-{lot8}&states=settled`, and `apply_review_batch` writes `correct` or
`incorrect` onto the spot when the verdict lands (`ApplyResult.spot_judged`). A human ruling on a
`settled` object still upgrades it to `accepted`, as `state_for` already did.

### The allocator

`settlement.py::expected_remaining(lot)` gives, per judging lot, the expected verdicts to a decision
(clamped to `cap_n - n`), minutes at the 10 verdicts per minute the estimate has always used, and
`value = population * OC(p_hat) / minutes`: objects a passed lot would settle per minute of a person's
time. `/autonomy/state` returns the ranked `worklist` and `verdict_minutes_open`; the autonomy page
draws each lot's llr between its two bounds. `p_hat` is smoothed with one pseudo-verdict at the
indifference midpoint `(p0+p1)/2`, because a Laplace prior made an unjudged lot look half defective
and priced it at zero.

### Measured

From the rule itself, evaluated at every `(defects, n)` up to the cap for each tier:

| far | cap (0106) | clean lot accepts at n | first whole increment | verdicts saved | straight defects to reject |
| --- | --- | --- | --- | --- | --- |
| 0.05 | 110 | 87 | 100 | 9% | 5 |
| 0.02 | 283 | 222 | 225 | 20% | 5 |
| 0.01 | 562 | 447 | 450 | 20% | 5 |

A lot with exactly one defect never accepts early under the conjunction at any tier; it runs to the cap
and Wilson accepts it there, as it did before. Five consecutive defects reject at any tier after 5
verdicts, where the fixed draw would have needed the whole sample.

**The two live lots produce no replay number.** `253cacf5` (motorcycle, far 0.01, 562 drawn) and
`be7b3273` (traffic sign, far 0.05, 110 drawn) both have 0 verdicts, so there is nothing to replay
through the SPRT. They were planned under 0106 and stay `rule='wilson', cap_n=0`; the tally treats that
as the fixed rule. Re-planning them under the sequential rule is the operator's call, and the first
verdict-minute numbers will come from whichever lot is judged first.

Tests: `tests/test_sprt.py` (21), `tests/test_settlement.py` (16, including the calibration umbrella
`test_settlement_changes_no_calibration_input`), `tests/test_migrations_roundtrip.py` (1).

---

## WP3. Synthetic data behind a quarantine, and the copy-paste generator

### The defect it fixes

Cattle and rider are the two classes the gate keeps blocking on, and both are starved of positives:
cattle has 6 human-accepted objects in the whole corpus, rider has 4 with a polygon mask. Pasting real
instances of a class onto real road scenes is the cheapest proven way to add positives, but it is only
safe if nothing that measures a model can ever see a composited pixel. A predicate sprinkled across
readers cannot deliver that: over 160 modules select `Object` outside the API, so "remember to exclude
synthetic" is a rule nobody can audit.

### The quarantine

Isolation is structural rather than a predicate everyone must remember. A synthetic object is written
with `state = 'synthetic'` and `source = 'synthetic'`, two values no real object has ever held, so every
allow-list reader excludes it without being changed: gold builds take `source == 'human'`, the precision
draw takes `MACHINE_STATES`, the control sample takes `auto_accept`, settlement takes `review`, triage
defaults to `review,annotate`. `OBJECT_STATES` gains the value and `state_for` refuses it for a human
actor, making it machine-only in the way `settled` is human-only in reverse.

The readers that select frames rather than objects cannot inherit that, so they carry the explicit
predicate `Frame.origin == REAL` (`core/origin.py`): `training/dataset_builder._select`,
`export/dataset`, `export/coverage`, `intelligence/embed/pending` (no GPU is spent embedding a
composite, and it stays out of novelty and dedup), `context/rarity`, `intelligence/search/rarity`,
`agent/scenario_miner`, `explore/query` (default on, with an opt-in flag) and `verdyx/blind_audit`.
The training builder is the one place that can opt in, `BuildSpec.include_synthetic`, and even then a
synthetic frame is never validation and is dropped when its source frame's session lands in val, so a
composite can never be scored against the model built from it.

Migration `0108_origin` adds `session.origin` and `frame.origin` (`real|synthetic|perturbed`) with
`frame.source_frame_id` pointing back at the background, and extends `ck_object_state` and
`ck_object_source` with the new value. The downgrade deletes synthetic frames and their sessions before
restoring the narrower CHECKs, so it leaves a database no wider than the one it started from.

### The generator

`services/synth/copy_paste.py` is CPU-only OpenCV, takes no GPU slot, and composes each frame inside
`asyncio.to_thread`. Donors are human-accepted objects of the wanted class on real frames that carry a
polygon mask and are at least 48 px on the short side. Backgrounds are real, selected frames that have
a drivable mask and at least one object already labelled.

That background rule is looser than the plan's "no `review` or `annotate` object", and the corpus is
the reason. The strict rule left 24 usable frames. Frames with no object at all are not clean
backgrounds either: 7,160 of them have zero predictions, meaning they were never labelled rather than
labelled empty, and training on them would re-teach the lesson that made recall 0.411. Requiring at
least one labelled object and no pending `annotate` work gives 14,136 backgrounds over 197 sessions.

Placement scale comes from the row: the horizon is `cy - fy*tan(pitch)` from the resolved calibration,
and a donor moved from its own horizon to the background's scales by the ratio of its distance below
each. Colour is matched with a Reinhard transfer in LAB at strength 0.5 with the chroma ratio clamped
to [0.5, 2], and the mask edge is feathered. A paste is refused rather than fudged when it would cover
more than 30% of an existing box, leave the frame, or imply a scale outside [0.35, 3.0].

Each build is a parent `AgentRun(kind='synth_build')` whose children are `synth_batch` runs of 200
frames, committed one at a time; `revert_run` cascades to the children, and each child deletes its own
frames, images and mask blobs. Memory is checked against `resources.host()` before every batch and the
build waits rather than pushing the host past 90%.

### The bonnet, which only looking at the output revealed

The first 500-frame build passed every test and put riders on the recording car's own bonnet. The
drivable segmenter reads the lower frame as road on most dashcams, because a bonnet is smooth, grey and
continuous with the road, so the placement band happily included it. The fix reuses the per-camera hood
mask the detector cleanup sweep already estimates (`autolabel/ego_mask`) and subtracts it from the
drivable surface before a footprint row is drawn. Cameras with no cached hood mask keep the raw surface
and the run report counts them separately, so the gap is visible rather than assumed away.

### Measured

One build of 500 frames on the live corpus, launched by the agent's own off-hours hook
(`maybe_synth_starved`, run `81a9cc00`, `created_by='synth_starved'`) against the gate deficit of run
`mr-real-v2-nano-7a66432d`:

| quantity | value |
| --- | --- |
| frames written | 500 in 3 batches |
| objects written | 6,747, every one `state` and `source` `synthetic` |
| class pasted | pedestrian, from 7 donors |
| backgrounds available | 14,136 of 41,752 real frames |
| composites refused | 67 |
| backgrounds with a hood mask | 498 |
| backgrounds without one | 26 |

The refusals, by reason: 43 had a drivable mask with no `drivable` polygon, 19 would have covered more
than 30% of an existing box, 2 would have left the frame, 2 implied a scale outside the band, and 1 had
no drivable surface left in the placement band once the bonnet was removed.

The quarantine was then checked against the live database rather than assumed: of 6,747 synthetic
objects, all sit on the 500 synthetic frames, none on a real frame, and no object in any other state
sits on a synthetic frame. Reverting the earlier 500-frame build took the whole thing back through the
same cascade, three child runs and 500 frames, leaving no synthetic session, frame, object, image or
mask behind and touching nothing real.

A cattle build is refused with its reason rather than attempted: no cattle object in the corpus is
human-accepted, on a real frame, and carries a polygon mask. The 153 cattle masks that exist were
accepted by the VLM judge, not by a person, and the donor rule deliberately does not take them.

### The opt-in path, measured on the live corpus

`BuildSpec.include_synthetic` was run both ways over the whole corpus, selection only, no training:

| | `include_synthetic=False` | `include_synthetic=True` |
| --- | --- | --- |
| candidate objects | 513,157 | 519,663 |
| frames | 34,504 | 34,903 |
| synthetic frames | 0 | 399 kept, 101 dropped |
| validation frames | 7,013 | 7,013 |

The validation side is identical to the object, which is the property the quarantine exists to give. The
101 dropped composites are the leak guard firing on real data: their background frame's session landed
in validation, so training on them would have shown the model val pixels under a different frame id.
One composite in five was built on a background that validation later claimed.

**The hier_ap50 delta is not reported, because this build cannot produce a resolvable one.** The
arithmetic, not a missing capability: the build added 500 pasted pedestrians to a corpus that already
holds 35,616 pedestrian objects on real frames, a 1.4% increase, spread over 399 of 27,491 training
frames. A single-seed A/B at that effect size measures the seed, not the synthetic data, and reporting
the difference between two 60-epoch runs as evidence would be exactly the kind of number this document
refuses to print.

The reason it landed on pedestrian is worth more than the delta would have been. `maybe_synth_starved`
fires on the champion gate's recall deficit, and the gate's deficit for pedestrian is a recall problem
on 35,616 existing labels, not a shortage of them. The class that is genuinely label-starved is cattle,
with 1,617 objects against motorcycle's 66,786, and cattle is precisely the class the donor rule cannot
serve: of its 6 human-accepted objects, none carries a polygon mask.

| cattle objects on real frames | total | with a polygon mask | and at least 48 px |
| --- | --- | --- | --- |
| `accepted` by a person | 6 | 0 | 0 |
| `accepted` by the VLM judge | 171 | 153 | 26 |

So the generator is correct, quarantined and reverted cleanly, and it currently cannot reach the one
class that needs it. Opening the donor rule to VLM-accepted masks would turn 0 cattle donors into 26.
That is a policy decision about what "accepted" is allowed to mean for a donor, not a defect to patch
quietly, and it is left for the operator: WP3 keeps the strict rule, in which a donor is something a
person ruled on.

**One guard was found failing open and was fixed here rather than noted for later.** The host's NVIDIA
kernel module is 595.84 and its userspace library is 595.91, so NVML fails to initialise. CUDA is
unaffected, and `VramGuard` is unaffected with it, because that guard reads `torch.cuda.mem_get_info`
rather than the driver library. What was blind is `services/hardening/resources.gpus()`, which shells
out to `nvidia-smi` and returned no cards, and with it `class_precision.free_vram_mb()`, which derived
free VRAM from that list and returned None. `wait_for_headroom` reads None as "no GPU to check here"
and proceeds, so on a host with a working, busy 16 GB card the headroom guard stopped guarding. That is
the wrong direction for a guard to fail, so `free_vram_mb` now falls back to the CUDA runtime and
returns None only when neither reading can see a device.

Tests: `tests/test_origin_quarantine.py` (12), `tests/test_synth_compose.py` (15),
`tests/test_synth_build.py` (3), `tests/test_migrations_roundtrip.py` (1), `tests/test_class_precision.py`
(the two new headroom readings). The quarantine tests were proven non-vacuous by removing the origin
predicate from `embed/pending` and from `dataset_builder` and watching each one fail. The full suite
runs 3,219 passed, 5 skipped against `labeloxav_test`, against a 3,124 baseline.

---

## WP2. Shadow mode: challenger inference and disagreement mining

### The defect it fixes

Every model this program has built was measured on frozen gold: 164 validation images sealed months ago.
The corpus has taken in 377 sessions and 41,752 frames since, and no model has ever been compared on any
of them. Worse, the frames that would teach the most are the ones two models read differently, and
nothing surfaced those to anybody. A promotion decision had exactly one kind of evidence, and that
evidence stopped being new the day it was sealed.

### The sweep

`services/govern/shadow_agent.py::maybe_shadow_sweep` runs off-hours. It takes real, selected,
non-duplicate frames newer than the last committed sweep, oldest first, scores the champion and each
challenger over them, and files every disagreement. It promotes nothing and labels nothing.

The GPU discipline matters more than usual because two detectors are involved. They run one at a time
inside `gpu_slot`, never both resident, in chunks of 256 frames, with `training_holds_gpu` and a VRAM
floor checked between chunks so a training job that starts mid-sweep gets the card back in seconds
rather than at the end of the sweep.

The high-water mark only advances on a **committed** sweep, and it is the maximum across committed
sweeps rather than the latest one. A failed or reverted sweep compared nothing on its frames, and
letting its mark stand would step the window past them permanently with nothing ever saying so. This
was not a hypothetical: reverting the first real sweep and running another one showed the second
starting after the frames the first had named rather than at them.

### Three things `run_inference` had to be fixed for first

It was already an idempotent batched writer, but not one that could carry two models over thousands of
frames.

1. **The idempotency key made a nightly sweep a no-op after the first night.** The key is
   `(model_version, gold_id, code_sha, params)`, and for a sweep `gold_id` is null and the params are
   identical every night. The second night would have matched the first night's key exactly, reused
   that run, and scored nothing. `scope` now names the sweep and is folded into the key, so a new night
   is a new run and a retry of the same night is a reuse. `force=True` is not the answer, because it
   would duplicate rows when a crashed sweep retries.
2. **The model was constructed per 16-frame batch.** A 2,000-frame sweep is 125 batches, each re-reading
   the checkpoint from disk. It is loaded once per run now.
3. **It took no GPU slot at all.** The sweep wraps each chunk in one.

### The matcher

`services/verdyx/shadow_run.py::compare_frame` is pure and takes two lists of detections. It pairs boxes
greedily by IoU, class-agnostically so a class flip pairs instead of reading as two misses, and names
four kinds: `champion_miss`, `challenger_miss`, `class_flip`, `conf_gap`.

Both models are cut at the champion's operating point rather than at the 0.001 inference floor.
Comparing raw floors would turn every low-confidence tail detection of one model into a "miss" by the
other, which is an artefact of the floor and not a disagreement about the picture.

### The artefact the first real sweep was made of

The first sweep ran clean and produced a number that was almost entirely wrong. Comparing the 15-class
champion against a 4-class challenger, 3,091 of its 3,240 disagreements were `challenger_miss`. The
challenger had not missed anything. It had never been taught those words, and a model that cannot say
`pedestrian` will register every pedestrian the champion finds as its own failure. A win share computed
from those rows would have measured vocabulary size.

Migration `0111_inference_class_vocab` records, on each inference run, the ontology class ids that
model's own class order maps onto: what it was able to say at all, which is a property of the checkpoint
and knowable only at inference time. The matcher compares on the intersection. Null stays null and means
"this run declared no vocabulary", which is not an empty one, and a comparison facing null covers every
class as it did before.

### Where the human verdict lands

A `champion_miss` has no Object for anyone to rule on, so this is not a Review-row flow. The sweep files
one labelling task over the 40 worst disagreement frames, and everything pending on a queued frame is
queued with it so one person opening one frame settles every disagreement on it at once. On
`submit_job`, `adjudicate_for_job` matches each disagreement's box against that frame's human objects at
IoU 0.5 and reads the verdict off the person's drawing rather than asking them to vote.

That direction was written backwards on the first attempt and a test caught it. A `champion_miss` where
the person also found nothing means the challenger invented a box, so the champion was right; the code
had it awarding the challenger. Left in, every challenger hallucination would have become evidence in
the challenger's favour, and the gate would have read a model that invents objects as one that finds
them.

### The gate clause

`champion_gate` gains one clause, fail-closed only. With at least 30 adjudicated discordant pairs and a
Wilson upper bound on the challenger's win share below 0.5, the promotion is blocked with a reason.
It can never promote: a model that wins on the frames two models argue about has not been shown to be
safe on the frames they agree on, which is nearly all of them. Below 30 pairs it reports "unmeasured"
and blocks nothing, because too little evidence is not evidence of failure.

### Measured

One sweep on the live corpus, 512 frames, champion `mr-idd-yolo11l-local-aa408c72b0` against two
challengers, at the ship-default threshold of 0.5 because no fitted threshold exists for this champion
and the run says so rather than implying one was measured.

| model | frames | GPU seconds |
| --- | --- | --- |
| champion (yolo11l, 15 classes) | 512 | 20.4 |
| `mr-real-v2-nano` (10 classes) | 512 | 19.0 |
| `dashlab-det9class` (4 classes) | 512 | 9.1 |

The whole sweep took 50 seconds wall clock and filed a task of 40 frames carrying 369 disagreements.

The vocabulary fix was then measured against itself, on the identical predictions, changing only whether
the comparison was restricted to shared classes:

| challenger | shared classes | all classes | shared only | artefact |
| --- | --- | --- | --- | --- |
| `dashlab-det9class` | 4 of 15 | 4,153 | 379 | 91% |
| `mr-real-v2-nano` | 10 of 15 | 2,953 | 2,884 | 2% |

The disagreements that survive, by kind and by class, for the challenger that shares most of the
champion's vocabulary: 2,239 `challenger_miss`, 417 `champion_miss`, 138 `conf_gap`, 90 `class_flip`.
The three classes the two argue about most are sedan, pedestrian and traffic sign. The commonest class
flips are sedan against truck and bus against truck, which is the large-vehicle boundary rather than
noise.

**No win share is reported, and the reason is that nobody has ruled yet.** The sweep filed its 40 frames
and they wait on a person. `win_share` returns `measured: false` with that reason rather than a number,
the gate reads it as unmeasured and blocks nothing, and the autonomy page prints the reason where the
share would go. The first real win share arrives when that task is submitted.

Tests: `tests/test_shadow_match.py` (17), `tests/test_shadow_run.py` (14, one of which caught the verdicts
being written backwards),
`tests/test_migrations_roundtrip.py` (1, extended over 0109 and 0110 with rows present). The full suite
runs 3,249 passed, 6 skipped against `labeloxav_test`; the web suite 682 passed with `tsc --noEmit`
clean.
