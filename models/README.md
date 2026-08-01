# Trained models

Weights trained on labels produced by this engine, tracked with git-lfs. Clone with git-lfs installed or
these files arrive as pointer text rather than checkpoints:

```bash
git lfs install
git lfs pull
```

| File | Backbone | Trained on | Micro recall @0.25 | Macro recall |
| --- | --- | --- | --- | --- |
| **`real-v2-nano.pt`** | YOLO11n | every detection a human did not refuse | **0.770** | **0.623** |
| `real-v1-nano.pt` | YOLO11n | gate-approved detections only | 0.411 | 0.282 |
| `real-v1-small.pt` | YOLO11s | gate-approved detections only | 0.386 | 0.281 |

Ten classes: motorcycle, autorickshaw, sedan, bus, truck, pedestrian, rider, cattle, traffic_signal,
traffic_sign.

**Use `real-v2-nano.pt`.** The v1 checkpoints are kept because the comparison between them is the finding,
not because either is worth serving.

## The finding: the labels were the constraint, not the model

`real-v1` was trained on objects the confidence gate approved, which sounded like the careful choice and was
the defect. On its own training frames that is 7.6% of what the detector found; the other 92.4% sat in
`review` and `annotate` and were therefore taught to the model as background. Per class the effect is stark:
the model was shown 14,819 riders labelled background against 134 labelled rider, and rider recall was
exactly 0.000.

`real-v2` changes one thing. Same frames, same session split, same validation key, same backbone, same
schedule; only the training labels differ, from 10,053 to 116,618. Objects a human explicitly rejected stay
out, because that is the corpus's only real negative evidence.

| class | val n | v1 | v2 |
| --- | --- | --- | --- |
| truck | 506 | 0.743 | **0.945** |
| sedan | 138 | 0.181 | **0.899** |
| motorcycle | 598 | 0.343 | **0.865** |
| bus | 237 | 0.684 | **0.865** |
| pedestrian | 116 | 0.216 | **0.750** |
| traffic_sign | 246 | 0.163 | **0.528** |
| rider | 77 | 0.000 | **0.468** |
| cattle | 115 | 0.209 | 0.209 |
| traffic_signal | 48 | 0.000 | 0.083 |
| autorickshaw | 6 | 0.000 | 0.167 (n<30) |

This also explains what three earlier experiments could not. A longer schedule and a larger backbone both
returned nothing, because neither addresses a training signal that says most of the traffic is not there.

### Why this is not just recall bought by predicting more

It is the first thing to suspect: recall is free if a model emits enough boxes. v2 emits 8.69 boxes per
image against v1's 0.92, on a key holding 1.16, so the key cannot answer the question and neither can
precision, which is unmeasurable against a 7.6% complete key.

The images can. Drawing both on held-out frames, v2's additional boxes are overwhelmingly real traffic the
key does not contain: cars, autorickshaws, riders, pedestrians, a worker in hi-vis. The number that should
have looked wrong all along is v1's, since 0.92 objects per frame in Bangalore traffic is not a plausible
scene, and the training frames average about 18.

### What is still not known

Precision. A complete answer key does not exist on this corpus, so the false-positive rate is unmeasured,
and "the extra boxes look right" is an observation rather than a measurement. Growing the human-reviewed set
is what turns it into one.

And v2 is still distilling the previous autolabeler, now more of it, including its mistakes. It is a better
student of the same teacher, not independent evidence that the teacher was right.

## The data

`real-v1`, built from the operational Bangalore dashcam fleet: 7,306 train frames / 10,053 objects and
1,831 val frames / 2,087 objects, from 193 sessions split at the session boundary. Consecutive dashcam
frames at 3fps are near duplicates, so a frame level split would put the same scene on both sides and report
memorisation as generalisation.

Only objects in `accepted` or `auto_accept` state are written as labels. Test fixture sessions are excluded
twice over, by vehicle id and by a pixel test that rejects flat colour fill and uniform static, and the two
filters agree independently: the pixel test caught nothing the id list had missed. See the corpus correction
in the root README for why that screening exists and what it was hiding.

## Read these as recall, not as mAP

The corpus is 2.9% reviewed. A correct detection of an object nobody has labelled yet is scored as a false
positive, so precision and average precision inherit that error and are not measurable on this data. Recall
does not inherit it: the only question asked is whether the model found a box that is in the key, which
holds however incomplete the key is.

Read "reviewed" carefully, though. Of the 12,256 objects in that pool, 11,673 are `auto_accept`, meaning the
previous autolabeler's own output gated by confidence rather than anything a person looked at. The corpus
holds **126 human reviewed objects** out of 570,378, which is 0.02%. So these checkpoints are largely
distilling the earlier autolabeler, and their ceiling is its accuracy.

Any mAP figure quoted for these checkpoints, including the one the training job records, is measured against
that same incomplete key and should not be relied on.

## What the numbers say

Recall tracks how much of each class was actually labelled, not how much of it exists. That is the whole
finding above, and it is worth keeping the superseded reading beside it, because it was wrong in an
instructive way.

On `real-v1` the numbers looked like a data-volume story: truck at 2,679 training instances reached 0.743,
rider at 134 reached 0.000, and the obvious conclusion was that the starved classes needed more labels.
Under that reading a longer schedule and a larger backbone should not have helped, and neither did, which
seemed to confirm it.

It was the wrong reading. Those instance counts are not how many riders were in the training frames; they
are how many were labelled. There were 14,953 riders present and 134 labelled, so the model was not short of
rider examples, it was being taught that almost every rider was background. Correcting that took rider from
0.000 to 0.468 without adding a single new label.

The other half of the old diagnosis survives and is now explained. `real-v1` recall rose from 0.411 to 0.591
when the confidence threshold dropped from 0.25 to 0.05: it was finding the objects and scoring them under
threshold. That is exactly what training against overwhelmingly false negatives does, and `real-v2` recovers
most of that gap at the original threshold rather than needing a lower one.

Localisation and classification recall stay within a point of each other on every class in both models, so
these remain failures to detect rather than failures to name. There is no class confusion to fix.

Two classes did not move. Cattle sits at 0.209 in both, and traffic_signal at 0.083, because they are
genuinely scarce rather than under-labelled: cattle has 122 instances present on the training frames in
total, of which 91 were already labelled. Those two are the real case for more human labels.

## None of these is a champion

All three are candidates. Nothing here has passed the promotion gate. `real-v2` moves rider from 0.000 to
0.468, which is the first time a VRU floor has been within reach at all, but cattle is unchanged at 0.209
and traffic_signal at 0.083, so the safety floors are still not met.

The gate should also not be taken at its word yet: it reads mAP against the same incomplete key, and a run
earlier in this work reported `promote: true` while safe_miou sat at 0.0. Do not serve these.
