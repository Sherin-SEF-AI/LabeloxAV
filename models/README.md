# Trained models

Weights trained on labels produced by this engine, tracked with git-lfs. Clone with git-lfs installed or
these files arrive as pointer text rather than checkpoints:

```bash
git lfs install
git lfs pull
```

| File | Backbone | Params | Trained | Recall @0.25 | Recall @0.05 |
| --- | --- | --- | --- | --- | --- |
| `real-v1-nano.pt` | YOLO11n | 2.6M | 60 epochs, 640px, batch 16 | **0.411** | 0.591 |
| `real-v1-small.pt` | YOLO11s | 9.4M | 60 epochs, 640px, batch 16 | 0.386 | - |

Ten classes: motorcycle, autorickshaw, sedan, bus, truck, pedestrian, rider, cattle, traffic_signal,
traffic_sign.

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
does not inherit it: every box in the key is a thing a human confirmed is there, and the only question asked
is whether the model found it, which holds however incomplete the key is.

Any mAP figure quoted for these checkpoints, including the one the training job records, is measured against
that same incomplete key and should not be relied on.

## What the numbers say

The larger backbone is not better, 0.386 against 0.411, so capacity is not the constraint. Recall tracks
training instance count instead:

| Class | Train instances | Recall @0.25 |
| --- | --- | --- |
| truck | 2,679 | 0.743 |
| bus | 2,089 | 0.684 |
| motorcycle | 2,752 | 0.343 |
| pedestrian | 752 | 0.216 |
| cattle | 91 | 0.209 |
| sedan | 193 | 0.181 |
| traffic_sign | 1,201 | 0.163 |
| traffic_signal | 158 | 0.000 |
| rider | 134 | 0.000 |
| autorickshaw | 4 | not measurable |

Above roughly two thousand instances the model works, below two hundred it does not fire at all. Motorcycle
is the exception that is genuinely hard rather than starved: 2,752 instances and 0.343, on small, densely
packed, heavily occluded targets.

Localisation and classification recall are within a point of each other on every class, so these are
failures to detect rather than failures to name. There is no class confusion to fix.

Recall rises from 0.411 to 0.591 when the threshold drops from 0.25 to 0.05, with motorcycle going 0.343 to
0.714. The model is finding these objects and scoring them under threshold, which is what training on a
partially annotated set does: unlabelled positives are taught as background and suppress confidence on real
ones. More labels or a loss that ignores unlabelled regions will move this. A bigger model will not.

## Neither is a champion

Both are candidates. Nothing here has passed the promotion gate, and the safety floors for VRU and cattle
recall are not met: rider is 0.000 and cattle 0.209. Do not serve these.
