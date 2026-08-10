# Canvas rework: Phase 0 discovery

Written before any implementation, as the build spec requires. Where the spec's assumptions conflict with
what is actually here, the behavioural requirement is kept and the implementation is adapted. Each conflict
is named below with what changes.

Six of the spec's premises turned out to be wrong about this codebase, and two of those change the size of
a workstream by a large factor in opposite directions: WS1 is smaller than specified, WS5 is much smaller,
and WS3 is larger.

---

## 1. Canvas rendering

`web/components/editor/EditorCanvas.tsx` (597 lines) is the whole 2D renderer. One react-konva `Stage`
containing exactly **one `Layer`**, which is one `<canvas>` element plus one offscreen hit canvas.

**Conflict 1: the renderer migration in WS1 is already done.** The spec says "if discovery finds per-object
DOM/SVG nodes on the hot path, migrate object rendering to a single Canvas2D or WebGL layer", and the Do Not
list says "do not leave per-object DOM nodes on the render hot path after WS1". There are none. DOM node
count is constant regardless of object count. The only per-object SVG in the tree is
`web/components/editor/RigView.tsx:71-79`, read-only multi-camera context tiles, not the editing surface.
A migration here would be churn against an acceptance criterion that already passes.

What scales is the Konva node count and the React element tree, which is rebuilt in full on every render:

| feature | nodes per object | line |
| --- | --- | --- |
| box | 1 `Rect` | 349-400 |
| mask | 1 `Shape` with a custom `sceneFunc` | 311-327 |
| cuboid | **12 `Line`s**, one per projected edge | 415-422 |
| pose | 1 `Group` + up to 16 edges + up to 17 joints | 497-521 |
| selected mask | 2 `Circle`s per vertex (vertex + midpoint) | 425-464 |

So a plain box is 1 node, a posed pedestrian about 30, a cuboid object 14, and a selected 40-vertex mask
adds 80 circles.

**The actual bottleneck, which the spec does not mention.** Three findings, in leverage order:

1. `onMove` calls `p.onCursor([x, y])` on every pointer move (`EditorCanvas.tsx:216`), which sets state on
   the 2,179-line `FrameEditor` page (`page.tsx:1152`) and re-renders the page and the entire Konva tree.
   That cursor value is read by one text span at `page.tsx:1445`.
2. `EditorCanvas` is not memoized and receives around 20 freshly allocated inline callbacks per render
   (`page.tsx:1109-1153`), so nothing downstream can skip work.
3. On that same path: an O(n log n) sort per render (`EditorCanvas.tsx:349-355`, whose comment says it
   exists for hit-test ordering rather than visuals) and O(R*N) relationship lookups (`403-405`).

There is no requestAnimationFrame loop in the 2D editor, no memoization, no viewport culling (the frustum is
known and unused), no level of detail, no mask caching, no pointer throttling, no `shape.cache()` anywhere,
and no virtualization on the object list panel, which renders every object as DOM rows.

Pan and zoom live in reducer state and are applied as the Konva stage transform. Stage dragging is the one
path already isolated from React: it commits once on `dragEnd` (`EditorCanvas.tsx:284-286`).

**Conflict 2: two WS1 sub-items are greenfield, not rework.**

- **There are no object labels on the canvas.** No class name, no id, no confidence is drawn next to any
  object. The only `KText` uses are the measure readout and the marquee count. So "label culling and
  clustering" means building label rendering and a placement solver from nothing. What stands in for labels
  today is an HTML popup in page space for the selected object only (`page.tsx:1377-1395`).
- **There is no hover state.** No `hoveredId`, no `onMouseEnter` on any shape. Nothing re-renders on hover
  because nothing shows on hover. "Hover response under 16ms" means building hover first.

Layering is hardcoded JSX paint order inside the single Layer, with 7 boolean flags
(`boxes, masks, lanes, drivable, adverse, cuboids, seg`). The LAYERS panel
(`web/components/shell/FloatingLayers.tsx`) iterates those 7 keys generically. There is no z-order system
and no per-class or per-group layer concept, so WS1's group-and-preset rebuild is additive.

Hit testing is Konva's colour-keyed hit canvas plus event bubbling. There is no spatial index anywhere
(grep for quadtree, rbush, flatbush, kd-tree returns nothing). Masks, lanes, cuboids and all previews are
explicitly `listening={false}`, so only boxes, polylines and handles are interactive. Cuboids being
non-listening means 3D boxes are not clickable on the 2D canvas today.

Masks are flat polygon arrays end to end, redrawn by re-walking every point each render, with no caching or
simplification. Server side they are polygon JSON in the object store, fetched one blob per object on frame
load (`services/api/routers/objects.py:119-127`, called per object at `:309`).

---

## 2. Annotation data model and provenance

`Object` (`db/models.py:159-217`) holds all five geometry kinds as columns on one row: `bbox` (mandatory,
doubles as the AABB for polyline and rotated shapes), `mask_uri`, `polyline`, `cuboid_3d`, `keypoints`, plus
`rot_deg`. Class is `class_id`, confidence is `conf`, attributes are `attrs` JSONB.

**Conflict 3: `provenance` is taken, and means something else.** `core/schemas.py:83-106` defines
`Provenance` as the fusion audit blob: which model paths proposed the object, their raw confidences,
agreement, entropy, quality flags, and the reasoner trace. `core/provenance.py` documents its walk as an
audit spine that "must never break". WS2 wants `provenance` to be a lifecycle state machine. These are
different concepts and the name collides.

**Resolution:** keep WS2's behavioural spec exactly and name the new field `lifecycle`, with
`lifecycle_history` for the append-only log. The visual grammar, the transitions and the summary chip are
unchanged; only the identifier moves.

What exists today in place of a lifecycle is the `source` and `state` pair:

- `source` is a bare `String(16)` with **no constraint and 9 distinct written values**, only 4 of which are
  `ObjectSource` enum members: `fused`, `auto_accept`, `human`, `recall`, `imported`, `interpolated`,
  `interp`, `propagated`, `relabel`. A human touch collapses it to `"human"` destructively
  (`services/api/routers/review.py:157`), losing the prior machine identity except via the `Review.before`
  snapshot.
- `state` is the queue: `auto_accept`, `review`, `annotate`, `accepted`, `rejected`, plus `submitted`, which
  is written by the client and the VLM QA path but is not in the `GateState` enum.

They are coupled by convention and constrained by nothing. Note a latent bug found in passing:
`services/intelligence/propagate.py:160` writes `"interp"` while `services/temporal/interpolate.py:130`
writes `"interpolated"`, and only the latter is in `_MACHINE_SOURCES`
(`services/autolabel/persist.py:36-40`), so propagated rows survive an autolabel re-run while interpolated
rows are deleted. Worth fixing during WS2 since the migration touches exactly this vocabulary.

**Conflict 4: the WS2 backfill can derive history for a very small minority of objects.** `Review`
(`db/models.py:317-330`) is genuinely append-only and its `before` snapshot deliberately carries `source`,
`conf` and `provenance`, so the pre-human machine identity is reconstructable. But:

- the agent path (frame, attribute, cuboid, copilot, relabel agents) writes no `Review` row, only
  `AgentRun.changes` plus a provenance stamp;
- the 3D edit path explicitly keeps no `Review` row (`services/api/routers/objects3d.py:173-176`);
- `Object.version` is a bare counter with no version-to-snapshot mapping.

Scale: `services/autolabel/reasoner/attribution.py:53-57` reports drawing 20,000 rows from 583,525 to find
2,007 carrying a human ruling. So the derivable subset is a fraction of one percent and the honest default
for everything else is `machine_proposed` with source unknown flagged explicitly, which is what the spec
asks for. The dry-run report must state the derived-versus-defaulted split rather than implying the
backfill reconstructed history it did not have.

---

## 3. Ontology

166 classes in `ontology/labelox_in_v0.yaml`, 83 flagged `india: true`. Per-class keys are exactly
`{id, name, l0, l1, india}`.

**Conflict 5: WS3 is larger than it looks, and the spec already anticipated this.** The YAML declares
`hierarchy_levels: 4` and encodes **two flat levels**. There is no `parent`, `superclass`, `is_a`, `l2` or
`l3` key, no edge set, no closure table, no `parent_id` on `OntologyClass`, and no ancestor or descendant
query anywhere in `services/autolabel/ontology.py`. The declared 4 is read by nothing that walks a
hierarchy.

This matters because WS3's `refinement` level is defined as "old is ancestor of new" and `coarsening` as
"new is ancestor of old". **Neither is computable today.** Encoding the hierarchy is explicitly in scope per
the spec, and it is the long pole of WS3, not the guard matrix.

The 4 `l0` values are `object` (83), `infra` (58), `surface` (22), `ignore` (3). The 13 `l1` values
partition cleanly under them, one `l0` each.

Class groups exist as **at least 8 independent hardcoded sets**, two of which disagree:
`packs/av/pack.py:43` has `AV_SAFETY_L1 = {vru, animal}` while `services/flywheel/signals.py:26` has
`SAFETY_L1 = {vru, animal, two_wheeler}`. Others live in `services/agent/nl.py:27-37`,
`services/agent/critic.py:24-30`, `services/autolabel/quality_reviewer.py:20-23`,
`services/verdyx/slice_eval.py:61`, and `services/autolabel/ontology.py:33-46`. `packs/av/pack.py:43` even
carries a comment naming six files the `{vru, animal}` literal was copy-pasted across.

So `configs/ontology_groups.yaml` is not a new idea being introduced; it is the consolidation of eight
existing ones, and the disagreement above has to be resolved deliberately rather than by picking whichever
file is read first.

---

## 4. Critic

`services/agent/critic.py` (185 lines) runs **5 checks**, not the 6 reason codes WS4 names:
`temporal` (track class flips, centroid teleport past 0.6 of the frame diagonal), `geometric` (box bottom
ray to the ego ground plane, flags above-horizon), `motion` (speed from `ObjectDynamics`, 45 km/h for VRU,
220 otherwise), `cross_modal` (LiDAR frustum point count, minimum 3), and `relationship` (a rider must have
a two-wheeler under its lower half).

Its design contract is veto-only: it can demote an auto-accept to review, never create one
(`critic.py:14-17`, enforced at `services/agent/policy.py:48-52`). Every threshold and every severity weight
is a hand-set constant. There is no measurement of how often a flag corresponds to a real error.

Mapping to WS4's six codes: `geometry_size_prior`, `class_margin_low`, `attribute_conflict`,
`duplicate_overlap` and `mask_boundary_outlier` are all new checks to build; `temporal_flicker` is close to
the existing `temporal` check but is specified against association, which already exists (see section 5).
The existing `geometric`, `motion`, `cross_modal` and `relationship` checks have no WS4 code and should not
be dropped silently; they need codes of their own or an explicit decision to retire them.

**Conflict 6, in the helpful direction: the calibration machinery WS4 requires already exists, applied to a
different check set.** `services/autolabel/reasoner/attribution.py:43-160` computes per-check precision with
Laplace smoothing, lift over the corpus base rate, and a minimum sample size of 25, for the reasoner's 8
checks. Its docstring records exactly the trap WS4 is guarding against: on this corpus 63% of reviewed
objects were corrected, so a rule firing at random scores 0.63 and looks respectable. It also states its own
bias honestly, that precision is measured over reviewed objects only and review is not a random sample.
`suggest_weights` deliberately reports without applying.

WS4 should reuse this rather than build a second calibration path, and inherit the base-rate framing: a
suspicion score has to be reported as lift over base rate, not as bare precision.

The blanket count WS4 deletes has three producers, not one: the per-frame agent panel
(`web/components/agent/AgentPanel.tsx:214-219`), the corpus fix queue (`web/app/agent/page.tsx:15-19`), and
the rig track panel (`web/components/editor/RigTrackPanel.tsx:44`). All three need to go.

**Batch operations report volume only.** `fit 3D boxes` reports attached and routed counts, `fill
attributes` reports attrs filled, `batch-fix` reports relabeled. None reports a correctness estimate. The
only measured precision in the system is the gate's auto-accept precision from `ControlSample`
(`services/govern/control_sample.py:69-80`), which mirrors 2% of auto-accepts to human review and is
surfaced at `GET /api/govern/control/precision`. The overnight auditor already narrates the unmeasured case
correctly ("not yet measurable, N controls awaiting review"), which is the tone WS4's unmeasured state
should match.

---

## 5. Tracking

**Conflict 7, the largest scope reduction: WS5's association module already exists.** The spec says "if
discovery found no track_id: implement an association module as its own package", specifying Hungarian
matching on IoU blended with appearance distance, and gap handling to a configured max age.

All of that is built:

- `Object.track_id` (`db/models.py:164`) with an index, and a `Track` table (`db/models.py:143-156`) holding
  `trajectory`, `id_switch_flags` and `tracker_version`.
- Two backends. `services/intelligence/tracking.py` is a deterministic greedy IoU tracker.
  `services/autolabel/track/tracker.py` is **BoT-SORT with DINOv3 appearance embeddings**, a
  constant-velocity Kalman filter on `[cx, cy, w, h]`, and Hungarian assignment via
  `scipy.optimize.linear_sum_assignment`. That is the specified algorithm, already using the detection
  embeddings the spec says to reuse.
- A driver at `services/autolabel/track/assign.py::retrack_session` and an endpoint at
  `POST /api/tracks/retrack`.
- A per-track review UI at `web/app/track/[id]/page.tsx` that already scans a track as a strip and
  highlights cells disagreeing with the dominant class, plus track relabel, merge, split, interpolate,
  smooth and attribute-timeline endpoints.

What is genuinely missing from WS5 is the analysis layer, not the association layer: lifespan bars across a
clip, gap and class-switch and size-discontinuity markers, per-track flicker metrics sortable, the keyframe
spot-check flow, `track_confirmed` propagation, and the filmstrip delta ticks. Also absent is any MOT
quality metric; `id_switch_flags` is self-reported by the tracker and never validated against ground truth,
so "flicker metrics" must be careful not to present tracker self-report as measurement.

---

## 6. History and the punch list

Undo is **snapshots, not commands**. `web/components/editor/useEditor.ts:37-41` stores whole pre-edit
snapshots capped at 100, with structural sharing so memory is references rather than deep copies. Labels are
generated prose ("relabelled sedan to truck").

**Conflict 8: WS6 item 1 is mostly a re-skin.** Jump-to-entry already works (`useEditor.ts:301-324`), and
`HistoryPanel.tsx:136-148` already renders one clickable button per entry with an up or down glyph and jumps
on click. What the spec asks for, a position marker with entries above it dimmed as undone, is a
presentation change over working machinery, not new behaviour. The undo and redo arrows it wants removed are
in the top bar at `page.tsx:1197-1200`.

Note `useEditor.ts:142-145` does a `JSON.stringify` comparison per object on every undo, redo and jump,
which is O(N) stringify on a path that will get slower as WS1 raises the object ceiling.

---

## 7. Cross-cutting gaps found

- **No feature flag mechanism exists anywhere.** The spec requires one flag per workstream so each ships
  independently. Nothing in `core/`, the API or `web/lib/` provides this, so it is a prerequisite for the
  stated rollout order rather than something to assume.
- **No client config channel exists.** `configs/` holds `default.yaml`, `class_priors.yaml` and
  `event_taxonomy.yaml`, read server-side only through `core/config.py`. The web app's only `process.env`
  use is the API proxy in `next.config.mjs:2`; there is no `NEXT_PUBLIC_*` anywhere. So
  `configs/canvas.yaml` and `configs/ontology_groups.yaml` need a delivery mechanism built, not just files
  written. The natural shape is a typed settings block plus a `GET /api/config/canvas` endpoint, matching
  how the ontology already reaches the client.
- Structured logging has a home already: `core/bus.py:31` provides an `EventBus` with `emit`, and
  `configs/event_taxonomy.yaml` is the precedent for declaring event shapes in config.

---

## Revised plan

Keeping the spec's rollout order, with the conflicts above folded in:

1. **Prerequisites** (neither is optional given the above): a feature flag mechanism, and a client config
   channel. Small, and everything else depends on them.
2. **WS1**, minus the renderer migration, plus the three re-render fixes first. Tiers reduce per-object draw
   cost, but the dominant cost today is re-rendering everything on pointer move; adding tiers on top of that
   would mask the real problem rather than fix it. Labels and hover are built, not tuned.
3. **WS2**, with `lifecycle` as the field name and a dry-run report that states the derived-versus-defaulted
   split honestly.
4. **WS6**, independent and small.
5. **WS3**, whose long pole is encoding the class hierarchy that `hierarchy_levels: 4` has been claiming for
   some time, and consolidating eight competing group definitions including one live disagreement.
6. **WS4**, reusing `attribution.py`'s calibration and its base-rate framing rather than building a second
   path, and deciding explicitly what happens to the four existing critic checks that have no WS4 code.
7. **WS5**, reduced to the analysis and review layer since association is already built and already uses the
   specified algorithm.
