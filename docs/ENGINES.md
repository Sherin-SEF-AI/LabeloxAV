# Engines: one default per job

LabeloxAV grew several implementations of the same few jobs. This page records which one is the default for
each, what was folded together, and what was left alone on purpose. It was written from an inventory of
every implementation and its callers (2026-11-02).

## Optical-flow propagation

**Default: `services/temporal/flow.py`** (`flow_box`, `flow_track`). The editor's forward propagation, the
SAM propagation and the agent's propagation walk all step boxes with it. It is the guarded version of the
three copies that existed: it handles frames of different sizes and refuses a box that grows or shrinks
more than 4x between adjacent frames. Every frame query that feeds it filters on the source camera.

Different methods, kept: lane control points (`services/autolabel/lane/curves.py`), projecting a box into
other cameras by geometry (`services/multicam/propagate.py`), ego-motion propagation
(`services/calyx/ego_propagate.py`, which refuses on this corpus for lack of pose data).

## Interpolation between keyframes

**Default: `services/temporal/interpolate.py`** (`interpolate_track_keyframed`). It refuses to bridge two
boxes that are not plausibly one object (`services/temporal/gap_gate.py`), filters by camera, and is
idempotent. The keyframe routes (`/objects/{id}/keyframe`, `/reinterpolate`) use it.

Left alone: `services/intelligence/propagate.py::interpolate_track`, the older linear fill without the
same-object check. It has no API caller, only its CLI and tests. Routing it through the gated version would
change what those tests expect (the gated one refuses holes), so it stays until someone retires the CLI.

## Tracking

**Default: BoT-SORT** (`services/autolabel/track/tracker.py`, Kalman filter + DINOv3 appearance), selected by
`intelligence.tracker.backend`. The greedy IoU tracker (`services/intelligence/tracking.py::track_camera`)
runs only if that setting is changed.

`POST /mine` (`services/intelligence/run.py::mine_session`) tracks a session the first time it is mined.
Mined again, it reuses the tracks already stored rather than adding a second set beside them, because
replacing tracks would cascade away the track events people wrote on them, and it replaces the session's
scenarios instead of duplicating them. `POST /tracks/retrack` is the explicit way to replace a session's
tracks, from detections only.

3D tracking, cross-camera rig tracks and person re-identification are different jobs and stay separate.

## Confidence calibration

**Serving path: `services/autolabel/calibrate.py::calibrate_confidence`**, which applies the curve named in
`configs/default.yaml` (`calibrate.isotonic_uri`).

**Fitting a curve to activate: `services/autolabel/gold_calibrate.py`** (held-out frame split, a
`trustworthy` verdict). `POST /quality/calibrate/fit` fits on the review trail instead; it now reports
`trustworthy: false` with a warning whenever that trail is too lopsided, which on this corpus is always.

Dormant and different, kept: density-conditioned (`services/oraclyx/density_calibration.py`) and
confidence x track-coherence (`services/oraclyx/joint_calibration.py`) calibration. Wiring either in would
change every auto-accept decision, so neither is on by default.
