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
`trustworthy` verdict). Run it with `make calibrate-gold ARGS="fit-gold --gold <id>"` (or `fit-vlm`), and
switch the gate to the result with `make calibrate-gold ARGS="activate --uri <uri>"`, which refuses a curve
whose held-out check did not pass. Fitting never activates. `POST /quality/calibrate/fit` fits on the review trail instead; it now reports
`trustworthy: false` with a warning whenever that trail is too lopsided, which on this corpus is always.

Dormant and different, kept: density-conditioned (`services/oraclyx/density_calibration.py`) and
confidence x track-coherence (`services/oraclyx/joint_calibration.py`) calibration. Wiring either in would
change every auto-accept decision, so neither is on by default.

## 3D dataset interchange

Added 2026-11-24. What reads and writes 3D data, and what each one is not.

**nuScenes import: `services/imports/adapter_nuscenes.py`** reads two things that share the table layout.
A standard release (`v1.0-mini`, `v1.0-trainval`) has only 3D boxes in the global frame, so each box is
taken global -> ego -> sensor for every camera keyframe of its sample and projected with the camera
intrinsic; the 2D box is the projected silhouette clipped to the image, after cutting the box at a 0.1 m
near plane. LabeloxAV's own export (the `lbx_bbox2d` field) is read as before. No devkit: the quaternion
and transform math is `services/imports/nuscenes_tables.py`.

A standard import carries the category name as the source class, the instance token as `track_ref`,
`nuscenes_visibility` and `nuscenes_attributes` in attrs, and in provenance `cuboid_3d` (ego frame,
`{center, size [w, l, h], yaw}`, the shape of `Object.cuboid_3d`) plus `nuscenes_box` with the full
rotation in the global, ego and sensor frames. A frame's `ts_ns` is its sample's timestamp, so the six
cameras of a sample form one rig group; the camera's own capture time is in provenance.

Not done by the import driver yet (`services/imports/run.py`): it stores the cuboid only inside provenance
and does not set `Object.cuboid_3d`, it does not use `track_ref` to build tracks, it writes every scene of a
dataset into one session, and it does not store the per-camera calibration. nuScenes category names are not
in the remap table, so they land on the fallback classes until one is added.

**nuScenes LiDAR: `services/lidar/ingest/nuscenes_sweeps.py`** lists the LIDAR_TOP sweeps (file, time,
sensor -> ego, ego -> global) and reads them into Clouds in the ego frame. Persisting them is
`store_cloud(cloud, session_id)`; nothing calls that for an import yet.

**PointCloud2: `services/lidar/ingest/pointcloud2.py`** decodes `sensor_msgs/PointCloud2` with numpy
(offsets, datatypes, point_step, row_step, endianness, organised clouds; non-finite points dropped) and
deserializes the message itself from ROS 2 CDR and from ROS 1 bytes, by its fixed layout rather than through
a schema-driven decoder. `read_pointclouds_mcap` (`services/lidar/ingest/mcap_pc.py`) reads the private
`labelox-npz` channel and both PointCloud2 forms from one file. Per-point time is decoded but the internal
Cloud has no field for it.

**ROS 1 bags: `services/lidar/ingest/rosbag1.py`** reads PointCloud2 topics from a format 2.0 `.bag` by a
sequential scan (none, bz2 and lz4 chunks). It is tested against bags assembled from the format
specification, not against a bag recorded by `rosbag record`: none was available.

**3D export: `services/lidar/export/adapters3d.py`**. OpenLABEL is the primary format. The `nuscenes`,
`kitti` and `waymo_json` outputs borrow those datasets' box field conventions and are not loadable by the
datasets' own tooling. In particular `waymo_json` (older key and function name: `waymo`,
`write_waymo_3d`, both still accepted) is a JSON file with Waymo `Label.Box` field names. It is not the
Waymo Open Dataset format, which is TFRecord files of `Frame` protobufs; nothing here writes that.
