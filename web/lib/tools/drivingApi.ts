// Client for the driving annotation tools (server: services/api/routers/tools_driving.py, paths /api/tools/driving/...).
// Background work (auto-alignment, long lane propagation) returns a job; follow it with useToolJob from ./toolJobs.

import { apiDelete, apiGet, apiPost, apiPut } from "@/lib/api";
import type { BevView } from "./drivingGeom";

const B = "/api/tools/driving";

export type QueuedJob = { job_id: string; kind: string; status: string; session_id: string | null };

// ---- #7 lanes in a top-down view ----

export const LANE_TYPES = ["solid", "dashed", "double", "road_edge", "implicit", "fallback", "stop_line", "unknown"] as const;
export const LANE_RELATIONS = ["successor", "predecessor", "left_neighbour", "right_neighbour", "stops_at", "controlled_by"] as const;
export type LaneRelationKind = (typeof LANE_RELATIONS)[number];

export type BevLane = {
  lane_id: string;
  track_ref: string | null;
  control_points: number[][];
  bev_points: number[][];
  dropped: number;
  lane_type: string;
  is_ego: boolean;
  source: string;
  marking_conf: number | null;
  method: string | null;
};
export type LaneRelation = {
  relation_id: string;
  lane_id: string;
  relation: LaneRelationKind;
  target_lane_id: string | null;
  target_object_id: string | null;
  target_element_id: string | null;
};
export type BevState = {
  frame_id: string;
  session_id: string;
  cam_id: string;
  view: BevView;
  calibration: { source: string; quality: number; model: string };
  lanes: BevLane[];
  relations: LaneRelation[];
  signals: { object_id: string; bbox: number[]; class_name: string }[];
  has_pose: boolean;
  pose_source: string | null;
  next_frames: number;
};
export type PropagateResult =
  | ({ queued: true } & QueuedJob)
  | { queued: false; created: number; frames: number; frames_with_lanes: number; skipped_no_pose: number;
      skipped_human: number; out_of_view: number; pose_source: string; reason?: string };
export type LaneExport = {
  lanelet2: string;
  opendrive: string;
  georeferenced: boolean;
  origin: { lat: number; lon: number } | null;
  frames: number;
  boundaries: number;
  stop_lines: number;
  lanelets: number;
  successors: number;
  unpaired: { id: string; reason: string }[];
  regulatory: number;
  hints: { pairs: number; successors: number };
  note: string | null;
};

export const bevLanes = (frameId: string) => apiGet<BevState>(`${B}/frames/${frameId}/bev-lanes`);
export const saveBevLane = (frameId: string, body: { lane_id?: string | null; bev_points: number[][]; lane_type: string; is_ego: boolean }) =>
  apiPut<{ lane_id: string; control_points: number[][]; lane_type: string; is_ego: boolean }>(`${B}/frames/${frameId}/bev-lanes`, body);
export const addLaneRelation = (laneId: string, body: { relation: LaneRelationKind; target_lane_id?: string; target_object_id?: string; target_element_id?: string }) =>
  apiPost<LaneRelation>(`${B}/lanes/${laneId}/relations`, body);
export const deleteLaneRelation = (relationId: string) => apiDelete<{ deleted: string }>(`${B}/lane-relations/${relationId}`);
export const propagateLanesByPose = (frameId: string, frames: number, laneIds?: string[]) =>
  apiPost<PropagateResult>(`${B}/frames/${frameId}/lanes/propagate-pose`, { frames, lane_ids: laneIds ?? null });
export const exportLanes = (frameIds: string[]) => apiPost<LaneExport>(`${B}/lanes/export`, { frame_ids: frameIds });

// ---- #8 paths and intent ----

export type PathSession = { session_id: string; route: string | null; vehicle_id: string | null; city: string | null; tracks: number; ego_poses: number };
export type PathLabel = {
  path_id: string;
  kind: "ego_path" | "future";
  track_id: string | null;
  anchor_frame_id: string | null;
  anchor_ts_ns: number;
  coord_frame: "world" | "ego";
  points: number[][];
  source: "human" | "baseline_ctrv" | "smoothed";
  seed: Record<string, unknown>;
  notes: string | null;
  updated_at: string | null;
};
export type PathTrack = {
  track_id: string;
  class_id: number;
  class_name: string;
  intents: { intent: string; source?: string; status?: string }[];
  n: number;
  first_ts_ns: number;
  last_ts_ns: number;
  /** [[t_s, x, y], ...] in the session's coord frame, t from t0_ns. */
  path: number[][];
  frames: string[];
  events: number;
  futures: PathLabel[];
};
export type EgoPose = { ts_ns: number; frame_id: string | null; x: number; y: number; yaw: number; source: string; measured: boolean };
export type PathSessionView = {
  session_id: string;
  route: string | null;
  vehicle_id: string | null;
  city: string | null;
  coord_frame: "world" | "ego";
  t0_ns: number;
  origin: [number, number] | null;
  frame_note: string;
  ego: { poses: EgoPose[]; label: PathLabel | null };
  tracks: PathTrack[];
  positions_beyond_range: number;
  calibration: string[];
  baseline_note: string;
};
export type Forecast = {
  track_id: string;
  anchor_frame_id: string;
  anchor_ts_ns: number;
  coord_frame: "world" | "ego";
  state: { x: number; y: number; speed_mps: number; heading_rad: number; yaw_rate_rps: number };
  forecast: number[][];
  observed_after: number[][];
  error_vs_observed: { ade_m: number; fde_m: number; n: number; horizon_s: number } | null;
  model: "ctrv";
  note: string;
};
export type EventType = { name: string; definition: string; applies_to: string; applicable: boolean };
export type TrackIntents = {
  track_id: string;
  class_l1: string | null;
  intents: { intent: string; source?: string; status?: string }[];
  quick: string[];
  missing_quick: string[];
  event_types: EventType[];
  events: { event_id: string; event_type: string; state: string; source: string; start_frame_id: string;
            end_frame_id: string; start_ts_ns: number; end_ts_ns: number; notes: string | null }[];
};

export const pathSessions = () => apiGet<PathSession[]>(`${B}/paths/sessions`);
export const pathSession = (sid: string) => apiGet<PathSessionView>(`${B}/paths/sessions/${sid}`);
export const trackForecast = (trackId: string, anchorFrameId: string | null, horizonS = 3, dtS = 0.25) => {
  const q = new URLSearchParams({ horizon_s: String(horizonS), dt_s: String(dtS) });
  if (anchorFrameId) q.set("anchor_frame_id", anchorFrameId);
  return apiGet<Forecast>(`${B}/paths/tracks/${trackId}/forecast?${q}`);
};
export const saveFuture = (trackId: string, body: { anchor_frame_id: string; points: number[][]; source: "human" | "baseline_ctrv"; seed?: Record<string, unknown>; notes?: string }) =>
  apiPut<PathLabel>(`${B}/paths/tracks/${trackId}/future`, body);
export const deletePathLabel = (pathId: string) => apiDelete<{ deleted: string }>(`${B}/paths/labels/${pathId}`);
export const smoothEgo = (sid: string, window = 5) =>
  apiPost<{ points: number[][]; anchor_ts_ns: number; window: number }>(`${B}/paths/sessions/${sid}/ego/smooth`, { window });
export const saveEgoPath = (sid: string, body: { points: number[][]; source: "human" | "smoothed"; notes?: string }) =>
  apiPut<PathLabel>(`${B}/paths/sessions/${sid}/ego`, body);
export const trackIntents = (trackId: string) => apiGet<TrackIntents>(`${B}/paths/tracks/${trackId}/intents`);
export const addTrackIntent = (trackId: string, body: { event_type: string; start_frame_id: string; end_frame_id: string; notes?: string }) =>
  apiPost<{ event_id: string; state: string; mirrored_to_intents: boolean }>(`${B}/paths/tracks/${trackId}/intents`, body);
export const deleteTrackIntent = (eventId: string) => apiDelete<{ deleted: string }>(`${B}/paths/intents/${eventId}`);
export const scenarioExport = (sid: string) =>
  apiGet<{ json: Record<string, unknown>; openscenario: string; actors: number; events: number }>(`${B}/paths/sessions/${sid}/scenario`);

// ---- #9 camera-LiDAR alignment ----

export type AlignSession = { session_id: string; route: string | null; vehicle_id: string | null; scans: number;
  cameras: { cam_id: string; synced_frames: number }[]; calibrations: { cam_id: string; source: string }[] };
export type AlignFrame = { frame_id: string; cloud_id: string; ts_ns: number; cam_id: string; width: number; height: number };
export type CalibSummary = { source: string; quality: number; model: string; fx: number; fy: number; cx: number; cy: number;
  rpy_deg: number[]; xyz_m: number[] };
export type AlignScore = { edge: number | null; mi: number | null; objective?: number; in_image: number; edge_points: number; views?: number };
export type AlignView = {
  frame_id: string;
  cloud_id: string;
  session_id: string;
  cam_id: string;
  width: number;
  height: number;
  points_b64: string;
  point_count: number;
  point_stride: number;
  edge_png: string | null;
  current: CalibSummary;
  base: CalibSummary;
  has_manual: boolean;
  score_current: AlignScore;
  score_base: AlignScore;
  scan_points: number;
};
export type RefineResult = {
  rpy_deg: number[];
  xyz_m: number[];
  before: AlignScore;
  after: AlignScore;
  improved: boolean;
  evaluations: number;
  held_out: { frames: string[]; before: AlignScore; after: AlignScore; improved: boolean } | null;
  recommend_save: boolean;
  shift_from_start: { mean_px: number | null; median_px: number | null };
  shift_from_base: { mean_px: number | null; median_px: number | null };
  at_bound: boolean;
  alignment_id: string;
};
export type SaveResult = { calibration_id: string; alignment_id: string; before: AlignScore; after: AlignScore; improved: boolean;
  base_source: string; resolver_source?: string };
export type AlignHistory = { alignment_id: string; cam_id: string; method: string; saved: boolean; base_source: string;
  rpy_deg: number[]; xyz_m: number[]; score_before: AlignScore; score_after: AlignScore; frames: number; notes: string | null;
  created_at: string | null };

export const alignSessions = () => apiGet<AlignSession[]>(`${B}/align/sessions`);
export const alignFrames = (sid: string, camId?: string) =>
  apiGet<{ session_id: string; frames: AlignFrame[] }>(`${B}/align/sessions/${sid}/frames${camId ? `?cam_id=${encodeURIComponent(camId)}` : ""}`);
export const alignView = (frameId: string) => apiGet<AlignView>(`${B}/align/frames/${frameId}/view`);
export const alignScore = (frameIds: string[], rpy: number[], xyz: number[]) =>
  apiPost<{ current: AlignScore; candidate: AlignScore }>(`${B}/align/score`, { frame_ids: frameIds, rpy_deg: rpy, xyz_m: xyz });
export const alignSolvePairs = (frameId: string, points: number[][], uv: number[][], rpy: number[], xyz: number[]) =>
  apiPost<{ rpy_deg: number[]; xyz_m: number[]; before: { rms: number | null }; after: { rms: number | null }; pairs: number }>(
    `${B}/align/solve-pairs`, { frame_id: frameId, points, uv, rpy_deg: rpy, xyz_m: xyz });
export const alignRefine = (frameIds: string[], rpy: number[], xyz: number[]) =>
  apiPost<QueuedJob>(`${B}/align/refine`, { frame_ids: frameIds, rpy_deg: rpy, xyz_m: xyz });
export const alignSave = (body: { frame_ids: string[]; rpy_deg: number[]; xyz_m: number[]; method: "manual" | "auto" | "pairs";
  notes?: string; pairs_points?: number[][]; pairs_uv?: number[][] }) => apiPost<SaveResult>(`${B}/align/save`, body);
export const alignRevert = (sid: string, camId: string) =>
  apiDelete<{ reverted: boolean }>(`${B}/align/sessions/${sid}/cameras/${encodeURIComponent(camId)}/manual`);
export const alignHistory = (sid: string, camId?: string) =>
  apiGet<{ session_id: string; alignments: AlignHistory[] }>(`${B}/align/sessions/${sid}/history${camId ? `?cam_id=${encodeURIComponent(camId)}` : ""}`);

/** Decode the view's base64 little-endian float32 rows. */
export function decodePoints(b64: string): Float32Array {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

/** Save text as a file in the browser. */
export function downloadText(name: string, text: string, type = "application/xml") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
