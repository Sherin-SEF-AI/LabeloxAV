// Client for the sensors tools (server: services/api/routers/tools_sensors.py, paths /api/tools/sensors/...):
// #10 radar and event cameras (/sensors), #15 scene reconstruction (/reconstruct), #16 synthetic variants
// (/synthetic). Background work starts through a route that returns a job; follow it with useToolJob.

import { apiDelete, apiGet, apiPost, getBinary } from "@/lib/api";

const B = "/api/tools/sensors";

export type JobRef = { job_id: string; kind: string; status: string; session_id: string | null };

function qs(params: Record<string, string | number | boolean | null | undefined>): string {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") p.set(k, String(v));
  const s = p.toString();
  return s ? `?${s}` : "";
}

// ---------------------------------------------------------------- radar and events

export type RadarSessionRow = {
  session_id: string; name: string | null; sweeps: number; sensors: number; first_ts_ns: number;
  last_ts_ns: number; points: number;
};
export type EventStreamRow = {
  stream_id: string; session_id: string; session_name: string | null; sensor: string; source: string;
  width: number; height: number; n_events: number; t0_ns: number; t1_ns: number;
  stats: { positive?: number; negative?: number; events_per_s?: number; duration_s?: number };
  accumulation: EventAccumulation | null; created_at: string | null;
};
export type EventAccumulation = {
  mode: "count" | "time_surface"; window_ms: number; tau_ms: number | null; frames: number;
  kept_labelled: number; removed: number; events_per_frame: number;
};
export type SensorsOverview = { radar: RadarSessionRow[]; events: EventStreamRow[] };
export const sensorsOverview = () => apiGet<SensorsOverview>(`${B}/overview`);

export type RadarImportBody = {
  session_id: string; source_uri: string; format: "nuscenes" | "mcap" | "pcd";
  keyframes_only?: boolean; filter_points?: boolean; topics?: string[] | null;
  extrinsics?: Record<string, number[][]> | null; sensor_to_ego?: number[][] | null;
  time_source?: "log" | "header"; sensor?: string | null; ts_ns?: number | null;
};
export const radarImport = (body: RadarImportBody) => apiPost<JobRef>(`${B}/radar/import`, body);

export type RadarSweepRow = {
  sweep_id: string; ts_ns: number; sensor: string; source: string; n_points: number;
  stats: { n_total?: number; compensated?: boolean; vector?: boolean; moving?: number; max_range_m?: number | null;
    max_speed_mps?: number | null };
};
export type RadarFrameRow = { frame_id: string; ts_ns: number; cam_id: string; radar_dt_ms: number | null };
export type RadarSweeps = { session_id: string; sweeps: RadarSweepRow[]; sensors: string[]; frames: RadarFrameRow[] };
export const radarSweeps = (sessionId: string) => apiGet<RadarSweeps>(`${B}/radar/sessions/${sessionId}/sweeps`);

export type RadarOverlayPoint = {
  u: number; v: number; depth_m: number; range_m: number; radial_mps: number; speed_mps: number; rcs: number;
  end: [number, number] | null;
};
export type RadarOverlay = {
  frame_id: string; cam_id: string; width: number; height: number; calibration: string; arrow_s: number;
  sensors: { sweep_id: string; sensor: string; ts_ns: number; dt_ms: number; n_points: number; in_image: number;
    points: RadarOverlayPoint[] }[];
};
export const radarOverlay = (frameId: string) => apiGet<RadarOverlay>(`${B}/radar/frames/${frameId}/overlay`);

export type RadarBevPoint = { x: number; y: number; z: number; vx: number; vy: number; radial_mps: number; rcs: number };
export type RadarBev = {
  session_id: string; ts_ns: number;
  sensors: { sweep_id: string; sensor: string; ts_ns: number; origin: number[]; points: RadarBevPoint[] }[];
  boxes: { object_3d_id: string; center: number[]; dims: number[]; yaw: number; class_id: number; state: string;
    radar?: { source: string; speed_mps: number; residual_mps: number | null; n_returns: number } | null }[];
};
export const radarBev = (sessionId: string, tsNs: number) =>
  apiGet<RadarBev>(`${B}/radar/sessions/${sessionId}/bev${qs({ ts_ns: tsNs })}`);

export const radarFuse = (sessionId: string, body: { tol_mps: number; apply: boolean }) =>
  apiPost<JobRef>(`${B}/radar/sessions/${sessionId}/fuse`, body);
export type RadarFusionTrack = {
  track_3d_id: string; class_name: string; first_ts_ns: number; last_ts_ns: number; median_speed_mps: number | null;
  fused: number; rejected: number; no_radar: number; tol_mps: number; source: string;
};
export const radarFusion = (sessionId: string) =>
  apiGet<{ session_id: string; tracks_3d: number; fused_tracks: RadarFusionTrack[] }>(
    `${B}/radar/sessions/${sessionId}/fusion`);

export type EventsImportBody = {
  source_uri: string; format?: "npy" | "h5" | "mcap" | null; session_id?: string | null; vehicle_id?: string | null;
  city?: string | null; sensor: string; width?: number | null; height?: number | null; columns?: string;
  time_unit?: "ns" | "us" | "ms" | "s"; time_offset_ns?: number;
};
export const eventsImport = (body: EventsImportBody) => apiPost<JobRef>(`${B}/events/import`, body);
export const eventsPreviewUrl = (streamId: string, p: { t_ns?: number; mode: string; window_ms: number; tau_ms: number }) =>
  `${B}/events/streams/${streamId}/preview.jpg${qs(p)}`;
export const eventsAccumulate = (streamId: string, body: {
  mode: "count" | "time_surface"; window_ms: number; tau_ms: number; max_frames: number; replace: boolean;
}) => apiPost<JobRef>(`${B}/events/streams/${streamId}/accumulate`, body);
export type EventFrames = {
  stream_id: string; session_id: string; accumulation: EventAccumulation | null;
  frames: { frame_id: string; ts_ns: number; tags: string[]; objects: number }[];
};
export const eventsFrames = (streamId: string) => apiGet<EventFrames>(`${B}/events/streams/${streamId}/frames`);

// ---------------------------------------------------------------- reconstruction

export type ReconCamera = { cam_id: string; frames: number; first_ts_ns: number; last_ts_ns: number };
export type ReconSession = { session_id: string; name: string | null; poses: number; clouds: number; cameras: ReconCamera[] };
export type ReconSessions = { sessions: ReconSession[]; gsplat: { installed: boolean; version: string | null } };
export const reconSessions = () => apiGet<ReconSessions>(`${B}/reconstruct/sessions`);

export type ReconView = {
  index: number; name: string; kind: "camera" | "novel" | "bev"; frame_id: string | null; ts_ns: number | null;
  frame_index: number | null; held_out: boolean; K: number[][]; T_world_cam: number[][]; width: number;
  height: number; ortho: boolean; ppm?: number; heading?: number[]; z_max?: number;
};
export type ReconMetrics = {
  window?: { frames_requested: number; frames_posed: number; frames_without_pose: number; scans: number };
  initial_points?: number; gaussians?: number; iterations?: number; psnr_train?: number | null;
  psnr_held_out?: number | null; psnr_camera_views?: number | null; held_out_frames?: number;
  train_frames?: number; seconds?: number; image_size?: number[]; calibration?: string; points?: number;
};
export type Reconstruction = {
  recon_id: string; session_id: string; method: "gsplat" | "pointmap"; status: string; cam_id: string | null;
  first_ts_ns: number | null; last_ts_ns: number | null; params: Record<string, unknown>; metrics: ReconMetrics;
  error: string | null; job_id: string | null; created_at: string | null; finished_at: string | null;
  has_ply: boolean; preview_points: number | null; n_views: number; views?: ReconView[]; frames?: string[];
};
export type ReconstructBody = {
  session_id: string; method: "gsplat" | "pointmap"; cam_id?: string | null; start_ts_ns?: number | null;
  n_frames: number; stride: number; iters: number; max_width?: number;
};
export const reconstruct = (body: ReconstructBody) =>
  apiPost<{ reconstruction: Reconstruction; job: JobRef }>(`${B}/reconstruct`, body);
export const reconstructions = (sessionId?: string) =>
  apiGet<{ reconstructions: Reconstruction[] }>(`${B}/reconstructions${qs({ session_id: sessionId })}`);
export const reconstruction = (id: string) => apiGet<Reconstruction>(`${B}/reconstructions/${id}`);
export const reconViewUrl = (id: string, index: number) => `${B}/reconstructions/${id}/views/${index}.jpg`;
export const reconPlyUrl = (id: string) => `${B}/reconstructions/${id}/splats.ply`;
export const deleteReconstruction = (id: string) => apiDelete<{ deleted: string }>(`${B}/reconstructions/${id}`);

/** The preview cloud: interleaved [x, y, z, r, g, b] relative to `anchor` (add it back for world coordinates). */
export async function reconPreview(id: string): Promise<{ points: Float32Array; count: number; anchor: number[] }> {
  const r = await getBinary(`${B}/reconstructions/${id}/preview.bin`);
  if (!r.ok) throw new Error(r.status === 404 ? "no preview cloud yet" : `preview failed (${r.status})`);
  const buf = await r.arrayBuffer();
  const anchor = (r.headers.get("X-Anchor") ?? "0,0,0").split(",").map(Number);
  return { points: new Float32Array(buf), count: Number(r.headers.get("X-Count") ?? buf.byteLength / 24), anchor };
}

export type PlaceBody = {
  center: number[]; dims: number[]; yaw: number; class_id: number; ground_snap: boolean; max_range_m: number;
};
export const placeInRecon = (id: string, body: PlaceBody) => apiPost<JobRef>(`${B}/reconstructions/${id}/place`, body);
export type PlacedRow = {
  job_id: string; status: string; progress: number; error: string | null; center: number[] | null;
  dims: number[] | null; yaw: number | null; class_id: number | null; created_at: string | null;
  result: { track_3d_id?: string; scans?: number; boxes_2d?: number; tracks_2d?: Record<string, string>;
    with_points?: number; skipped?: Record<string, number> } | null;
};
export const reconPlaced = (id: string) => apiGet<{ placed: PlacedRow[] }>(`${B}/reconstructions/${id}/placed`);

// ---------------------------------------------------------------- synthetic variants

export type Perturbation = { name: string; default_strength: number };
export const variantPerturbations = () => apiGet<{ perturbations: Perturbation[] }>(`${B}/variants/perturbations`);
export type HardFrame = {
  frame_id: string; session_id: string; cam_id: string; ts_ns: number; signal: number; why: string;
  eligible: boolean; refused: string | null;
};
export const hardFrames = (p: { source: "errors" | "corrected" | "session"; session_id?: string; limit?: number;
  include_review?: boolean }) => apiGet<{ frames: HardFrame[] }>(`${B}/variants/hard-frames${qs(p)}`);
export const variantPreviewUrl = (frameId: string, perturbation: string, strength: number, seed = 7) =>
  `${B}/variants/preview.jpg${qs({ frame_id: frameId, perturbation, strength, seed })}`;
export const variantCheck = (frameIds: string[], includeReview: boolean) =>
  apiPost<{ eligible: string[]; refused: Record<string, string>; objects: number }>(
    `${B}/variants/check`, { frame_ids: frameIds, include_review: includeReview });
export const buildVariants = (body: { frame_ids: string[]; perturbations: string[]; strengths: Record<string, number>;
  seed: number; include_review: boolean }) =>
  apiPost<JobRef & { eligible: number; refused: Record<string, string>; variants: number }>(`${B}/variants`, body);
export type VariantRun = {
  job_id: string; status: string; progress: number; stage: string | null; error: string | null;
  perturbations: string[] | null; sources: number; frames: number | null; objects: number | null;
  session_id: string | null; per_perturbation: Record<string, number> | null;
  reverted: { removed: number } | null; first_frame_id: string | null; created_at: string | null;
};
export const variantRuns = () => apiGet<{ runs: VariantRun[] }>(`${B}/variants/runs`);
export const variantSummary = () =>
  apiGet<{ frames: number; by_perturbation: Record<string, number>; labels: number }>(`${B}/variants/summary`);
export const revertVariants = (jobId: string) =>
  apiPost<{ removed: number; skipped: number; objects_left: number }>(`${B}/variants/runs/${jobId}/revert`, {});

// The copy-paste generator's existing routes (services/api/routers/synth.py), offered on the same page.
export type SynthPlan = {
  classes: { class_name: string; class_id: number | null; donors: number; reason?: string }[];
  backgrounds: number; n_frames: number; batches: number; feasible: boolean; reason: string | null;
};
export const synthPlan = (classNames: string[], nFrames: number) =>
  apiPost<SynthPlan>("/api/synth/plan", { class_names: classNames, n_frames: nFrames });
export const synthBuild = (classNames: string[], nFrames: number) =>
  apiPost<{ run_id: string; plan: SynthPlan; seed: string }>("/api/synth/build", { class_names: classNames, n_frames: nFrames });
export type SynthRun = {
  run_id: string; status: string; policy: { class_names?: string[]; n_frames?: number; seed?: string };
  report: { frames?: number; requested?: number; reason?: string; stopped?: string; pasted?: Record<string, number> };
  created_at: string | null; batches: { run_id: string; status: string; frames: number }[];
};
export const synthRuns = () => apiGet<{ runs: SynthRun[] }>("/api/synth/runs?limit=20");
