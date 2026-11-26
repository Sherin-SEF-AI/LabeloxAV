// Client for the scene3d annotation tools (server: services/api/routers/tools_scene3d.py, paths /api/tools/scene3d/...).
// #3 label once in 3D, #5 point labels, #6 occupancy editing. Point clouds, labels and masks travel as binary
// (Float32 / Int16 / Uint8), through getBinary so the auth header rides along; everything else is JSON.

import { apiGet, apiPost, getBinary } from "@/lib/api";

const B = "/api/tools/scene3d";

export type ToolJobRef = { job_id: string; kind: string; status: string; session_id: string | null; mode?: string };

async function binary(path: string, init?: RequestInit): Promise<Response> {
  const r = await getBinary(path, init);
  if (!r.ok) {
    let msg = `${init?.method ?? "GET"} ${path.replace(B, "")} failed (${r.status})`;
    try {
      const j = await r.json();
      if (j?.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r;
}

const hnum = (r: Response, k: string, d = 0) => Number(r.headers.get(k) ?? d);

// ---- session overview ----

export type Scene3dOverview = {
  session_id: string; vehicle_id: string; route: string | null;
  clouds: Record<string, number>; lidar: boolean; poses: number; cameras: Record<string, number>;
  calibration: Record<string, string>; ego_frame: string; lidar_height_m: number | null;
  occupancy: Record<string, number>;
};
export const sceneOverview = (sid: string) => apiGet<Scene3dOverview>(`${B}/sessions/${sid}/overview`);

// ---- #3 label once ----

export type ScenePoints = { points: Float32Array; count: number; scans: number; anchor: number[] };
export async function scenePoints(sid: string, opts: { every?: number; max?: number; range?: number } = {}): Promise<ScenePoints> {
  const q = new URLSearchParams();
  if (opts.every) q.set("every", String(opts.every));
  if (opts.max) q.set("max", String(opts.max));
  if (opts.range) q.set("range_m", String(opts.range));
  const r = await binary(`${B}/sessions/${sid}/scene/points?${q.toString()}`);
  return {
    points: new Float32Array(await r.arrayBuffer()), count: hnum(r, "X-Point-Count"), scans: hnum(r, "X-Scans"),
    anchor: (r.headers.get("X-Anchor") || "0,0,0").split(",").map(Number),
  };
}

export type ScenePose = { cloud_id: string; ts_ns: number; x: number; y: number; z: number; yaw: number };
export const scenePoses = (sid: string) =>
  apiGet<{ anchor: number[]; summary: Record<string, unknown>; poses: ScenePose[] }>(`${B}/sessions/${sid}/scene/poses`);

export type LabelOnceBody = {
  center: number[]; dims: number[]; yaw: number; class_id: number;
  ground_snap?: boolean; max_range_m?: number; project?: boolean;
};
export const labelOnce = (sid: string, body: LabelOnceBody) => apiPost<ToolJobRef>(`${B}/sessions/${sid}/label-once`, body);

export type LabelOnceTrack = {
  track_3d_id: string; class_id: number; class_name: string; first_ts_ns: number; last_ts_ns: number;
  scene_label: { center: number[]; dims: number[]; yaw: number } | null;
  states: Record<string, number>; scans: number;
};
export const labelOnceTracks = (sid: string) => apiGet<{ tracks: LabelOnceTrack[] }>(`${B}/sessions/${sid}/label-once`);

export type ReviewBox = {
  object_id: string; frame_id: string; cam_id: string; bbox: number[]; state: string; width: number; height: number;
  version: number;
};
export type ReviewScan = {
  object_3d_id: string; cloud_id: string; ts_ns: number; frame_id: string | null; center: number[]; dims: number[];
  yaw: number; state: string; version: number; n_points: number | null; range_m: number | null;
  ground_snapped: boolean | null; boxes: ReviewBox[];
};
export type TrackReview = {
  track_3d_id: string; class_id: number; session_id: string; track_id: string | null;
  scene_label: { center: number[]; dims: number[]; yaw: number } | null; scans: ReviewScan[]; n: number;
};
export const trackReview = (tid: string) => apiGet<TrackReview>(`${B}/tracks3d/${tid}/review`);
export const trackReviewApply = (tid: string, body: { action: "accept" | "reject"; object_3d_ids?: string[]; include_2d?: boolean }) =>
  apiPost<{ updated_3d: number; updated_2d: number; state: string; run_id: string | null }>(`${B}/tracks3d/${tid}/review`, body);

// ---- #5 point labels ----

export type SegPoints = {
  points: Float32Array; count: number; total: number; stride: number; source: string; sessionId: string;
  tsNs: number; intensityMin: number; intensityMax: number;
};
export async function segPoints(cloudId: string, max = 400000): Promise<SegPoints> {
  const r = await binary(`${B}/clouds/${cloudId}/points?max=${max}`);
  return {
    points: new Float32Array(await r.arrayBuffer()), count: hnum(r, "X-Point-Count"), total: hnum(r, "X-Total"),
    stride: hnum(r, "X-Stride", 1), source: r.headers.get("X-Source") || "", sessionId: r.headers.get("X-Session") || "",
    tsNs: hnum(r, "X-Ts"), intensityMin: hnum(r, "X-Intensity-Min"), intensityMax: hnum(r, "X-Intensity-Max", 1),
  };
}

export type SegVersion = {
  seg_id: string; source: "auto" | "model" | "human" | string; version: number; method: string | null;
  model_version: string; kind: string; n_points: number | null; low_conf_frac: number | null;
  base_seg_id: string | null; created_by: string | null; created_at: string | null;
};
export const segVersions = (cloudId: string) => apiGet<{ versions: SegVersion[] }>(`${B}/clouds/${cloudId}/segmentations`);

export async function segLabels(cloudId: string, stride: number, segId?: string):
  Promise<{ labels: Int16Array; segId: string; source: string; version: number }> {
  const q = new URLSearchParams({ stride: String(stride) });
  if (segId) q.set("seg_id", segId);
  const r = await binary(`${B}/clouds/${cloudId}/labels?${q.toString()}`);
  return { labels: new Int16Array(await r.arrayBuffer()), segId: r.headers.get("X-Seg-Id") || "",
           source: r.headers.get("X-Seg-Source") || "", version: hnum(r, "X-Seg-Version") };
}

export async function saveSegLabels(cloudId: string, stride: number, labels: Int16Array, baseSegId?: string | null):
  Promise<{ seg_id: string; version: number; labelled: number; n_points: number; classes: Record<string, number> }> {
  const q = new URLSearchParams({ stride: String(stride) });
  if (baseSegId) q.set("base_seg_id", baseSegId);
  const body = new Uint8Array(labels.buffer, labels.byteOffset, labels.byteLength).slice().buffer;
  const r = await binary(`${B}/clouds/${cloudId}/labels?${q.toString()}`, {
    method: "PUT", body, headers: { "Content-Type": "application/octet-stream" } });
  return r.json();
}

export type AssistMask = { mask: Uint8Array; selected: number; info: string };
export async function assistGround(cloudId: string, stride: number): Promise<AssistMask> {
  const r = await binary(`${B}/clouds/${cloudId}/assist/ground?stride=${stride}`, { method: "POST" });
  return { mask: new Uint8Array(await r.arrayBuffer()), selected: hnum(r, "X-Selected"),
           info: `${r.headers.get("X-Method") || "ground"} plane` };
}
export async function assistCluster(cloudId: string, stride: number, point: number[], eps = 0.5): Promise<AssistMask> {
  const r = await binary(`${B}/clouds/${cloudId}/assist/cluster?stride=${stride}`, {
    method: "POST", body: JSON.stringify({ point, eps }), headers: { "Content-Type": "application/json" } });
  return { mask: new Uint8Array(await r.arrayBuffer()), selected: hnum(r, "X-Selected"),
           info: r.headers.get("X-Info") || "" };
}
export async function assistLabels(cloudId: string, stride: number, kind: "from-2d" | "from-cuboids"):
  Promise<{ labels: Int16Array; selected: number; info: string }> {
  const r = await binary(`${B}/clouds/${cloudId}/assist/${kind}?stride=${stride}`, { method: "POST" });
  return { labels: new Int16Array(await r.arrayBuffer()), selected: hnum(r, "X-Selected"),
           info: r.headers.get("X-Info") || "" };
}
export const ptv3Status = () => apiGet<{ available: boolean; reason: string; classes?: string[] }>(`${B}/ptv3/status`);
export const assistAi = (cloudId: string) => apiPost<ToolJobRef>(`${B}/clouds/${cloudId}/assist/ai`, {});

// ---- #6 occupancy ----

export type OccGridRow = {
  grid_id: string; ts_ns: number; source: string; occupied: number; flow_voxels: number; placed: boolean;
  voxel_m: number; dims: number[]; frame_id: string | null; edited: boolean; occupied_after: number;
};
export const occGrids = (sid: string) => apiGet<{ grids: OccGridRow[] }>(`${B}/sessions/${sid}/occupancy`);

export type OccGrid = {
  grid_id: string; session_id: string; ts_ns: number; source: string; origin: number[]; voxel_m: number;
  dims: number[]; placed: boolean; frame_id: string | null; voxels: number[][]; flow: number[][]; edited: number[];
  total: number; truncated: boolean; base_occupied: number; set_occupied: number[][]; set_free: number[][];
  edited_by: string | null;
};
export const occGrid = (gridId: string) => apiGet<OccGrid>(`${B}/occupancy/${gridId}`);

export type OccOp =
  | { kind: "paint"; value: "occupied" | "free" | "reset"; voxels: number[][] }
  | { kind: "box"; value: "occupied" | "free" | "reset"; lo: number[]; hi: number[] };
export const occEdit = (gridId: string, ops: OccOp[]) =>
  apiPost<{ touched: number; delta_voxels: number; occupied_before: number; occupied_after: number }>(
    `${B}/occupancy/${gridId}/edit`, { ops });
export const occPropagate = (gridId: string, count: number, ops?: OccOp[]) =>
  apiPost<{ propagated_to: number; skipped_unplaced: number; skipped_outside: number; reason?: string }>(
    `${B}/occupancy/${gridId}/propagate`, { count, ops: ops ?? null });
export const occBuild = (sid: string, body: { mode: "auto" | "lidar" | "camera"; every?: number; limit?: number }) =>
  apiPost<ToolJobRef>(`${B}/sessions/${sid}/occupancy/build`, body);
