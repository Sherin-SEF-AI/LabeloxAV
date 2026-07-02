import type {
  AlItem,
  AssignmentRow,
  AuditRow,
  Keypoints,
  ErrorCandidateRow,
  GovState,
  MergeRequestRow,
  RegistryRow,
  Relationship,
  AdverseRegion,
  ProjectedCuboid,
  CalibDetail,
  CalibResolved,
  CalibSession,
  EgoState,
  InertialEvents,
  Confusions,
  CorrectionCoverage,
  CorrectionSuggestion,
  DiscoveryCandidate,
  CurationSummary,
  DatasetDetail,
  DatasetRow,
  FrameMeta,
  LaneRow,
  MapCommitRow,
  MapFeature,
  MapProvenance,
  MulticamGroups,
  SimilarResponse,
  FrameObject,
  ObjectDynamicsRow,
  JobRow,
  ObjectDetail,
  Ontology,
  OntologyClass,
  Scenario,
  SegmentResult,
  SessionRow,
  Track,
  TriageRow,
  UserRow,
  CloudStatus,
  CloudOrphan,
} from "./types";

// Same-origin: next.config rewrites /api/* to the FastAPI backend. Every request carries the current
// user (X-Lbx-User-Id) for attribution.
import { userHeaders } from "./user";
import { begin, end } from "./progress";

async function get<T>(path: string): Promise<T> {
  begin();
  try {
    const r = await fetch(path, { cache: "no-store", headers: { ...userHeaders() } });
    if (!r.ok) throw new Error(`GET ${path} -> ${r.status}`);
    return r.json();
  } finally {
    end();
  }
}

async function post<T>(path: string, body: unknown): Promise<T> {
  begin();
  try {
    const r = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...userHeaders() },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`POST ${path} -> ${r.status} ${await r.text()}`);
    return r.json();
  } finally {
    end();
  }
}

async function put<T>(path: string, body: unknown): Promise<T> {
  begin();
  try {
    const r = await fetch(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json", ...userHeaders() },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`PUT ${path} -> ${r.status} ${await r.text()}`);
    return r.json();
  } finally {
    end();
  }
}

async function del<T>(path: string): Promise<T> {
  begin();
  try {
    const r = await fetch(path, { method: "DELETE", headers: { ...userHeaders() } });
    if (!r.ok) throw new Error(`DELETE ${path} -> ${r.status} ${await r.text()}`);
    return r.json();
  } finally {
    end();
  }
}

// SAM segment supports point prompts (with fg/bg labels) and/or a box prompt. Returns 503 if a
// training job holds the GPU; callers surface that as a non-blocking notice.
export type SegmentPrompt = { points?: number[][]; labels?: number[]; box?: number[]; precise?: boolean };

export type AgentPolicy = { auto_accept_conf?: number; review_low?: number; require_agreement?: boolean };
export type AgentCounts = { total: number; auto_accept: number; review: number; annotate: number; unchanged: number; demoted_by_critic: number };
export type AgentPlanItem = { object_id: string; class_name: string; conf: number; current_state: string; action: string; changes_state: boolean; reason: string; tier: string; critic_ok: boolean; critic_reasons: string[] };
export type AgentPlan = { frame_id: string; policy: AgentPolicy; counts: AgentCounts; critic_flags: Record<string, number>; items: AgentPlanItem[] };
export type AuditReport = {
  window_hours: number; sampled: number; vlm_checked: number; vlm_disagreements: number;
  among_sample_agreement: number | null;
  control_precision: { precision: number | null; reviewed: number; pending: number };
  confusion_movers: { from: string; to: string; n: number; concentrated_in: string | null }[];
  critic_flags: Record<string, number>; suspects_queued: number;
  budget: { max_calls: number; used: number; remaining: number }; notes: string[];
};

// LiDAR 3D viewer
export type LidarBounds = { min: number[]; max: number[]; n: number };
export type LidarCloud = {
  cloud_id: string;
  ts_ns: number;
  source: string;
  point_count: number;
  depth_model: string | null;
  bounds: LidarBounds | null;
  variants: string[];
};
export type LidarPoints = {
  points: Float32Array; // interleaved [x, y, z, intensity]
  count: number;
  decimated: boolean;
  source: string;
  frame: string;
  intensityMin: number;
  intensityMax: number;
};
export type Cuboid3D = {
  object_3d_id: string;
  cloud_id: string;
  frame_id: string | null;
  object_id: string | null;
  track_3d_id: string | null;
  class_id: number;
  class_name: string;
  center: number[];
  dims: number[];
  yaw: number;
  pitch: number;
  roll: number;
  conf: number;
  box_source: string;
  source: string;
  state: string;
  is_keyframe: boolean;
  attrs: Record<string, unknown>;
  version: number;
};
export type Cuboid3DInput = {
  class_id: number;
  center: number[];
  dims: number[];
  yaw: number;
  pitch?: number;
  roll?: number;
  attrs?: Record<string, unknown>;
  object_id?: string | null;
  ground_snap?: boolean;
};

// The segmentation overlay stream is Float32 [x, y, z, semantic_class], decimated with labels aligned.
export async function lidarSegmentationPoints(
  cloudId: string,
  max = 300000,
): Promise<{ points: Float32Array; count: number; classes: number[]; lowConfFrac: number }> {
  const r = await fetch(`/api/lidar/clouds/${cloudId}/segmentation/points?max=${max}`, {
    cache: "no-store",
    headers: { ...userHeaders() },
  });
  if (!r.ok) throw new Error(`GET segmentation points -> ${r.status}`);
  const buf = await r.arrayBuffer();
  const classesHeader = r.headers.get("X-Classes") || "";
  return {
    points: new Float32Array(buf),
    count: Number(r.headers.get("X-Point-Count") || 0),
    classes: classesHeader ? classesHeader.split(",").map(Number) : [],
    lowConfFrac: Number(r.headers.get("X-Low-Conf-Frac") ?? 0),
  };
}

// The point stream is a binary ArrayBuffer (Float32 xyzi), not JSON, so it is fetched directly.
export async function lidarCloudPoints(
  cloudId: string,
  opts?: { variant?: string; max?: number; full?: boolean },
): Promise<LidarPoints> {
  const q = new URLSearchParams();
  if (opts?.variant && opts.variant !== "raw") q.set("variant", opts.variant);
  if (opts?.max) q.set("max", String(opts.max));
  if (opts?.full) q.set("full", "true");
  const r = await fetch(`/api/lidar/clouds/${cloudId}/points?${q.toString()}`, {
    cache: "no-store",
    headers: { ...userHeaders() },
  });
  if (!r.ok) throw new Error(`GET points -> ${r.status}`);
  const buf = await r.arrayBuffer();
  return {
    points: new Float32Array(buf),
    count: Number(r.headers.get("X-Point-Count") || 0),
    decimated: r.headers.get("X-Decimated") === "True",
    source: r.headers.get("X-Source") || "",
    frame: r.headers.get("X-Frame") || "",
    intensityMin: Number(r.headers.get("X-Intensity-Min") ?? 0),
    intensityMax: Number(r.headers.get("X-Intensity-Max") ?? 1),
  };
}

export const api = {
  lidarClouds: (sessionId: string) =>
    get<{ session_id: string; clouds: LidarCloud[] }>(`/api/lidar/sessions/${sessionId}/clouds`),
  lidarCloudMeta: (cloudId: string) => get<LidarCloud & { calibration_version: string | null }>(`/api/lidar/clouds/${cloudId}`),
  lidarBuild: (sessionId: string, limit = 1) =>
    post<{ session_id: string; clouds: number; groups_total: number }>(
      `/api/lidar/sessions/${sessionId}/build`, { limit }),
  lidarTrajectory: (sessionId: string, refTsNs?: number) =>
    get<{ session_id: string; anchor_ts_ns?: number; heading_rad?: number; path: { x: number; y: number }[] }>(
      `/api/lidar/sessions/${sessionId}/trajectory${refTsNs != null ? `?ref_ts_ns=${refTsNs}` : ""}`),
  // 3D cuboid annotation
  lidarObjects3d: (cloudId: string) =>
    get<{ cloud_id: string; objects: Cuboid3D[] }>(`/api/lidar/clouds/${cloudId}/objects3d`),
  lidarCreateCuboid: (cloudId: string, body: Cuboid3DInput) =>
    post<Cuboid3D>(`/api/lidar/clouds/${cloudId}/objects3d`, body),
  lidarPatchCuboid: async (id: string, body: Partial<Cuboid3DInput> & { expected_version?: number }) => {
    const r = await fetch(`/api/lidar/objects3d/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", ...userHeaders() },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`PATCH cuboid -> ${r.status}`);
    return (await r.json()) as Cuboid3D;
  },
  lidarDeleteCuboid: (id: string) => del<{ deleted: string }>(`/api/lidar/objects3d/${id}`),
  lidarLiftFrame: (frameId: string) =>
    post<{ frame_id: string; cuboids: number; objects: unknown[] }>(`/api/lidar/frames/${frameId}/lift`, {}),
  lidarLiftCloud: (cloudId: string) =>
    post<{ frame_id: string; cuboids: number; objects: unknown[] }>(`/api/lidar/clouds/${cloudId}/lift`, {}),
  lidarSegment: (cloudId: string) =>
    post<{ seg_id: string; method: string; classes_present: number[]; low_conf_frac: number; n_instances: number }>(
      `/api/lidar/clouds/${cloudId}/segment`, {}),
  lidarTrack3d: (sessionId: string) =>
    post<{ session_id: string; tracks: number; detections: number }>(`/api/lidar/sessions/${sessionId}/track3d`, {}),
  lidarLinkCloud: (cloudId: string) =>
    post<{ cloud_id: string; linked: number; cuboids: number }>(`/api/lidar/clouds/${cloudId}/link`, {}),
  lidarObject3dProjection: (id: string, camId = "cam_f", w = 1280, h = 960) =>
    get<{ corners_uv: number[][]; in_front: boolean[]; in_image: boolean[]; edges: number[][]; any_in_image: boolean }>(
      `/api/lidar/objects3d/${id}/projection?cam_id=${camId}&w=${w}&h=${h}`),
  lidarObject3dLinked: (id: string) =>
    get<{ object_3d_id: string; object_id: string | null; class_id: number;
      projections: Record<string, number[]>; object_2d: { object_id: string; bbox: number[]; cam_id: string } | null }>(
      `/api/lidar/objects3d/${id}/linked`),
  lidarObject3dProperties: (id: string) =>
    post<{ object_3d_id: string; properties: Record<string, number | null> }>(
      `/api/lidar/objects3d/${id}/properties`, {}),
  lidarSimilar3d: (id: string, k = 10) =>
    get<{ object_3d_id: string; class_id: number;
      similar: { object_3d_id: string; dims: number[]; dims_dist: number; state: string }[] }>(
      `/api/lidar/objects3d/${id}/similar?k=${k}`),
  lidarBatchCorrect: (ids: string[], classId?: number, dims?: number[]) =>
    post<{ updated: number }>(`/api/lidar/objects3d/batch_correct`,
      { object_3d_ids: ids, class_id: classId ?? null, dims: dims ?? null }),
  ontology: () => get<Ontology>("/api/ontology"),
  addClass: (name: string) => post<OntologyClass & { existed: boolean }>("/api/ontology/classes", { name }),
  sessions: () => get<SessionRow[]>("/api/sessions"),
  sessionStats: (id: string) => get<{ session_id: string; frames: number; objects: number; by_state: Record<string, number>; done: number; progress: number }>(`/api/sessions/${id}/stats`),
  firstFrame: (id: string) => get<{ frame_id: string }>(`/api/sessions/${id}/first-frame`),
  // M4.0/M4.1 review queue
  alScore: (sessionId?: string, limit = 50) => get<{ pool: number; items: AlItem[] }>(`/api/activelearn/score?limit=${limit}${sessionId ? `&session_id=${sessionId}` : ""}`),
  errorCandidates: (status = "pending", limit = 100) => get<ErrorCandidateRow[]>(`/api/errordetect/candidates?status=${status}&limit=${limit}`),
  errorRun: (kinds?: string[]) => post<{ persisted: number; by_kind: Record<string, number> }>("/api/errordetect/run", kinds ? { kinds } : {}),
  errorConfirm: (id: string) => post(`/api/errordetect/candidates/${id}/confirm`, {}),
  errorDismiss: (id: string) => post(`/api/errordetect/candidates/${id}/dismiss`, {}),
  // M4.4 governance
  governState: () => get<GovState>("/api/govern/state"),
  governRegistry: () => get<RegistryRow[]>("/api/govern/registry"),
  governPrecision: () => get<{ reviewed: number; incorrect: number; precision: number | null; pending: number }>("/api/govern/control/precision"),
  governAudit: (limit = 60) => get<AuditRow[]>(`/api/govern/audit?limit=${limit}`),
  governTick: () => post<Record<string, unknown>>("/api/govern/controller/tick", {}),
  governDriftScan: () => post<{ breached: string[]; paused: boolean }>("/api/govern/drift/scan", {}),
  governPromote: (v: string) => post(`/api/govern/promote?model_version=${v}`, {}),
  governKill: (reason: string) => post("/api/govern/killswitch/engage", { reason }),
  governRelease: () => post("/api/govern/killswitch/release", {}),
  // M4.3 collaboration
  collabBranches: () => get<{ branches: string[] }>("/api/collaborate/branches"),
  collabAssignments: () => get<AssignmentRow[]>("/api/collaborate/assignments"),
  collabMRs: () => get<MergeRequestRow[]>("/api/collaborate/merge_requests"),
  collabApprove: (id: string, reviewer_id: string) => post(`/api/collaborate/merge_requests/${id}/approve`, { reviewer_id }),
  collabMerge: (id: string, reviewer_id: string) => post(`/api/collaborate/merge_requests/${id}/merge`, { reviewer_id }),
  collabRevert: (id: string, reviewer_id: string) => post(`/api/collaborate/merge_requests/${id}/revert`, { reviewer_id }),
  // M3.3 HD map
  hdmapGeoref: (sid: string) => post<{ elements: number; lanes: number; signs: number }>(`/api/hdmap/georef?session_id=${sid}`, {}),
  hdmapFuse: (sids: string, compute = "local") => post<{ job_id: string; commit_id?: string; fused_elements?: number; status?: string }>(`/api/hdmap/fuse?session_ids=${sids}&compute_target=${compute}`, {}),
  hdmapCommits: () => get<MapCommitRow[]>("/api/hdmap/commits"),
  hdmapElements: (q: string) => get<{ type: string; features: MapFeature[] }>(`/api/hdmap/elements?${q}`),
  hdmapProvenance: (eid: string) => get<MapProvenance>(`/api/hdmap/provenance?element_id=${eid}`),
  triage: (params: Record<string, string>) =>
    get<TriageRow[]>("/api/triage?" + new URLSearchParams(params).toString()),
  object: (id: string) => get<ObjectDetail>(`/api/objects/${id}`),
  segment: (frame_id: string, point: number[]) =>
    post<SegmentResult>("/api/segment", { frame_id, points: [point], labels: [1] }),
  // Frame-centric editor
  frame: (id: string) => get<FrameMeta>(`/api/frames/${id}`),
  frameObjects: (id: string) => get<FrameObject[]>(`/api/frames/${id}/objects`),
  // P3 per-object dynamics (derived: distance/speed/heading/ttc/risk)
  frameDynamics: (id: string) => get<{ frame_id: string; dynamics: ObjectDynamicsRow[] }>(`/api/dynamics/frame/${id}`),
  computeDynamics: (session_id: string) => post<{ objects: number; tracked_with_speed: number; with_distance: number }>(`/api/dynamics/compute?session_id=${session_id}`, {}),
  computeLidarCuboids: (frameId: string) => post<{ frame_id: string; cuboids: number; objects: { object_id: string; cuboid_3d: Record<string, unknown> }[] }>(`/api/lidar/cuboids/${frameId}`, {}),
  segmentPrompt: (frame_id: string, p: SegmentPrompt) =>
    post<SegmentResult>("/api/segment", { frame_id, ...p }),
  classifyObject: (frame_id: string, box: number[]) =>
    post<{ predictions: { class_id: number; class_name: string; conf: number }[] }>("/api/objects/classify", { frame_id, box }),
  createObject: (
    frame_id: string,
    body: { class_name: string; bbox: number[]; attrs?: Record<string, unknown>; mask_polygons?: number[][]; state?: string; idem_key?: string; rot_deg?: number; keypoints?: Keypoints | null; polyline?: number[][]; cuboid_3d?: { center: number[]; size: number[]; yaw: number } },
  ) => post<ObjectDetail>(`/api/frames/${frame_id}/objects`, body),
  updateMask: (object_id: string, polygons: number[][], width?: number, height?: number) =>
    put<{ object_id: string }>(`/api/objects/${object_id}/mask`, { polygons, width, height }),
  deleteObject: (object_id: string) => del<{ deleted: string }>(`/api/objects/${object_id}`),
  // object relationships / grouping (rider_of, towed_by, part_of, member_of, occludes)
  relateObject: (object_id: string, body: { to_object_id: string; kind: string }) =>
    post<{ relationship_id: string }>(`/api/objects/${object_id}/relate`, body),
  deleteRelationship: (relationship_id: string) =>
    del<{ deleted: string }>(`/api/relationships/${relationship_id}`),
  frameRelationships: (frame_id: string) =>
    get<Relationship[]>(`/api/frames/${frame_id}/relationships`),
  // adverse-condition region tags (glare, reflection, shadow, rain, fog, lowlight)
  createAdverse: (frame_id: string, body: { geometry: number[]; condition: string }) =>
    post<AdverseRegion>(`/api/frames/${frame_id}/adverse`, body),
  listAdverse: (frame_id: string) => get<AdverseRegion[]>(`/api/frames/${frame_id}/adverse`),
  deleteAdverse: (region_id: string) => del<{ deleted: string }>(`/api/adverse/${region_id}`),
  // in-image cuboids: projected wireframes + lift a pixel to the ego ground point
  frameCuboids: (frame_id: string) => get<ProjectedCuboid[]>(`/api/frames/${frame_id}/cuboids`),
  liftGround: (frame_id: string, u: number, v: number) =>
    get<{ ego: number[] | null; reason?: string }>(`/api/frames/${frame_id}/lift_ground?u=${u}&v=${v}`),
  // annotation agent: dry-run plan, reversible commit, revert
  agentPlan: (frame_id: string, policy: AgentPolicy = {}) =>
    post<AgentPlan>(`/api/agent/frames/${frame_id}/plan`, policy),
  agentRun: (frame_id: string, policy: AgentPolicy = {}) =>
    post<{ run_id: string; applied: number; counts: AgentCounts; policy: AgentPolicy }>(`/api/agent/frames/${frame_id}/run`, policy),
  agentRevert: (run_id: string) =>
    post<{ run_id: string; reverted: number; skipped: number }>(`/api/agent/runs/${run_id}/revert`, {}),
  agentAttributesPlan: (frame_id: string) =>
    post<{ counts: { objects: number; attrs_filled: number; by_attr: Record<string, number> } }>(`/api/agent/frames/${frame_id}/attributes/plan`, {}),
  agentAttributes: (frame_id: string) =>
    post<{ run_id: string; objects_updated: number; counts: { attrs_filled: number; by_attr: Record<string, number> } }>(`/api/agent/frames/${frame_id}/attributes`, {}),
  agentAsk: (text: string) =>
    post<{ understood: string; count: number; frames: { frame_id: string; session_id: string }[] }>(`/api/agent/ask`, { text }),
  agentReport: () =>
    get<{ size: { sessions: number; objects: number; human_labeled: number }; class_balance: { missing: number; rare: number }; coverage_gaps: string[]; fix_queue: Record<string, number>; fix_queue_total: number; scenarios: Record<string, number>; geo: Record<string, number> }>(`/api/agent/report`),
  agentSuggest: (frame_id: string) =>
    get<{ suggestions: { action: string; label: string; n: number; score: number }[] }>(`/api/agent/frames/${frame_id}/suggest`),
  agentTrainingCycle: (dry_run = true) =>
    post<{ run_id: string; tick: { frames: number; auto_accept: number; review: number; annotate: number }; retrain: { attempted: boolean; triggered?: boolean } }>(`/api/agent/training/cycle`, { dry_run }),
  agentGoldDrift: () =>
    post<{ status: string; champion?: string; baseline_map?: number; current_map?: number; drop?: number }>(`/api/agent/gold-drift`, {}),
  agentMineScenarios: () =>
    post<{ persisted: number; by_kind: Record<string, number>; top: { kind: string; score: number; tag: string }[] }>(`/api/agent/scenarios/mine`, {}),
  agentMineDisagreements: () =>
    post<{ persisted: number; top: { score: number; tag: string }[] }>(`/api/agent/disagreements/mine`, {}),
  agentCoverage: () =>
    get<{ scene_frames: number; class_balance: { median: number; missing: string[]; rare: string[] }; scene_coverage: Record<string, Record<string, number>>; geo: Record<string, number>; gaps: string[] }>(`/api/agent/coverage`),
  agentErrorSweep: (max_sessions = 10, kinds?: string[]) =>
    post<{ run_id: string; status: string }>(`/api/agent/errors/sweep`, kinds ? { max_sessions, kinds } : { max_sessions }),
  agentErrorQueue: (status = "pending", limit = 60) =>
    get<{ summary: Record<string, number>; candidates: { candidate_id: string; object_id: string; kind: string; score: number; detail: Record<string, unknown>; proposed_label?: { class_name?: string } | null }[] }>(`/api/agent/errors/queue?status=${status}&limit=${limit}`),
  agentTemporalRepairPlan: (session_id?: string) =>
    post<{ counts: { tracks: number; flipped_tracks: number; relabels: number; skipped_static?: number } }>(`/api/agent/temporal-repair/plan`, session_id ? { session_id } : {}),
  agentTemporalRepair: (session_id?: string) =>
    post<{ run_id: string; relabeled: number; counts: Record<string, number> }>(`/api/agent/temporal-repair`, session_id ? { session_id } : {}),
  agentCuboidsPlan: (frame_id: string) =>
    post<{ counts: { total: number; auto_accept: number; review: number; skip: number } }>(`/api/agent/frames/${frame_id}/cuboids/plan`, {}),
  agentCuboids: (frame_id: string) =>
    post<{ run_id: string; attached: number; counts: { auto_accept: number; review: number; skip: number } }>(`/api/agent/frames/${frame_id}/cuboids`, {}),
  agentCrossCamPlan: (object_id: string) =>
    post<{ counts: { targets: number; auto_accept: number; review: number; skip: number }; class_name?: string; reason?: string }>(`/api/agent/objects/${object_id}/crosscam/plan`, {}),
  agentCrossCam: (object_id: string) =>
    post<{ run_id: string; created: number; counts: { auto_accept: number; review: number; skip: number } }>(`/api/agent/objects/${object_id}/crosscam`, {}),
  agentPropagatePlan: (object_id: string, span = 24) =>
    post<{ object_id: string; counts: { total_steps: number; auto_accept: number; review: number; stops: number; appearance_used: boolean }; forward: number; backward: number }>(`/api/agent/objects/${object_id}/propagate/plan`, { span }),
  agentPropagate: (object_id: string, span = 24) =>
    post<{ run_id: string; track_id: string; created: number; counts: { auto_accept: number; review: number; stops: number } }>(`/api/agent/objects/${object_id}/propagate`, { span }),
  agentCommand: (frame_id: string, text: string) =>
    post<{ intent: { action: string; classes: string[] | string; conf_min: number | null }; result: unknown; summary: string; blocked?: boolean }>(`/api/agent/command`, { text, frame_id }),
  // relabel: an independent model re-reads existing boxes and corrects the class where it decisively disagrees
  agentRelabelPlan: (frame_id: string) =>
    post<{ frame_id: string; counts: { total: number; relabel_keep: number; relabel_review: number }; items: { object_id: string; from_name: string; to_name: string; conf: number; action: string }[] }>(`/api/agent/frames/${frame_id}/relabel/plan`, {}),
  agentRelabel: (frame_id: string) =>
    post<{ run_id: string; relabeled: number; counts: { total: number; relabel_keep: number; relabel_review: number } }>(`/api/agent/frames/${frame_id}/relabel`, {}),
  agentRelabelAll: (opts: { max_frames?: number; session_id?: string } = {}) =>
    post<{ run_id: string; status: string }>(`/api/agent/relabel/all`, opts),
  agentRunStatus: (run_id: string) =>
    get<{ run_id: string; kind: string; status: string; counts: Record<string, number>; changed: number }>(`/api/agent/runs/${run_id}`),
  // Overnight Auditor: run the nightly patrol, read the morning report
  agentAuditRun: (opts: { sample_size?: number; vlm_calls?: number; since_hours?: number } = {}) =>
    post<{ run_id: string; status: string }>(`/api/agent/audit/run`, opts),
  agentAuditLatest: () =>
    get<{ run_id?: string; status?: string; created_at?: string; report: AuditReport | null }>(`/api/agent/audit/latest`),
  // Drift Investigator: scan for drift now, root-cause a breach; read the latest diagnosis
  agentDriftInvestigate: () =>
    post<{ breached: string[]; ran?: boolean; run_id?: string }>(`/api/agent/drift/investigate`, {}),
  agentDriftLatest: () =>
    get<{ status?: string; created_at?: string; report: { breached: string[]; hypothesis: string; proposed_action: { kind: string; detail?: string } } | null }>(`/api/agent/drift/latest`),
  // Documentation Agent: draft datasheet / weekly quality report from the platform's own metrics
  agentDocDatasheet: (gold_id?: string) =>
    post<{ uri: string; markdown: string }>(`/api/agent/docs/datasheet`, gold_id ? { gold_id } : {}),
  agentDocWeekly: () =>
    post<{ uri: string; markdown: string }>(`/api/agent/docs/weekly`, {}),
  // pixel-assist: brush/eraser mask composition + SLIC superpixels
  composeMask: (body: { polygons: number[][]; ops: { op: string; center: number[]; radius: number }[]; width: number; height: number }) =>
    post<{ polygons: number[][] }>(`/api/mask/compose`, body),
  superpixels: (frame_id: string, n = 300) =>
    post<{ superpixels: number[][] }>(`/api/superpixels/${frame_id}?n=${n}`, {}),
  // dense semantic/panoptic segmentation: run auto, fetch metadata (the overlay is an image URL)
  autoSegment: (frame_id: string, kind = "semantic") =>
    post<{ kind: string; coverage: Record<string, number>; n_instances: number }>(`/api/frames/${frame_id}/segment?kind=${kind}`, {}),
  getSegment: (frame_id: string, kind = "semantic") =>
    get<{ found: boolean; coverage?: Record<string, number>; has_overlay?: boolean; source?: string; model_version?: string | null }>(`/api/frames/${frame_id}/segment?kind=${kind}`),
  // M2.1 lanes
  framesLanes: (frameId: string) => get<LaneRow[]>(`/api/frames/${frameId}/lanes`),
  proposeLanes: (frameId: string) => post<{ proposed: number; lanes: LaneRow[]; model: string }>(`/api/frames/${frameId}/lanes/propose`, {}),
  createLane: (frameId: string, body: { control_points: number[][]; lane_type: string; is_ego: boolean }) =>
    post<LaneRow>(`/api/frames/${frameId}/lanes`, body),
  updateLane: (laneId: string, body: { control_points: number[][]; lane_type: string; is_ego: boolean }) =>
    put<LaneRow>(`/api/lanes/${laneId}`, body),
  deleteLane: (laneId: string) => del<{ deleted: string }>(`/api/lanes/${laneId}`),
  propagateLanes: (frameId: string, frames = 8) => post<{ created: number; to_frames: number }>(`/api/frames/${frameId}/lanes/propagate?frames=${frames}`, {}),
  // M2.2 drivable
  segmentDrivable: (frameId: string) => post<{ coverage: Record<string, number>; model: string }>(`/api/frames/${frameId}/drivable`, {}),
  getDrivable: (frameId: string) => get<{ found: boolean; classes?: Record<string, number[][]>; coverage?: Record<string, number>; source?: string; model_version?: string | null }>(`/api/frames/${frameId}/drivable`),
  propagateObject: (object_id: string, frames = 12) =>
    post<{ created: number; track_id?: string; object_ids?: string[]; reason?: string }>(
      `/api/objects/${object_id}/propagate?frames=${frames}`, {}),
  interpolateTrack: (track_id: string) =>
    post<{ created: number; track_id: string }>(`/api/tracks/${track_id}/interpolate`, {}),
  // Track (tracklet) editor
  // Operational layer: unified jobs, bulk review, UI-triggered autolabel
  users: () => get<UserRow[]>("/api/users"),
  createUser: (name: string, role: string) => post<UserRow>("/api/users", { name, role }),
  curationSummary: (session_id?: string) =>
    get<CurationSummary>("/api/curation/summary" + (session_id ? `?session_id=${session_id}` : "")),
  curationEmbed: (session_id?: string) =>
    post<{ started: boolean }>("/api/curation/embed" + (session_id ? `?session_id=${session_id}` : ""), {}),
  datasets: () => get<DatasetRow[]>("/api/datasets"),
  dataset: (id: string) => get<DatasetDetail>(`/api/datasets/${id}`),
  startExport: (body: { name: string; states?: string[]; class_names?: string[]; cities?: string[]; session_id?: string; formats: string[] }) =>
    post<{ job_id: string; status: string }>("/api/datasets/export", body),
  jobs: () => get<JobRow[]>("/api/jobs"),
  ingestProgress: () => get<{ active: boolean; finished: boolean; done: number; total: number; current: string | null; frames: number }>("/api/ingest/progress"),
  bulkReview: (object_ids: string[], action: string, class_name?: string, state?: string, attrs?: Record<string, unknown>) =>
    post<{ updated: number }>("/api/objects/bulk-review", { object_ids, action, class_name, state, attrs }),
  // Interactive AI correction: correct one -> find similar -> bulk apply
  correctionSuggest: (body: {
    object_id: string;
    kind: "class" | "attr";
    old_class_name?: string;
    new_class_name?: string;
    attr_key?: string;
    old_value?: unknown;
    new_value?: unknown;
    filters?: Record<string, unknown>;
    limit?: number;
    threshold?: number;
  }) => post<CorrectionSuggestion>("/api/corrections/suggest", body),
  confusions: (by = "class") => get<Confusions>(`/api/corrections/confusions?by=${by}`),
  discoveryQueue: (state = "pending") => get<DiscoveryCandidate[]>(`/api/discovery/queue?state=${state}`),
  discoveryRun: (session_id: string) =>
    post<{ candidates: number; by_kind: Record<string, number> }>(`/api/discovery/run?session_id=${session_id}`, {}),
  discoverySetState: (candidate_id: string, state: string, tag?: string) =>
    post<{ state: string }>(`/api/discovery/${candidate_id}/state`, { state, tag }),
  searchSimilar: (body: { frame_id?: string; object_id?: string; image_b64?: string; mode?: "visual" | "semantic"; k?: number }) =>
    post<SimilarResponse>("/api/search/similar", body),
  searchSemantic: (q: string, k = 24) =>
    get<{ query: string; filters: Record<string, string>; classes: string[]; count: number; results: SimilarResponse["results"] }>(
      `/api/search/semantic?q=${encodeURIComponent(q)}&k=${k}`),
  correctionCoverage: () => get<CorrectionCoverage>("/api/corrections/coverage"),
  computeObjectEmbeddings: (session_id?: string) =>
    post<{ started: boolean }>("/api/corrections/embed" + (session_id ? `?session_id=${session_id}` : ""), {}),
  startAutolabel: (session_id: string, limit?: number, compute_target: "local" | "cloud" = "local") =>
    post<{ job_id: string; status: string }>("/api/autolabel/start", { session_id, limit, compute_target }),
  estimateEgoMasks: (force = false) =>
    post<{ cameras: number; with_hood: number; no_hood: string[] }>(`/api/autolabel/ego-masks/estimate?force=${force}`, {}),
  piiBackfill: (limit = 2000) =>
    post<{ status: string; limit: number }>(`/api/autolabel/pii-backfill?limit=${limit}`, {}),
  redetectAll: (backfill_pii = true) =>
    post<{ run_id: string; status: string }>(`/api/autolabel/redetect-all?backfill_pii=${backfill_pii}`, {}),
  startVlmQa: (session_id: string, limit = 40) =>
    post<{ started: boolean }>(`/api/qa/vlm?session_id=${session_id}&limit=${limit}`, {}),
  recognizeSigns: (session_id: string, limit = 200) =>
    post<{ recognized: number; text_bearing: number }>(`/api/signs/recognize?session_id=${session_id}&limit=${limit}`, {}),
  track: (id: string) => get<Track>(`/api/tracks/${id}`),
  // M3.2 map-assisted
  mapMatch: (sid: string) => post<{ matched: number; no_road: number; road_classes: Record<string, number> }>(`/api/mapassist/match?session_id=${sid}`, {}),
  framePriors: (fid: string) => get<{ found: boolean; has_map: boolean; road_class?: string; lane_count?: number | null; speed_limit?: number | null; hints: { kind: string }[] }>(`/api/mapassist/priors?frame_id=${fid}`),
  // M3.0 calibration
  calibrationValidate: (sid: string) => post<CalibDetail>(`/api/calibration/validate?session_id=${sid}`, {}),
  calibrationSessions: () => get<CalibSession[]>("/api/calibration/sessions"),
  calibrationDetail: (sid: string) => get<CalibDetail>(`/api/calibration/${sid}`),
  // M-CAL.3 real-calibration ingestion + the resolved/trust surface
  calibrationResolved: (sid: string) => get<CalibResolved>(`/api/calibration/${sid}/resolved`),
  // M-IMU Plane-4 inertial
  egoState: (sid: string) => get<EgoState>(`/api/sessions/${sid}/egostate`),
  inertialEvents: (sid: string) => get<InertialEvents>(`/api/sessions/${sid}/inertial_events`),
  calibrationSetSpec: (sid: string, camSpecs: Record<string, Record<string, number>>, source = "measured") =>
    post<Record<string, unknown>>(`/api/calibration/${sid}/calibrate`, { cam_specs: camSpecs, source }),
  calibrationEstimate: (sid: string) => post<Record<string, unknown>>(`/api/calibration/${sid}/estimate`, {}),
  calibrationExtrinsics: (sid: string) => post<{ checked: boolean; reason?: string; worst_sampson_px?: number | null }>(`/api/calibration/${sid}/extrinsics`, {}),
  calibrationImport: (sid: string, body: { cam_id: string; format: string; ref_width?: number; calib_text?: string; camera_intrinsic?: number[][]; translation?: number[] }) =>
    post<Record<string, unknown>>(`/api/calibration/${sid}/import`, body),
  // M3.1 multi-camera
  multicamGroups: (sid: string) => get<MulticamGroups>(`/api/multicam/groups?session_id=${sid}`),
  multicamAssociate: (sid: string) => post<{ associated: number; rig_tracks: number; cameras: string[]; reason?: string }>(`/api/multicam/associate?session_id=${sid}`, {}),
  // M2.5 keyframe + interpolation
  setKeyframe: (objectId: string, value = true) => post<{ is_keyframe: boolean; track_id: string | null }>(`/api/objects/${objectId}/keyframe?value=${value}`, {}),
  interpolateKeyframed: (trackId: string, method = "linear") => post<{ created: number; method: string; keyframes: number }>(`/api/tracks/${trackId}/interpolate-keyframed?method=${method}`, {}),
  reinterpolate: (objectId: string, method = "linear") => post<{ created: number }>(`/api/objects/${objectId}/reinterpolate?method=${method}`, {}),
  relabelTrack: (id: string, class_name: string) =>
    post<{ relabeled: number }>(`/api/tracks/${id}/relabel`, { class_name }),
  deleteTrack: (id: string) => del<{ n_objects: number }>(`/api/tracks/${id}`),
  review: (
    id: string,
    payload: {
      reviewer?: string;
      action: string;
      class_name?: string;
      bbox?: number[];
      attrs?: Record<string, unknown>;
      state?: string;
      time_spent_ms?: number;
      expected_version?: number;
      rot_deg?: number;
      keypoints?: Keypoints | null;
      mask_polygons?: number[][];
      polyline?: number[][];
      cuboid_3d?: { center: number[]; size: number[]; yaw: number };
    },
  ) => post<ObjectDetail & { version?: number; rot_deg?: number }>(`/api/objects/${id}/review`, payload),
  scenarios: (params: Record<string, string>) =>
    get<Scenario[]>("/api/scenarios?" + new URLSearchParams(params).toString()),
  scenarioSearch: (q: string, semantic = false) =>
    get<{ query: string; count: number; results: Scenario[]; semantic: boolean }>(
      "/api/scenarios/search?" + new URLSearchParams({ q, semantic: String(semantic) }).toString(),
    ),
  objectSimilar: (id: string) =>
    get<{ object_id: string; results: SimilarObject[] }>(`/api/objects/${id}/similar`),
  searchObjects: (q: string) =>
    get<{ query: string; count: number; results: SimilarObject[] }>(
      "/api/search/objects?" + new URLSearchParams({ q }).toString(),
    ),
  // Gate A evidence + Gate B (M9) quality sheet
  analyticsPii: (session_id?: string) =>
    get<PiiCoverage>("/api/analytics/pii" + (session_id ? `?session_id=${session_id}` : "")),
  goldSets: () => get<GoldSetRow[]>("/api/quality/gold-sets"),
  qualitySheet: (gold_id: string) =>
    get<QualitySheet>("/api/quality/sheet?" + new URLSearchParams({ gold_id }).toString()),
  sealGold: (body: { name: string; cities?: string[]; session_id?: string; limit?: number }) =>
    post<{ gold_id: string; n_objects: number; n_frames: number }>("/api/quality/gold/seal", body),
  fitCalibration: (body: { gold_id?: string; session_id?: string }) =>
    post<{ uri: string; n_train: number; report: { ece: number | null } }>(
      "/api/quality/calibrate/fit",
      body,
    ),
  // Upload + multi-format import
  uploadMultipart,
  startImport: (body: {
    format: string;
    source_uri: string;
    target_vehicle?: string;
    city?: string;
    options?: Record<string, unknown>;
  }) => post<{ job_id: string; status: string }>("/api/imports/start", body),
  importStatus: (jobId: string) => get<ImportJob>(`/api/imports/${jobId}`),
  listImports: () => get<ImportJob[]>("/api/imports"),
  // In-app training platform
  trainingTasks: () => get<{ task_type: string; default_base_weights: string }[]>("/api/training/tasks"),
  startTraining: (body: {
    purpose: string;
    task_type?: string;
    compute_target?: string;
    dataset_spec?: Record<string, unknown>;
    base_weights?: string | null;
    hparams?: Record<string, unknown>;
    gate?: Record<string, unknown>;
    promote?: boolean;
    notes?: string;
  }) => post<{ job_id: string; status: string }>("/api/training/start", body),
  trainingStatus: (jobId: string) => get<TrainingJob>(`/api/training/${jobId}`),
  listTraining: () => get<TrainingJob[]>("/api/training"),
  cancelTraining: (jobId: string) => post<TrainingJob>(`/api/training/${jobId}/cancel`, {}),
  trainingRegistry: () => get<ModelLine[]>("/api/training/registry"),

  // Warm cloud-GPU control. connect carries the acknowledged hourly rate (the backend rejects a mismatch).
  cloudStatus: () => get<CloudStatus>("/api/cloud/status"),
  cloudConnect: (ackHourlyUsd: number) => post<CloudStatus>("/api/cloud/connect", { ack_hourly_usd: ackHourlyUsd }),
  cloudDisconnect: (pause = false) => post<CloudStatus>("/api/cloud/disconnect", { pause }),
  cloudOrphans: () => get<{ orphans: CloudOrphan[] }>("/api/cloud/orphans"),
  cloudTerminateOrphan: (podId: string) => post<{ terminated: string }>("/api/cloud/orphans/terminate", { pod_id: podId }),
};

export type TrainingJob = {
  job_id: string;
  status: string; // pending|running|done|error|canceled
  purpose: string;
  task_type: string;
  compute_target: string; // local | cloud
  stage: string | null;
  progress: number;
  counts: Record<string, number>;
  metrics: Record<string, unknown>;
  result: Record<string, unknown>;
  error: string | null;
  run_id: string | null;
  config: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
};

export type ModelLineRun = {
  run_id: string;
  dataset_name: string;
  epochs: number;
  map50: number | null;
  safe_miou: number | null;
  promoted: boolean;
  weights_uri: string | null;
  created_at: string | null;
};

export type ModelLine = {
  purpose: string;
  task_type: string;
  runs: ModelLineRun[];
  promoted: ModelLineRun | null;
};

export type ImportJob = {
  job_id: string;
  status: string;
  format: string;
  source_uri: string | null;
  target_vehicle: string;
  city: string | null;
  progress: number;
  counts: Record<string, number>;
  error: string | null;
  session_id: string | null;
  created_at: string | null;
  updated_at: string | null;
};

// Presigned-multipart direct-to-storage: bytes go browser -> MinIO/S3, never through the API.
// 64 MB parts; the API only signs each part. Returns the s3:// uri of the assembled object.
export async function uploadMultipart(
  file: File,
  onProgress?: (frac: number) => void,
): Promise<string> {
  const PART = 64 * 1024 * 1024;
  const nParts = Math.max(1, Math.ceil(file.size / PART));
  const { key, upload_id } = await post<{ key: string; upload_id: string }>("/api/upload/init", {
    filename: file.name,
    content_type: file.type || "application/octet-stream",
  });
  const parts: { PartNumber: number; ETag: string }[] = [];
  try {
    for (let i = 0; i < nParts; i++) {
      const blob = file.slice(i * PART, Math.min(file.size, (i + 1) * PART));
      const { url } = await post<{ url: string }>("/api/upload/sign", {
        key,
        upload_id,
        part_number: i + 1,
      });
      const r = await fetch(url, { method: "PUT", body: blob });
      if (!r.ok) throw new Error(`part ${i + 1} PUT -> ${r.status}`);
      const etag = r.headers.get("ETag") || r.headers.get("etag");
      if (!etag) throw new Error(`no ETag on part ${i + 1} (check MinIO CORS ExposeHeaders)`);
      parts.push({ PartNumber: i + 1, ETag: etag });
      onProgress?.((i + 1) / nParts);
    }
    const { uri } = await post<{ uri: string }>("/api/upload/complete", { key, upload_id, parts });
    return uri;
  } catch (e) {
    await post("/api/upload/abort", { key, upload_id }).catch(() => {});
    throw e;
  }
}

export type PiiCoverage = {
  total_frames: number;
  frames_anonymized: number;
  coverage_pct: number;
  faces_blurred: number;
  plates_blurred: number;
  method_versions: Record<string, number>;
};

export type GoldSetRow = {
  gold_id: string;
  name: string;
  n_objects: number;
  n_frames: number;
  ontology_version: string;
  measured: boolean;
  created_at: string | null;
};

export type QualitySheet = {
  gold_id: string;
  found: boolean;
  name?: string;
  n_objects?: number;
  n_frames?: number;
  measured?: boolean;
  metrics?: {
    weights?: string;
    map50?: number;
    map?: number;
    precision?: number;
    recall?: number;
    safe_miou?: number | null;
    safety_weight?: number;
    per_class_pr?: Record<string, { precision: number; recall: number; ap50: number }>;
    calibration?: { ece: number | null; n_train: number } | null;
  };
};

export type SimilarObject = {
  object_id: string;
  frame_id: string;
  class_id: number;
  class_name: string;
  conf: number;
  state: string;
  score: number;
  image_url: string;
};
