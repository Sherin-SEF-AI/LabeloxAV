// Client for the quality annotation tools (server: services/api/routers/tools_quality.py, paths /api/tools/quality/...).
// #11 consensus, #12 explained errors, #13 VLM reason per label, #14 rare scenes. Background work returns a
// ToolJob reference; follow it with useToolJob from ./toolJobs.

import { apiGet, apiPost } from "@/lib/api";

export type JobRef = { job_id: string; kind: string; status: string; session_id: string | null };

// ---- #11 consensus --------------------------------------------------------------------------------------

export type MergeParams = { iou_thresh: number; quorum: number; include_model: boolean; require_human: boolean };
export const DEFAULT_MERGE: MergeParams = { iou_thresh: 0.5, quorum: 2, include_model: true, require_human: true };

export type ConsensusTask = {
  task_id: string; name: string; project_id: string; project: string; replicas: number;
  n_groups: number; ready_groups: number; consensus_objects: number; created_at: string | null;
};
export type ClassTally = { accepted: number; disputed: number; dropped: number };
export type HeatCell = (ClassTally & { rate: number | null }) | null;
export type ConsensusGroup = {
  replica_group: string; ready: boolean;
  jobs: { job_id: string; replica_index: number; state: string; assignee_id: string | null; n_frames: number }[];
  scored_frames: number; detection_agreement: number | null; consensus_objects: number;
  per_class: Record<string, ClassTally>;
  counts: Record<string, number> | null;
};
export type ConsensusBoard = {
  task_id: string; name: string; replicas: number; params: MergeParams; n_groups: number;
  groups: ConsensusGroup[];
  heatmap: { classes: string[]; groups: string[]; cells: HeatCell[][] };
};
export type Disagreement = {
  issue_id: string; frame_id: string; object_id: string | null; job_id: string | null;
  replica_index: number | null; class_name: string | null; bbox: number[] | null; source: string | null;
  reason: string; created_at: string | null;
};
export type ClusterMember = {
  voter: string; object_id: string; class_name: string; bbox: number[]; human: boolean; agree: boolean;
  iou: number | null;
};
export type Cluster = {
  status: "accepted" | "disputed" | "dropped"; class_name: string; bbox: number[]; reason: string;
  agree_n: number; n_voters: number; consensus_score: number; class_votes: Record<string, number>;
  members: ClusterMember[]; object_id?: string;
};
export type MergePreview = {
  replica_group: string; task_id: string; voters: string[]; params: MergeParams; dry_run: boolean;
  counts: Record<string, number>; per_class: Record<string, ClassTally>;
  frames: { frame_id: string; clusters: Cluster[] }[];
};

const mergeQs = (p: MergeParams) =>
  `iou_thresh=${p.iou_thresh}&quorum=${p.quorum}&include_model=${p.include_model}&require_human=${p.require_human}`;

export const listConsensusTasks = () => apiGet<{ tasks: ConsensusTask[] }>("/api/tools/quality/consensus/tasks");
export const getConsensusBoard = (taskId: string, p: MergeParams) =>
  apiGet<ConsensusBoard>(`/api/tools/quality/consensus/tasks/${taskId}?${mergeQs(p)}`);
export const listDisagreements = (taskId: string, status: "open" | "resolved" = "open") =>
  apiGet<{ items: Disagreement[] }>(`/api/tools/quality/consensus/tasks/${taskId}/disagreements?status=${status}`);
export const previewMerge = (group: string, p: MergeParams) =>
  apiPost<MergePreview>(`/api/tools/quality/consensus/groups/${group}/preview`, p);
export const runMerge = (target: { task_id?: string; replica_group?: string }, p: MergeParams) =>
  apiPost<JobRef>("/api/tools/quality/consensus/merge", { ...target, ...p });

// ---- #12 explained errors -------------------------------------------------------------------------------

export type EvidenceBox = { bbox: number[]; role: "subject" | "overlaps" | "contains" | string; label: string; object_id: string | null };
export type Neighbour = {
  object_id: string; frame_id: string; class_name: string; bbox: number[]; similarity: number | null;
  role: string; crop_url: string;
};
export type ErrorWhy = {
  candidate_id: string; object_id: string; kind: string; detector: string; rule: string | null;
  score: number; status: string; proposed: { class_name: string; class_id: number | null } | null;
  subject: { object_id: string; frame_id: string; class_name: string; bbox: number[]; conf: number; state: string; source: string };
  frame: { frame_id: string; width: number | null; height: number | null; image_url: string };
  reason: string; facts: { label: string; value: string }[];
  evidence: EvidenceBox[]; neighbours: Neighbour[];
  twin_frame: {
    frame_id: string; similarity: number | null; width: number | null; height: number | null; image_url: string;
    objects: { bbox: number[]; class_name: string; object_id: string }[];
  } | null;
};
export type FrameWhy = { frame_id: string; width: number | null; height: number | null; items: ErrorWhy[] };

export const getErrorWhy = (candidateId: string) => apiGet<ErrorWhy>(`/api/tools/quality/errors/${candidateId}/why`);
export const getFrameWhy = (frameId: string, status = "pending") =>
  apiGet<FrameWhy>(`/api/tools/quality/frames/${frameId}/why?status=${status}`);

// ---- #13 VLM reason per label ---------------------------------------------------------------------------

export type VlmCheck = {
  object_id: string; verdict: "correct" | "incorrect" | "unsure"; reason: string; alternative: string | null;
  given_class: string | null; stale: boolean; provider: string; model: string; latency_ms: number | null;
  cost_usd: number | null; checked_at: string | null; checked_by: string | null;
};
export type VlmCheckResult = {
  requested: number; found: number; checked: number; failed: number; errors: string[];
  results: Record<string, VlmCheck>; model: string;
};
export const MAX_SYNC_CHECKS = 5;

export const checkLabels = (objectIds: string[]) =>
  apiPost<VlmCheckResult>("/api/tools/quality/vlm/check", { object_ids: objectIds.slice(0, MAX_SYNC_CHECKS) });
export const checkLabelsBatch = (q: { object_ids?: string[]; session_id?: string; limit?: number }) =>
  apiPost<JobRef>("/api/tools/quality/vlm/check-batch", q);
export const getVlmChecks = (objectIds: string[]) =>
  apiGet<{ checks: Record<string, VlmCheck> }>(`/api/tools/quality/vlm/checks?ids=${objectIds.join(",")}`);

// ---- #14 rare scenes ------------------------------------------------------------------------------------

export type RareFrame = {
  frame_id: string; session_id: string | null; score: number | null; percentile?: number;
  match?: number; nn_frame_id: string | null; nn_sim: number | null;
  cluster?: number; cluster_size?: number; cluster_sessions?: number;
  scene: Record<string, unknown>; vehicle_id: string | null; city: string | null;
  n_objects: number; n_reviewed: number; image_url: string; why: string[];
};
export type RareRanking = {
  space: string; k: number; n_frames: number; n_sessions: number; excluded_not_real: number; not_embedded: number;
  median_nn_sim?: number | null; score_quantiles?: Record<string, number>; n_clusters?: number;
  smallest_clusters?: { cluster: number; size: number; sessions: number }[];
  frames: RareFrame[]; detail?: string; params: Record<string, unknown>;
  /** Shortlisted frames the pixel screen set aside as not a scene (blank, corrupt, a fill), with the reason. */
  suspect_frames?: { frame_id: string; session_id: string; score: number; reason: string; image_url: string }[];
  n_suspect?: number;
};
export type RareLatest = {
  job: (JobRef & { finished_at: string | null }) | null; result: RareRanking | null; running: JobRef | null;
};
export type RareSearch = { mode: "text" | "example"; space: string; query: string; results: RareFrame[]; detail?: string };

export const rankRareScenes = (q: { space?: "dino" | "siglip"; k?: number; top?: number; per_session?: number;
  per_cluster?: number; session_id?: string; city?: string }) =>
  apiPost<JobRef>("/api/tools/quality/rare-scenes/rank", q);
export const getRareLatest = (space?: string) =>
  apiGet<RareLatest>(`/api/tools/quality/rare-scenes/latest${space ? `?space=${space}` : ""}`);
export const searchRareScenes = (q: { text?: string; frame_id?: string; space?: "dino" | "siglip"; k?: number;
  rarest_first?: boolean; session_id?: string }) =>
  apiPost<RareSearch>("/api/tools/quality/rare-scenes/search", q);
export const createRareTask = (q: { project_id: string; name: string; frame_ids: string[]; jobs_of?: number }) =>
  apiPost<{ task_id: string; n_frames: number; n_jobs: number; name: string }>("/api/tools/quality/rare-scenes/task", q);
export const listProjects = () =>
  apiGet<{ projects: { project_id: string; name: string }[] }>("/api/labelops/projects?limit=500");

/** A task whose frames go to several annotators independently (existing POST /api/labelops/tasks). */
export const MAX_REPLICAS = 5;
export const createReplicatedTask = (q: { project_id: string; name: string; session_id?: string; replicas: number;
  jobs_of: number }) =>
  apiPost<{ task_id: string; n_frames: number; n_jobs: number; replicas: number }>("/api/labelops/tasks", q);
