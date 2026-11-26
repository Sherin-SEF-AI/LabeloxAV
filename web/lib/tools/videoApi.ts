// Client for the video annotation tools (server: services/api/routers/tools_video.py, paths /api/tools/video/...).
// Build calls on apiGet/apiPost/apiPut/apiDelete; start background work through a route that returns a ToolJob
// and follow it with useToolJob from ./toolJobs.

import { apiPost } from "@/lib/api";

/** What a tool route that queues work answers with; follow `job_id` with useToolJob. */
export type QueuedJob = { job_id: string; kind: string; status: string; session_id: string | null };

// ---- #1 SAM 2 video mask tracking ----

export type TrackPrompt = {
  frame_id: string;
  /** [[x, y], ...] in image pixels. */
  points?: number[][];
  /** 1 = this is the object, 0 = this is not, one per point. */
  labels?: number[];
  box?: number[] | null;
};

export type TrackRequest = {
  /** The frame on screen; the tracked window is centred on it. */
  frame_id: string;
  /** Track an existing label (its box is the prompt on its own frame). */
  object_id?: string;
  /** Re-prompt a track this tool made, with a corrective click on another frame. */
  track_id?: string;
  /** For a fresh click with no label behind it. */
  class_name?: string;
  prompts: TrackPrompt[];
  before?: number;
  after?: number;
};

/** The `result` of a finished `sam2_video` job. */
export type TrackResult = {
  method: "sam2_video" | "sam_propagate_fallback";
  track_id?: string;
  frames?: number;
  window_frame_ids?: string[];
  created: string[];
  updated: string[];
  removed: string[];
  kept: string[];
  empty_frame_ids: string[];
  fallback_reason?: string | null;
  note?: string;
  seconds?: number;
};

export const startTrack = (body: TrackRequest) => apiPost<QueuedJob>("/api/tools/video/track", body);

// ---- #2 text prompt to labels ----

export type DescribeProposal = { bbox: number[]; conf: number; polygons: number[][] };
export type DescribeResponse = {
  frame_id: string;
  phrase: string;
  class_id: number | null;
  proposals: DescribeProposal[];
  note: string;
};

/** Proposals for one frame from a phrase. Nothing is written. (The route predates this tool: objects.py.) */
export const describeFrame = (frameId: string, phrase: string, maxProposals?: number) =>
  apiPost<DescribeResponse>(`/api/frames/${frameId}/describe-label`,
    maxProposals ? { phrase, max_proposals: maxProposals } : { phrase });

export const acceptDescribed = (frameId: string, body: {
  phrase: string; class_name: string; proposals: { bbox: number[]; polygons: number[][]; conf: number }[];
}) => apiPost<{ created: string[]; state: string; class_name: string; phrase: string }>(
  `/api/tools/video/frames/${frameId}/describe/accept`, body);

export type DescribeSessionRequest = {
  session_id: string; phrase: string; class_name: string;
  min_conf?: number; cam_id?: string; max_frames?: number; dry_run?: boolean;
};

export const describeSession = (body: DescribeSessionRequest) =>
  apiPost<QueuedJob>("/api/tools/video/describe/session", body);

// ---- #4 correction fan-out ----

export type FanoutObject = {
  object_id: string;
  frame_id: string;
  class_name: string;
  state: string;
  source: string;
  conf: number | null;
  track_id: string | null;
  score: number | null;
  crop_url: string;
  human: boolean;
  already?: boolean;
  /** False for a person's label or one already at the new class: shown, never ticked by default. */
  selectable: boolean;
};

export type FanoutPreview = {
  object_id: string;
  from_class: string;
  to_class: string;
  track: { track_id: string | null; objects: FanoutObject[] };
  lookalikes: { objects: FanoutObject[]; reason: string | null; threshold: number };
  confusion: { objects: FanoutObject[]; rule: string };
};

export type FanoutApplied = {
  track: number;
  others: number;
  updated: number;
  run_id: string | null;
  skipped_human?: string[];
  skipped_stale?: { object_id: string }[];
  refused?: { object_id: string; reason: string }[];
  skipped_missing: string[];
  class_name: string;
};

export const fanoutPreview = (body: { object_id: string; from_class: string; to_class: string; threshold?: number;
                                      limit?: number }) =>
  apiPost<FanoutPreview>("/api/tools/video/fanout/preview", body);

export const fanoutApply = (body: { object_id: string; to_class: string; track_object_ids: string[];
                                    lookalike_object_ids: string[]; confusion_object_ids: string[] }) =>
  apiPost<FanoutApplied>("/api/tools/video/fanout/apply", body);

export const fanoutUndo = (runId: string) =>
  apiPost<{ run_id: string; reverted: number; skipped: number }>(`/api/tools/video/fanout/${runId}/undo`, {});
