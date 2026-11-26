// The annotation tools' background jobs: start one through a tool's own API module, then follow it here.
// Server side: services/tools/registry.py (queue) and services/api/routers/tool_jobs.py (these routes).

import { useEffect, useRef, useState } from "react";

import { apiGet, apiPost } from "@/lib/api";

export type ToolJob = {
  job_id: string;
  kind: string;
  label: string;
  status: "pending" | "running" | "done" | "error" | "canceled" | string;
  gpu: boolean;
  session_id: string | null;
  params: Record<string, unknown>;
  result: Record<string, unknown> | null;
  progress: number;
  stage: string | null;
  error: string | null;
  created_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  /** Waiting because auto-labelling is paused for office hours (10:00-18:00 IST). */
  quiet_hours: boolean;
};

export const TERMINAL = new Set(["done", "error", "canceled"]);

export const getToolJob = (id: string) => apiGet<ToolJob>(`/api/tools/jobs/${id}`);
export const listToolJobs = (q: { kind?: string; session_id?: string; limit?: number } = {}) => {
  const p = new URLSearchParams();
  if (q.kind) p.set("kind", q.kind);
  if (q.session_id) p.set("session_id", q.session_id);
  if (q.limit) p.set("limit", String(q.limit));
  const qs = p.toString();
  return apiGet<ToolJob[]>(`/api/tools/jobs${qs ? `?${qs}` : ""}`);
};
export const cancelToolJob = (id: string) => apiPost<ToolJob>(`/api/tools/jobs/${id}/cancel`, {});

/**
 * Follow a job until it finishes: polls every `everyMs` while it is pending or running, then stops.
 * `onDone` fires once with the finished row (done, error or canceled).
 */
export function useToolJob(jobId: string | null, { everyMs = 1500, onDone }: {
  everyMs?: number; onDone?: (job: ToolJob) => void;
} = {}): { job: ToolJob | null; error: unknown } {
  const [job, setJob] = useState<ToolJob | null>(null);
  const [error, setError] = useState<unknown>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;
  useEffect(() => {
    setJob(null); setError(null);
    if (!jobId) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const j = await getToolJob(jobId);
        if (stop) return;
        setJob(j);
        if (TERMINAL.has(j.status)) { doneRef.current?.(j); return; }
      } catch (e) {
        if (stop) return;
        setError(e);
      }
      timer = setTimeout(tick, everyMs);
    };
    void tick();
    return () => { stop = true; if (timer) clearTimeout(timer); };
  }, [jobId, everyMs]);
  return { job, error };
}
