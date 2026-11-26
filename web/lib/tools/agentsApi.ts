// Client for the agents annotation tools (server: services/api/routers/tools_agents.py, paths /api/tools/agents/...).
// #17 bulk edits in plain language (preview, propose, revert; approval goes through the assistant's approve route)
// and #18 the overnight autopilot (settings, kill switch, dry run, run now, history).

import { apiGet, apiPost, apiPut } from "@/lib/api";

// ---- #17 bulk edits ---------------------------------------------------------------------------------------------

export type EditSample = {
  object_id: string; frame_id: string; session_id: string; class_name: string; bbox: number[];
  state: string; conf: number; crop_url: string;
};
export type EditSession = {
  session_id: string; vehicle?: string | null; city?: string | null; route?: string | null; date?: string | null;
  classes: Record<string, number>; total: number;
};
export type LeftAlone = { human_confirmed: number; category_change: number; already_target: number };
export type EditPreview = {
  plan: { operation: string; raw: string; to_class_id: number | null; select_class_ids: number[] };
  interpretation: string[];
  count: number;
  sessions: number;
  by_class: Record<string, number>;
  by_session: EditSession[];
  left_alone: LeftAlone;
  left_alone_by_class: Record<string, Partial<LeftAlone>>;
  sample: EditSample[];
  warnings: string[];
  too_many: boolean;
  can_propose: boolean;
};
export type EditProposal = {
  proposal_id: string; tool: string; summary: string; status: "pending" | "approved" | "rejected" | "failed";
  request: string | null; created_at: string | null; decided_at: string | null;
  preview: Partial<EditPreview> & { count?: number; human_labels_left_alone?: number };
  result: { run_id?: string; edited?: number; error?: string; skipped?: Record<string, number> } | null;
  run: { run_id: string; status: string; edited: number; reverted_at: string | null } | null;
};

export const previewEdit = (command: string, sessionId?: string) =>
  apiPost<EditPreview>("/api/tools/agents/edit/preview", { command, session_id: sessionId || undefined });
export const proposeEdit = (command: string, sessionId?: string) =>
  apiPost<EditProposal>("/api/tools/agents/edit/propose", { command, session_id: sessionId || undefined });
export const getEditProposal = (id: string) => apiGet<EditProposal>(`/api/tools/agents/edit/proposals/${id}`);
export const listEditProposals = (limit = 20) =>
  apiGet<{ proposals: EditProposal[] }>(`/api/tools/agents/edit/proposals?limit=${limit}`);
export const approveEdit = (id: string) => apiPost<EditProposal>(`/api/assistant/proposals/${id}/approve`, {});
export const rejectEdit = (id: string) => apiPost<EditProposal>(`/api/assistant/proposals/${id}/reject`, {});
export const revertEdit = (id: string) =>
  apiPost<{ reverted: number; skipped: number; proposal: EditProposal }>(`/api/tools/agents/edit/proposals/${id}/revert`, {});

// ---- #18 autopilot ----------------------------------------------------------------------------------------------

export type AutopilotSteps = { autolabel: boolean; vlm_check: boolean; error_sweep: boolean; auto_accept: boolean };
export type AutopilotSettings = {
  enabled: boolean; killed: boolean; killed_reason: string | null; killed_at: string | null;
  gpu_minutes: number; max_sessions: number; vlm_max_checks: number; earliest_start_ist: string;
  steps: AutopilotSteps; updated_at: string | null;
};
export type AutopilotWindow = {
  open: boolean; quiet: boolean; quiet_hours_ist: string; earliest_start_ist: string; closes_at_ist: string;
  reason: string | null; next_open: string | null; night: string;
};
export type PlanSession = {
  session_id: string; vehicle: string | null; city: string | null; frames: number; est_seconds?: number; gated?: boolean;
};
export type AutopilotPlan = {
  at: string; night: string; window: AutopilotWindow; steps: AutopilotSteps;
  governance: { loop_enabled: boolean; auto_accept_enabled: boolean };
  would_start: boolean; blockers: string[]; actions: string[];
  sessions: { would_label: PlanSession[]; over_budget: PlanSession[]; over_count: PlanSession[]; gated: PlanSession[];
              candidates: number };
  estimate: { gpu_minutes: number; autolabel_minutes?: number; vlm_minutes?: number; budget_minutes: number; sec_per_frame: number; sec_per_frame_basis: string;
              sec_per_vlm_check: number; sec_per_vlm_basis: string };
  ladder: { allowed: { class_name: string; level: number; threshold: number | null }[]; propose_only: number;
            champion: string | null };
};
export type NightSessionRow = {
  session_id: string; vehicle?: string | null; city?: string | null; frames: number; status: string;
  objects?: number | null; seconds?: number; error?: string | null; detail?: string;
};
export type AutopilotReport = {
  dry_run?: boolean; actions?: string[];
  steps?: AutopilotSteps; sessions?: NightSessionRow[]; notes?: string[];
  labelled_sessions?: number; labels_created?: number; proposed_auto_accepts?: number; left_for_later?: number;
  vlm?: { checked: number; disagreed: number; errors: number; seconds: number; unavailable?: string };
  error_sweep?: { candidates: number; by_kind: Record<string, number>; errors: string[] };
  auto_accepted?: { total: number; by_class: Record<string, number> };
  sent_to_review?: { from_auto_accept: number; reasons: Record<string, number>; total: number; by_class: Record<string, number> };
  gpu_minutes?: number; budget_minutes?: number; stopped?: string | null;
};
export type AutopilotNight = {
  night_id: string; night_date: string; dry_run: boolean; status: string; trigger: string;
  tool_job_id: string | null; agent_run_id: string | null; gpu_minutes: number; stopped_reason: string | null;
  created_at: string | null; started_at: string | null; finished_at: string | null; notes: string[];
  plan?: AutopilotPlan; report?: AutopilotReport; settings?: AutopilotSettings;
  summary?: { labelled_sessions?: number | null; labels_created?: number | null; auto_accepted?: number | null;
              sent_to_review?: number | null; would_label?: number };
};
export type AutopilotStatus = {
  settings: AutopilotSettings; window: AutopilotWindow; active: AutopilotNight | null; last: AutopilotNight | null;
};

export const getAutopilot = () => apiGet<AutopilotStatus>("/api/tools/agents/autopilot");
export const saveAutopilotSettings = (s: Partial<Omit<AutopilotSettings, "killed" | "killed_reason" | "killed_at" | "updated_at">>) =>
  apiPut<AutopilotSettings>("/api/tools/agents/autopilot/settings", s);
export const killAutopilot = (reason: string) =>
  apiPost<{ killed: boolean; stopped_nights: string[] }>("/api/tools/agents/autopilot/kill", { reason });
export const rearmAutopilot = () => apiPost<AutopilotSettings>("/api/tools/agents/autopilot/rearm", {});
export const dryRunAutopilot = () => apiPost<AutopilotNight>("/api/tools/agents/autopilot/dry-run", {});
export const runAutopilotNow = () => apiPost<AutopilotNight & { starts: string }>("/api/tools/agents/autopilot/run-now", {});
export const listNights = (limit = 30) => apiGet<{ nights: AutopilotNight[] }>(`/api/tools/agents/autopilot/nights?limit=${limit}`);
export const getNight = (id: string) => apiGet<AutopilotNight>(`/api/tools/agents/autopilot/nights/${id}`);

// ---- pure helpers (tested in agentsApi.test.ts) -------------------------------------------------------------------

/** Rows of a per-session x per-class count table: the class columns in order of total, then one row per session. */
export function editTable(p: Pick<EditPreview, "by_class" | "by_session">, maxClasses = 6): {
  classes: string[]; rows: { session: EditSession; cells: number[]; other: number }[];
} {
  const classes = Object.entries(p.by_class).sort((a, b) => b[1] - a[1]).slice(0, maxClasses).map(([k]) => k);
  const rows = p.by_session.map((s) => {
    const cells = classes.map((c) => s.classes[c] ?? 0);
    return { session: s, cells, other: s.total - cells.reduce((a, b) => a + b, 0) };
  });
  return { classes, rows };
}

/** One sentence for what an edit leaves alone, or null when it touches everything it matched. */
export function leftAloneText(l: LeftAlone | undefined): string | null {
  if (!l) return null;
  const parts: string[] = [];
  if (l.human_confirmed) parts.push(`${l.human_confirmed.toLocaleString()} human-confirmed label${l.human_confirmed === 1 ? "" : "s"}`);
  if (l.category_change) parts.push(`${l.category_change.toLocaleString()} that would change what kind of thing it is`);
  if (l.already_target) parts.push(`${l.already_target.toLocaleString()} already that class`);
  return parts.length ? `Left alone: ${parts.join(", ")}.` : null;
}

/** The state an approval flow is in, for the badge beside a proposal. */
export function approvalState(p: Pick<EditProposal, "status" | "run">): { label: string; tone: "warn" | "pass" | "block" | "muted" } {
  if (p.status === "pending") return { label: "Waiting for a reviewer", tone: "warn" };
  if (p.status === "approved" && p.run?.status === "reverted") return { label: "Applied, then undone", tone: "muted" };
  if (p.status === "approved") return { label: "Approved and applied", tone: "pass" };
  if (p.status === "failed") return { label: "Could not apply", tone: "block" };
  return { label: "Rejected", tone: "muted" };
}

/** "21:00" style HH:MM validation for the start-time field. */
export function validHHMM(s: string): boolean {
  const m = /^(\d{1,2}):(\d{2})$/.exec(s.trim());
  return !!m && Number(m[1]) < 24 && Number(m[2]) < 60;
}
