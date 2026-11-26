"use client";

import { useCallback, useEffect, useState } from "react";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Bar, Panel, Stat } from "@/components/ui/kit";
import { humanizeError } from "@/lib/api";
import { ago, humanize, num } from "@/lib/format";
import { toast } from "@/lib/toast";
import { useCurrentUser } from "@/lib/user";
import { useToolJob } from "@/lib/tools/toolJobs";
import {
  dryRunAutopilot, getAutopilot, getNight, killAutopilot, listNights, rearmAutopilot, runAutopilotNow,
  saveAutopilotSettings, validHHMM,
  type AutopilotNight, type AutopilotPlan, type AutopilotReport, type AutopilotSettings, type AutopilotStatus,
  type AutopilotSteps,
} from "@/lib/tools/agentsApi";

// The overnight autopilot (services/agent/autopilot.py): each night outside office hours it auto-labels the new
// drives, has the VLM and the error detectors check what the gate wanted to auto-accept, keeps only what the
// autonomy ladder allows, and sends everything else to the review queue. This page holds its settings, the kill
// switch, a dry run, and the morning reports.

const STEP_LABEL: Record<keyof AutopilotSteps, { label: string; desc: string }> = {
  autolabel: { label: "Auto-label new sessions", desc: "Sessions with frames and no labels yet, oldest first" },
  vlm_check: { label: "VLM check", desc: "A vision model looks at each label the gate wants to auto-accept" },
  error_sweep: { label: "Error sweep", desc: "The label-error detectors run over tonight's sessions" },
  auto_accept: { label: "Ladder-gated auto-accept", desc: "Off: every label goes to review" },
};

const STATUS_TONE: Record<string, "pass" | "warn" | "block" | "muted" | "info"> = {
  done: "pass", running: "info", planned: "warn", stopped: "warn", error: "block",
};

function rank(role: string | undefined): number {
  return role === "admin" ? 3 : role === "reviewer" ? 2 : role === "annotator" ? 1 : 0;
}

function SettingsPanel({ s, canEdit, onSaved }: { s: AutopilotSettings; canEdit: boolean; onSaved: () => void }) {
  const [draft, setDraft] = useState(s);
  const [busy, setBusy] = useState(false);
  useEffect(() => { setDraft(s); }, [s]);
  const dirty = JSON.stringify(draft) !== JSON.stringify(s);
  const startOk = validHHMM(draft.earliest_start_ist);
  const save = async () => {
    setBusy(true);
    try {
      await saveAutopilotSettings({
        enabled: draft.enabled, gpu_minutes: draft.gpu_minutes, max_sessions: draft.max_sessions,
        vlm_max_checks: draft.vlm_max_checks, earliest_start_ist: draft.earliest_start_ist, steps: draft.steps,
      });
      toast("Autopilot settings saved", "success");
      onSaved();
    } catch (e) { toast(humanizeError(e), "error"); }
    finally { setBusy(false); }
  };
  const numField = (key: "gpu_minutes" | "max_sessions" | "vlm_max_checks", label: string, hint: string) => (
    <label className="flex items-center justify-between gap-3 py-1.5">
      <span className="min-w-0">
        <span className="block text-[12.5px] text-ink">{label}</span>
        <span className="block text-[11px] text-ink-3">{hint}</span>
      </span>
      <input type="number" className="field w-24 text-right tabular-nums" value={draft[key]} disabled={!canEdit}
             onChange={(e) => setDraft({ ...draft, [key]: Number(e.target.value) })} aria-label={label} />
    </label>
  );
  return (
    <Panel title="Settings" desc={canEdit ? "What the autopilot may do at night" : "Only an admin can change these"}>
      <div className="flex flex-col divide-y divide-line/60">
        <label className="flex items-center justify-between gap-3 py-1.5 cursor-pointer">
          <span>
            <span className="block text-[12.5px] text-ink">Run every night</span>
            <span className="block text-[11px] text-ink-3">Off means it only runs when somebody presses Run tonight now</span>
          </span>
          <input type="checkbox" checked={draft.enabled} disabled={!canEdit}
                 onChange={(e) => setDraft({ ...draft, enabled: e.target.checked })} aria-label="Run every night" />
        </label>
        {numField("gpu_minutes", "GPU minutes a night", "It stops starting work once this is spent")}
        {numField("max_sessions", "Sessions a night", "The rest wait for the next night")}
        {numField("vlm_max_checks", "VLM checks a night", "Labels it cannot check stay in review")}
        <label className="flex items-center justify-between gap-3 py-1.5">
          <span>
            <span className="block text-[12.5px] text-ink">Start after (IST)</span>
            <span className="block text-[11px] text-ink-3">Never inside office hours, whatever this says</span>
          </span>
          <input className={`field w-24 text-right tabular-nums ${startOk ? "" : "border-block"}`} value={draft.earliest_start_ist}
                 disabled={!canEdit} onChange={(e) => setDraft({ ...draft, earliest_start_ist: e.target.value })}
                 aria-label="Start after" placeholder="20:00" />
        </label>
        <div className="py-2">
          <div className="label-caps mb-1.5">Steps</div>
          {(Object.keys(STEP_LABEL) as (keyof AutopilotSteps)[]).map((k) => (
            <label key={k} className="flex items-start gap-2 py-1 cursor-pointer">
              <input type="checkbox" className="mt-0.5" checked={draft.steps[k]} disabled={!canEdit}
                     onChange={(e) => setDraft({ ...draft, steps: { ...draft.steps, [k]: e.target.checked } })} />
              <span>
                <span className="block text-[12.5px] text-ink">{STEP_LABEL[k].label}</span>
                <span className="block text-[11px] text-ink-3">{STEP_LABEL[k].desc}</span>
              </span>
            </label>
          ))}
        </div>
      </div>
      {canEdit && (
        <div className="flex gap-2 pt-2">
          <button className="btn btn-primary" disabled={!dirty || busy || !startOk} onClick={() => void save()}>Save</button>
          {dirty && <button className="btn" onClick={() => setDraft(s)} disabled={busy}>Discard</button>}
        </div>
      )}
    </Panel>
  );
}

function ActiveNight({ night }: { night: AutopilotNight }) {
  const { job } = useToolJob(night.tool_job_id);
  return (
    <div className="panel px-4 py-3 flex flex-col gap-2" role="status">
      <div className="flex items-center gap-2 text-[13px]">
        <Badge state="info">{humanize(night.status)}</Badge>
        <span className="text-ink">Tonight&apos;s run ({night.night_date})</span>
        <span className="text-ink-3 text-[12px] ml-auto">{night.trigger === "manual" ? "started by hand" : "started by the scheduler"}</span>
      </div>
      {job && (
        <div className="flex items-center gap-3 text-[12px] text-ink-2">
          <Bar frac={job.progress} state="accent" />
          <span className="tabular-nums w-10 text-right">{Math.round(job.progress * 100)}%</span>
          <span className="truncate max-w-[50%]">{job.quiet_hours ? "waiting for office hours to end" : job.stage ?? humanize(job.status)}</span>
        </div>
      )}
    </div>
  );
}

function PlanView({ plan }: { plan: AutopilotPlan }) {
  const s = plan.sessions;
  return (
    <div className="flex flex-col gap-3">
      {plan.blockers.length > 0 && (
        <div className="text-[12.5px] border border-warn/40 bg-warn/5 rounded px-3 py-2">
          <div className="text-warn font-medium mb-0.5">It would not start right now</div>
          <ul className="list-disc pl-5 text-ink-2">{plan.blockers.map((b) => <li key={b}>{b}</li>)}</ul>
        </div>
      )}
      <ol className="list-decimal pl-5 text-[12.5px] text-ink-2 flex flex-col gap-1">
        {plan.actions.map((a) => <li key={a}>{a}</li>)}
      </ol>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat label="Would label" value={num(s.would_label.length)} sub={`${num(s.would_label.reduce((a, b) => a + b.frames, 0))} frames`} />
        <Stat label="Left for later" value={num(s.over_budget.length + s.over_count.length)} sub={`${s.gated.length} held by health checks`} />
        <Stat label="GPU estimate" value={`${plan.estimate.gpu_minutes} min`} sub={plan.estimate.vlm_minutes ? `${plan.estimate.autolabel_minutes} labelling + up to ${plan.estimate.vlm_minutes} VLM · budget ${plan.estimate.budget_minutes}` : `budget ${plan.estimate.budget_minutes} min`}
              tone={plan.estimate.gpu_minutes > plan.estimate.budget_minutes ? "warn" : undefined} />
        <Stat label="Ladder allows" value={num(plan.ladder.allowed.length)} sub={`${plan.ladder.propose_only} classes propose-only`} />
      </div>
      <p className="text-[11.5px] text-ink-3">
        {plan.estimate.sec_per_frame} s a frame ({plan.estimate.sec_per_frame_basis}); {plan.estimate.sec_per_vlm_check} s a VLM check ({plan.estimate.sec_per_vlm_basis}).
      </p>
      {s.would_label.length > 0 && (
        <div className="overflow-x-auto">
          <table className="data-table w-full">
            <thead><tr><th>Session</th><th>City</th><th className="text-right">Frames</th><th className="text-right">Estimate</th></tr></thead>
            <tbody>
              {s.would_label.slice(0, 25).map((r) => (
                <tr key={r.session_id}>
                  <td><a className="link" href={`/?session=${r.session_id}`}>{r.vehicle ?? r.session_id.slice(0, 8)}</a></td>
                  <td>{r.city ?? "-"}</td>
                  <td className="text-right tabular-nums">{num(r.frames)}</td>
                  <td className="text-right tabular-nums">{r.est_seconds != null ? `${Math.max(1, Math.round(r.est_seconds / 60))} min` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {s.would_label.length > 25 && <p className="text-[11.5px] text-ink-3 mt-1">and {s.would_label.length - 25} more</p>}
        </div>
      )}
      {plan.ladder.allowed.length > 0 && (
        <div>
          <div className="label-caps mb-1">Classes it may auto-accept</div>
          <div className="flex flex-wrap gap-1.5">
            {plan.ladder.allowed.map((c) => (
              <span key={c.class_name} className="chip" title={c.threshold != null ? `threshold ${c.threshold}` : "no measured threshold"}>
                {humanize(c.class_name)} · L{c.level}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function CountList({ title, counts, empty }: { title: string; counts: Record<string, number> | undefined; empty: string }) {
  const rows = Object.entries(counts ?? {});
  return (
    <div className="min-w-0">
      <div className="label-caps mb-1">{title}</div>
      {rows.length === 0 ? <p className="text-[12px] text-ink-3">{empty}</p> : (
        <ul className="text-[12.5px] flex flex-col gap-0.5">
          {rows.slice(0, 12).map(([k, v]) => (
            <li key={k} className="flex justify-between gap-3"><span className="text-ink-2 truncate">{humanize(k)}</span><span className="tabular-nums text-ink">{num(v)}</span></li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ReportView({ night }: { night: AutopilotNight }) {
  const r: AutopilotReport = night.report ?? {};
  if (night.dry_run) {
    return night.plan ? <PlanView plan={night.plan} /> : <p className="text-[12px] text-ink-3">This dry run kept no plan.</p>;
  }
  if (night.status === "planned" || night.status === "running") {
    return <p className="text-[12.5px] text-ink-3">The report is written when the night finishes.</p>;
  }
  return (
    <div className="flex flex-col gap-3">
      {r.notes && r.notes.length > 0 && (
        <ul className="text-[12.5px] text-ink-2 flex flex-col gap-1 list-disc pl-5">{r.notes.map((n) => <li key={n}>{n}</li>)}</ul>
      )}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        <Stat label="Sessions labelled" value={num(r.labelled_sessions ?? 0)} sub={`${num(r.labels_created ?? 0)} labels`} />
        <Stat label="Auto-accepted" value={num(r.auto_accepted?.total ?? 0)} tone="pass" sub={`of ${num(r.proposed_auto_accepts ?? 0)} proposed`} />
        <Stat label="Sent to review" value={num(r.sent_to_review?.total ?? 0)} tone="warn" sub={`${num(r.sent_to_review?.from_auto_accept ?? 0)} were proposed auto-accepts`} />
        <Stat label="GPU used" value={`${r.gpu_minutes ?? night.gpu_minutes} min`} sub={`budget ${r.budget_minutes ?? "-"} min`} />
      </div>
      <div className="grid gap-4 md:grid-cols-3">
        <CountList title="Auto-accepted, by class" counts={r.auto_accepted?.by_class} empty="Nothing was auto-accepted." />
        <CountList title="Sent to review, why" counts={r.sent_to_review?.reasons} empty="No proposed auto-accept was turned back." />
        <CountList title="Waiting for review, by class" counts={r.sent_to_review?.by_class} empty="Nothing new waits for review." />
      </div>
      <div className="grid gap-4 md:grid-cols-2 text-[12.5px]">
        <div>
          <div className="label-caps mb-1">VLM check</div>
          {r.vlm?.unavailable ? <p className="text-warn">{r.vlm.unavailable}</p>
            : <p className="text-ink-2">{num(r.vlm?.checked ?? 0)} checked, {num(r.vlm?.disagreed ?? 0)} disagreed{r.vlm?.errors ? `, ${r.vlm.errors} errors` : ""}</p>}
        </div>
        <div>
          <div className="label-caps mb-1">Error sweep</div>
          <p className="text-ink-2">{num(r.error_sweep?.candidates ?? 0)} likely errors filed
            {r.error_sweep && Object.keys(r.error_sweep.by_kind).length > 0 ? ` (${Object.entries(r.error_sweep.by_kind).map(([k, v]) => `${v} ${humanize(k)}`).join(", ")})` : ""}
            {r.error_sweep?.candidates ? <> · <a className="link" href="/quality">open the queue</a></> : null}</p>
        </div>
      </div>
      {r.sessions && r.sessions.length > 0 && (
        <div className="overflow-x-auto">
          <table className="data-table w-full">
            <thead><tr><th>Session</th><th>City</th><th className="text-right">Frames</th><th className="text-right">Labels</th><th>Result</th><th className="text-right">Time</th></tr></thead>
            <tbody>
              {r.sessions.map((s) => (
                <tr key={s.session_id}>
                  <td><a className="link" href={`/?session=${s.session_id}`}>{s.vehicle ?? s.session_id.slice(0, 8)}</a></td>
                  <td>{s.city ?? "-"}</td>
                  <td className="text-right tabular-nums">{num(s.frames)}</td>
                  <td className="text-right tabular-nums">{s.objects != null ? num(s.objects) : "-"}</td>
                  <td className={s.status === "done" ? "text-pass" : s.status === "error" ? "text-block" : "text-ink-3"} title={s.error ?? s.detail ?? undefined}>
                    {s.status === "left" ? "left for later" : humanize(s.status)}
                  </td>
                  <td className="text-right tabular-nums">{s.seconds != null ? `${Math.round(s.seconds)} s` : "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {night.agent_run_id && (
        <p className="text-[11.5px] text-ink-3">Every label it sent to review is one agent run ({night.agent_run_id.slice(0, 8)}); it can be undone from <a className="link" href="/agent">Agents</a>.</p>
      )}
    </div>
  );
}

export default function Page() {
  const me = useCurrentUser();
  const [status, setStatus] = useState<AutopilotStatus | null>(null);
  const [nights, setNights] = useState<AutopilotNight[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [selected, setSelected] = useState<AutopilotNight | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const role = rank(me?.role);

  const load = useCallback(async () => {
    try {
      const [st, hist] = await Promise.all([getAutopilot(), listNights(30)]);
      setStatus(st); setNights(hist.nights); setError(null);
    } catch (e) { setError(e); }
  }, []);
  useEffect(() => { void load(); }, [load]);
  useEffect(() => {
    // Default view: the newest night in the history, real or dry run.
    if (!selected && nights.length) void getNight(nights[0].night_id).then(setSelected).catch(() => undefined);
  }, [nights, selected]);

  const act = async (name: string, fn: () => Promise<unknown>, ok: string) => {
    setBusy(name);
    try { await fn(); toast(ok, "success"); await load(); }
    catch (e) { toast(humanizeError(e), "error"); }
    finally { setBusy(null); }
  };
  const dryRun = () => act("dry", async () => { const n = await dryRunAutopilot(); setSelected(n); }, "Dry run done; nothing was changed");
  const kill = () => {
    const reason = window.prompt("Stop the autopilot now? It stays stopped until an admin re-arms it. Reason:", "stopped from the autopilot page");
    if (reason === null) return;
    void act("kill", () => killAutopilot(reason || "stopped from the autopilot page"), "Autopilot stopped");
  };

  const s = status?.settings;
  const w = status?.window;
  const last = status?.last;

  return (
    <PageShell active="Autopilot" title="Autopilot"
      subtitle="Labels new drives overnight and hands people only what it is unsure of"
      right={s && (
        <div className="flex items-center gap-2">
          {role >= 2 && <button className="btn" disabled={!!busy} onClick={() => void dryRun()}>{busy === "dry" ? "Planning..." : "Dry run"}</button>}
          {role >= 3 && <button className="btn btn-primary" disabled={!!busy || s.killed || !!status?.active}
            onClick={() => void act("run", runAutopilotNow, "Tonight's run is queued")}
            title={status?.active ? "A night is already queued or running" : undefined}>Run tonight now</button>}
          {role >= 2 && !s.killed && <button className="btn btn-danger" disabled={!!busy} onClick={kill}>Kill switch</button>}
        </div>
      )}>
      <div className="page-wrap flex flex-col gap-4">
        <LoadState loading={!status && !error} error={error} onRetry={() => void load()} rows={4} />
        {s && w && (
          <>
            {s.killed && (
              <div role="alert" className="panel border-block/50 bg-block/5 px-4 py-3 flex flex-wrap items-center gap-3 text-[13px]">
                <span className="text-block font-medium">Kill switch on</span>
                <span className="text-ink-2">{s.killed_reason ?? "no reason given"}{s.killed_at ? ` · ${ago(s.killed_at)}` : ""}</span>
                {role >= 3
                  ? <button className="btn ml-auto" disabled={!!busy} onClick={() => void act("rearm", rearmAutopilot, "Autopilot re-armed")}>Re-arm</button>
                  : <span className="ml-auto text-[12px] text-ink-3">An admin re-arms it</span>}
              </div>
            )}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Stat label="Autopilot" value={s.killed ? "Stopped" : s.enabled ? "On" : "Off"}
                    tone={s.killed ? "block" : s.enabled ? "pass" : undefined}
                    sub={s.enabled ? `every night after ${s.earliest_start_ist} IST` : "runs only when started by hand"} />
              <Stat label="Tonight's window" value={w.open ? "Open" : "Closed"} tone={w.open ? "pass" : undefined}
                    sub={w.open ? `until ${w.closes_at_ist} IST` : w.reason ?? ""} />
              <Stat label="Last night" value={last ? num(last.report?.auto_accepted?.total ?? 0) : "-"}
                    sub={last ? `auto-accepted · ${num(last.report?.sent_to_review?.total ?? 0)} to review · ${last.night_date}` : "no night has run yet"} />
              <Stat label="GPU budget" value={`${s.gpu_minutes} min`} sub={`up to ${s.max_sessions} sessions, ${s.vlm_max_checks} VLM checks`} />
            </div>
            {status?.active && <ActiveNight night={status.active} />}
            <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,360px)] items-start">
              <div className="flex flex-col gap-4 min-w-0">
                <Panel title={selected ? (selected.dry_run ? `Dry run, ${ago(selected.created_at)}` : `Morning report, night of ${selected.night_date}`) : "Morning report"}
                       desc={selected?.dry_run ? "What a night would do if it started at that moment. Nothing was changed." : "What the autopilot did, and what it left for people"}
                       right={selected && <Badge state={STATUS_TONE[selected.status] ?? "muted"}>{selected.dry_run ? "dry run" : humanize(selected.status)}</Badge>}>
                  {selected ? <ReportView night={selected} /> : (
                    <EmptyState compact icon="play" title="No nights yet"
                      action={role >= 2 ? <button className="btn" onClick={() => void dryRun()}>Dry run</button> : undefined}>
                      A dry run shows what tonight would do without changing anything.
                    </EmptyState>
                  )}
                </Panel>
                <Panel title="History" desc="Every night it ran, and every dry run">
                  {nights.length === 0 ? <p className="text-[12px] text-ink-3">Nothing yet.</p> : (
                    <div className="overflow-x-auto">
                      <table className="data-table w-full">
                        <thead><tr><th>Night</th><th>Kind</th><th>Status</th><th className="text-right">Labelled</th><th className="text-right">Accepted</th><th className="text-right">To review</th><th className="text-right">GPU</th></tr></thead>
                        <tbody>
                          {nights.map((n) => (
                            <tr key={n.night_id} className={`cursor-pointer ${selected?.night_id === n.night_id ? "bg-accent/10" : ""}`}
                                onClick={() => void getNight(n.night_id).then(setSelected).catch((e) => toast(humanizeError(e), "error"))}>
                              <td>{n.night_date}<span className="text-ink-3 text-[11px] ml-1.5">{ago(n.created_at)}</span></td>
                              <td>{n.dry_run ? "dry run" : n.trigger === "manual" ? "by hand" : "scheduled"}</td>
                              <td className={n.status === "error" ? "text-block" : n.status === "stopped" ? "text-warn" : ""} title={n.stopped_reason ?? undefined}>{humanize(n.status)}</td>
                              <td className="text-right tabular-nums">{n.dry_run ? `${num(n.summary?.would_label ?? 0)} planned` : num(n.summary?.labelled_sessions ?? 0)}</td>
                              <td className="text-right tabular-nums">{n.dry_run ? "-" : num(n.summary?.auto_accepted ?? 0)}</td>
                              <td className="text-right tabular-nums">{n.dry_run ? "-" : num(n.summary?.sent_to_review ?? 0)}</td>
                              <td className="text-right tabular-nums">{n.dry_run ? "-" : `${n.gpu_minutes} min`}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </Panel>
              </div>
              <div className="flex flex-col gap-4">
                <SettingsPanel s={s} canEdit={role >= 3} onSaved={() => void load()} />
                <Panel title="How it decides" desc="It proposes; the ladder and the checks dispose">
                  <ul className="text-[12px] text-ink-2 flex flex-col gap-1.5 list-disc pl-4">
                    <li>Nothing starts in office hours ({w.quiet_hours_ist} IST), and a running night stops when they begin.</li>
                    <li>A label stays auto-accepted only if its class is at level 1 or above on the <a className="link" href="/autonomy">autonomy ladder</a>, the VLM agreed, and no error detector flagged it.</li>
                    <li>Anything it could not check goes to review, not to accepted.</li>
                    <li>Human labels are never touched. Every change is one agent run that can be undone.</li>
                  </ul>
                </Panel>
              </div>
            </div>
          </>
        )}
      </div>
    </PageShell>
  );
}
