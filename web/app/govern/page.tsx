"use client";

import { useCallback, useState } from "react";
import { api } from "@/lib/api";
import { invalidate, useQuery } from "@/lib/useQuery";
import type { AuditRow, GovState, RegistryRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { ConfBar } from "@/components/StateBadge";
import EmptyState from "@/components/ui/EmptyState";
import { Badge } from "@/components/ui/kit";
import { useConfirm } from "@/components/ConfirmProvider";
import { ago, humanize, num, plural, shortId } from "@/lib/format";

// M4.4 governance console: champion status, control-sample precision, drift, the audit log, and the kill
// switch. The one place a governor watches the unattended loop. Color is earned: green when healthy, warn
// when paused, block-red for the kill switch.

const NO_REGISTRY: RegistryRow[] = [];
const NO_AUDIT: AuditRow[] = [];

// The drift monitor names its signals by their storage keys. Said in words, with the raw text kept as a tip.
const DRIFT_NAMES: Record<string, string> = {
  label_distribution: "label mix",
  input_embedding: "image content",
  confidence: "model confidence",
};

function reasonInWords(reason: string): string {
  const m = /^drift breach:\s*(.*)$/i.exec(reason.trim());
  if (!m) return reason;
  const parts = m[1].split(",").map((kv) => {
    const [k, v] = kv.split("=").map((x) => x.trim());
    const n = Number(v);
    return `${DRIFT_NAMES[k] ?? humanize(k).toLowerCase()}${Number.isFinite(n) ? ` (${n.toFixed(2)})` : ""}`;
  });
  return `The incoming data has drifted: ${parts.join(" and ")}.`;
}

// Audit decisions are stored as codes; the log reads as what happened.
const DECISION_WORDS: Record<string, string> = {
  delete_object: "Deleted a label",
  erase_session: "Erased a session",
  kill: "Stopped automation",
  release: "Resumed automation",
};

function decisionTone(d: string): string {
  return d.includes("promote") && !d.includes("paused") ? "text-pass border-pass/40"
    : d.includes("reject") || d.includes("engage") || d.includes("kill") || d.includes("erase") ? "text-block border-block/40"
    : d.includes("pause") ? "text-warn border-warn/40"
    : "text-ink-2 border-line";
}

function auditDetail(a: AuditRow): React.ReactNode {
  const r = a.rationale ?? {};
  if (a.decision === "delete_object") {
    const frame = typeof r.frame_id === "string" ? r.frame_id : null;
    return (
      <>
        Label <span className="font-mono text-ink-3">{shortId(a.subject)}</span>
        {frame && <> on <a href={`/frame/${frame}`} className="link">frame {shortId(frame)}</a></>}
      </>
    );
  }
  if (a.decision === "erase_session") {
    const cert = (r.certificate ?? {}) as { frames?: number };
    return (
      <>
        Session <span className="font-mono text-ink-3">{shortId(a.subject)}</span>
        {typeof cert.frames === "number" && <>, {plural(cert.frames, "frame")}</>}
        {typeof r.reason === "string" && <span className="text-ink-3"> · {r.reason}</span>}
      </>
    );
  }
  return a.subject ? <span className="font-mono text-ink-3">{a.subject.length > 24 ? shortId(a.subject) : a.subject}</span> : null;
}

const metric = (v: unknown) => (typeof v === "number" ? v.toFixed(2) : "-");

export default function GovernPage() {
  // Four reads, cached separately so the console's gate panel and the autonomy page share them. Any one
  // failing is shown: this page holds the kill switch and the promotion gate, and a silent failure here
  // shows stale or absent state for controls somebody is about to act on, which is the worst place in the
  // app to be quiet.
  const stateQ = useQuery(["governState"], () => api.governState());
  const registryQ = useQuery(["governRegistry"], () => api.governRegistry());
  const precQ = useQuery(["governPrecision"], () => api.governPrecision());
  const auditQ = useQuery(["governAudit"], () => api.governAudit());
  const state: GovState | null = stateQ.data ?? null;
  const registry: RegistryRow[] = registryQ.data ?? NO_REGISTRY;
  const prec: { reviewed: number; incorrect: number; precision: number | null; pending?: number; interval?: { lo: number; hi: number; p: number | null } } | null = precQ.data ?? null;
  const audit: AuditRow[] = auditQ.data ?? NO_AUDIT;
  const [msg, setMsg] = useState<string | null>(null);
  const err = stateQ.loadState.error ?? registryQ.loadState.error ?? precQ.loadState.error ?? auditQ.loadState.error ?? null;

  const load = useCallback(
    // Everything a governance action can move, here and on the pages that cache the same reads.
    () => invalidate("governState", "governRegistry", "governPrecision", "governAudit", "autonomyState"),
    []);

  const confirm = useConfirm();

  const act = async (fn: () => Promise<unknown>, label: string) => {
    const r = (await fn()) as { paused?: boolean; breached?: string[]; status?: string };
    setMsg(`${label}: ${JSON.stringify(r).slice(0, 120)}`);
    await load();
  };

  const stopAll = async () => {
    const ok = await confirm({
      title: "Stop all automation?",
      body: "The unattended loop stops: nothing is auto-accepted and no model is promoted until someone resumes it.",
      confirmLabel: "Stop automation",
      danger: true,
    });
    if (ok) await act(() => api.governKill("manual kill switch"), "Automation stopped");
  };

  const champion = registry.find((r) => r.is_champion);
  const paused = state && (!state.loop_enabled || !state.auto_promote_enabled);
  const loading = stateQ.loading && !state;

  const primaryAction = (
    <div className="flex gap-2">
      <button onClick={() => act(() => api.governDriftScan(), "Drift scan finished")} className="btn"
        data-tip="Compare recent data with the training data and pause the loop if it has drifted">Check for drift</button>
      <button onClick={() => act(() => api.governTick(), "Controller step finished")} className="btn"
        data-tip="Run one step of the automation controller now instead of waiting for its schedule">Run controller step</button>
    </div>
  );

  return (
    <PageShell active="GOVERN" title="Model registry"
      subtitle="Which model is in charge, how well automation is doing, and every automated decision it made."
      primaryAction={primaryAction}>
      <div className="page-wrap space-y-4">
        {err != null && <LoadState error={err} onRetry={() => void load()} />}

        {/* State, with the emergency stop on its own, away from the routine buttons above. */}
        <section className={`panel p-4 flex flex-wrap items-center gap-x-6 gap-y-3 ${paused ? "border-warn/50" : ""}`}>
          <div className="flex items-center gap-2.5">
            <span className={`w-2.5 h-2.5 rounded-full ${loading ? "bg-ink-3" : state?.loop_enabled ? "bg-pass" : "bg-block"}`} aria-hidden />
            <span className="text-[13.5px] font-medium text-ink">
              {loading ? "Loading..." : state?.loop_enabled ? "Automation running" : "Automation stopped"}
            </span>
          </div>
          {state && (
            <div className="flex items-center gap-2 text-[12.5px]">
              <Badge state={state.auto_accept_enabled ? "pass" : "muted"}>Auto-accept {state.auto_accept_enabled ? "on" : "off"}</Badge>
              <Badge state={state.auto_promote_enabled ? "pass" : "muted"}>Auto-promote {state.auto_promote_enabled ? "on" : "off"}</Badge>
            </div>
          )}
          {state?.paused_reason && (
            <span className="text-[12.5px] text-warn" data-tip={state.paused_reason}>{reasonInWords(state.paused_reason)}</span>
          )}
          {msg && <span className="text-[12px] text-ink-3" data-tip={msg}>{msg.split(":")[0]}.</span>}
          <div className="ml-auto flex items-center gap-3">
            {state && (state.loop_enabled
              ? <>
                  <span className="text-[12px] text-ink-3 hidden xl:inline">Emergency stop for all automated labelling and promotion</span>
                  <button onClick={() => void stopAll()} className="btn btn-danger">Stop all automation</button>
                </>
              : <button onClick={() => act(() => api.governRelease(), "Automation resumed")} className="btn btn-primary">Resume automation</button>)}
          </div>
        </section>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* champion + control precision */}
          <section className="panel">
            <div className="panel-head"><h2 className="section-title">Champion model</h2></div>
            <div className="p-4 space-y-4">
              <div>
                <div className="text-[13px] text-ink font-medium break-all">{champion ? champion.model_version : "No champion yet"}</div>
                {champion && (
                  <div className="text-[12px] text-ink-3 mt-1 space-y-0.5">
                    <div>{champion.promoted_from ? <>Replaced <span className="font-mono">{champion.promoted_from}</span></> : "The first champion"}</div>
                    <div>Safe-mIoU {metric((champion.gold_metrics as Record<string, unknown>)?.safe_miou)}{champion.created_at ? ` · registered ${ago(champion.created_at)}` : ""}</div>
                  </div>
                )}
              </div>
              <div className="border-t hairline pt-3">
                <div className="label-caps">True auto-accept precision</div>
                <div className="mt-1.5 text-[20px] font-semibold text-ink">
                  {prec?.precision != null ? <ConfBar conf={prec.precision} /> : <span className="text-ink-3 text-[15px] font-normal">Not measured yet</span>}
                </div>
                <div className="text-[12px] text-ink-3 mt-1">
                  {prec ? (prec.reviewed ? `${num(prec.incorrect)} of ${num(prec.reviewed)} checked samples were wrong` : "No control samples have been checked yet") : ""}
                </div>
                {prec?.interval?.p != null && (
                  <div className="text-[12px] text-ink-3">95% interval {prec.interval.lo.toFixed(2)} to {prec.interval.hi.toFixed(2)}</div>
                )}
                {/* The queue that produces this number. Without a door to it, the measurement starves:
                    601 samples sat pending with the verdict route uncalled. */}
                {(prec?.pending ?? 0) > 0 && (
                  <a href="/review/rapid?control=1" className="btn btn-primary mt-3">
                    Check {num(prec!.pending ?? 0)} control samples &rarr;
                  </a>
                )}
              </div>
            </div>
          </section>

          {/* registry */}
          <section className="panel lg:col-span-2 min-w-0">
            <div className="panel-head">
              <h2 className="section-title">Registered models</h2>
              <span className="ml-auto text-[12px] text-ink-3 font-normal">{plural(registry.length, "model")}</span>
            </div>
            {registryQ.loading && !registryQ.data ? (
              <div className="p-3"><LoadState loading rows={5} /></div>
            ) : registry.length ? (
              <div className="overflow-x-auto max-h-[480px] overflow-y-auto">
                <table className="data-table">
                  <thead><tr><th>Version</th><th className="num">mAP</th><th className="num">Safe-mIoU</th><th>Role</th><th><span className="sr-only">Actions</span></th></tr></thead>
                  <tbody>
                    {registry.map((r) => {
                      const m = r.gold_metrics as Record<string, unknown>;
                      return (
                        <tr key={r.model_version}>
                          <td className="text-ink truncate max-w-[340px]" title={r.model_version}>{r.model_version}</td>
                          <td className="num">{metric(m?.map ?? m?.map50 ?? 0)}</td>
                          <td className="num">{metric(m?.safe_miou)}</td>
                          <td><Badge state={r.is_champion ? "pass" : "muted"}>{r.is_champion ? "Champion" : "Challenger"}</Badge></td>
                          <td className="text-right">{!r.is_champion && (
                            <button onClick={() => act(() => api.governPromote(r.model_version), "Evaluation finished")} className="btn"
                              data-tip="Run the promotion gate: it becomes champion only if it beats the current one">Evaluate</button>
                          )}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState compact icon="shield" title="No models registered">
                A model is registered here when a training run finishes.
              </EmptyState>
            )}
          </section>
        </div>

        {/* audit log */}
        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Audit log</h2>
            <span className="text-[12px] text-ink-3 font-normal">Every automated and administrative decision, newest first</span>
          </div>
          {auditQ.loading && !auditQ.data ? (
            <div className="p-3"><LoadState loading rows={6} /></div>
          ) : audit.length ? (
            <div className="max-h-[420px] overflow-auto">
              <table className="data-table">
                <thead><tr><th>When</th><th>Who</th><th>Decision</th><th>Detail</th></tr></thead>
                <tbody>
                  {audit.map((a) => (
                    <tr key={a.audit_id}>
                      <td className="whitespace-nowrap text-ink-3" title={a.created_at ?? undefined}>{ago(a.created_at)}</td>
                      <td className="whitespace-nowrap">{humanize(a.actor)}</td>
                      <td className="whitespace-nowrap">
                        {/* audit decisions are free-form multi-word strings; color by substring (StateBadge only
                            matches an exact vocabulary), so a kill-engage / reject still reads red at a glance. */}
                        <span className={`inline-flex text-[12px] border rounded px-1.5 py-0.5 ${decisionTone(a.decision)}`}>
                          {DECISION_WORDS[a.decision] ?? humanize(a.decision)}
                        </span>
                      </td>
                      <td className="text-ink-2">{auditDetail(a)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState compact icon="shield" title="No decisions recorded yet">
              Promotions, pauses, deletions and erasures are logged here as they happen.
            </EmptyState>
          )}
        </section>
      </div>
    </PageShell>
  );
}
