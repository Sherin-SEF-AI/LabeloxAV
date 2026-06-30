"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AuditRow, GovState, RegistryRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { StateBadge, ConfBar } from "@/components/StateBadge";

// M4.4 governance console: champion status, control-sample precision, drift, the audit log, and the kill
// switch. The one place a governor watches the unattended loop. Color is earned: green when healthy, warn
// when paused, block-red for the kill switch.

export default function GovernPage() {
  const [state, setState] = useState<GovState | null>(null);
  const [registry, setRegistry] = useState<RegistryRow[]>([]);
  const [prec, setPrec] = useState<{ reviewed: number; incorrect: number; precision: number | null } | null>(null);
  const [audit, setAudit] = useState<AuditRow[]>([]);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [s, r, p, a] = await Promise.all([api.governState(), api.governRegistry(), api.governPrecision(), api.governAudit()]);
    setState(s); setRegistry(r); setPrec(p); setAudit(a);
  }, []);

  useEffect(() => { load(); }, [load]);

  const act = async (fn: () => Promise<unknown>, label: string) => {
    const r = (await fn()) as { paused?: boolean; breached?: string[]; status?: string };
    setMsg(`${label}: ${JSON.stringify(r).slice(0, 120)}`);
    await load();
  };

  const champion = registry.find((r) => r.is_champion);
  const paused = state && (!state.loop_enabled || !state.auto_promote_enabled);

  const primaryAction = (
    <div className="flex gap-2 font-mono text-[11px]">
      <button onClick={() => act(() => api.governDriftScan(), "drift scan")} className="border border-line px-2 py-1 hover:border-accent">drift scan</button>
      <button onClick={() => act(() => api.governTick(), "controller tick")} className="border border-line px-2 py-1 hover:border-accent">controller tick</button>
      {state?.loop_enabled
        ? <button onClick={() => act(() => api.governKill("manual kill switch"), "kill")} className="border border-block text-block px-2 py-1 hover:bg-block/10">KILL SWITCH</button>
        : <button onClick={() => act(() => api.governRelease(), "release")} className="border border-pass text-pass px-2 py-1 hover:bg-pass/10">release</button>}
    </div>
  );

  return (
    <PageShell active="GOVERN" title="GOVERN" primaryAction={primaryAction}>
      <div className="p-4 space-y-4 font-mono text-[11px]">
        {/* state banner */}
        <div className={`panel p-3 flex items-center gap-3 ${paused ? "border-warn" : ""}`}>
          <span className={`w-2.5 h-2.5 rounded-full ${state?.loop_enabled ? "bg-pass" : "bg-block"}`} />
          <span className="text-ink-2">loop {state?.loop_enabled ? "ENABLED" : "PAUSED"}</span>
          <span className="text-ink-3">auto-accept {state?.auto_accept_enabled ? "on" : "off"}</span>
          <span className="text-ink-3">auto-promote {state?.auto_promote_enabled ? "on" : "off"}</span>
          {state?.paused_reason && <span className="text-warn">{state.paused_reason}</span>}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* champion + control precision */}
          <section className="panel p-3 space-y-2">
            <div className="uppercase text-[10px] text-ink-3">champion</div>
            <div className="text-ink-2 truncate">{champion ? champion.model_version : "none"}</div>
            {champion && <div className="text-ink-3">promoted from {champion.promoted_from || "(first)"}</div>}
            {champion && <div className="text-ink-3">safe-mIoU {String((champion.gold_metrics as Record<string, unknown>)?.safe_miou ?? "-")}</div>}
            <div className="uppercase text-[10px] text-ink-3 pt-2">true auto-accept precision</div>
            <div className="text-lg">
              {prec?.precision != null ? <ConfBar conf={prec.precision} /> : "n/a"}
            </div>
            <div className="text-ink-3">{prec ? `${prec.incorrect}/${prec.reviewed} controls incorrect` : ""}</div>
          </section>

          {/* registry */}
          <section className="panel lg:col-span-2">
            <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2">model registry</div>
            <table className="w-full text-[11px]">
              <thead><tr className="text-ink-3 text-left"><th className="px-3 py-1">version</th><th>mAP</th><th>safe-mIoU</th><th>role</th><th></th></tr></thead>
              <tbody>
                {registry.map((r) => {
                  const m = r.gold_metrics as Record<string, number>;
                  return (
                    <tr key={r.model_version} className="border-b hairline">
                      <td className="px-3 py-1 text-ink-2 truncate max-w-[160px]">{r.model_version}</td>
                      <td className="text-ink-3">{(m?.map ?? m?.map50 ?? 0).toFixed?.(2) ?? "-"}</td>
                      <td className="text-ink-3">{m?.safe_miou ?? "-"}</td>
                      <td><StateBadge state={r.is_champion ? "champion" : "challenger"} /></td>
                      <td className="text-right pr-3">{!r.is_champion && <button onClick={() => act(() => api.governPromote(r.model_version), "promote")} className="text-info hover:text-accent">evaluate</button>}</td>
                    </tr>
                  );
                })}
                {!registry.length && <tr><td colSpan={5} className="text-ink-3 text-center py-4">no models registered</td></tr>}
              </tbody>
            </table>
          </section>
        </div>

        {/* audit log */}
        <section className="panel">
          <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2">audit log (every automated decision)</div>
          <div className="p-2 space-y-0.5 max-h-80 overflow-auto">
            {audit.map((a) => (
              <div key={a.audit_id} className="flex items-center gap-2">
                <span className="text-ink-3 w-24 shrink-0">{a.actor}</span>
                {/* audit decisions are free-form multi-word strings; color by substring (StateBadge only
                    matches an exact vocabulary), so a kill-engage / reject still reads red at a glance. */}
                <span className={`font-mono text-[11px] uppercase tracking-wide border px-1.5 py-0.5 ${
                  a.decision.includes("promote") && !a.decision.includes("paused") ? "text-pass border-pass"
                    : a.decision.includes("reject") || a.decision.includes("engage") || a.decision.includes("kill") ? "text-block border-block"
                    : a.decision.includes("pause") ? "text-warn border-warn"
                    : "text-ink-2 border-line"}`}>{a.decision}</span>
                <span className="text-ink-3 truncate flex-1">{a.subject || ""}</span>
                <span className="text-ink-3 shrink-0">{a.created_at?.slice(11, 19)}</span>
              </div>
            ))}
            {!audit.length && <div className="text-ink-3 text-center py-4">no decisions yet</div>}
          </div>
        </section>
      </div>
    </PageShell>
  );
}
