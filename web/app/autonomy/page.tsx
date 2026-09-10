"use client";

import { useCallback, useEffect, useState } from "react";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import PulseDot from "@/components/PulseDot";
import { api, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import type {
  AutonomyState, SettlementLotRow, SettlementWorkItem, ShadowSummary,
} from "@/lib/types";

// The autonomy console: the machine's answer to "what are you allowed to do right now, and why".
//
// Everything here existed as scattered truth - switches on governance_state, the ladder in
// class_autonomy, staleness inside each measurement's rows, the night's story across agent runs - and
// reading it meant knowing where each piece lived. The whole measurement stack had zero web callers
// before this page.
//
// Two rules of the design. Evidence next to permission: every rung shows the basis that earned it, and
// a rung with no evidence says so instead of looking equal to one with a passed lot. And staleness is a
// first-class fact: a measurement's age is printed beside it, because a six-month-old precision is a
// different thing from yesterday's wearing the same digits.

const LEVEL_LABEL: Record<number, string> = {
  0: "propose only", 1: "auto-accept band", 2: "settlement",
};

// The log-likelihood ratio between its two bounds: left is accept, right is reject, the middle is
// "keep asking". Drawn rather than printed because a person reading -1.7 has to remember which way
// is good; a marker two thirds of the way to the accept edge does not need remembering.
function LlrBar({ llr, lo, hi }: { llr: number | null; lo: number | null; hi: number | null }) {
  if (llr == null || lo == null || hi == null || hi <= lo) {
    return <span className="text-ink-3">not computed</span>;
  }
  const pct = Math.max(0, Math.min(100, ((llr - lo) / (hi - lo)) * 100));
  return (
    <span className="inline-flex items-center gap-1" title={`llr ${llr.toFixed(2)} in [${lo.toFixed(2)}, ${hi.toFixed(2)}]`}>
      <span className="text-[9px] text-pass">accept</span>
      <span className="relative inline-block h-[6px] w-[72px] rounded-sm bg-bg-2 border hairline">
        <span className="absolute top-[-2px] h-[10px] w-[2px] bg-ink" style={{ left: `calc(${pct}% - 1px)` }} />
      </span>
      <span className="text-[9px] text-block">reject</span>
    </span>
  );
}

function Worklist({ items, minutes }: { items: SettlementWorkItem[]; minutes: number }) {
  return (
    <section className="panel">
      <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2 flex justify-between">
        <span>verdict worklist · where the next minute goes</span>
        <span>{minutes} min open</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[11px]">
          <thead><tr className="text-ink-3 text-left">
            <th className="px-3 py-1">class</th><th>evidence</th><th>llr</th>
            <th>still to judge</th><th>objects / min</th><th></th>
          </tr></thead>
          <tbody>
            {items.map((w) => (
              <tr key={w.lot_id} className="border-t hairline">
                <td className="px-3 py-1 text-ink-2">{w.class_name}</td>
                <td className="text-ink-3">
                  {w.sample_n ? `${w.defects}/${w.sample_n}` : "0 verdicts"} · {w.drawn}
                  {w.rule === "sprt" && w.cap_n ? ` of ${w.cap_n}` : ""} drawn
                </td>
                <td><LlrBar llr={w.llr} lo={w.bound_accept} hi={w.bound_reject} /></td>
                <td className="text-ink-3">~{w.remaining_verdicts} verdicts · {w.minutes} min</td>
                <td className="text-ink-2 tabular-nums" title={w.oc != null ? `P(accept) ${w.oc}` : undefined}>
                  {w.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </td>
                <td className="px-2">
                  {w.review_at && <a className="text-accent hover:underline" href={w.review_at}>judge</a>}
                </td>
              </tr>
            ))}
            {items.length === 0 && (
              <tr><td className="px-3 py-2 text-ink-3" colSpan={6}>
                no lot is asking for verdicts
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const KIND_LABEL: Record<string, string> = {
  champion_miss: "champion missed",
  challenger_miss: "challenger missed",
  class_flip: "read as another class",
  conf_gap: "same box, different confidence",
};

function ShadowPanel({ shadow }: { shadow: ShadowSummary | undefined }) {
  const kinds = Object.entries(shadow?.disagreements_by_kind ?? {}).sort((a, b) => b[1] - a[1]);
  const states = shadow?.disagreements_by_state ?? {};
  const open = (states.pending ?? 0) + (states.queued ?? 0);
  const last = shadow?.sweeps?.[0];
  return (
    <section className="panel">
      <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2 flex justify-between">
        <span>shadow mode · where two models read the same pixels differently</span>
        <span>{open} open</span>
      </div>
      {!shadow?.sweeps?.length ? (
        <div className="px-3 py-2 text-[11px] text-ink-3">
          no sweep has run. gold is frozen, so nothing yet compares the models on the sessions
          ingested since it was sealed.
        </div>
      ) : (
        <div className="px-3 py-2 text-[11px] space-y-2">
          <div className="text-ink-3">
            last sweep {age(last?.created_at)} · {String(last?.status ?? "")}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {kinds.map(([k, n]) => (
              <span key={k} className="text-ink-2">
                {KIND_LABEL[k] ?? k} <span className="tabular-nums text-ink-3">{n}</span>
              </span>
            ))}
            {kinds.length === 0 && <span className="text-ink-3">no disagreement recorded</span>}
          </div>
          <div className="space-y-1">
            {(shadow?.win_shares ?? []).map((w) => (
              <div key={w.challenger_run_id} className="flex justify-between gap-3">
                <span className="text-ink-3 truncate">{w.challenger_run_id.slice(0, 8)}</span>
                {w.measured ? (
                  <span className="text-ink-2 tabular-nums">
                    challenger won {w.challenger_right}/{w.discordant}
                    {w.share != null && ` · ${(w.share * 100).toFixed(0)}%`}
                    {w.lo != null && w.hi != null &&
                      ` [${(w.lo * 100).toFixed(0)}, ${(w.hi * 100).toFixed(0)}]`}
                  </span>
                ) : (
                  // Unmeasured, said out loud. A win share of zero and nobody having looked are
                  // different facts and this panel refuses to draw them the same way.
                  <span className="text-ink-3">unmeasured · {w.reason ?? "nobody has ruled yet"}</span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function age(iso: string | null | undefined): string {
  if (!iso) return "never";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export default function AutonomyPage() {
  const [state, setState] = useState<AutonomyState | null>(null);
  const [lots, setLots] = useState<SettlementLotRow[]>([]);
  const [err, setErr] = useState<unknown>(null);
  const [showAllClasses, setShowAllClasses] = useState(false);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const [s, l] = await Promise.all([api.autonomyState(), api.settlementLots()]);
      setState(s); setLots(l);
    } catch (e) {
      setErr(e);
    }
  }, []);

  useEffect(() => {
    load();
    // The daemon dot is a liveness claim, so it re-checks itself; a dot that only reflects page load
    // would go on saying "live" for as long as the tab stayed open.
    const t = setInterval(load, 60_000);
    return () => clearInterval(t);
  }, [load]);

  const act = async (fn: () => Promise<unknown>, label: string) => {
    try {
      await fn();
      toast(`${label}: done`);
      await load();
    } catch (e) {
      toast(`${label}: ${humanizeError(e)}`);
    }
  };

  const sw = state?.switches;
  const daemon = state?.daemon;
  const ladder = state?.ladder ?? [];
  const raised = ladder.filter((r) => r.level > 0 || r.explicit);
  const shown = showAllClasses ? ladder : raised;
  const meas = state?.measurements;
  const settle = state?.settlement;
  const acks = lots.filter((l) => l.status === "accepted" && !l.decision?.human_ack);

  return (
    <PageShell active="AUTONOMY" title="AUTONOMY"
      subtitle="what the machine may do right now, per class, and the evidence behind it">
      <div className="p-4 space-y-4 font-mono text-[11px]">
        {err != null && <LoadState error={err} onRetry={() => void load()} />}

        {/* the switches and the heartbeat */}
        <div className="panel p-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="flex items-center gap-1.5">
            <PulseDot tone={daemon ? (daemon.alive ? "live" : "bad") : "idle"}
              label={daemon?.alive
                ? `daemon live, last tick ${age(daemon.last_tick_at)}`
                : `daemon STALE - last tick ${age(daemon?.last_tick_at)}`} />
            <span className={daemon?.alive ? "text-ink-2" : "text-block"}>
              daemon {daemon ? (daemon.alive ? "live" : `stale (${age(daemon.last_tick_at)})`) : "…"}
            </span>
          </span>
          <span className="text-ink-3">loop {sw?.loop_enabled ? "on" : "OFF"}</span>
          <span className="text-ink-3">auto-accept {sw?.auto_accept_enabled ? "on" : "off"}</span>
          <span className="text-ink-3">
            promote {sw?.auto_promote_enabled ? "autonomous" : "by approval"}
          </span>
          <span className={sw?.settlement_enabled ? "text-pass" : "text-ink-3"}>
            settlement {sw?.settlement_enabled ? "ARMED" : "off"}
          </span>
          {sw?.paused_reason && <span className="text-warn">{sw.paused_reason}</span>}
        </div>

        {/* one-click acks the agent is waiting on */}
        {acks.length > 0 && (
          <div className="panel p-3 border-warn space-y-1">
            <div className="uppercase text-[10px] text-warn">awaiting a person</div>
            {acks.map((l) => (
              <div key={l.lot_id} className="flex items-center gap-2">
                <span className="text-ink-2">
                  {l.class_name} lot passed (far {l.far_bound}, {l.sample_n} verdicts,
                  {" "}{l.defects} defects)
                </span>
                {l.tier !== "default" && <span className="text-warn">{l.tier}</span>}
                <button className="border border-line px-2 py-0.5 hover:border-accent"
                  onClick={() => act(() => api.settlementAck(l.lot_id), `ack ${l.class_name}`)}>
                  ack settle
                </button>
              </div>
            ))}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* the ladder */}
          <section className="panel lg:col-span-2">
            <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2 flex justify-between">
              <span>per-class ladder</span>
              <button className="text-ink-3 hover:text-ink-1"
                onClick={() => setShowAllClasses((v) => !v)}>
                {showAllClasses ? "raised only" : `all ${ladder.length} classes`}
              </button>
            </div>
            <div className="overflow-x-auto max-h-[420px] overflow-y-auto">
              <table className="w-full text-[11px]">
                <thead><tr className="text-ink-3 text-left">
                  <th className="px-3 py-1">class</th><th>level</th><th>set by</th><th>basis</th>
                </tr></thead>
                <tbody>
                  {shown.map((r) => (
                    <tr key={r.class_name} className="border-t hairline align-top">
                      <td className="px-3 py-1 text-ink-2">
                        {r.class_name}{r.pinned ? " 📌" : ""}
                      </td>
                      <td className={r.level === 2 ? "text-pass" : r.level === 1 ? "text-ink-2" : "text-ink-3"}>
                        L{r.level} {LEVEL_LABEL[r.level]}
                      </td>
                      <td className="text-ink-3">{r.set_by}</td>
                      <td className="text-ink-3 max-w-[26rem]">
                        {String(r.basis?.reason ?? r.basis?.step_down_reason ?? "")}
                        {r.cooldown_until && <span className="text-warn"> · cooldown until {r.cooldown_until.slice(0, 10)}</span>}
                      </td>
                    </tr>
                  ))}
                  {shown.length === 0 && (
                    <tr><td className="px-3 py-2 text-ink-3" colSpan={4}>
                      no class is above propose-only; a rung is earned by a measured threshold (L1)
                      and a passed lot (L2)
                    </td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          {/* measurements + their age */}
          <section className="panel p-3 space-y-2">
            <div className="uppercase text-[10px] text-ink-3">the evidence, and its age</div>
            <div className="text-ink-2">
              auto-accept precision{" "}
              {meas?.control_precision.precision != null
                ? `${meas.control_precision.precision} (${meas.control_precision.interval.lo.toFixed(2)}-${meas.control_precision.interval.hi.toFixed(2)})`
                : "unmeasured"}
              <span className="text-ink-3"> · {meas?.control_precision.pending ?? 0} pending · measured {age(meas?.control_precision.measured_at)}</span>
            </div>
            <div className="text-ink-2">
              judge calibration
              <span className="text-ink-3"> · {meas?.judge_calibration.verdicts ?? 0} verdicts · {age(meas?.judge_calibration.measured_at)}</span>
            </div>
            <div className="text-ink-2">
              fitted thresholds active: {meas?.active_fitted_thresholds ?? 0}
              {(meas?.active_fitted_thresholds ?? 0) === 0 &&
                <span className="text-warn"> (no class can hold L1 without one)</span>}
            </div>
            <div className="uppercase text-[10px] text-ink-3 pt-2">class precision, oldest first</div>
            <div className="max-h-[180px] overflow-y-auto space-y-0.5">
              {(meas?.class_precision ?? []).map((c) => (
                <div key={c.class_name} className="flex justify-between text-ink-3">
                  <span>{c.class_name}</span>
                  <span className={c.age_days > 14 ? "text-warn" : ""}>{c.age_days}d</span>
                </div>
              ))}
            </div>
          </section>
        </div>

        <Worklist items={settle?.worklist ?? []} minutes={settle?.verdict_minutes_open ?? 0} />

        <ShadowPanel shadow={state?.shadow} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* lots */}
          <section className="panel lg:col-span-2">
            <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2">
              settlement lots · {settle?.settled_objects ?? 0} objects settled
              {settle?.revert_rate != null && ` · revert rate ${settle.revert_rate}`}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[11px]">
                <thead><tr className="text-ink-3 text-left">
                  <th className="px-3 py-1">class</th><th>tier</th><th>population</th>
                  <th>sample</th><th>status</th><th></th>
                </tr></thead>
                <tbody>
                  {lots.map((l) => (
                    <tr key={l.lot_id} className="border-t hairline">
                      <td className="px-3 py-1 text-ink-2">{l.class_name}</td>
                      <td className={l.tier === "critical" ? "text-block" : l.tier === "safety" ? "text-warn" : "text-ink-3"}>{l.tier}</td>
                      <td className="text-ink-3">{l.population.toLocaleString()}</td>
                      <td className="text-ink-3">
                        {l.sample_n ? `${l.defects}/${l.sample_n} defects` : `${l.sample_drawn} awaiting verdicts`}
                        {l.rule === "sprt" && l.cap_n ? ` · ${l.sample_drawn}/${l.cap_n} drawn` : ""}
                      </td>
                      <td className="text-ink-3">{l.status}</td>
                      <td className="px-2 whitespace-nowrap">
                        {l.review_at && l.status === "judging" && (
                          <a className="text-accent hover:underline" href={l.review_at}>judge</a>
                        )}
                        {l.spot_review_at && l.status === "settled" && (
                          <a className="text-accent hover:underline mr-2" href={l.spot_review_at}>spot check</a>
                        )}
                        {l.status === "settled" && (
                          <button className="text-warn hover:underline"
                            onClick={() => act(() => api.settlementRevert(l.lot_id, "manual"), `revert ${l.class_name}`)}>
                            revert
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                  {lots.length === 0 && (
                    <tr><td className="px-3 py-2 text-ink-3" colSpan={6}>
                      no lots yet; the nightly agent plans one for the best-measured eligible class
                    </td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          {/* the journal */}
          <section className="panel">
            <div className="uppercase text-[10px] text-ink-3 border-b hairline px-3 py-2">agent journal</div>
            <div className="max-h-[300px] overflow-y-auto">
              {(state?.journal ?? []).map((j) => (
                <div key={j.run_id} className="flex justify-between px-3 py-1 border-t hairline">
                  <span className="text-ink-2">{j.kind}</span>
                  <span className={j.status === "error" ? "text-block" : "text-ink-3"}>
                    {j.status} · {age(j.created_at)}
                  </span>
                </div>
              ))}
              {(state?.journal ?? []).length === 0 && (
                <div className="px-3 py-2 text-ink-3">no agent has run yet</div>
              )}
            </div>
          </section>
        </div>
      </div>
    </PageShell>
  );
}
