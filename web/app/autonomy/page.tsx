"use client";

import { useCallback, useState } from "react";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import PulseDot from "@/components/PulseDot";
import EmptyState from "@/components/ui/EmptyState";
import { ago, humanize, num, pct, shortId } from "@/lib/format";
import { api, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
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
  0: "Proposes only", 1: "Accepts confident labels", 2: "Settles whole batches",
};

// A panel header in the shared style, with an optional right-hand note.
function Head({ title, note, children }: { title: string; note?: React.ReactNode; children?: React.ReactNode }) {
  return (
    <div className="panel-head">
      <h2 className="section-title">{title}</h2>
      {note != null && <span className="text-[12px] text-ink-3">{note}</span>}
      {children && <div className="ml-auto flex items-center gap-2">{children}</div>}
    </div>
  );
}

// The log-likelihood ratio between its two bounds: left is accept, right is reject, the middle is
// "keep asking". Drawn rather than printed because a person reading -1.7 has to remember which way
// is good; a marker two thirds of the way to the accept edge does not need remembering.
function LlrBar({ llr, lo, hi }: { llr: number | null; lo: number | null; hi: number | null }) {
  if (llr == null || lo == null || hi == null || hi <= lo) {
    return <span className="text-ink-3">No verdicts yet</span>;
  }
  const pct = Math.max(0, Math.min(100, ((llr - lo) / (hi - lo)) * 100));
  return (
    <span className="inline-flex items-center gap-1.5" data-tip={`Evidence so far ${llr.toFixed(2)}, accept at ${lo.toFixed(2)}, reject at ${hi.toFixed(2)}`}>
      <span className="text-[11px] text-pass">accept</span>
      <span className="relative inline-block h-[6px] w-[80px] rounded-sm bg-bg-2 border hairline">
        <span className="absolute top-[-2px] h-[10px] w-[2px] bg-ink" style={{ left: `calc(${pct}% - 1px)` }} />
      </span>
      <span className="text-[11px] text-block">reject</span>
    </span>
  );
}

function Worklist({ items, minutes }: { items: SettlementWorkItem[]; minutes: number }) {
  return (
    <section className="panel">
      <Head title="Waiting for your verdicts" note="Batches that need a person to judge a sample before they can be settled">
        <span className="text-[12px] text-ink-3">About {num(minutes)} min of judging left</span>
      </Head>
      {items.length === 0 ? (
        <EmptyState compact icon="check" title="Nothing to judge">No batch is waiting for verdicts right now.</EmptyState>
      ) : (
      <div className="overflow-x-auto">
        <table className="data-table">
          <thead><tr>
            <th>Class</th><th>Sample so far</th><th>Leaning</th>
            <th>Left to judge</th><th className="num" title="How many objects one minute of judging settles">Objects per minute</th><th></th>
          </tr></thead>
          <tbody>
            {items.map((w) => (
              <tr key={w.lot_id}>
                <td className="text-ink">{humanize(w.class_name)}</td>
                <td>
                  {w.sample_n ? `${num(w.defects)} wrong of ${num(w.sample_n)} judged` : "None judged"} · {num(w.drawn)}
                  {w.rule === "sprt" && w.cap_n ? ` of ${num(w.cap_n)}` : ""} drawn
                </td>
                <td><LlrBar llr={w.llr} lo={w.bound_accept} hi={w.bound_reject} /></td>
                <td>About {num(w.remaining_verdicts)} verdicts · {num(w.minutes)} min</td>
                <td className="num text-ink" data-tip={w.oc != null ? `Chance it passes: ${pct(w.oc)}` : undefined}>
                  {w.value.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </td>
                <td className="text-right">
                  {w.review_at && <a className="btn btn-primary" href={w.review_at}>Judge</a>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      )}
    </section>
  );
}

const KIND_LABEL: Record<string, string> = {
  champion_miss: "Live model missed",
  challenger_miss: "Candidate missed",
  class_flip: "Read as another class",
  conf_gap: "Same box, different confidence",
};

function ShadowPanel({ shadow }: { shadow: ShadowSummary | undefined }) {
  const kinds = Object.entries(shadow?.disagreements_by_kind ?? {}).sort((a, b) => b[1] - a[1]);
  const states = shadow?.disagreements_by_state ?? {};
  const open = (states.pending ?? 0) + (states.queued ?? 0);
  const last = shadow?.sweeps?.[0];
  return (
    <section className="panel">
      <Head title="Live model against the candidate" note="Where two models read the same frames differently">
        <span className="text-[12px] text-ink-3">{num(open)} disagreements open</span>
      </Head>
      {!shadow?.sweeps?.length ? (
        <EmptyState compact icon="layers" title="No comparison has run">
          Nothing yet compares the live model with a candidate on the recordings added since the gold set
          was frozen.
        </EmptyState>
      ) : (
        <div className="p-3.5 text-[12.5px] space-y-3">
          <div className="text-ink-3">
            Last compared {since(last?.created_at)} · {humanize(String(last?.status ?? ""))}
          </div>
          <div className="flex flex-wrap gap-x-5 gap-y-1">
            {kinds.map(([k, n]) => (
              <span key={k} className="text-ink-2">
                {KIND_LABEL[k] ?? humanize(k)} <span className="tabular-nums text-ink font-medium">{num(n)}</span>
              </span>
            ))}
            {kinds.length === 0 && <span className="text-ink-3">No disagreement recorded.</span>}
          </div>
          {(shadow?.win_shares ?? []).length > 0 && (
            <div className="space-y-1 border-t hairline pt-2.5">
              {(shadow?.win_shares ?? []).map((w) => (
                <div key={w.challenger_run_id} className="flex justify-between gap-3">
                  <span className="text-ink-3 truncate">Candidate <span className="font-mono">{shortId(w.challenger_run_id)}</span></span>
                  {w.measured ? (
                    <span className="text-ink-2 tabular-nums">
                      Candidate right in {num(w.challenger_right)} of {num(w.discordant)}
                      {w.share != null && ` · ${(w.share * 100).toFixed(0)}%`}
                      {w.lo != null && w.hi != null &&
                        ` (likely ${(w.lo * 100).toFixed(0)}-${(w.hi * 100).toFixed(0)}%)`}
                    </span>
                  ) : (
                    // Unmeasured, said out loud. A win share of zero and nobody having looked are
                    // different facts and this panel refuses to draw them the same way.
                    <span className="text-ink-3">Not measured: {w.reason ?? "nobody has ruled yet"}</span>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

// "never" for a measurement that has not happened, rather than the dash ago() gives an absent time.
function since(iso: string | null | undefined): string {
  return iso ? ago(iso) : "never";
}

export default function AutonomyPage() {
  // The daemon dot is a liveness claim, so it re-checks itself; a dot that only reflects page load
  // would go on saying "live" for as long as the tab stayed open.
  const stateQ = useQuery(["autonomyState"], () => api.autonomyState(), { pollMs: 60_000 });
  const lotsQ = useQuery(["settlementLots"], () => api.settlementLots(), { pollMs: 60_000 });
  const state: AutonomyState | null = stateQ.data ?? null;
  const lots: SettlementLotRow[] = lotsQ.data ?? [];
  const err = stateQ.loadState.error ?? lotsQ.loadState.error ?? null;
  const [showAllClasses, setShowAllClasses] = useState(false);

  const reloadState = stateQ.reload;
  const reloadLots = lotsQ.reload;
  const load = useCallback(async () => {
    await Promise.all([reloadState(), reloadLots()]);
  }, [reloadState, reloadLots]);

  const act = async (fn: () => Promise<unknown>, label: string) => {
    try {
      await fn();
      toast(`${label}: done`);
      // An ack or a revert moves a lot and can move a class's rung: refresh both here and wherever else
      // they are cached.
      await invalidate("autonomyState", "settlementLots");
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

  const live = daemon?.alive;
  const Switch = ({ label, on, onText = "On", offText = "Off", tip }: { label: string; on: boolean | undefined; onText?: string; offText?: string; tip?: string }) => (
    <span className="flex items-center gap-1.5" data-tip={tip}>
      <span className="text-ink-3">{label}</span>
      <span className={on ? "text-ink" : "text-ink-2"}>{sw ? (on ? onText : offText) : "-"}</span>
    </span>
  );

  return (
    <PageShell active="AUTONOMY" title="AUTONOMY"
      subtitle="What the system may do without a person, class by class, and the evidence that earned it">
      <div className="page-wrap space-y-4 text-[12.5px]">
        {err != null && <LoadState error={err} onRetry={() => void load()} />}

        {/* the switches and the heartbeat */}
        <div className="panel px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-2">
          <span className="flex items-center gap-2">
            <PulseDot tone={daemon ? (live ? "live" : "bad") : "idle"}
              label={live
                ? `background agent running, last ran ${since(daemon?.last_tick_at)}`
                : `background agent stopped, last ran ${since(daemon?.last_tick_at)}`} />
            <span className={daemon && !live ? "text-block font-medium" : "text-ink font-medium"}>
              {daemon ? (live ? "Background agent running" : "Background agent stopped") : "Background agent"}
            </span>
            {daemon && <span className="text-ink-3">last ran {since(daemon.last_tick_at)}</span>}
          </span>
          {Switch({ label: "Improvement loop", on: sw?.loop_enabled, tip: "Mines, labels and retrains on its own" })}
          {Switch({ label: "Auto-accept", on: sw?.auto_accept_enabled, tip: "Confident machine labels are accepted without review" })}
          {Switch({ label: "Model promotion", on: sw?.auto_promote_enabled, onText: "Automatic", offText: "Needs approval" })}
          {Switch({ label: "Batch settlement", on: sw?.settlement_enabled, onText: "Armed", offText: "Off",
            tip: "Whole batches of a class can be accepted from a judged sample" })}
          {sw?.paused_reason && <span className="text-warn" data-tip="Why the loop is paused">Paused: {sw.paused_reason}</span>}
        </div>

        {/* one-click acks the agent is waiting on */}
        {acks.length > 0 && (
          <section className="panel border-warn/50">
            <Head title="Waiting for you" note="These batches passed their sample check and need a person to sign off" />
            <div className="p-3.5 space-y-2">
              {acks.map((l) => (
                <div key={l.lot_id} className="flex flex-wrap items-center gap-3">
                  <span className="text-ink">{humanize(l.class_name)}</span>
                  <span className="text-ink-3">
                    passed: {num(l.defects)} wrong in {num(l.sample_n)} judged (allowed error {l.far_bound})
                  </span>
                  {l.tier !== "default" && <span className="text-warn">{humanize(l.tier)}</span>}
                  <button type="button" className="btn btn-primary ml-auto"
                    onClick={() => act(() => api.settlementAck(l.lot_id), `Sign off ${humanize(l.class_name)}`)}>
                    Sign off and settle
                  </button>
                </div>
              ))}
            </div>
          </section>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* the ladder */}
          <section className="panel lg:col-span-2 min-w-0">
            <Head title="What each class may do">
              <button type="button" className="link text-[12px]"
                onClick={() => setShowAllClasses((v) => !v)}>
                {showAllClasses ? "Show raised classes only" : `Show all ${num(ladder.length)} classes`}
              </button>
            </Head>
            {shown.length === 0 ? (
              <EmptyState compact icon="target" title="Every class still needs a person">
                No class may act on its own yet. A class earns &ldquo;accepts confident labels&rdquo; once its
                precision has been measured, and &ldquo;settles whole batches&rdquo; once a judged sample passes.
              </EmptyState>
            ) : (
            <div className="overflow-x-auto max-h-[420px] overflow-y-auto">
              <table className="data-table">
                <thead><tr>
                  <th>Class</th><th>Allowed to</th><th>Set by</th><th>Why</th>
                </tr></thead>
                <tbody>
                  {shown.map((r) => (
                    <tr key={r.class_name} className="align-top">
                      <td className="text-ink">
                        {humanize(r.class_name)}{r.pinned ? <span className="text-ink-3" data-tip="Pinned by a person"> (pinned)</span> : null}
                      </td>
                      <td className={r.level === 2 ? "text-pass" : r.level === 1 ? "text-ink" : "text-ink-3"}>
                        {LEVEL_LABEL[r.level] ?? `Level ${r.level}`}
                      </td>
                      <td>{humanize(r.set_by)}</td>
                      <td className="max-w-[26rem]">
                        {String(r.basis?.reason ?? r.basis?.step_down_reason ?? "")}
                        {r.cooldown_until && <span className="text-warn"> · paused until {r.cooldown_until.slice(0, 10)}</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            )}
          </section>

          {/* measurements + their age */}
          <section className="panel min-w-0">
            <Head title="The evidence, and how old it is" />
            <div className="p-3.5 space-y-2.5">
              <div>
                <div className="text-ink">Auto-accept precision</div>
                <div className="text-ink-3">
                  {meas?.control_precision.precision != null
                    ? `${pct(meas.control_precision.precision)} (likely ${pct(meas.control_precision.interval.lo)}-${pct(meas.control_precision.interval.hi)})`
                    : "Not measured yet"}
                  {" "}· {num(meas?.control_precision.pending ?? 0)} checks waiting · measured {since(meas?.control_precision.measured_at)}
                </div>
              </div>
              <div>
                <div className="text-ink">Judge model calibration</div>
                <div className="text-ink-3">{num(meas?.judge_calibration.verdicts ?? 0)} verdicts · {since(meas?.judge_calibration.measured_at)}</div>
              </div>
              <div>
                <div className="text-ink">Per-class thresholds in use: {num(meas?.active_fitted_thresholds ?? 0)}</div>
                {(meas?.active_fitted_thresholds ?? 0) === 0 &&
                  <div className="text-warn">No class can accept labels on its own until one is fitted.</div>}
              </div>
              <div className="border-t hairline pt-2.5">
                <div className="label-caps mb-1.5">Class precision, oldest first</div>
                <div className="max-h-[180px] overflow-y-auto space-y-1">
                  {(meas?.class_precision ?? []).map((c) => (
                    <div key={c.class_name} className="flex justify-between gap-3">
                      <span className="text-ink-2 truncate">{humanize(c.class_name)}</span>
                      <span className={`shrink-0 tabular-nums ${c.age_days > 14 ? "text-warn" : "text-ink-3"}`}
                        data-tip={c.age_days > 14 ? "Older than two weeks" : undefined}>
                        {Math.round(c.age_days)} days old
                      </span>
                    </div>
                  ))}
                  {(meas?.class_precision ?? []).length === 0 && <div className="text-ink-3">No class measured yet.</div>}
                </div>
              </div>
            </div>
          </section>
        </div>

        <Worklist items={settle?.worklist ?? []} minutes={settle?.verdict_minutes_open ?? 0} />

        <ShadowPanel shadow={state?.shadow} />

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* lots */}
          <section className="panel lg:col-span-2 min-w-0">
            <Head title="Batches" note={<>
              {num(settle?.settled_objects ?? 0)} objects settled
              {settle?.revert_rate != null && ` · ${pct(settle.revert_rate)} reverted`}
            </>} />
            {lots.length === 0 ? (
              <EmptyState compact icon="layers" title="No batches yet">
                Each night the agent plans a batch for the best-measured class that qualifies.
              </EmptyState>
            ) : (
            <div className="overflow-x-auto">
              <table className="data-table">
                <thead><tr>
                  <th>Class</th><th>Risk tier</th><th className="num">Objects</th>
                  <th>Sample</th><th>Status</th><th></th>
                </tr></thead>
                <tbody>
                  {lots.map((l) => (
                    <tr key={l.lot_id}>
                      <td className="text-ink">{humanize(l.class_name)}</td>
                      <td className={l.tier === "critical" ? "text-block" : l.tier === "safety" ? "text-warn" : "text-ink-3"}>{humanize(l.tier)}</td>
                      <td className="num">{num(l.population)}</td>
                      <td>
                        {l.sample_n ? `${num(l.defects)} wrong of ${num(l.sample_n)}` : `${num(l.sample_drawn)} waiting for verdicts`}
                        {l.rule === "sprt" && l.cap_n ? ` · ${num(l.sample_drawn)} of ${num(l.cap_n)} drawn` : ""}
                      </td>
                      <td>{humanize(l.status)}</td>
                      <td className="text-right whitespace-nowrap space-x-3">
                        {l.review_at && l.status === "judging" && (
                          <a className="link" href={l.review_at}>Judge</a>
                        )}
                        {l.spot_review_at && l.status === "settled" && (
                          <a className="link" href={l.spot_review_at}>Spot check</a>
                        )}
                        {l.status === "settled" && (
                          <button type="button" className="text-block hover:underline"
                            data-tip="Undo the settlement: every object in the batch goes back to needing review"
                            onClick={() => act(() => api.settlementRevert(l.lot_id, "manual"), `Revert ${humanize(l.class_name)}`)}>
                            Revert
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            )}
          </section>

          {/* the journal */}
          <section className="panel min-w-0">
            <Head title="Agent activity" />
            <div className="max-h-[300px] overflow-y-auto">
              {(state?.journal ?? []).map((j) => (
                <div key={j.run_id} className="flex justify-between gap-3 px-3.5 py-2 border-t hairline first:border-t-0">
                  <span className="text-ink-2 truncate">{humanize(j.kind)}</span>
                  <span className={`shrink-0 ${j.status === "error" ? "text-block" : "text-ink-3"}`}>
                    {humanize(j.status)} · {since(j.created_at)}
                  </span>
                </div>
              ))}
              {(state?.journal ?? []).length === 0 && (
                <EmptyState compact icon="robot" title="No agent has run yet" />
              )}
            </div>
          </section>
        </div>
      </div>
    </PageShell>
  );
}
