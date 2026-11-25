"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import { describeFailure } from "@/lib/actionError";
import { pollDelay } from "@/lib/pollDelay";
import type { AgentRunRow } from "@/lib/types";
import { invalidate, useQuery } from "@/lib/useQuery";
import { humanize } from "@/lib/format";

// What the agent is doing right now, and what it just finished.
//
// /jobs already streams import, training, export and autolabel. AgentRuns are the other half, and they are
// everything this console launches: sweeps, audits, relabels, flywheel ticks, ontology scans. They were
// watched nowhere, so a launch produced one line of text and then silence until somebody reloaded.
//
// Progress is the job's own cursor, never elapsed time. A job that never recorded a total shows no bar,
// because "we do not know how far it got" is a different statement from zero, and a bar that moves whether
// or not work is happening is what let a run sit at "running" for 863 hours unnoticed.

const TERMINAL = new Set(["committed", "reverted", "error", "interrupted"]);
const NO_RUNS: AgentRunRow[] = [];

const TONE: Record<string, string> = {
  running: "text-warn border-warn/50",
  committed: "text-pass border-pass/50",
  reverted: "text-ink-3 border-line",
  error: "text-block border-block/50",
  interrupted: "text-block border-block/50",
  planned: "text-ink-3 border-line",
};

const STATUS_LABEL: Record<string, string> = {
  running: "Running", committed: "Done", reverted: "Undone", error: "Failed", interrupted: "Interrupted", planned: "Planned",
};

function dur(fromIso: string | null, toIso: string | null): string | null {
  if (!fromIso) return null;
  const ms = (toIso ? new Date(toIso).getTime() : Date.now()) - new Date(fromIso).getTime();
  if (ms < 0) return null;
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.round(s / 60);
  return m < 60 ? `${m}m` : `${Math.round(m / 60)}h`;
}

/** The numbers a run produced, which are the reason anyone launched it. */
function results(counts: Record<string, unknown>): [string, number][] {
  return Object.entries(counts)
    .filter((e): e is [string, number] => typeof e[1] === "number" && e[1] > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);
}

export default function JobWatcher({ limit = 12 }: { limit?: number }) {
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [, setTick] = useState(0);

  // Polled at the cadence the cloud control already uses: briskly while something is running, slowly when
  // idle, and backing off after a failed request. A failed poll keeps the last list on screen. The cadence
  // is worked out from the response that just arrived, and the poll stops while the tab is hidden: this
  // page is left open all day.
  const runsQ = useQuery(["agentRuns", limit], () => api.agentRuns(limit), {
    pollMs: (data, error) =>
      pollDelay({ ok: !error, live: (data ?? []).some((r) => !TERMINAL.has(r.status)) }),
  });
  const runs = runsQ.data ?? NO_RUNS;

  // A one-second tick so an elapsed time on a running job advances between polls. Local arithmetic on
  // values already held, not a request.
  useEffect(() => {
    const live = runs.some((r) => !TERMINAL.has(r.status));
    if (!live) return;
    const t = setInterval(() => setTick((x) => x + 1), 1000);
    return () => clearInterval(t);
  }, [runs]);

  const revert = async (r: AgentRunRow) => {
    setBusy(r.run_id); setNote(null);
    try {
      const out = await api.agentRevertRun(r.run_id);
      setNote(`${humanize(r.kind)} undone: ${out.reverted} restored, ${out.skipped} left alone because a person has changed them since`);
      // Every cached run list, whatever its limit: the console dialog reads the same runs.
      await invalidate("agentRuns");
    } catch (e) {
      const f = describeFailure(`revert ${r.kind}`, e);
      setNote(f.hint ? `${f.message} (${f.hint})` : f.message);
    } finally { setBusy(null); }
  };

  if (runs.length === 0) return null;
  const liveCount = runs.filter((r) => !TERMINAL.has(r.status)).length;

  return (
    <div className="panel">
      <div className="panel-head">
        {liveCount > 0 && <span className="running-dot" aria-hidden />}
        <h2 className="section-title">Agent jobs</h2>
        <span className="text-[12px] text-ink-3">
          {liveCount > 0 ? `${liveCount} running` : `Last ${runs.length}`}
        </span>
      </div>

      <div className="divide-y hairline">
        {runs.map((r, i) => {
          const live = !TERMINAL.has(r.status);
          const took = dur(r.created_at, live ? null : (r.reverted_at || r.heartbeat_at));
          const rs = results(r.counts || {});
          return (
            <div key={r.run_id} className={`px-4 py-2 fade-up ${live ? "running" : ""}`}
              style={{ ["--d" as string]: `${Math.min(i, 8) * 35}ms` }}>
              <div className="flex items-center gap-2.5 flex-wrap">
                <span className={`text-[11px] border px-1.5 py-px rounded ${TONE[r.status] || "border-line text-ink-3"}`}>
                  {STATUS_LABEL[r.status] ?? humanize(r.status)}
                </span>
                <span className="text-[12.5px] text-ink">{humanize(r.kind)}</span>
                {took && <span className="text-[12px] text-ink-3 tabular-nums">{took}</span>}
                {r.fraction !== null && r.fraction !== undefined && (
                  <span className="text-[12px] text-accent-2 tabular-nums">
                    {Math.round(r.fraction * 100)}%
                  </span>
                )}
                {r.changed > 0 && (
                  <span className="text-[12px] text-ink-3">{r.changed.toLocaleString()} objects changed</span>
                )}
                {r.status === "committed" && r.changed > 0 && (
                  <button onClick={() => revert(r)} disabled={busy !== null}
                    title="restore every object this run touched, unless a person has since"
                    type="button" className="btn btn-danger ml-auto">
                    {busy === r.run_id ? "Undoing..." : "Undo"}
                  </button>
                )}
              </div>

              {r.fraction !== null && r.fraction !== undefined && (
                <div className="mt-1.5 h-[3px] w-full bg-line/40 rounded overflow-hidden">
                  <div className={`h-full ${live ? "bg-warn" : "bg-accent"} transition-[width] duration-700 ease-out`}
                    style={{ width: `${Math.max(2, r.fraction * 100)}%` }} />
                </div>
              )}

              {rs.length > 0 && (
                <div className="mt-1 flex gap-3 flex-wrap text-[12px] text-ink-3">
                  {rs.map(([k, v]) => (
                    <span key={k}><span className="text-ink-2 tabular-nums">{v.toLocaleString()}</span> {humanize(k).toLowerCase()}</span>
                  ))}
                </div>
              )}

              {r.error && (
                <div className="mt-1 text-[12px] text-block break-words">{r.error}</div>
              )}
            </div>
          );
        })}
      </div>

      {note && <div role="status" className="px-4 py-2 border-t hairline text-[12.5px] text-ink-2">{note}</div>}
    </div>
  );
}
