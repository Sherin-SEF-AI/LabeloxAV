"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";
import { describeFailure } from "@/lib/actionError";
import { pollDelay } from "@/lib/pollDelay";
import type { AgentRunRow } from "@/lib/types";

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

const TONE: Record<string, string> = {
  running: "text-warn border-warn/50",
  committed: "text-pass border-pass/50",
  reverted: "text-ink-3 border-line",
  error: "text-block border-block/50",
  interrupted: "text-block border-block/50",
  planned: "text-ink-3 border-line",
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
  const [runs, setRuns] = useState<AgentRunRow[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const alive = useRef(true);

  const load = useCallback(async (): Promise<boolean> => {
    try { setRuns(await api.agentRuns(limit)); return true; }
    catch { return false; }
  }, [limit]);

  // Poll only while something is running, at the cadence the cloud control already uses. An idle console
  // must not sit on a timer: this page is left open all day.
  useEffect(() => {
    alive.current = true;
    let timer: ReturnType<typeof setTimeout>;
    const loop = async () => {
      if (!alive.current) return;
      const ok = await load();
      if (!alive.current) return;
      const live = runs.some((r) => !TERMINAL.has(r.status));
      timer = setTimeout(loop, pollDelay({ ok, live }));
    };
    loop();
    return () => { alive.current = false; clearTimeout(timer); };
    // `runs` is deliberately absent: including it restarts the timer on every poll, which turns an adaptive
    // poll into a tight loop. Liveness is read from the ref-fresh closure on the next tick instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load]);

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
      setNote(`${r.kind} reverted: ${out.reverted} restored, ${out.skipped} left alone`);
      await load();
    } catch (e) {
      const f = describeFailure(`revert ${r.kind}`, e);
      setNote(f.hint ? `${f.message} (${f.hint})` : f.message);
    } finally { setBusy(null); }
  };

  if (runs.length === 0) return null;
  const liveCount = runs.filter((r) => !TERMINAL.has(r.status)).length;

  return (
    <div className="panel">
      <div className="flex items-center gap-3 px-4 py-2 border-b hairline">
        {liveCount > 0 && <span className="running-dot" aria-hidden />}
        <div className="text-ink font-medium text-sm">Agent jobs</div>
        <div className="text-ink-3 text-xs">
          {liveCount > 0 ? `${liveCount} running` : `last ${runs.length}`}
        </div>
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
                <span className={`font-mono text-[9px] uppercase border px-1.5 rounded ${TONE[r.status] || "border-line text-ink-3"}`}>
                  {r.status}
                </span>
                <span className="font-mono text-[11px] text-ink">{r.kind}</span>
                {took && <span className="font-mono text-[10px] text-ink-3 tabular-nums">{took}</span>}
                {r.fraction !== null && r.fraction !== undefined && (
                  <span className="font-mono text-[10px] text-accent tabular-nums">
                    {Math.round(r.fraction * 100)}%
                  </span>
                )}
                {r.changed > 0 && (
                  <span className="font-mono text-[10px] text-ink-3">{r.changed} objects changed</span>
                )}
                {r.status === "committed" && r.changed > 0 && (
                  <button onClick={() => revert(r)} disabled={busy !== null}
                    title="restore every object this run touched, unless a person has since"
                    className="lift ml-auto font-mono text-[10px] border border-line text-ink-3 px-2 py-0.5
                               rounded hover:text-ink hover:border-block disabled:opacity-40">
                    {busy === r.run_id ? "reverting" : "revert"}
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
                <div className="mt-1 flex gap-3 flex-wrap font-mono text-[10px] text-ink-3">
                  {rs.map(([k, v]) => (
                    <span key={k}><span className="text-ink-2 tabular-nums">{v.toLocaleString()}</span> {k}</span>
                  ))}
                </div>
              )}

              {r.error && (
                <div className="mt-1 font-mono text-[10px] text-block break-words">{r.error}</div>
              )}
            </div>
          );
        })}
      </div>

      {note && <div className="px-4 py-2 border-t hairline font-mono text-[11px] text-ink-2">{note}</div>}
    </div>
  );
}
