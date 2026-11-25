"use client";

import { useState } from "react";

import { api } from "@/lib/api";
import type { InterruptedRun } from "@/lib/types";
import { describeFailure } from "@/lib/actionError";
import { invalidate, useQuery } from "@/lib/useQuery";
import { ago, humanize, num } from "@/lib/format";

// Jobs whose process stopped before they finished. Background work runs as a task inside the API process,
// so a restart or a crash takes it with no trace but a row still marked running. Until this existed the
// corpus quietly accumulated them: one had been "running" for 863 hours.
//
// The design decision worth stating: the progress bar is driven by the job's own cursor, not by elapsed
// time. A bar that animates while nothing is happening is the thing that made a dead job look alive in the
// first place, so a run that never recorded a total shows no bar at all rather than a plausible one.

function countSummary(counts: Record<string, unknown>): string {
  const parts = Object.entries(counts)
    .filter(([, v]) => typeof v === "number" && v > 0)
    .slice(0, 4)
    .map(([k, v]) => `${num(v as number)} ${humanize(k).toLowerCase()}`);
  return parts.join(" · ");
}

const NO_RUNS: InterruptedRun[] = [];

export default function InterruptedRuns({ onResumed }: { onResumed?: () => void }) {
  // A failed read shows nothing here; the console reports it.
  const runsQ = useQuery(["agentInterruptedRuns"], () => api.agentInterruptedRuns());
  const runs: InterruptedRun[] = runsQ.data?.runs ?? NO_RUNS;
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const resume = async (r: InterruptedRun) => {
    setBusy(r.run_id); setNote(null);
    try {
      const out = await api.agentResumeRun(r.run_id);
      setNote(out.restarted
        ? `${humanize(r.kind)} resumed from where it stopped`
        : `${humanize(r.kind)} claimed, but it cannot continue from where it stopped`);
      // The resumed run leaves this list and reappears in the run lists as running.
      await invalidate("agentInterruptedRuns", "agentRuns");
      onResumed?.();
    } catch (e) {
      const f = describeFailure(`resume ${r.kind}`, e);
      setNote(f.hint ? `${f.message} (${f.hint})` : f.message);
    } finally { setBusy(null); }
  };

  const visible = runs.filter((r) => !dismissed.has(r.run_id));
  if (visible.length === 0) return null;

  return (
    <div className="panel reveal border-warn/40">
      <div className="panel-head">
        <span className="w-1.5 h-1.5 rounded-full bg-warn shrink-0" aria-hidden />
        <h2 className="section-title">Interrupted jobs</h2>
        <span className="text-[12px] text-ink-3">
          {visible.length} stopped before finishing, usually because the server restarted
        </span>
      </div>

      <div className="divide-y hairline">
        {visible.map((r, i) => (
          <div key={r.run_id} className="px-4 py-2.5 fade-up" style={{ ["--d" as string]: `${i * 40}ms` }}>
            <div className="flex items-center gap-3">
              <span className="text-[12.5px] text-ink">{humanize(r.kind)}</span>
              <span className="text-[12px] text-ink-3">stopped {(r.heartbeat_at || r.created_at) ? ago(r.heartbeat_at || r.created_at) : "at an unknown time"}</span>
              {r.fraction !== null && r.fraction !== undefined && (
                <span className="text-[12px] text-ink-2 tabular-nums">
                  {Math.round(r.fraction * 100)}% done
                </span>
              )}
              <div className="ml-auto flex items-center gap-2">
                <button
                  onClick={() => resume(r)}
                  disabled={busy !== null || !r.resumable}
                  title={r.resumable
                    ? "continue this job from its cursor"
                    : "this job recorded no cursor, so it can be started again but not continued"}
                  type="button" className="btn btn-primary disabled:cursor-not-allowed">
                  {busy === r.run_id ? "Resuming..." : r.resumable ? "Resume" : "Cannot resume"}
                </button>
                <button
                  type="button"
                  onClick={() => setDismissed((d) => new Set(d).add(r.run_id))}
                  aria-label={`dismiss ${r.kind}`}
                  className="btn">
                  Hide
                </button>
              </div>
            </div>

            {/* Determinate, and only when the cursor can say. An absent total means "we do not know how far
                it got", which is not the same claim as zero. */}
            {r.fraction !== null && r.fraction !== undefined && (
              <div className="mt-1.5 h-[3px] w-full bg-line/40 rounded overflow-hidden">
                <div className="h-full bg-accent transition-[width] duration-500 ease-out"
                  style={{ width: `${Math.max(2, r.fraction * 100)}%` }} />
              </div>
            )}

            {countSummary(r.counts) && (
              <div className="mt-1 text-[12px] text-ink-3">
                Done before it stopped: {countSummary(r.counts)}
              </div>
            )}
          </div>
        ))}
      </div>

      {note && <div role="status" className="px-4 py-2 border-t hairline text-[12.5px] text-ink-2">{note}</div>}
    </div>
  );
}
