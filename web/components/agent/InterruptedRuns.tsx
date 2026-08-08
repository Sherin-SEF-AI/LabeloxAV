"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { InterruptedRun } from "@/lib/types";
import { describeFailure } from "@/lib/actionError";

// Jobs whose process stopped before they finished. Background work runs as a task inside the API process,
// so a restart or a crash takes it with no trace but a row still marked running. Until this existed the
// corpus quietly accumulated them: one had been "running" for 863 hours.
//
// The design decision worth stating: the progress bar is driven by the job's own cursor, not by elapsed
// time. A bar that animates while nothing is happening is the thing that made a dead job look alive in the
// first place, so a run that never recorded a total shows no bar at all rather than a plausible one.

function ago(iso: string | null): string {
  if (!iso) return "unknown";
  const ms = Date.now() - new Date(iso).getTime();
  const m = Math.round(ms / 60000);
  if (m < 60) return `${m}m ago`;
  const h = Math.round(m / 60);
  return h < 48 ? `${h}h ago` : `${Math.round(h / 24)}d ago`;
}

function countSummary(counts: Record<string, unknown>): string {
  const parts = Object.entries(counts)
    .filter(([, v]) => typeof v === "number" && v > 0)
    .slice(0, 4)
    .map(([k, v]) => `${k} ${v}`);
  return parts.join("  ");
}

export default function InterruptedRuns({ onResumed }: { onResumed?: () => void }) {
  const [runs, setRuns] = useState<InterruptedRun[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const load = useCallback(async () => {
    try { setRuns((await api.agentInterruptedRuns()).runs); } catch { /* the console reports it */ }
  }, []);
  useEffect(() => { load(); }, [load]);

  const resume = async (r: InterruptedRun) => {
    setBusy(r.run_id); setNote(null);
    try {
      const out = await api.agentResumeRun(r.run_id);
      setNote(out.restarted
        ? `${r.kind} resumed from where it stopped`
        : `${r.kind} claimed, but it has no resume path`);
      await load();
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
      <div className="flex items-center gap-3 px-4 py-2 border-b hairline">
        <span className="w-1.5 h-1.5 rounded-full bg-warn shrink-0" aria-hidden />
        <div className="text-ink font-medium text-sm">Interrupted jobs</div>
        <div className="text-ink-3 text-xs">
          {visible.length} stopped before finishing
        </div>
      </div>

      <div className="divide-y hairline">
        {visible.map((r, i) => (
          <div key={r.run_id} className="px-4 py-2.5 fade-up" style={{ ["--d" as string]: `${i * 40}ms` }}>
            <div className="flex items-center gap-3">
              <span className="font-mono text-[11px] text-ink">{r.kind}</span>
              <span className="font-mono text-[10px] text-ink-3">{ago(r.heartbeat_at || r.created_at)}</span>
              {r.fraction !== null && r.fraction !== undefined && (
                <span className="font-mono text-[10px] text-ink-2 tabular-nums">
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
                  className="lift font-mono text-[10px] border border-accent/60 text-accent px-2 py-1 rounded
                             hover:bg-accent/10 disabled:opacity-40 disabled:cursor-not-allowed">
                  {busy === r.run_id ? "resuming" : r.resumable ? "resume" : "not resumable"}
                </button>
                <button
                  onClick={() => setDismissed((d) => new Set(d).add(r.run_id))}
                  aria-label={`dismiss ${r.kind}`}
                  className="font-mono text-[10px] text-ink-3 hover:text-ink px-1">
                  hide
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
              <div className="mt-1 font-mono text-[10px] text-ink-3">
                kept: {countSummary(r.counts)}
              </div>
            )}
          </div>
        ))}
      </div>

      {note && <div className="px-4 py-2 border-t hairline font-mono text-[11px] text-ink-2">{note}</div>}
    </div>
  );
}
