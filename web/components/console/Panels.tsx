"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { type CanvasOp, subscribeOps } from "@/lib/canvasOps";
import { level, mb, uptime, workInFlight } from "@/lib/console";
import type { AgentRunRow } from "@/lib/types";
import type { JobStream, SystemStream } from "@/lib/useEventStream";
import { useQuery } from "@/lib/useQuery";
import { ago, humanize, num, plural, shortId } from "@/lib/format";

// The console's panels, as pieces rather than a page.
//
// They are shared because the console is now two surfaces: a dialog that opens over whatever raised the
// question, and the route that is still in the menu and still deep-linkable. Two copies of "how much VRAM is
// left" would drift, and the one people happened not to be looking at would be the one that went wrong.

const TONE: Record<string, string> = { ok: "bg-accent", warn: "bg-warn", critical: "bg-block" };
const TEXT: Record<string, string> = { ok: "text-ink-2", warn: "text-warn", critical: "text-block" };

export function Meter({ label, frac, detail }: { label: string; frac: number | null; detail: string }) {
  const tone = level(frac);
  return (
    <div className="space-y-1">
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="text-ink-3">{label}</span>
        <span className={TEXT[tone]}>{detail}</span>
      </div>
      <div className="h-1.5 bg-line rounded overflow-hidden">
        <div className={`h-full ${TONE[tone]} transition-[width] duration-700 ease-out`}
          style={{ width: `${Math.round((frac ?? 0) * 100)}%` }} />
      </div>
    </div>
  );
}

export function GpuPanel({ sys }: { sys: SystemStream | null }) {
  const gpu = sys?.gpus?.[0];
  if (!gpu) {
    return (
      <div className="text-[12.5px] text-ink-3 leading-relaxed">
        No GPU is visible to this process. Auto-labelling and training will not start without one.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="text-[12.5px] text-ink-2">{gpu.name}</div>
      <Meter label="Memory" frac={gpu.memory_used_frac}
        detail={`${mb(gpu.memory_used_mb)} / ${mb(gpu.memory_total_mb)}`} />
      <Meter label="Utilisation" frac={(gpu.utilization_pct ?? 0) / 100}
        detail={`${Math.round(gpu.utilization_pct ?? 0)}%`} />
      <div className="text-[12px] text-ink-3">
        {gpu.temperature_c != null && <span>{Math.round(gpu.temperature_c)}°C</span>}
        {gpu.power_w != null && <span> · {Math.round(gpu.power_w)} W</span>}
      </div>
      {/* The tenant list is the actionable part. "7.2 GB used" is not; "llama-server holds 7.2 GB" is the
          difference between waiting and killing something. */}
      {gpu.processes.length > 0 && (
        <div className="pt-2 border-t hairline space-y-0.5">
          {gpu.processes.map((p) => (
            <div key={`${p.pid}-${p.name}`} className="flex justify-between text-[12px]">
              <span className="text-ink-3 truncate" title={p.name}>{p.name.split("/").pop()}</span>
              <span className="text-ink-2 shrink-0 ml-2">{mb(p.used_mb)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function HostPanel({ sys }: { sys: SystemStream | null }) {
  const host = sys?.host;
  if (!host) return <NoReading />;
  return (
    <div className="space-y-3">
      <div className="text-[12.5px] text-ink-2">
        Load {host.load["1m"]} across {host.cpu_count} cores
      </div>
      <Meter label="Memory" frac={host.memory_used_frac}
        detail={`${mb(host.memory_used_mb)} / ${mb(host.memory_total_mb)}`} />
      <Meter label="CPU" frac={(host.cpu_pct ?? 0) / 100} detail={`${Math.round(host.cpu_pct)}%`} />
      <Meter label="Disk" frac={host.disk.used_frac}
        detail={`${mb(host.disk.used_mb)} / ${mb(host.disk.total_mb)}`} />
      <div className="text-[11.5px] text-ink-3 font-mono truncate" title={host.disk.path}>{host.disk.path}</div>
    </div>
  );
}

export function ProcessPanel({ sys, waitingTotal }: { sys: SystemStream | null; waitingTotal: number }) {
  if (!sys) return <NoReading />;
  return (
    <div className="text-[12.5px] space-y-1.5 text-ink-3">
      {/* Its own share, so "the machine is busy" can be told from "we are busy". */}
      <div className="flex justify-between"><span>Memory in use</span>
        <span className="text-ink-2 tabular-nums">{mb(sys.process.rss_mb)}</span></div>
      <div className="flex justify-between"><span>Uptime</span>
        <span className="text-ink-2">{uptime(sys.process.uptime_s)}</span></div>
      <div className="flex justify-between"><span>Queued jobs</span>
        <span className={`tabular-nums ${waitingTotal ? "text-warn" : "text-ink-2"}`}>{num(waitingTotal)}</span></div>
      <div className="flex justify-between"><span>Threads</span>
        <span className="text-ink-2 tabular-nums">{sys.process.threads}</span></div>
      <div className="flex justify-between"><span>Process id</span>
        <span className="text-ink-3 font-mono">{sys.process.pid}</span></div>
    </div>
  );
}

export function RunningNowPanel({ jobs }: { jobs: JobStream | null }) {
  const { running, waitingTotal } = workInFlight(jobs);
  if (!running.length) {
    return (
      <div className="py-6 text-center text-[12.5px] text-ink-3">
        {waitingTotal
          ? `Nothing is running. ${plural(waitingTotal, "job")} queued, mostly waiting for hardware that is not connected.`
          : "Nothing is running."}
      </div>
    );
  }
  return (
    <div className="divide-y divide-line">
      {running.map((w) => (
        <div key={w.id} className="flex items-center gap-3 py-2.5 text-[12.5px]">
          <span className="text-ink w-24">{humanize(w.kind)}</span>
          <span className="text-ink-3 font-mono text-[11.5px] w-20 truncate" title={w.id}>{shortId(w.id)}</span>
          <span className="flex-1 h-1.5 bg-line rounded overflow-hidden">
            {/* Unmeasured progress shows no bar at all. A bar at zero reads as stalled, which is a
                different claim from "this job does not report progress". */}
            {w.progress != null && (
              <span className="block h-full bg-accent transition-[width] duration-700 ease-out"
                style={{ width: `${Math.round(w.progress * 100)}%` }} />
            )}
          </span>
          <span className="text-ink-2 w-12 text-right tabular-nums">
            {w.progress != null ? `${Math.round(w.progress * 100)}%` : "-"}
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * How far the auto-accept gate is from being measured at all.
 *
 * The gate decides, unattended, which machine labels enter the corpus without a person seeing them. The
 * only measurement of whether it is right is the control sample, and 601 of them sit unjudged because the
 * one place that shows them is a tab on another page. A number that lives only where somebody has already
 * decided to look is not a signal.
 */
export function GatePanel() {
  // A failed read renders as "no reading", which is what it is; the error is deliberately not shown here.
  const { data: p } = useQuery(["governPrecision"], () => api.governPrecision());
  if (!p) return <NoReading />;
  return (
    <div className="text-[12.5px] space-y-1.5 text-ink-3">
      <div className="flex justify-between">
        <span>Gate precision</span>
        {p.precision == null
          ? <span className="text-warn">Not measured yet</span>
          : <span className="text-ink-2">{(p.precision * 100).toFixed(1)}% over {num(p.reviewed)} judged</span>}
      </div>
      <div className="flex justify-between">
        <span>Awaiting a verdict</span>
        <span className={`tabular-nums ${p.pending ? "text-warn" : "text-ink-2"}`}>{num(p.pending)}</span>
      </div>
      {p.precision == null && p.pending > 0 && (
        <a href="/review/queue" className="link block pt-1">
          Judge them in the review queue
        </a>
      )}
    </div>
  );
}


/** Agent runs, polled rather than streamed: a corpus sweep going for an hour is no more informative for
 *  being re-read every three seconds. */
const NO_RUNS: AgentRunRow[] = [];

export function useAgentRuns(enabled = true) {
  // A failed poll keeps the last list rather than blanking it, and the next beat tries again. The poll
  // stops while the tab is hidden. keepPrevious, so switching it off leaves the last list in place as the
  // state it replaces did.
  const { data } = useQuery(["agentRuns", 12], () => api.agentRuns(12),
    { enabled, pollMs: 20_000, keepPrevious: true });
  return data ?? NO_RUNS;
}

/** One entry of an agent run's `counts`, rendered so a value that is not a number still says something.
 *
 * `counts` is `Record<string, unknown>`: free-form JSON written by whichever agent produced the run, and
 * several of them put a list there. `drift_investigator` writes `findings` as an array of objects and
 * `pseudo_lift` writes `sessions` the same way. Interpolating one of those into a template string
 * produced `[object Object]`, which is what the console actually showed for every drift and lift run.
 *
 * A list is reported by its length, which is the number a reader wants from a list anyway, and a nested
 * object by how many keys it has. Neither is guessed at: both are facts about the value.
 */
export function countText(k: string, v: unknown): string {
  if (Array.isArray(v)) return `${k} ${v.length}`;
  if (v && typeof v === "object") return `${k} ${Object.keys(v).length}`;
  return `${k} ${String(v)}`;
}

/** The same entry as a phrase a person reads: `objects: 7` -> "7 objects", `findings: [..]` -> "2 findings". */
export function countPhrase(k: string, v: unknown): string {
  const name = humanize(k).toLowerCase();
  if (Array.isArray(v)) return `${num(v.length)} ${name}`;
  if (typeof v === "number") return `${num(v)} ${name}`;
  if (v && typeof v === "object") return `${name}: ${Object.keys(v).length}`;
  return `${name}: ${String(v)}`;
}

const RUN_TONE: Record<string, string> = {
  running: "text-accent-2 border-accent/50", planned: "text-ink-2 border-line",
  error: "text-block border-block/40", interrupted: "text-block border-block/40",
  committed: "text-pass border-pass/40", reverted: "text-ink-3 border-line",
};

export function BackgroundPanel({ runs }: { runs: AgentRunRow[] }) {
  if (!runs.length) {
    return (
      <div className="py-6 text-center text-[12.5px] text-ink-3">
        No background runs yet. Corpus sweeps, relabels and review batches appear here while they run.
      </div>
    );
  }
  return (
    <div className="divide-y divide-line">
      {runs.slice(0, 10).map((r) => (
        <div key={r.run_id} className="flex items-center gap-3 py-2 text-[12.5px]">
          <span className="text-ink w-40 truncate" title={r.kind}>{humanize(r.kind)}</span>
          <span className="w-28 shrink-0">
            <span className={`inline-flex items-center px-1.5 py-0.5 rounded border text-[11.5px] ${
              RUN_TONE[r.status] ?? "text-ink-3 border-line"}`}>
              {humanize(r.status)}
            </span>
          </span>
          <span className="text-ink-2 flex-1 truncate">
            {Object.entries(r.counts ?? {}).slice(0, 3).map(([k, v]) => countPhrase(k, v)).join(" · ")}
          </span>
          <span className="text-ink-3 text-[12px] shrink-0 text-right whitespace-nowrap">
            {[r.created_by, r.created_at ? ago(r.created_at) : null].filter(Boolean).join(" · ")}
          </span>
        </div>
      ))}
    </div>
  );
}

function NoReading() {
  return <div className="text-[12.5px] text-ink-3">No reading yet. It updates every few seconds once the stream connects.</div>;
}

const DOT: Record<CanvasOp["status"], string> = { running: "bg-accent", ok: "bg-pass", failed: "bg-block" };

/** What the editor is doing, from the same store the in-canvas panel reads, so the two never disagree. */
export function CanvasPanel() {
  const [ops, setOps] = useState<CanvasOp[]>([]);
  useEffect(() => subscribeOps(setOps), []);
  if (!ops.length) {
    return (
      <div className="py-6 text-center text-[12.5px] text-ink-3">
        Nothing from the canvas yet. Open a frame and segment, propagate or save, and the steps show here.
      </div>
    );
  }
  return (
    <div className="divide-y divide-line">
      {ops.slice().reverse().map((op) => (
        <div key={op.id} className="flex items-center gap-2 py-1.5 text-[12px]">
          <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${DOT[op.status]}
                            ${op.status === "running" ? "running-dot" : ""}`} />
          <span className="text-ink-2 truncate">{op.label}</span>
          <span className="ml-auto shrink-0 text-ink-3">
            {(((op.endedAt ?? Date.now()) - op.startedAt) / 1000).toFixed(1)}s
          </span>
          {op.detail && (
            <span className={`shrink-0 max-w-[160px] truncate ${
              op.status === "failed" ? "text-block" : "text-ink-3"}`} title={op.detail}>{op.detail}</span>
          )}
        </div>
      ))}
    </div>
  );
}
