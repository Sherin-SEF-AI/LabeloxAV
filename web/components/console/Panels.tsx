"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { type CanvasOp, subscribeOps } from "@/lib/canvasOps";
import { level, mb, uptime, workInFlight } from "@/lib/console";
import type { AgentRunRow } from "@/lib/types";
import type { JobStream, SystemStream } from "@/lib/useEventStream";

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
      <div className="flex items-baseline justify-between font-mono text-[11px]">
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
      <div className="font-mono text-[11px] text-ink-3">
        no GPU visible to this process. Auto-labelling and training refuse without one.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div className="font-mono text-[11px] text-ink-3">{gpu.name}</div>
      <Meter label="memory" frac={gpu.memory_used_frac}
        detail={`${mb(gpu.memory_used_mb)} / ${mb(gpu.memory_total_mb)}`} />
      <Meter label="utilisation" frac={(gpu.utilization_pct ?? 0) / 100}
        detail={`${Math.round(gpu.utilization_pct ?? 0)}%`} />
      <div className="font-mono text-[11px] text-ink-3">
        {gpu.temperature_c != null && <span>{Math.round(gpu.temperature_c)}°C</span>}
        {gpu.power_w != null && <span> · {Math.round(gpu.power_w)} W</span>}
      </div>
      {/* The tenant list is the actionable part. "7.2 GB used" is not; "llama-server holds 7.2 GB" is the
          difference between waiting and killing something. */}
      {gpu.processes.length > 0 && (
        <div className="pt-2 border-t hairline space-y-0.5">
          {gpu.processes.map((p) => (
            <div key={`${p.pid}-${p.name}`} className="flex justify-between font-mono text-[10px]">
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
  if (!host) return <div className="font-mono text-[11px] text-ink-3">no reading</div>;
  return (
    <div className="space-y-3">
      <div className="font-mono text-[11px] text-ink-3">
        load {host.load["1m"]} over {host.cpu_count} cores
      </div>
      <Meter label="memory" frac={host.memory_used_frac}
        detail={`${mb(host.memory_used_mb)} / ${mb(host.memory_total_mb)}`} />
      <Meter label="cpu" frac={(host.cpu_pct ?? 0) / 100} detail={`${Math.round(host.cpu_pct)}%`} />
      <Meter label="disk" frac={host.disk.used_frac}
        detail={`${mb(host.disk.used_mb)} / ${mb(host.disk.total_mb)}`} />
      <div className="font-mono text-[10px] text-ink-3 truncate" title={host.disk.path}>{host.disk.path}</div>
    </div>
  );
}

export function ProcessPanel({ sys, waitingTotal }: { sys: SystemStream | null; waitingTotal: number }) {
  if (!sys) return <div className="font-mono text-[11px] text-ink-3">no reading</div>;
  return (
    <div className="font-mono text-[11px] space-y-1 text-ink-3">
      {/* Its own share, so "the machine is busy" can be told from "we are busy". */}
      <div className="flex justify-between"><span>pid</span><span className="text-ink-2">{sys.process.pid}</span></div>
      <div className="flex justify-between"><span>resident</span>
        <span className="text-ink-2">{mb(sys.process.rss_mb)}</span></div>
      <div className="flex justify-between"><span>threads</span>
        <span className="text-ink-2">{sys.process.threads}</span></div>
      <div className="flex justify-between"><span>uptime</span>
        <span className="text-ink-2">{uptime(sys.process.uptime_s)}</span></div>
      <div className="flex justify-between"><span>queued work</span>
        <span className={waitingTotal ? "text-warn" : "text-ink-2"}>{waitingTotal}</span></div>
    </div>
  );
}

export function RunningNowPanel({ jobs }: { jobs: JobStream | null }) {
  const { running, waitingTotal } = workInFlight(jobs);
  if (!running.length) {
    return (
      <div className="py-6 text-center font-mono text-[11px] text-ink-3">
        {waitingTotal
          ? `nothing running. ${waitingTotal} queued, mostly parked for hardware that is not here.`
          : "nothing running."}
      </div>
    );
  }
  return (
    <div className="divide-y divide-line">
      {running.map((w) => (
        <div key={w.id} className="flex items-center gap-3 py-2 font-mono text-[11px]">
          <span className="text-ink-2 w-20">{w.kind}</span>
          <span className="text-ink-3 w-20 truncate">{w.id.slice(0, 8)}</span>
          <span className="flex-1 h-1.5 bg-line rounded overflow-hidden">
            {/* Unmeasured progress shows no bar at all. A bar at zero reads as stalled, which is a
                different claim from "this job does not report progress". */}
            {w.progress != null && (
              <span className="block h-full bg-accent transition-[width] duration-700 ease-out"
                style={{ width: `${Math.round(w.progress * 100)}%` }} />
            )}
          </span>
          <span className="text-ink-3 w-12 text-right">
            {w.progress != null ? `${Math.round(w.progress * 100)}%` : "-"}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Agent runs, polled rather than streamed: a corpus sweep going for an hour is no more informative for
 *  being re-read every three seconds. */
export function useAgentRuns(enabled = true) {
  const [runs, setRuns] = useState<AgentRunRow[]>([]);
  useEffect(() => {
    if (!enabled) return;
    const load = () => api.agentRuns(12).then(setRuns).catch(() => {});
    load();
    const t = setInterval(load, 20_000);
    return () => clearInterval(t);
  }, [enabled]);
  return runs;
}

export function BackgroundPanel({ runs }: { runs: AgentRunRow[] }) {
  if (!runs.length) {
    return <div className="py-6 text-center font-mono text-[11px] text-ink-3">no agent runs recorded.</div>;
  }
  return (
    <div className="divide-y divide-line">
      {runs.slice(0, 10).map((r) => (
        <div key={r.run_id} className="flex items-center gap-3 py-1.5 font-mono text-[11px]">
          <span className="text-ink-2 w-32 truncate">{r.kind}</span>
          <span className={`w-24 ${r.status === "running" ? "text-accent"
            : r.status === "error" || r.status === "interrupted" ? "text-block" : "text-ink-3"}`}>
            {r.status}
          </span>
          <span className="text-ink-3 flex-1 truncate">
            {Object.entries(r.counts ?? {}).slice(0, 3).map(([k, v]) => `${k} ${v}`).join(" · ")}
          </span>
        </div>
      ))}
    </div>
  );
}

const DOT: Record<CanvasOp["status"], string> = { running: "bg-accent", ok: "bg-pass", failed: "bg-block" };

/** What the editor is doing, from the same store the in-canvas panel reads, so the two never disagree. */
export function CanvasPanel() {
  const [ops, setOps] = useState<CanvasOp[]>([]);
  useEffect(() => subscribeOps(setOps), []);
  if (!ops.length) {
    return (
      <div className="py-6 text-center font-mono text-[11px] text-ink-3">
        nothing from the canvas. Open a frame and segment, propagate or save.
      </div>
    );
  }
  return (
    <div className="divide-y divide-line">
      {ops.slice().reverse().map((op) => (
        <div key={op.id} className="flex items-center gap-2 py-1.5 font-mono text-[11px]">
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
