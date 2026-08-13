"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { AgentRunRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { headline, level, mb, uptime, workInFlight } from "@/lib/console";
import { useJobStream, useSystemStream } from "@/lib/useEventStream";

// One place that answers "is anything happening".
//
// It was three places before, and two of them were not in the application. Job state was on /jobs, the
// machine was in a terminal, and long-running background work was in a log file on the box. So the answer to
// "why is this slow" required a shell, which the people doing the labelling do not have.
//
// What goes on it is decided by what actually stalls work here: a GPU another tenant is holding, a disk
// filling up, work queued for hardware that is not present, and an agent run that died without saying so.
// Everything else is on a page of its own already and does not need repeating.

const TONE: Record<string, string> = {
  ok: "bg-accent", warn: "bg-warn", critical: "bg-block",
};
const TEXT: Record<string, string> = {
  ok: "text-ink-2", warn: "text-warn", critical: "text-block",
};

function Meter({ label, frac, detail }: { label: string; frac: number | null; detail: string }) {
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

export default function ConsolePage() {
  const router = useRouter();
  const { data: jobs, connected: jobsLive } = useJobStream();
  const { data: sys, connected: sysLive } = useSystemStream();
  const [runs, setRuns] = useState<AgentRunRow[]>([]);

  // Agent runs are not on a stream: they change on the scale of minutes, and a corpus sweep that has been
  // going for an hour is not more informative for being re-read every three seconds.
  useEffect(() => {
    const load = () => api.agentRuns(12).then(setRuns).catch(() => {});
    load();
    const t = setInterval(load, 20_000);
    return () => clearInterval(t);
  }, []);

  const { running, waitingTotal } = workInFlight(jobs);
  const gpu = sys?.gpus?.[0];
  const host = sys?.host;
  const live = jobsLive && sysLive;
  const activeRuns = runs.filter((r) => r.status === "running" || r.status === "planned");

  return (
    <PageShell
      active="CONSOLE"
      title="Console"
      subtitle={headline(jobs, sys)}
      right={
        <span className="flex items-center gap-2 font-mono text-[11px] text-ink-3">
          {/* A console whose numbers freeze looks exactly like a quiet machine, so it says which it is. */}
          <span className={`w-1.5 h-1.5 rounded-full ${live ? "bg-pass" : "bg-ink-3"}`} />
          <span>{live ? "live" : "not connected"}</span>
          {sys && <span>· up {uptime(sys.process.uptime_s)}</span>}
        </span>
      }
    >
      <div className="p-4 space-y-4">
        <div className="grid gap-4 md:grid-cols-3">
          {/* GPU */}
          <div className="panel p-3 space-y-3">
            <div className="flex items-baseline justify-between">
              <span className="text-ink font-medium text-sm">GPU</span>
              <span className="font-mono text-[11px] text-ink-3">{gpu?.name ?? "none visible"}</span>
            </div>
            {gpu ? (
              <>
                <Meter label="memory" frac={gpu.memory_used_frac}
                  detail={`${mb(gpu.memory_used_mb)} / ${mb(gpu.memory_total_mb)}`} />
                <Meter label="utilisation" frac={(gpu.utilization_pct ?? 0) / 100}
                  detail={`${Math.round(gpu.utilization_pct ?? 0)}%`} />
                <div className="font-mono text-[11px] text-ink-3">
                  {gpu.temperature_c != null && <span>{Math.round(gpu.temperature_c)}°C</span>}
                  {gpu.power_w != null && <span> · {Math.round(gpu.power_w)} W</span>}
                </div>
                {/* The tenant list is the actionable part. "7.2 GB used" is not; "llama-server holds 7.2 GB"
                    is the difference between waiting and killing something. */}
                {gpu.processes.length > 0 && (
                  <div className="pt-1 border-t hairline space-y-0.5">
                    {gpu.processes.map((p) => (
                      <div key={`${p.pid}-${p.name}`} className="flex justify-between font-mono text-[10px]">
                        <span className="text-ink-3 truncate" title={p.name}>
                          {p.name.split("/").pop()}
                        </span>
                        <span className="text-ink-2 shrink-0 ml-2">{mb(p.used_mb)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="font-mono text-[11px] text-ink-3">
                no GPU visible to this process. Auto-labelling and training refuse without one.
              </div>
            )}
          </div>

          {/* Host */}
          <div className="panel p-3 space-y-3">
            <div className="flex items-baseline justify-between">
              <span className="text-ink font-medium text-sm">Machine</span>
              {host && (
                <span className="font-mono text-[11px] text-ink-3">
                  load {host.load["1m"]} over {host.cpu_count} cores
                </span>
              )}
            </div>
            {host ? (
              <>
                <Meter label="memory" frac={host.memory_used_frac}
                  detail={`${mb(host.memory_used_mb)} / ${mb(host.memory_total_mb)}`} />
                <Meter label="cpu" frac={(host.cpu_pct ?? 0) / 100}
                  detail={`${Math.round(host.cpu_pct)}%`} />
                <Meter label="disk" frac={host.disk.used_frac}
                  detail={`${mb(host.disk.used_mb)} / ${mb(host.disk.total_mb)}`} />
                <div className="font-mono text-[10px] text-ink-3 truncate" title={host.disk.path}>
                  {host.disk.path}
                </div>
              </>
            ) : <div className="font-mono text-[11px] text-ink-3">no reading</div>}
          </div>

          {/* This process */}
          <div className="panel p-3 space-y-3">
            <div className="flex items-baseline justify-between">
              <span className="text-ink font-medium text-sm">API process</span>
              <span className="font-mono text-[11px] text-ink-3">pid {sys?.process.pid ?? "-"}</span>
            </div>
            {sys ? (
              <div className="font-mono text-[11px] space-y-1 text-ink-3">
                {/* Its own share, so "the machine is busy" can be told from "we are busy". */}
                <div className="flex justify-between"><span>resident</span>
                  <span className="text-ink-2">{mb(sys.process.rss_mb)}</span></div>
                <div className="flex justify-between"><span>threads</span>
                  <span className="text-ink-2">{sys.process.threads}</span></div>
                <div className="flex justify-between"><span>uptime</span>
                  <span className="text-ink-2">{uptime(sys.process.uptime_s)}</span></div>
                <div className="flex justify-between"><span>queued work</span>
                  <span className={waitingTotal ? "text-warn" : "text-ink-2"}>{waitingTotal}</span></div>
              </div>
            ) : <div className="font-mono text-[11px] text-ink-3">no reading</div>}
          </div>
        </div>

        {/* Work in flight */}
        <div className="panel">
          <div className="flex items-center gap-3 px-4 py-2 border-b hairline">
            <span className="text-ink font-medium text-sm">Running now</span>
            <span className="text-ink-3 text-xs">
              {running.length ? `${running.length} job${running.length === 1 ? "" : "s"}` : "nothing"}
            </span>
            <button onClick={() => router.push("/jobs")}
              className="ml-auto font-mono text-[10px] border border-line px-1.5 py-0.5 text-ink-3 hover:border-accent">
              all jobs
            </button>
          </div>
          {running.length ? (
            <div className="divide-y divide-line">
              {running.map((w) => (
                <div key={w.id} className="flex items-center gap-3 px-4 py-2 font-mono text-[11px]">
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
          ) : (
            <div className="px-4 py-6 text-center font-mono text-[11px] text-ink-3">
              {waitingTotal
                ? `nothing running. ${waitingTotal} queued, mostly parked for hardware that is not here.`
                : "nothing running."}
            </div>
          )}
        </div>

        {/* Background agent work, which is the part that used to live only in a log file. */}
        <div className="panel">
          <div className="flex items-center gap-3 px-4 py-2 border-b hairline">
            <span className="text-ink font-medium text-sm">Background</span>
            <span className="text-ink-3 text-xs">
              {activeRuns.length ? `${activeRuns.length} active` : "corpus sweeps, relabels, batches"}
            </span>
            <button onClick={() => router.push("/agent")}
              className="ml-auto font-mono text-[10px] border border-line px-1.5 py-0.5 text-ink-3 hover:border-accent">
              agent console
            </button>
          </div>
          {runs.length ? (
            <div className="divide-y divide-line">
              {runs.slice(0, 8).map((r) => (
                <div key={r.run_id} className="flex items-center gap-3 px-4 py-1.5 font-mono text-[11px]">
                  <span className="text-ink-2 w-32 truncate">{r.kind}</span>
                  <span className={`w-24 ${r.status === "running" ? "text-accent"
                    : r.status === "error" || r.status === "interrupted" ? "text-block" : "text-ink-3"}`}>
                    {r.status}
                  </span>
                  <span className="text-ink-3 flex-1 truncate">
                    {Object.entries(r.counts ?? {}).slice(0, 3)
                      .map(([k, v]) => `${k} ${v}`).join(" · ")}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div className="px-4 py-6 text-center font-mono text-[11px] text-ink-3">
              no agent runs recorded.
            </div>
          )}
        </div>
      </div>
    </PageShell>
  );
}
