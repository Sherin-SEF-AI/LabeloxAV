"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { JobRow } from "@/lib/types";
import TopNav from "@/components/TopNav";

// Unified jobs dashboard: import, training, and autolabel jobs in one live stream. The single place
// to watch everything the engine is doing.

const STATUS_COLOR: Record<string, string> = {
  done: "text-pass",
  error: "text-block",
  canceled: "text-ink-3",
  running: "text-warn",
  pending: "text-info",
};

const KIND_COLOR: Record<string, string> = {
  import: "text-info",
  training: "text-accent",
  autolabel: "text-pass",
};

export default function JobsPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [kind, setKind] = useState<string>("all");

  useEffect(() => {
    const refresh = () => api.jobs().then(setJobs).catch(() => {});
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, []);

  const kinds = ["all", "import", "training", "autolabel"];
  const shown = kind === "all" ? jobs : jobs.filter((j) => j.kind === kind);
  const active = jobs.filter((j) => j.status === "running" || j.status === "pending").length;

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="JOBS" right={<span className="text-ink-3">{active} active</span>} />
      <main className="flex-1 overflow-auto p-4">
        <div className="flex items-center gap-1 mb-3 font-mono text-[11px]">
          {kinds.map((k) => (
            <button key={k} onClick={() => setKind(k)}
              className={`px-2 py-1 border ${kind === k ? "border-accent text-ink" : "border-line text-ink-3"}`}>
              {k}
            </button>
          ))}
        </div>

        <div className="panel">
          <div className="grid grid-cols-[80px_1fr_90px_1fr_140px_70px] gap-2 px-3 py-2 border-b hairline font-mono text-[10px] uppercase text-ink-3">
            <span>kind</span><span>label</span><span>status</span><span>progress</span><span>detail</span><span></span>
          </div>
          {shown.map((j) => (
            <div key={j.job_id} className="grid grid-cols-[80px_1fr_90px_1fr_140px_70px] gap-2 px-3 py-1.5 border-b hairline items-center font-mono text-[11px]">
              <span className={KIND_COLOR[j.kind] ?? "text-ink-3"}>{j.kind}</span>
              <span className="text-ink-2 truncate" title={j.label}>{j.label}</span>
              <span className={STATUS_COLOR[j.status] ?? "text-ink-3"}>{j.status}</span>
              <div className="flex items-center gap-2">
                <div className="flex-1 h-1.5 bg-line relative">
                  <div className="absolute left-0 top-0 h-full bg-accent" style={{ width: `${(j.progress || 0) * 100}%` }} />
                </div>
                <span className="text-ink-3 w-8 text-right">{Math.round((j.progress || 0) * 100)}%</span>
              </div>
              <span className="text-ink-3 truncate" title={j.error || j.detail}>
                {j.error ? <span className="text-block">{j.error.slice(0, 40)}</span> : j.detail}
              </span>
              <button onClick={() => router.push(j.link)} className="border border-line px-1.5 py-0.5 text-ink-3 hover:border-accent">open</button>
            </div>
          ))}
          {!shown.length && <div className="px-3 py-6 text-center font-mono text-xs text-ink-3">no jobs</div>}
        </div>
      </main>
    </div>
  );
}
