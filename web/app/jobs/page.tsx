"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { JobRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { StateBadge, ConfBar } from "@/components/StateBadge";
import { Spinner, SkeletonRows } from "@/components/Spinner";

// Unified jobs dashboard: import, training, and autolabel jobs in one live stream. The single place
// to watch everything the engine is doing.

export default function JobsPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [kind, setKind] = useState<string>("all");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const refresh = () => api.jobs().then((j) => { setJobs(j); setLoading(false); }).catch(() => setLoading(false));
    refresh();
    const t = setInterval(refresh, 2000);
    return () => clearInterval(t);
  }, []);

  const kinds = ["all", "import", "training", "autolabel"];
  const shown = kind === "all" ? jobs : jobs.filter((j) => j.kind === kind);
  const active = jobs.filter((j) => j.status === "running" || j.status === "pending").length;

  const filters = (
    <div className="flex items-center gap-1 font-mono text-[11px]">
      {kinds.map((k) => (
        <button key={k} onClick={() => setKind(k)}
          className={`px-2 py-1 border ${kind === k ? "border-accent text-ink" : "border-line text-ink-3"}`}>
          {k}
        </button>
      ))}
    </div>
  );

  return (
    <PageShell
      active="JOBS"
      right={loading ? <Spinner label="loading jobs" /> : <span className="text-ink-3">{active} active</span>}
      filters={filters}
    >
      <div className="p-4">
        <div className="panel">
          <div className="grid grid-cols-[80px_1fr_90px_1fr_140px_70px] gap-2 px-3 py-2 border-b hairline font-mono text-[10px] uppercase text-ink-3">
            <span>kind</span><span>label</span><span>status</span><span>progress</span><span>detail</span><span></span>
          </div>
          {shown.map((j) => (
            <div key={j.job_id} className="grid grid-cols-[80px_1fr_90px_1fr_140px_70px] gap-2 px-3 py-1.5 border-b hairline items-center font-mono text-[11px]">
              <span><StateBadge state={j.kind} /></span>
              <span className="text-ink-2 truncate" title={j.label}>{j.label}</span>
              <span><StateBadge state={j.status} /></span>
              <div className="flex items-center gap-2">
                <ConfBar conf={j.progress || 0} />
              </div>
              <span className="text-ink-3 truncate" title={j.error || j.detail}>
                {j.error ? <span className="text-block">{j.error.slice(0, 40)}</span> : j.detail}
              </span>
              <button onClick={() => router.push(j.link)} className="border border-line px-1.5 py-0.5 text-ink-3 hover:border-accent">open</button>
            </div>
          ))}
          {loading && !shown.length
            ? <SkeletonRows rows={8} cols="grid-cols-[80px_1fr_90px_1fr_140px_70px]" />
            : !shown.length && <div className="px-3 py-6 text-center font-mono text-xs text-ink-3">no jobs</div>}
        </div>
      </div>
    </PageShell>
  );
}
