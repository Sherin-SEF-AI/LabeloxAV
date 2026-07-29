"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { JobRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { StateBadge, ConfBar } from "@/components/StateBadge";
import { Spinner, SkeletonRows } from "@/components/Spinner";
import { useJobStream } from "@/lib/useEventStream";

// Unified jobs dashboard: import, training, and autolabel jobs in one live stream. The single place
// to watch everything the engine is doing.

export default function JobsPage() {
  const router = useRouter();
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [kind, setKind] = useState<string>("all");
  const [loading, setLoading] = useState(true);

  // Live via server-sent events. This page used to re-fetch every two seconds forever, whether or not
  // anything had changed and whether or not the tab was even visible. The stream pushes only on change; the
  // one-off fetch below still runs so the table renders immediately on a cold load and so the page still
  // works if the stream cannot connect.
  const { data: stream, connected } = useJobStream();

  useEffect(() => {
    api.jobs().then((j) => { setJobs(j); setLoading(false); }).catch(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!stream) return;
    setJobs((prev) => {
      // The stream carries status and progress per job id; merge onto the rows already rendered rather than
      // replacing them, so the richer fields the REST shape provides are not lost on every tick.
      const live = new Map<string, { status: string; progress?: number | null }>();
      // Only the job-shaped keys. `ingest` rides the same stream and is a progress summary rather than a
      // list of jobs, so folding it in here would key the map on undefined.
      for (const key of ["training", "import", "export", "autolabel"] as const) {
        for (const r of stream[key] ?? []) live.set(r.job_id, { status: r.status, progress: r.progress });
      }
      return prev.map((j) => {
        const u = live.get(j.job_id);
        return u ? { ...j, status: u.status, progress: u.progress ?? j.progress } : j;
      });
    });
    setLoading(false);
  }, [stream]);

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
      title="Jobs"
      right={loading ? <Spinner label="loading jobs" /> : (
        <span className="flex items-center gap-2 text-ink-3">
          {/* Say whether the page is live. A silently dead stream looks identical to an idle system, so a
              stalled dot is the difference between "nothing is happening" and "you are not being told". */}
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-pass" : "bg-ink-3"}`}
                title={connected ? "live" : "not connected; showing last known state"} />
          <span>{active} active</span>
        </span>
      )}
      filters={filters}
    >
      <div className="p-4">
        <div className="panel">
          {/* Detail carries the fraction, progress does not. It used to be the other way round: a
              fixed-width bar held a full 1fr while the only explanation of a failure was squeezed into
              140px and truncated, so the one column that matters when something breaks was the one you
              could not read. */}
          <div className="grid grid-cols-[80px_minmax(0,14rem)_100px_120px_minmax(0,1fr)_70px] gap-2 px-3 py-2 border-b hairline font-mono text-[10px] uppercase text-ink-3">
            <span>kind</span><span>label</span><span>status</span><span>progress</span><span>detail</span><span></span>
          </div>
          {shown.map((j) => (
            <div key={j.job_id} className="grid grid-cols-[80px_minmax(0,14rem)_100px_120px_minmax(0,1fr)_70px] gap-2 px-3 py-2 border-b hairline items-center font-mono text-[11px]">
              <span><StateBadge state={j.kind} /></span>
              <span className="text-ink-2 truncate" title={j.label}>{j.label}</span>
              <span><StateBadge state={j.status} /></span>
              <div className="flex items-center gap-2">
                <ConfBar conf={j.progress || 0} />
              </div>
              {/* Wraps to two lines rather than truncating. An error sliced to 40 characters names the
                  exception and never the cause, which is the half a reader does not need. */}
              <span className="text-ink-3 leading-tight min-w-0" title={j.error || j.detail}>
                {j.error ? <span className="text-block line-clamp-2">{j.error}</span> : j.detail}
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
