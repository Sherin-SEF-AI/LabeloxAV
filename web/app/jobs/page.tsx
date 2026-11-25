"use client";

import { useEffect, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { JobRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import { StateBadge } from "@/components/StateBadge";
import { Spinner } from "@/components/Spinner";
import LoadState from "@/components/shell/LoadState";
import { ago, humanize, num, pct, plural } from "@/lib/format";
import type { SessionRow } from "@/lib/types";
import { useJobStream } from "@/lib/useEventStream";
import { canCancel, cancelPath, jobCounts, jobKinds } from "@/lib/jobActions";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";

// Unified jobs dashboard: import, training, and autolabel jobs in one live stream. The single place
// to watch everything the engine is doing.

// Kind codes as names. Only the label changes: the filter still compares the code.
const KIND_NAMES: Record<string, string> = { map_fusion: "Map fusion", autolabel: "Autolabel", import: "Import",
  training: "Training", export: "Export", relabel: "Relabel" };
const kindName = (k: string) => (k === "all" ? "All" : KIND_NAMES[k] ?? humanize(k));

// The API spells one finished state several ways (map fusion says "completed", the rest "done"). The badge
// shows them as one word so the column reads as one vocabulary; the row's own status is left as it came.
const DONE_ALIASES = new Set(["completed", "complete", "succeeded", "success"]);
const shownStatus = (s: string) => (DONE_ALIASES.has(s) ? "done" : s);

// The server's detail strings are compact storage forms ("12fr / 0obj", "ep 15/15", "40 obj -> abc").
// Read them back as words; anything not recognised is shown as it came.
function readDetail(d: string): string {
  if (!d) return "";
  const fo = d.match(/^(\d+)fr \/ (\d+)obj$/);
  if (fo) return `${plural(Number(fo[1]), "frame")} · ${plural(Number(fo[2]), "object")}`;
  const ep = d.match(/^ep (\d+)\/(\d+)$/);
  if (ep) return `Epoch ${ep[1]} of ${ep[2]}`;
  const ob = d.match(/^(\d+) obj(.*)$/);
  if (ob) return `${plural(Number(ob[1]), "object")}${ob[2].replace(/->/g, "→")}`;
  const t = d.replace(/->/g, "→").trim();
  return t ? t[0].toUpperCase() + t.slice(1) : t;
}

// An autolabel job is labelled with the first eight characters of its session id. When the session list
// knows that session, the job is named after its vehicle instead, and the id stays as secondary meta.
const isShortId = (s: string) => /^[0-9a-f]{8}$/.test(s);

export default function JobsPage() {
  const router = useRouter();
  // The REST list is cached, so coming back to this page paints the last table at once and refreshes
  // behind it. It is copied into state rather than read straight from the cache because the stream below
  // merges onto the rendered rows, and a fresh list has to replace that merge, not be overridden by it.
  const q = useQuery(["jobs"], () => api.jobs());
  const [jobs, setJobs] = useState<JobRow[]>(q.data ?? []);
  useEffect(() => { if (q.data) setJobs(q.data); }, [q.data]);
  const [kind, setKind] = useState<string>("all");
  // The same cached session list the queue and calibration pages read; only used to name rows.
  const sessionsQ = useQuery(["sessions"], () => api.sessions());
  const sessionByPrefix = new Map<string, SessionRow>();
  for (const s of sessionsQ.data ?? []) sessionByPrefix.set(s.session_id.slice(0, 8), s);
  const [busy, setBusy] = useState<string | null>(null);

  // Live via server-sent events. This page used to re-fetch every two seconds forever, whether or not
  // anything had changed and whether or not the tab was even visible. The stream pushes only on change; the
  // one-off fetch below still runs so the table renders immediately on a cold load and so the page still
  // works if the stream cannot connect.
  const { data: stream, connected } = useJobStream();

  // Loading until the list settles either way, or the stream has spoken first.
  const loading = q.loading && !stream;

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
  }, [stream]);

  // Derived from the rows that arrived rather than from a hardcoded list, which is how export, map_fusion
  // and relabel came to be visible under "all" and impossible to filter to.
  const kinds = jobKinds(jobs);
  const shown = kind === "all" ? jobs : jobs.filter((j) => j.kind === kind);
  const counts = jobCounts(jobs);

  const cancel = async (j: JobRow) => {
    setBusy(j.job_id);
    try {
      const r = await api.cancelJob(cancelPath(j));
      toast(r.detail || "Cancelled");
      await q.reload();
      // Home counts queued and running jobs, and would otherwise show the one just stopped.
      void invalidate("/api/home");
    } catch (e) {
      toast(`Could not cancel: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(null);
    }
  };

  const filters = (
    <div className="flex items-center gap-1.5" role="group" aria-label="Filter by kind">
      {kinds.map((k) => {
        const n = k === "all" ? jobs.length : jobs.filter((j) => j.kind === k).length;
        return (
          <button key={k} onClick={() => setKind(k)} aria-pressed={kind === k}
            className={`chip ${kind === k ? "chip-on" : ""}`}>
            {kindName(k)}
            <span className="text-ink-3 tabular-nums">{num(n)}</span>
          </button>
        );
      })}
    </div>
  );

  return (
    <PageShell
      active="JOBS"
      title="Jobs"
      subtitle="Imports, autolabelling, training and exports, live."
      right={loading ? <Spinner label="loading jobs" /> : (
        <span className="flex items-center gap-2 text-[12px] text-ink-2">
          {/* Say whether the page is live. A silently dead stream looks identical to an idle system, so a
              stalled dot is the difference between "nothing is happening" and "you are not being told". */}
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-pass" : "bg-ink-3"}`}
                title={connected ? "live" : "not connected; showing last known state"} />
          {/* Three numbers rather than one called "active". This deployment parks 67 auto-label jobs for a
              GPU pod that is not provisioned, and reporting 68 active on an idle machine is how a status
              line stops being read. */}
          <span>{counts.running} running</span>
          {counts.queued > 0 && <span className="text-ink-3">{counts.queued} queued</span>}
          {counts.parked > 0 && (
            <span className="text-ink-3" title="parked for a GPU pod; nothing local will pick these up">
              {counts.parked} parked
            </span>
          )}
        </span>
      )}
      filters={filters}
    >
      <div className="page-wrap">
        <LoadState loading={loading && !shown.length} error={!jobs.length ? q.loadState.error : undefined}
          onRetry={q.loadState.onRetry} rows={8}
          empty={!shown.length}
          emptyIcon="server"
          emptyTitle={kind === "all" ? "No jobs yet" : `No ${kindName(kind).toLowerCase()} jobs`}
          emptyText={kind === "all"
            ? "Uploads, autolabel runs, training and exports appear here while they run and after they finish."
            : "Nothing of this kind has run recently. Pick All to see every job."}>
        <div className="panel overflow-x-auto">
          {/* Detail carries the fraction, progress does not. It used to be the other way round: a
              fixed-width bar held a full 1fr while the only explanation of a failure was squeezed into
              140px and truncated, so the one column that matters when something breaks was the one you
              could not read. */}
          <table className="data-table">
            <thead>
              <tr>
                <th className="text-left w-[110px]">Kind</th>
                <th className="text-left">Job</th>
                <th className="text-left w-[110px]">Status</th>
                <th className="text-left w-[150px]">Progress</th>
                <th className="text-left">Detail</th>
                <th className="text-left w-[110px]">Started</th>
                <th className="w-[130px]"><span className="sr-only">Actions</span></th>
              </tr>
            </thead>
            <tbody>
          {shown.map((j) => {
            const sess = j.kind === "autolabel" && isShortId(j.label) ? sessionByPrefix.get(j.label) : undefined;
            const status = shownStatus(j.status);
            const failed = status === "error" || status === "failed";
            return (
            <tr key={j.job_id}>
              <td><StateBadge state={j.kind} /></td>
              <td className="max-w-[22rem]">
                {sess ? (
                  <>
                    <div className="text-ink truncate" title={sess.session_id}>{sess.vehicle_id}</div>
                    <div className="text-[11.5px] text-ink-3 truncate">
                      {sess.city ? `${sess.city} · ` : ""}<span className="font-mono">{j.label}</span>
                    </div>
                  </>
                ) : (
                  <div className="truncate text-ink" title={j.label}>
                    {j.kind === "autolabel" && isShortId(j.label)
                      ? <>Session <span className="font-mono text-ink-2">{j.label}</span></>
                      : j.label}
                  </div>
                )}
              </td>
              <td><StateBadge state={status} /></td>
              <td>
                <span className="inline-flex items-center gap-2">
                  <span className="w-16 h-1.5 rounded-full bg-bg-2 overflow-hidden">
                    <span className={`block h-full ${failed ? "bg-block/70" : status === "done" ? "bg-pass" : "bg-accent"}`}
                      style={{ width: `${Math.min(1, j.progress || 0) * 100}%` }} />
                  </span>
                  <span className="tabular-nums text-[12px] w-9">{pct(j.progress || 0)}</span>
                </span>
              </td>
              {/* Wraps to two lines rather than truncating. An error sliced to 40 characters names the
                  exception and never the cause, which is the half a reader does not need. */}
              <td className="min-w-0" title={j.error || j.detail}>
                {j.error ? <span className="text-block line-clamp-2">{j.error}</span> : <span className="text-ink-2">{readDetail(j.detail)}</span>}
              </td>
              <td className="text-ink-3 whitespace-nowrap" title={j.created_at ?? undefined}>{ago(j.created_at)}</td>
              <td className="text-right whitespace-nowrap">
                <span className="inline-flex items-center gap-1.5">
                {canCancel(j) && (
                  <button onClick={() => cancel(j)} disabled={busy === j.job_id}
                    title="stop this job"
                    className="btn btn-danger">
                    {busy === j.job_id ? "Stopping..." : "Stop"}
                  </button>
                )}
                <button onClick={() => router.push(j.link)}
                  className="btn"><T k="action.open">Open</T></button>
                </span>
              </td>
            </tr>
            );
          })}
            </tbody>
          </table>
        </div>
        </LoadState>
      </div>
    </PageShell>
  );
}
