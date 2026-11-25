"use client";

import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { Spinner } from "@/components/Spinner";
import { useQuery } from "@/lib/useQuery";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { Badge } from "@/components/ui/kit";
import { duration, plural, shortId } from "@/lib/format";

// The Session Inspector home: pick an MCAP session to inspect. The health verdict chip answers "did my
// recording work" before a single panel is opened.

type Row = { session_id: string; vehicle_id: string; city: string | null; start_ts_ns: number; end_ts_ns: number; verdict: string | null; has_mcap?: boolean; n_clouds?: number };

const V: Record<string, "pass" | "warn" | "block"> = { pass: "pass", warn: "warn", fail: "block" };

export default function InspectorHome() {
  const router = useRouter();
  // A failed read shows the empty list, as it always has; the cached list paints at once on return.
  const q = useQuery(["inspectorSessions"], () => api.inspectorSessions());
  const rows: Row[] = q.error ? [] : q.data ?? [];
  const loading = q.loading;

  return (
    <PageShell active="INSPECT" title="Session Inspector"
      subtitle="Play back a recording on one clock: camera, IMU, CAN, GPS, point clouds and raw messages, with labels on top."
      right={loading ? <Spinner label="Loading" /> : <span className="text-[12.5px] text-ink-3">{plural(rows.length, "session")}</span>}>
      <div className="page-wrap-narrow">
        {q.error ? (
          <LoadState error={q.error} onRetry={() => void q.reload()} />
        ) : loading && rows.length === 0 ? (
          <LoadState loading rows={8} />
        ) : rows.length === 0 ? (
          <div className="panel">
            <EmptyState icon="grid" title="No sessions to inspect yet"
              action={<a href="/import" className="btn btn-primary">Import a recording</a>}>
              A session appears here once it has an MCAP recording or point clouds. Import an .mcap file to add one.
            </EmptyState>
          </div>
        ) : (
          <section className="panel overflow-hidden">
            <div className="panel-head">
              <h2 className="section-title">Sessions</h2>
            </div>
            <ul className="divide-y divide-line/60">
              {rows.map((r) => {
                const secs = (r.end_ts_ns - r.start_ts_ns) / 1e9;
                const what = [
                  r.has_mcap ? "MCAP recording" : null,
                  r.n_clouds ? plural(r.n_clouds, "point cloud") : null,
                ].filter(Boolean).join(" · ");
                return (
                  <li key={r.session_id}>
                    <button onClick={() => router.push(`/inspect/${r.session_id}`)}
                      className="w-full flex items-center gap-4 px-4 py-3 hover:bg-head/60 text-left group">
                      <span className="min-w-0 flex-1">
                        <span className="flex items-baseline gap-2">
                          <span className="text-[13px] text-ink font-medium">{r.vehicle_id}</span>
                          {r.city && <span className="text-[12px] text-ink-3">{r.city}</span>}
                        </span>
                        <span className="block text-[11.5px] text-ink-3 mt-0.5 truncate">
                          {[what, Number.isFinite(secs) && secs > 0 ? duration(secs) : null].filter(Boolean).join(" · ")}
                          <span className="font-mono ml-2 text-ink-3/80">{shortId(r.session_id)}</span>
                        </span>
                      </span>
                      {r.verdict && (
                        <Badge state={V[r.verdict] ?? "muted"}>
                          {r.verdict === "fail" ? "Health check failed" : r.verdict === "warn" ? "Health warning" : "Healthy"}
                        </Badge>
                      )}
                      <span className="text-[12.5px] link shrink-0">Inspect &rarr;</span>
                    </button>
                  </li>
                );
              })}
            </ul>
          </section>
        )}
      </div>
    </PageShell>
  );
}
