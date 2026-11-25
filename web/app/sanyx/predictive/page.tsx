"use client";

// M10 SANYX predictive maintenance: catch a rig degrading before it fails. The ingest board lists sessions
// with their health; picking a vehicle pulls its per-component health trend across sessions and any alerts on
// components trending toward failure. Operational Materialism: an alert is amber (warn) or red (critical); a
// steady component is neutral.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, Bar } from "@/components/engine/prim";
import { getJSON } from "@/lib/engine";
import LoadState from "@/components/shell/LoadState";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, num, shortId } from "@/lib/format";

type BoardRow = { session_id: string; vehicle_id: string | null; city: string | null; score: number | null;
  decision: string | null; root_cause: unknown };
type Alert = { component: string; metric: string; trend: string; severity: string; evidence?: unknown };
type Trends = { vehicle_id: string; n_sessions: number; alerts: Alert[]; series: Record<string, number[]> };

const DEC_TONE: Record<string, string> = { pass: "text-pass", degraded: "text-warn", quarantine: "text-block" };
const DEC_LABEL: Record<string, string> = { pass: "Healthy", degraded: "Degraded", quarantine: "Quarantined" };

export default function PredictivePage() {
  const [rows, setRows] = useState<BoardRow[]>([]);
  const [vehicle, setVehicle] = useState<string | null>(null);
  const [trends, setTrends] = useState<Trends | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [err, setErr] = useState<unknown>(null);

  const loadBoard = () => {
    setErr(null);
    getJSON<{ sessions: BoardRow[] }>("/api/sanyx/board?limit=200").then((d) => {
      setRows(d.sessions);
      const first = d.sessions.find((r) => r.vehicle_id)?.vehicle_id ?? null;
      setVehicle(first);
    }).catch(setErr).finally(() => setLoaded(true));
  };
  useEffect(() => { loadBoard(); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!vehicle) return;
    getJSON<Trends>(`/api/sanyx/rig/${encodeURIComponent(vehicle)}/trends`).then(setTrends).catch(() => setTrends(null));
  }, [vehicle]);

  const vehicles = Array.from(new Set(rows.map((r) => r.vehicle_id).filter(Boolean))) as string[];

  const recent = rows.filter((r) => r.vehicle_id === vehicle).slice(0, 12);

  return (
    <PageShell active="SANYX" title="Predictive"
      subtitle="Catch a camera or sensor wearing out across a vehicle's drives, before it fails"
      right={<span className="text-[12.5px] text-ink-3">{num(vehicles.length)} vehicles</span>}>
      <div className="page-wrap space-y-4">
        {err != null && <LoadState error={err} onRetry={loadBoard} />}
        {!loaded ? <LoadState loading rows={5} /> : !vehicles.length ? (
          err == null && <LoadState empty emptyIcon="activity" emptyTitle="No vehicles yet"
            emptyText="Once recordings have been through the health checks, each vehicle's sensor health shows up here." />
        ) : (
      <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr] gap-4">
        {/* vehicle picker */}
        <Panel title="Vehicles">
          <div className="space-y-0.5 -mx-1.5" role="listbox" aria-label="Vehicles">
            {vehicles.map((v) => (
              <button key={v} type="button" role="option" aria-selected={v === vehicle} onClick={() => setVehicle(v)}
                className={`block w-full text-left text-[12.5px] px-2 py-1.5 rounded ${
                  v === vehicle ? "bg-accent/15 text-ink" : "text-ink-2 hover:bg-head"}`}>{v}</button>
            ))}
          </div>
        </Panel>

        {/* trends + alerts */}
        <div className="space-y-4 min-w-0">
          <Panel title="Early warnings" hint={trends?.n_sessions ? `Across ${num(trends.n_sessions)} drives` : undefined}>
            {trends?.alerts.length ? (
              <div className="space-y-2">
                {trends.alerts.map((a, i) => (
                  <div key={i} className={`rounded border p-2.5 ${a.severity === "critical" ? "border-block/40" : "border-warn/40"}`}>
                    <div className="flex items-center justify-between text-[12.5px]">
                      <span className="text-ink">{humanize(a.component)} · {humanize(a.metric).toLowerCase()}</span>
                      <span className={a.severity === "critical" ? "text-block" : "text-warn"}>{humanize(a.severity)}</span>
                    </div>
                    <div className="text-[12px] text-ink-3 mt-0.5">Trend: {humanize(a.trend).toLowerCase()}</div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState compact icon="check" title="Nothing wearing out">
                No sensor on {vehicle ?? "this vehicle"} is trending toward failure.
              </EmptyState>
            )}
          </Panel>

          <Panel title="Sensor health over time" hint="Oldest drive on the left">
            {trends && Object.keys(trends.series).length ? (
              <div className="space-y-3">
                {Object.entries(trends.series).map(([name, vs]) => (
                  <div key={name}>
                    <div className="text-[12px] text-ink-2 mb-1">{humanize(name)}</div>
                    <div className="flex items-end gap-0.5 h-10">
                      {vs.map((v, i) => (
                        <span key={i} data-tip={`Health ${v.toFixed(2)}`}
                          className={`flex-1 rounded-t-sm ${v >= 0.8 ? "bg-pass" : v >= 0.6 ? "bg-warn" : "bg-block"}`}
                          style={{ height: `${Math.max(4, v * 100)}%` }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState compact icon="chart" title="No health history yet">
                Run the health checks on this vehicle&apos;s recordings (Ingest health, Health board) and the trend builds up here.
              </EmptyState>
            )}
          </Panel>

          <Panel title="Recent drives" hint="From the health board">
            {recent.length ? (
              <div className="space-y-1.5">
                {recent.map((r) => (
                  <div key={r.session_id} className="flex items-center gap-3 text-[12px]">
                    <span className="w-40 shrink-0 truncate text-ink-3"><span className="font-mono">{shortId(r.session_id)}</span> · {r.city ?? "-"}</span>
                    <Bar frac={r.score ?? 0} tone={(r.score ?? 0) >= 0.8 ? "pass" : (r.score ?? 0) >= 0.6 ? "warn" : "block"} />
                    <span className={`w-24 shrink-0 text-right ${DEC_TONE[r.decision ?? ""] ?? "text-ink-3"}`}>
                      {r.decision ? (DEC_LABEL[r.decision] ?? humanize(r.decision)) : "Not checked"}
                    </span>
                  </div>
                ))}
              </div>
            ) : <div className="text-[12.5px] text-ink-3">No drives for this vehicle on the board.</div>}
          </Panel>
        </div>
      </div>
        )}
      </div>
    </PageShell>
  );
}
