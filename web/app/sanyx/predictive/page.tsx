"use client";

// M10 SANYX predictive maintenance: catch a rig degrading before it fails. The ingest board lists sessions
// with their health; picking a vehicle pulls its per-component health trend across sessions and any alerts on
// components trending toward failure. Operational Materialism: an alert is amber (warn) or red (critical); a
// steady component is neutral.

import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, Bar } from "@/components/engine/prim";
import { getJSON } from "@/lib/engine";

type BoardRow = { session_id: string; vehicle_id: string | null; city: string | null; score: number | null;
  decision: string | null; root_cause: unknown };
type Alert = { component: string; metric: string; trend: string; severity: string; evidence?: unknown };
type Trends = { vehicle_id: string; n_sessions: number; alerts: Alert[]; series: Record<string, number[]> };

const DEC_TONE: Record<string, string> = { pass: "text-pass", degraded: "text-warn", quarantine: "text-block" };

export default function PredictivePage() {
  const [rows, setRows] = useState<BoardRow[]>([]);
  const [vehicle, setVehicle] = useState<string | null>(null);
  const [trends, setTrends] = useState<Trends | null>(null);

  useEffect(() => {
    getJSON<{ sessions: BoardRow[] }>("/api/sanyx/board?limit=200").then((d) => {
      setRows(d.sessions);
      const first = d.sessions.find((r) => r.vehicle_id)?.vehicle_id ?? null;
      setVehicle(first);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (!vehicle) return;
    getJSON<Trends>(`/api/sanyx/rig/${encodeURIComponent(vehicle)}/trends`).then(setTrends).catch(() => setTrends(null));
  }, [vehicle]);

  const vehicles = Array.from(new Set(rows.map((r) => r.vehicle_id).filter(Boolean))) as string[];

  return (
    <PageShell active="SANYX" title="Predictive maintenance"
      subtitle="per-component health trends and pre-failure alerts across a vehicle's sessions"
      right={<span className="font-mono text-[11px] text-ink-3">{vehicles.length} vehicles</span>}>
      <div className="p-4 grid grid-cols-1 lg:grid-cols-[240px_1fr] gap-4 max-w-6xl">
        {/* vehicle picker */}
        <Panel title="vehicles">
          <div className="space-y-0.5">
            {vehicles.map((v) => (
              <button key={v} onClick={() => setVehicle(v)}
                className={`block w-full text-left font-mono text-[11px] px-1.5 py-1 hover:bg-bg-2 ${
                  v === vehicle ? "text-accent" : "text-ink-2"}`}>{v}</button>
            ))}
            {!vehicles.length && <div className="font-mono text-[11px] text-ink-3">no vehicles on the board</div>}
          </div>
        </Panel>

        {/* trends + alerts */}
        <div className="space-y-4">
          <Panel title="pre-failure alerts" hint={trends ? `${trends.n_sessions} sessions` : ""}>
            {trends?.alerts.length ? (
              <div className="space-y-1">
                {trends.alerts.map((a, i) => (
                  <div key={i} className={`border p-2 ${a.severity === "critical" ? "border-block/40" : "border-warn/40"}`}>
                    <div className="flex items-center justify-between font-mono text-[11px]">
                      <span className="text-ink">{a.component} · {a.metric}</span>
                      <span className={a.severity === "critical" ? "text-block" : "text-warn"}>{a.severity}</span>
                    </div>
                    <div className="font-mono text-[10px] text-ink-3">trend: {a.trend}</div>
                  </div>
                ))}
              </div>
            ) : <div className="font-mono text-[11px] text-ink-3 py-2">no components trending toward failure</div>}
          </Panel>

          <Panel title="component health series" hint="oldest to newest session">
            {trends && Object.keys(trends.series).length ? (
              <div className="space-y-2">
                {Object.entries(trends.series).map(([name, vs]) => (
                  <div key={name}>
                    <div className="font-mono text-[10px] text-ink-2 mb-1">{name}</div>
                    <div className="flex items-end gap-0.5 h-10">
                      {vs.map((v, i) => (
                        <span key={i} title={String(v)}
                          className={`flex-1 ${v >= 0.8 ? "bg-pass" : v >= 0.6 ? "bg-warn" : "bg-block"}`}
                          style={{ height: `${Math.max(4, v * 100)}%` }} />
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            ) : <div className="font-mono text-[11px] text-ink-3 py-2">no SANYX health series for this vehicle yet</div>}
          </Panel>

          <Panel title="recent sessions" hint="ingest board">
            <div className="space-y-0.5">
              {rows.filter((r) => r.vehicle_id === vehicle).slice(0, 12).map((r) => (
                <div key={r.session_id} className="flex items-center gap-2 font-mono text-[10px]">
                  <span className="w-40 shrink-0 truncate text-ink-3">{r.session_id.slice(0, 8)} · {r.city ?? "-"}</span>
                  <Bar frac={r.score ?? 0} tone={(r.score ?? 0) >= 0.8 ? "pass" : (r.score ?? 0) >= 0.6 ? "warn" : "block"} />
                  <span className={`w-20 shrink-0 text-right ${DEC_TONE[r.decision ?? ""] ?? "text-ink-3"}`}>{r.decision ?? "-"}</span>
                </div>
              ))}
            </div>
          </Panel>
        </div>
      </div>
    </PageShell>
  );
}
