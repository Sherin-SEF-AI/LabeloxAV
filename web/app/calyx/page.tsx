"use client";

// CALYX rig calibration timeline: the drift-over-time for a rig, so a slowly loosening mount is caught before
// it blocks. Plots the SE(3) drift rotation and translation across the rig's sessions; the alert row lists the
// sessions CALYX flagged or blocked. Reuses the shared LineChart. Operational Materialism.

import { useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import LineChart from "@/components/charts/LineChart";
import EmptyState from "@/components/ui/EmptyState";
import { api, apiGet } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";
import { ago, humanize, shortId } from "@/lib/format";

type Point = { session_id: string; created_at: string | null; severity: string; rotation_deg: number; translation_m: number };
const SEV: Record<string, string> = { ok: "text-pass", drift_detected: "text-warn", block: "text-block" };
const SEV_LABEL: Record<string, string> = { ok: "Within tolerance", drift_detected: "Drifting", block: "Blocked" };

export default function CalyxTimeline() {
  const [vehicle, setVehicle] = useState("TIGOR-07");
  const [timeline, setTimeline] = useState<Point[]>([]);
  const [loadErr, setLoadErr] = useState<unknown>(null);
  const [shown, setShown] = useState<string | null>(null);
  // The vehicles the corpus already knows, offered as suggestions so nobody has to remember a rig's id.
  const sessionsQ = useQuery(["sessions"], () => api.sessions());
  const knownVehicles = Array.from(new Set((sessionsQ.data ?? []).map((s) => s.vehicle_id).filter(Boolean))).sort();

  const load = useCallback(async () => {
    setLoadErr(null);
    try {
        const r = await apiGet<{ timeline?: Point[] }>(`/api/calyx/rig/${encodeURIComponent(vehicle)}/history`);
        setTimeline(r.timeline ?? []);
        setShown(vehicle);
    } catch (e) {
      // Without this the page kept rendering an empty shell after a
      // failed request, which is indistinguishable from having no data.
      setLoadErr(e);
    }
  }, [vehicle]);
  useEffect(() => { load(); }, [load]);

  const x = timeline.map((_, i) => i + 1);
  const flagged = timeline.filter((p) => p.severity !== "ok");

  return (
    <PageShell active="CALYX" title="Rig timeline"
      subtitle="How far each camera has moved on its mount, drive by drive, so a loosening mount is caught early"
      right={<span className="text-[12.5px] text-ink-3">{timeline.length} drives · {flagged.length} flagged</span>}>
      <div className="page-wrap-narrow space-y-4">
        <form className="panel p-3.5 flex flex-wrap items-end gap-3"
          onSubmit={(e) => { e.preventDefault(); void load(); }}>
          <label className="flex flex-col gap-1">
            <span className="text-[12px] text-ink-2">Vehicle</span>
            <input value={vehicle} onChange={(e) => setVehicle(e.target.value)} list="calyx-vehicles"
              className="field w-56" />
            <datalist id="calyx-vehicles">
              {knownVehicles.map((v) => <option key={v} value={v} />)}
            </datalist>
          </label>
          <button type="submit" className="btn btn-primary">Show timeline</button>
        </form>

        {loadErr != null && <LoadState error={loadErr} onRetry={() => void load()} />}

        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Camera movement over time</h2>
            <span className="text-[12px] text-ink-3">Rotation in degrees, shift in metres</span>
          </div>
          <div className="p-3.5">
            {timeline.length ? (
              <LineChart x={x} height={180} series={[
                { key: "rot", label: "rotation (degrees)", color: "#FF7A2F", values: timeline.map((p) => p.rotation_deg) },
                { key: "trans", label: "shift (metres)", color: "#58A6FF", values: timeline.map((p) => p.translation_m) },
              ]} />
            ) : (
              <EmptyState compact icon="ruler" title={`No history for ${shown ?? vehicle}`}>
                A point is added each time one of this vehicle&apos;s drives has its calibration validated. Check the
                vehicle name, or validate a drive under Calibration, Camera validation.
              </EmptyState>
            )}
          </div>
        </section>

        {flagged.length > 0 && (
          <section className="panel">
            <div className="panel-head"><h2 className="section-title">Flagged drives</h2></div>
            <table className="data-table">
              <thead><tr><th>Drive</th><th>When</th><th>Result</th><th className="num">Rotation</th><th className="num">Shift</th></tr></thead>
              <tbody>
                {flagged.map((p) => (
                  <tr key={p.session_id}>
                    <td className="font-mono text-ink-2">{shortId(p.session_id)}</td>
                    <td className="text-ink-3">{p.created_at ? ago(p.created_at) : "-"}</td>
                    <td className={SEV[p.severity]}>{SEV_LABEL[p.severity] ?? humanize(p.severity)}</td>
                    <td className="num">{p.rotation_deg.toFixed(2)}&deg;</td>
                    <td className="num">{p.translation_m.toFixed(3)} m</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}
      </div>
    </PageShell>
  );
}
