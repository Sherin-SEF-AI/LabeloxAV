"use client";

// CALYX rig calibration timeline: the drift-over-time for a rig, so a slowly loosening mount is caught before
// it blocks. Plots the SE(3) drift rotation and translation across the rig's sessions; the alert row lists the
// sessions CALYX flagged or blocked. Reuses the shared LineChart. Operational Materialism.

import { useCallback, useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import LineChart from "@/components/charts/LineChart";
import { apiGet } from "@/lib/api";

type Point = { session_id: string; created_at: string | null; severity: string; rotation_deg: number; translation_m: number };
const SEV: Record<string, string> = { ok: "text-pass", drift_detected: "text-warn", block: "text-block" };

export default function CalyxTimeline() {
  const [vehicle, setVehicle] = useState("TIGOR-07");
  const [timeline, setTimeline] = useState<Point[]>([]);
  const [loadErr, setLoadErr] = useState<unknown>(null);

  const load = useCallback(async () => {
    setLoadErr(null);
    try {
        const r = await apiGet<{ timeline?: Point[] }>(`/api/calyx/rig/${encodeURIComponent(vehicle)}/history`);
        setTimeline(r.timeline ?? []);
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
    <PageShell active="CALYX" title="CALYX rig calibration"
      right={<span className="font-mono text-[11px] text-ink-3">{timeline.length} sessions · {flagged.length} flagged</span>}>
      <div className="p-4 max-w-4xl space-y-4 font-mono text-[11px]">
      {loadErr != null && <LoadState error={loadErr} onRetry={() => void load()} />}
        <div className="flex items-end gap-2">
          <label className="flex flex-col gap-1"><span className="text-ink-3 text-[10px] uppercase">vehicle</span>
            <input value={vehicle} onChange={(e) => setVehicle(e.target.value)} className="bg-panel border border-line px-2 py-1 text-ink w-48" /></label>
          <button onClick={load} className="border border-line px-3 py-1 hover:border-accent">load timeline</button>
        </div>

        <section className="border border-line p-3">
          <div className="text-[10px] uppercase text-ink-3 mb-1">extrinsic drift over time</div>
          {timeline.length ? (
            <LineChart x={x} height={150} series={[
              { key: "rot", label: "rotation deg", color: "#FF7A2F", values: timeline.map((p) => p.rotation_deg) },
              { key: "trans", label: "translation m", color: "#58A6FF", values: timeline.map((p) => p.translation_m) },
            ]} />
          ) : <div className="text-ink-3 py-6 text-center">no drift history for this rig; CALYX records drift as sessions are validated</div>}
        </section>

        {flagged.length > 0 && (
          <section className="border border-line p-3">
            <div className="text-[10px] uppercase text-ink-3 mb-2">alerts</div>
            {flagged.map((p) => (
              <div key={p.session_id} className="flex items-center gap-3 mb-1">
                <span className="text-ink-2">{p.session_id.slice(0, 8)}</span>
                <span className={SEV[p.severity]}>{p.severity}</span>
                <span className="text-ink-3">rot {p.rotation_deg.toFixed(2)} deg · trans {p.translation_m.toFixed(3)} m</span>
              </div>
            ))}
          </section>
        )}
      </div>
    </PageShell>
  );
}
