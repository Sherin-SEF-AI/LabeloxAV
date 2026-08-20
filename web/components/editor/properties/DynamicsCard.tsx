"use client";

// The P3 derived-dynamics readout for the selected object: the planning and prediction signals computed
// from monocular inverse perspective mapping. Every row is an estimate and the footer says so, because a
// distance in metres printed without qualification reads as a measurement.

import type { ObjectDynamicsRow } from "@/lib/types";

export default function DynamicsCard({ row, onRecompute }: {
  row: ObjectDynamicsRow | undefined;
  onRecompute: () => void;
}) {
  const line = (label: string, val: string, cls = "text-ink-2") => (
    <div className="flex justify-between"><span className="text-ink-3">{label}</span><span className={cls}>{val}</span></div>
  );

  return (
    <>
      <div className="flex justify-end mb-1">
        <button onClick={onRecompute} title="compute distance/speed/heading/TTC/risk for this session"
          className="font-mono text-[10px] text-info hover:text-accent">recompute</button>
      </div>
      {!row ? (
        <div className="font-mono text-[10px] text-ink-3">no dynamics yet (save the object, then recompute)</div>
      ) : (
        <div className="font-mono text-[10px] space-y-0.5">
          {line("distance", row.distance_m != null ? `${row.distance_m} m` : "-")}
          {line("speed", row.speed_kmh != null ? `${row.speed_kmh} km/h` : "-")}
          {line("closing", row.closing_speed_kmh != null ? `${row.closing_speed_kmh} km/h` : "-")}
          {line("heading", row.heading_deg != null ? `${row.heading_deg}°` : "-")}
          {line("TTC", row.ttc_s != null ? `${row.ttc_s} s` : "-")}
          {line("risk", row.risk_level ?? "-",
            row.risk_level === "high" ? "text-block" : row.risk_level === "medium" ? "text-warn" : "text-pass")}
          {row.track_id && line("track", row.track_id.slice(0, 8))}
          <div className="text-ink-3 pt-0.5">estimate · IPM monocular</div>
        </div>
      )}
    </>
  );
}
