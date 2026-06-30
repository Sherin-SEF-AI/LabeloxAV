"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import type { EgoSample, EgoState, InertialEvents, SessionRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";

// M-IMU.3: the inertial timeline. The derived ego-state (speed, longitudinal + lateral accel, yaw rate,
// jerk) plotted over the clip, with tagged events as shaded windows, anomaly pre-marks as dots, and the
// maneuver segmentation as a band. The signal is derived until a measured IMU is ingested.

const METRICS: { key: keyof EgoSample; label: string; color: string }[] = [
  { key: "speed_mps", label: "speed", color: "#58A6FF" },
  { key: "long_accel", label: "long a", color: "#FF7A2F" },
  { key: "lat_accel", label: "lat a", color: "#8B5CF6" },
  { key: "yaw_rate", label: "yaw rate", color: "#56D364" },
  { key: "jerk", label: "jerk", color: "#E3B341" },
];
const EVENT_COLOR: Record<string, string> = { hard_brake: "#F85149", hard_accel: "#FF7A2F", swerve: "#8B5CF6", impact: "#E3B341" };
const MANEUVER_COLOR: Record<string, string> = { stationary: "#3a3f47", turn: "#8B5CF6", brake: "#F85149", accelerate: "#FF7A2F", cruise: "#2a2e35" };

const W = 1000, H = 280, PAD = 8, BAND_H = 16;

export default function InertialPage() {
  const [sessions, setSessions] = useState<SessionRow[]>([]);
  const [pick, setPick] = useState("");
  const [ego, setEgo] = useState<EgoState | null>(null);
  const [ev, setEv] = useState<InertialEvents | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => { api.sessions().then((s) => { setSessions(s); if (s[0]) setPick(s[0].session_id); }); }, []);
  const load = useCallback(async (sid: string) => {
    if (!sid) return;
    setBusy(true);
    try {
      const [e, v] = await Promise.all([api.egoState(sid).catch(() => null), api.inertialEvents(sid).catch(() => null)]);
      setEgo(e); setEv(v);
    } finally { setBusy(false); }
  }, []);
  useEffect(() => { load(pick); }, [pick, load]);

  const series = ego?.series ?? [];
  const t0 = series[0]?.ts_ns ?? 0;
  const span = Math.max(1, (series[series.length - 1]?.ts_ns ?? 1) - t0);
  const xOf = useCallback((ts: number) => PAD + ((ts - t0) / span) * (W - 2 * PAD), [t0, span]);

  const lines = useMemo(() => METRICS.map((m) => {
    const present = series.map((s) => s[m.key] as number | null).filter((v): v is number => v != null);
    if (!present.length) return { ...m, d: "" };
    const lo = Math.min(...present), hi = Math.max(...present), rng = hi - lo || 1;
    const top = PAD, bot = H - BAND_H - PAD;
    let d = "", pen = false;
    series.forEach((s) => {
      const v = s[m.key] as number | null;
      if (v == null) { pen = false; return; }
      const x = PAD + ((s.ts_ns - t0) / span) * (W - 2 * PAD);
      const y = bot - ((v - lo) / rng) * (bot - top);
      d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)} `; pen = true;
    });
    return { ...m, d };
  }), [series, t0, span]);

  return (
    <PageShell active="INERTIAL" title="Inertial">
      <div className="p-4 space-y-4 font-mono text-[11px]">
        <div className="flex items-center gap-3">
          <select value={pick} onChange={(e) => setPick(e.target.value)} className="bg-bg border border-line px-2 py-1 text-ink">
            {sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.vehicle_id} / {s.session_id.slice(0, 8)}</option>)}
          </select>
          {ego && <span className="text-ink-3">{ego.source} | {ego.n_samples} samples | {ego.n_with_motion} with motion</span>}
          {busy && <span className="text-ink-3">loading...</span>}
        </div>

        <div className="panel p-2">
          <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 300 }}>
            {ev?.maneuvers.map((m, i) => (
              <rect key={`m${i}`} x={xOf(m.t_in_ns)} y={H - BAND_H} width={Math.max(1, xOf(m.t_out_ns) - xOf(m.t_in_ns))} height={BAND_H}
                fill={MANEUVER_COLOR[m.kind] ?? "#2a2e35"} opacity={0.85}><title>{m.kind}</title></rect>
            ))}
            {ev?.events.map((e, i) => (
              <rect key={`e${i}`} x={xOf(e.t_in_ns)} y={PAD} width={Math.max(2, xOf(e.t_out_ns) - xOf(e.t_in_ns))} height={H - BAND_H - 2 * PAD}
                fill={EVENT_COLOR[e.kind] ?? "#888"} opacity={0.13}><title>{`${e.kind} sev ${e.severity}`}</title></rect>
            ))}
            {lines.map((l) => l.d && <path key={l.key as string} d={l.d} fill="none" stroke={l.color} strokeWidth={1.2} opacity={0.9} />)}
            {ev?.anomalies.map((a, i) => (
              <circle key={`a${i}`} cx={xOf(a.ts_ns)} cy={PAD + 4} r={3} fill="#E3B341" stroke="#0B0C0E"><title>{`${a.metric} z${a.z}`}</title></circle>
            ))}
          </svg>
          <div className="flex flex-wrap gap-3 px-1 pt-1">
            {METRICS.map((m) => <span key={m.key as string} className="flex items-center gap-1"><span className="w-3 h-[2px] inline-block" style={{ background: m.color }} /><span className="text-ink-3">{m.label}</span></span>)}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div className="panel p-2 space-y-1">
            <div className="uppercase text-ink-3 border-b hairline pb-1">events ({ev?.events.length ?? 0}) + anomalies ({ev?.anomalies.length ?? 0})</div>
            {ev?.events.map((e, i) => (
              <div key={`ev${i}`} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: EVENT_COLOR[e.kind] }} />
                <span className="text-ink-2 w-20">{e.kind}</span>
                <span className="text-ink-3">peak {e.peak}</span>
                <span className="ml-auto text-ink-3">sev {e.severity}</span>
              </div>
            ))}
            {ev?.anomalies.map((a, i) => (
              <div key={`an${i}`} className="flex items-center gap-2 text-ink-3">
                <span className="w-2 h-2 rounded-full bg-warn" /><span className="w-20">anomaly</span><span>{a.metric} {a.value}</span><span className="ml-auto">z {a.z} ({a.status})</span>
              </div>
            ))}
            {!ev?.events.length && !ev?.anomalies.length && <div className="text-ink-3 text-center py-3">no events (a smooth segment, or a session without GNSS)</div>}
          </div>
          <div className="panel p-2 space-y-1">
            <div className="uppercase text-ink-3 border-b hairline pb-1">maneuvers ({ev?.maneuvers.length ?? 0})</div>
            {ev?.maneuvers.map((m, i) => (
              <div key={`mv${i}`} className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full" style={{ background: MANEUVER_COLOR[m.kind] }} />
                <span className="text-ink-2">{m.kind}</span>
                <span className="ml-auto text-ink-3">{((m.t_out_ns - m.t_in_ns) / 1e9).toFixed(1)}s</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </PageShell>
  );
}
