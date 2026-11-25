"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";
import type { EgoSample, EgoState, InertialEvents, SessionRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import SessionSelect from "@/components/shell/SessionSelect";
import EmptyState from "@/components/ui/EmptyState";
import { Badge } from "@/components/ui/kit";
import { duration, humanize, num } from "@/lib/format";

// Ego motion: the inertial timeline. The derived ego-state (speed, longitudinal + lateral accel, yaw rate,
// jerk) plotted over the clip, with tagged events as shaded windows, anomaly pre-marks as dots, and the
// maneuver segmentation as a band. The signal is derived until a measured IMU is ingested.

// Series colours come from the theme tokens, so the chart reads the same in the light palette.
const tok = (name: string) => `rgb(var(--c-${name}))`;
const METRICS: { key: keyof EgoSample; label: string; color: string }[] = [
  { key: "speed_mps", label: "Speed", color: tok("accent") },
  { key: "long_accel", label: "Braking and acceleration", color: "#FF7A2F" },
  { key: "lat_accel", label: "Sideways acceleration", color: "#8B5CF6" },
  { key: "yaw_rate", label: "Turn rate", color: tok("pass") },
  { key: "jerk", label: "Jerk", color: tok("warn") },
];
const EVENT_COLOR: Record<string, string> = { hard_brake: "#F85149", hard_accel: "#FF7A2F", swerve: "#8B5CF6", impact: "#E3B341" };
const MANEUVER_COLOR: Record<string, string> = { stationary: "#4B5260", turn: "#8B5CF6", brake: "#F85149", accelerate: "#FF7A2F", cruise: "#3F6E9E" };

const W = 1000, H = 280, PAD = 8, BAND_H = 16;

const NO_SESSIONS: SessionRow[] = [];

export default function InertialPage() {
  const sessionsQ = useQuery(["sessions"], () => api.sessions());
  const sessions: SessionRow[] = sessionsQ.data ?? NO_SESSIONS;
  const [pick, setPick] = useState("");
  // Start on the newest session, without overriding one already picked when the list refreshes.
  useEffect(() => {
    if (sessions[0]) setPick((cur) => cur || sessions[0].session_id);
  }, [sessions]);
  // Either read failing leaves that half absent rather than failing the page, as before. keepPrevious:
  // the last session's traces stay up while the next one loads.
  const egoQ = useQuery(pick ? ["egoState", pick] : null, () => api.egoState(pick), { keepPrevious: true });
  const evQ = useQuery(pick ? ["inertialEvents", pick] : null, () => api.inertialEvents(pick), { keepPrevious: true });
  const ego: EgoState | null = egoQ.error ? null : egoQ.data ?? null;
  const ev: InertialEvents | null = evQ.error ? null : evQ.data ?? null;
  const busy = egoQ.validating || evQ.validating;

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

  const spanS = series.length > 1 ? span / 1e9 : 0;
  const hasData = series.length > 0;
  return (
    <PageShell active="Ego motion" title="Ego motion"
      subtitle="The vehicle's own speed, braking and turning over a drive, with sharp manoeuvres marked."
      filters={
        <>
          <span className="text-ink-3">Session</span>
          <SessionSelect value={pick} onChange={setPick} className="w-96" />
          {ego && (
            <Badge state={ego.source === "derived" ? "muted" : "pass"}>
              {ego.source === "derived" ? "Estimated from GPS positions" : humanize(ego.source)}
            </Badge>
          )}
          {ego && <span className="text-ink-3">{num(ego.n_samples)} samples, {num(ego.n_with_motion)} with motion</span>}
          {busy && <span className="text-ink-3">Loading...</span>}
        </>
      }>
      <div className="page-wrap space-y-4 text-[12.5px]">
        <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Motion over the drive</h2>
            <span className="ml-auto text-[12px] text-ink-3">Each line is scaled to its own range. Shaded bands are events; the strip below is the manoeuvre.</span>
          </div>
          <div className="p-3">
            {hasData ? (
              <>
                <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 300 }} preserveAspectRatio="none" role="img"
                  aria-label="Speed, acceleration, turn rate and jerk over the session">
                  {ev?.maneuvers.map((m, i) => (
                    <rect key={`m${i}`} x={xOf(m.t_in_ns)} y={H - BAND_H} width={Math.max(1, xOf(m.t_out_ns) - xOf(m.t_in_ns))} height={BAND_H}
                      fill={MANEUVER_COLOR[m.kind] ?? "#2a2e35"} opacity={0.85}><title>{humanize(m.kind)}</title></rect>
                  ))}
                  {ev?.events.map((e, i) => (
                    <rect key={`e${i}`} x={xOf(e.t_in_ns)} y={PAD} width={Math.max(2, xOf(e.t_out_ns) - xOf(e.t_in_ns))} height={H - BAND_H - 2 * PAD}
                      fill={EVENT_COLOR[e.kind] ?? "#888"} opacity={0.13}><title>{`${humanize(e.kind)}, severity ${e.severity}`}</title></rect>
                  ))}
                  {lines.map((l) => l.d && <path key={l.key as string} d={l.d} fill="none" style={{ stroke: l.color }} strokeWidth={1.5} opacity={0.9} vectorEffect="non-scaling-stroke" />)}
                  {ev?.anomalies.map((a, i) => (
                    <circle key={`a${i}`} cx={xOf(a.ts_ns)} cy={PAD + 4} r={3} fill="#E3B341" stroke="#0B0C0E"><title>{`${humanize(a.metric)} unusual (z ${a.z})`}</title></circle>
                  ))}
                </svg>
                <div className="flex justify-between text-[11px] text-ink-3 px-1 tabular-nums">
                  <span>0 s</span><span>{duration(spanS)}</span>
                </div>
                <div className="flex flex-wrap gap-4 px-1 pt-2">
                  {METRICS.map((m) => (
                    <span key={m.key as string} className="flex items-center gap-1.5 text-[12px]">
                      <span className="w-4 h-[2px] inline-block rounded" style={{ background: m.color }} />
                      <span className="text-ink-2">{m.label}</span>
                    </span>
                  ))}
                </div>
              </>
            ) : busy || sessionsQ.loading ? (
              <div className="skeleton h-[300px] w-full" />
            ) : (
              <EmptyState compact icon="activity" title={pick ? "No motion for this session" : "Choose a session"}>
                {pick
                  ? "Motion is worked out from the GPS positions recorded with each frame. This session has none, so there is nothing to plot."
                  : "Pick a recorded drive above to see its speed, braking and turning."}
              </EmptyState>
            )}
          </div>
        </section>

        {hasData && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Events and unusual readings</h2>
                <span className="text-[12px] text-ink-3">{num((ev?.events.length ?? 0) + (ev?.anomalies.length ?? 0))}</span>
              </div>
              {!ev?.events.length && !ev?.anomalies.length ? (
                <EmptyState compact icon="check" title="No events in this session">
                  No hard braking, sharp acceleration, swerve or impact was found. Either the drive was smooth or
                  the session has too few positions to tell.
                </EmptyState>
              ) : (
                <table className="data-table">
                  <thead><tr><th>Kind</th><th>Detail</th><th className="num">Severity</th></tr></thead>
                  <tbody>
                    {ev?.events.map((e, i) => (
                      <tr key={`ev${i}`}>
                        <td className="text-ink">
                          <span className="inline-block w-2 h-2 rounded-full mr-2 align-middle" style={{ background: EVENT_COLOR[e.kind] }} />
                          {humanize(e.kind)}
                        </td>
                        <td>Peak {e.peak}</td>
                        <td className="num">{e.severity}</td>
                      </tr>
                    ))}
                    {ev?.anomalies.map((a, i) => (
                      <tr key={`an${i}`}>
                        <td className="text-ink">
                          <span className="inline-block w-2 h-2 rounded-full mr-2 align-middle bg-warn" />
                          Unusual {humanize(a.metric).toLowerCase()}
                        </td>
                        <td>{a.value} ({humanize(a.status).toLowerCase()})</td>
                        <td className="num">z {a.z}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </section>
            <section className="panel">
              <div className="panel-head">
                <h2 className="section-title">Manoeuvres</h2>
                <span className="text-[12px] text-ink-3">{num(ev?.maneuvers.length ?? 0)}</span>
              </div>
              {ev?.maneuvers.length ? (
                <table className="data-table">
                  <thead><tr><th>Manoeuvre</th><th className="num">Starts at</th><th className="num">Lasts</th></tr></thead>
                  <tbody>
                    {ev.maneuvers.map((m, i) => (
                      <tr key={`mv${i}`}>
                        <td className="text-ink">
                          <span className="inline-block w-2 h-2 rounded-full mr-2 align-middle" style={{ background: MANEUVER_COLOR[m.kind] }} />
                          {humanize(m.kind)}
                        </td>
                        <td className="num">{duration((m.t_in_ns - t0) / 1e9)}</td>
                        <td className="num">{duration((m.t_out_ns - m.t_in_ns) / 1e9)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <EmptyState compact icon="route" title="No manoeuvres found" />
              )}
            </section>
          </div>
        )}
      </div>
    </PageShell>
  );
}
