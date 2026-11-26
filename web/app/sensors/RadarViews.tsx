"use client";

// Radar drawn two ways. On the camera frame: each return at its projected pixel, coloured by Doppler, with an
// arrow to where its velocity takes it in half a second. From above: the ego frame (x forward up the screen,
// y left), each sensor's returns with velocity arrows, the radar positions, and the 3D boxes of the nearest
// scan with the speed fusion gave them.

import { cuboidCorners, dopplerColour } from "@/lib/tools/sensorsGeom";
import type { RadarBev, RadarOverlay } from "@/lib/tools/sensorsApi";

export function RadarOnFrame({ frameId, overlay, hidden }: { frameId: string; overlay: RadarOverlay | null;
  hidden: Set<string> }) {
  const w = overlay?.width ?? 1280, h = overlay?.height ?? 720;
  const r = Math.max(3, w / 220);
  return (
    <div className="relative w-full" style={{ aspectRatio: `${w} / ${h}` }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={`/api/frames/${frameId}/image`} alt="Camera frame" className="absolute inset-0 w-full h-full rounded" />
      {overlay && (
        <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full pointer-events-none">
          {overlay.sensors.filter((s) => !hidden.has(s.sensor)).flatMap((s) => s.points.map((p, i) => {
            const c = dopplerColour(p.radial_mps);
            return (
              <g key={`${s.sweep_id}-${i}`}>
                {p.end && <line x1={p.u} y1={p.v} x2={p.end[0]} y2={p.end[1]} stroke={c} strokeWidth={2} vectorEffect="non-scaling-stroke" />}
                <circle cx={p.u} cy={p.v} r={r} fill={c} stroke="rgba(0,0,0,0.6)" strokeWidth={1} vectorEffect="non-scaling-stroke">
                  <title>{`${s.sensor}: ${p.range_m} m, radial ${p.radial_mps} m/s, ${p.speed_mps} m/s over ground, RCS ${p.rcs}`}</title>
                </circle>
              </g>
            );
          }))}
        </svg>
      )}
    </div>
  );
}

export function RadarFromAbove({ bev, hidden, rangeM = 60 }: { bev: RadarBev | null; hidden: Set<string>; rangeM?: number }) {
  // ego frame -> screen: x forward is up, y left is left. The ego sits a fifth of the way up from the bottom.
  const W = 400, H = 520;
  const s = H / (rangeM * 1.25);
  const toS = (x: number, y: number) => [W / 2 - y * s, H * 0.8 - x * s];
  const rings = [10, 20, 40, 60].filter((m) => m <= rangeM);
  const boxes = (bev?.boxes ?? []).map((b) => ({
    ...b, corners: cuboidCorners(b.center, b.dims, b.yaw).slice(0, 4).map((p) => toS(p[0], p[1])),
  }));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto rounded bg-bg-2" role="img" aria-label="Radar from above">
      {rings.map((m) => {
        const [cx, cy] = toS(0, 0);
        return <g key={m}>
          <circle cx={cx} cy={cy} r={m * s} fill="none" stroke="currentColor" className="text-line" strokeDasharray="3 4" />
          <text x={cx + 3} y={cy - m * s - 3} className="fill-ink-3" fontSize={11}>{m} m</text>
        </g>;
      })}
      {(() => { const [x, y] = toS(0, 0); return <rect x={x - 6} y={y - 12} width={12} height={24} rx={2} className="fill-accent" />; })()}
      {boxes.map((b) => (
        <g key={b.object_3d_id}>
          <polygon points={b.corners.map((p) => p.join(",")).join(" ")} fill="none" strokeWidth={1.5}
            stroke={b.radar?.source === "radar" ? "#4ade80" : b.radar ? "#fbbf24" : "#94a3b8"} />
          {b.radar && (() => { const [x, y] = toS(b.center[0], b.center[1]); return (
            <text x={x + 8} y={y} fontSize={11} className="fill-ink">{b.radar.speed_mps.toFixed(1)} m/s</text>); })()}
        </g>
      ))}
      {(bev?.sensors ?? []).filter((sn) => !hidden.has(sn.sensor)).map((sn) => (
        <g key={sn.sweep_id}>
          {(() => { const [x, y] = toS(sn.origin[0], sn.origin[1]); return <rect x={x - 3} y={y - 3} width={6} height={6} className="fill-info"><title>{sn.sensor}</title></rect>; })()}
          {sn.points.map((p, i) => {
            const [x, y] = toS(p.x, p.y);
            const [x2, y2] = toS(p.x + p.vx * 0.5, p.y + p.vy * 0.5);
            const c = dopplerColour(p.radial_mps);
            return <g key={i}>
              <line x1={x} y1={y} x2={x2} y2={y2} stroke={c} strokeWidth={1.5} />
              <circle cx={x} cy={y} r={2.8} fill={c}><title>{`${sn.sensor}: radial ${p.radial_mps} m/s, RCS ${p.rcs}`}</title></circle>
            </g>;
          })}
        </g>
      ))}
    </svg>
  );
}

export function DopplerLegend() {
  return (
    <div className="flex items-center gap-3 text-[11px] text-ink-3">
      <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: dopplerColour(-10) }} />approaching</span>
      <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: dopplerColour(0) }} />still</span>
      <span className="flex items-center gap-1"><span className="inline-block w-2.5 h-2.5 rounded-full" style={{ background: dopplerColour(10) }} />receding</span>
      <span>arrows: where it is in 0.5 s</span>
    </div>
  );
}
