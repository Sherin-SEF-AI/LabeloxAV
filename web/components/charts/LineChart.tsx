"use client";

// Hand-rolled SVG multi-series line chart (the codebase has no chart lib; SVG is the house style).
// Plots one or more series over a shared x axis (epochs), with a light grid, min/max y labels, a legend,
// and a hover crosshair that reads out each series' value at the nearest x. Colors come from the
// Operational Materialism palette.

import { useState } from "react";

export type Series = { key: string; label: string; color: string; values: number[] };

const PAD = { l: 34, r: 8, t: 8, b: 16 };

export default function LineChart({
  x, series, height = 150, yDomain, yFormat = (v) => v.toFixed(2), unit = "",
}: {
  x: number[];
  series: Series[];
  height?: number;
  yDomain?: [number, number];
  yFormat?: (v: number) => string;
  unit?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 640, H = height;
  const iw = W - PAD.l - PAD.r, ih = H - PAD.t - PAD.b;

  const all = series.flatMap((s) => s.values).filter((v) => Number.isFinite(v));
  const lo = yDomain ? yDomain[0] : Math.min(...all, 0);
  const hi = yDomain ? yDomain[1] : Math.max(...all, 0.0001);
  const span = hi - lo || 1;
  const n = x.length;
  const px = (i: number) => PAD.l + (n <= 1 ? 0 : (i / (n - 1)) * iw);
  const py = (v: number) => PAD.t + ih - ((v - lo) / span) * ih;

  const gridY = [0, 0.25, 0.5, 0.75, 1].map((f) => lo + f * span);

  if (!n || !series.length) {
    return <div className="font-mono text-[10px] text-ink-3 py-6 text-center">no curve data</div>;
  }

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const rel = ((e.clientX - r.left) / r.width) * W - PAD.l;
          const i = Math.round((rel / iw) * (n - 1));
          setHover(i >= 0 && i < n ? i : null);
        }}
        onMouseLeave={() => setHover(null)}>
        {/* grid + y labels */}
        {gridY.map((v, k) => (
          <g key={k}>
            <line x1={PAD.l} x2={W - PAD.r} y1={py(v)} y2={py(v)} stroke="#23262B" strokeWidth={1} />
            <text x={PAD.l - 4} y={py(v) + 3} textAnchor="end" fontSize={8} fill="#6C727A" fontFamily="monospace">
              {yFormat(v)}
            </text>
          </g>
        ))}
        {/* x endpoints */}
        <text x={PAD.l} y={H - 4} fontSize={8} fill="#6C727A" fontFamily="monospace">{x[0]}</text>
        <text x={W - PAD.r} y={H - 4} textAnchor="end" fontSize={8} fill="#6C727A" fontFamily="monospace">{x[n - 1]}</text>
        {/* series polylines */}
        {series.map((s) => (
          <polyline key={s.key} fill="none" stroke={s.color} strokeWidth={1.5} strokeLinejoin="round"
            points={s.values.map((v, i) => `${px(i)},${py(v)}`).join(" ")} />
        ))}
        {/* hover crosshair + dots */}
        {hover != null && (
          <g>
            <line x1={px(hover)} x2={px(hover)} y1={PAD.t} y2={PAD.t + ih} stroke="#6C727A" strokeWidth={1} strokeDasharray="2 2" />
            {series.map((s) => (
              <circle key={s.key} cx={px(hover)} cy={py(s.values[hover])} r={2.5} fill={s.color} />
            ))}
          </g>
        )}
      </svg>
      {/* legend + hover readout */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 mt-1 font-mono text-[9px]">
        {series.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-1 text-ink-3">
            <span className="inline-block w-2.5 h-0.5" style={{ background: s.color }} />
            {s.label}
            {hover != null && <span className="text-ink">{yFormat(s.values[hover])}{unit}</span>}
          </span>
        ))}
        {hover != null && <span className="text-ink-3 ml-auto">epoch {x[hover]}</span>}
      </div>
    </div>
  );
}
