"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { AnnotationRow, LabelConfig } from "@/lib/types";

// Time series: multi-channel plot, brush a span to segment it.
//
// Each channel is normalized to its OWN range rather than a shared one. Sensor channels differ by orders of
// magnitude (accel in m/s2 next to a gyro in rad/s next to a speed in km/h); on a shared axis the small
// channel becomes a flat line and the event you are labeling is invisible. Per-channel scaling keeps every
// trace readable, and the real min/max are printed so the normalization is not mistaken for the data.

type Series = { name: string; values: number[]; t?: number[] };

type Props = {
  series: Series[];
  duration: number;                 // seconds
  annotations: AnnotationRow[];
  config: LabelConfig;
  activeLabel: string | null;
  onCreate: (tStart: number, tEnd: number, channel?: string) => void;
  onSelect: (id: string | null) => void;
  selectedId: string | null;
};

const CH_COLORS = ["#4c8dff", "#f2994a", "#27ae60", "#eb5757", "#bb6bd9", "#56ccf2"];

function colorFor(config: LabelConfig, label: string | null): string {
  return (config.labels ?? []).find((x) => x.name === label)?.color || "#4c8dff";
}

export default function TimeSeriesEditor({
  series, duration, annotations, config, activeLabel, onCreate, onSelect, selectedId,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [drag, setDrag] = useState<{ x0: number; x1: number } | null>(null);
  const W = 900;
  const ROW = 90;
  const H = Math.max(ROW, series.length * ROW);

  const ranges = useMemo(
    () => series.map((s) => {
      const vs = s.values.filter((v) => Number.isFinite(v));
      return vs.length ? { lo: Math.min(...vs), hi: Math.max(...vs) } : { lo: 0, hi: 1 };
    }),
    [series],
  );

  const regions = annotations.filter((a) => a.kind === "region");

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = `${W}px`; cv.style.height = `${H}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    series.forEach((s, si) => {
      const top = si * ROW;
      const { lo, hi } = ranges[si];
      const span = hi - lo || 1;

      // baseline + label
      ctx.strokeStyle = "#2a2f3a";
      ctx.beginPath(); ctx.moveTo(0, top + ROW - 0.5); ctx.lineTo(W, top + ROW - 0.5); ctx.stroke();
      ctx.fillStyle = "#8b93a1";
      ctx.font = "10px monospace";
      ctx.fillText(`${s.name}  [${lo.toFixed(2)}, ${hi.toFixed(2)}]`, 4, top + 12);

      ctx.strokeStyle = CH_COLORS[si % CH_COLORS.length];
      ctx.lineWidth = 1;
      ctx.beginPath();
      const n = s.values.length;
      for (let i = 0; i < n; i++) {
        const x = (i / Math.max(1, n - 1)) * W;
        const y = top + ROW - 14 - ((s.values[i] - lo) / span) * (ROW - 24);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    });

    for (const r of regions) {
      if (!duration) continue;
      const x0 = (Number(r.payload.t_start ?? 0) / duration) * W;
      const x1 = (Number(r.payload.t_end ?? 0) / duration) * W;
      const c = colorFor(config, r.label);
      ctx.fillStyle = `${c}26`;
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), H);
      ctx.strokeStyle = r.annotation_id === selectedId ? "#ffffff" : c;
      ctx.lineWidth = r.annotation_id === selectedId ? 2 : 1;
      ctx.strokeRect(x0, 0, Math.max(1, x1 - x0), H);
    }

    if (drag) {
      ctx.fillStyle = "rgba(76,141,255,0.18)";
      ctx.fillRect(Math.min(drag.x0, drag.x1), 0, Math.abs(drag.x1 - drag.x0), H);
    }
  }, [series, ranges, regions, duration, drag, selectedId, config, H]);

  const xToT = useCallback((x: number) => (duration ? (x / W) * duration : 0), [duration]);
  const localX = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(W, e.clientX - r.left));
  };

  if (!series.length) {
    return <div className="p-4 font-mono text-[11px] text-ink-3">this asset carries no series data</div>;
  }

  return (
    <div className="p-4 space-y-2">
      <canvas ref={canvasRef}
        className="block border border-line cursor-crosshair"
        onMouseDown={(e) => setDrag({ x0: localX(e), x1: localX(e) })}
        onMouseMove={(e) => setDrag((d) => (d ? { ...d, x1: localX(e) } : null))}
        onMouseUp={(e) => {
          const d = drag;
          setDrag(null);
          if (!d) return;
          const x1 = localX(e);
          const a = Math.min(d.x0, x1), b = Math.max(d.x0, x1);
          if (b - a < 3) {
            const t = xToT(a);
            const hit = regions.find((r) => t >= Number(r.payload.t_start) && t <= Number(r.payload.t_end));
            onSelect(hit ? hit.annotation_id : null);
            return;
          }
          onCreate(xToT(a), xToT(b));
        }}
        onMouseLeave={() => setDrag(null)} />
      <div className="font-mono text-[11px] text-ink-3">
        {series.length} channels, {duration.toFixed(2)}s, {regions.length} segments
        {!activeLabel && " - pick a label, then brush a span"}
      </div>
    </div>
  );
}
