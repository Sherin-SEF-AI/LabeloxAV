"use client";

// Top-down plot for /trajectories: the ego path, every road user's observed path, the selected track's futures,
// the kinematic baseline, and draggable handles for whichever path is being edited. Metres are square and +y is
// up (north in the world frame, left in the ego frame). Scroll to zoom, drag the background to pan.

import { useMemo, useRef, useState } from "react";
import { pathBounds, plotTransform } from "@/lib/tools/drivingGeom";

export type PlotPath = {
  id: string;
  points: number[][];          // [t, x, y]
  kind: "ego" | "track" | "track-selected" | "future" | "baseline" | "observed-after" | "ego-edit" | "future-edit";
  label?: string;
};

type Props = {
  paths: PlotPath[];
  fitTo?: number[][][];
  editable?: { id: string; points: number[][]; every?: number } | null;
  onEdit?: (points: number[][]) => void;
  onPick?: (id: string) => void;
  marker?: number[] | null;
  height?: number;
};

const STYLE: Record<PlotPath["kind"], { stroke: string; width: number; dash?: string; opacity?: number }> = {
  ego: { stroke: "rgb(var(--c-pass))", width: 2.5 },
  "ego-edit": { stroke: "rgb(var(--c-warn))", width: 2.5 },
  track: { stroke: "rgb(var(--c-ink-3))", width: 1.2, opacity: 0.7 },
  "track-selected": { stroke: "rgb(var(--c-accent))", width: 3 },
  future: { stroke: "rgb(var(--c-info))", width: 2.5, dash: "6 4" },
  "future-edit": { stroke: "rgb(var(--c-warn))", width: 2.5, dash: "6 4" },
  baseline: { stroke: "rgb(var(--c-ink-2))", width: 1.5, dash: "2 4" },
  "observed-after": { stroke: "rgb(var(--c-accent))", width: 1.5, dash: "1 3", opacity: 0.8 },
};

export default function PathPlot({ paths, fitTo, editable, onEdit, onPick, marker, height = 560 }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const W = 900;
  const H = height;
  const bounds = useMemo(() => pathBounds(fitTo && fitTo.length ? fitTo : paths.map((p) => p.points), 6), [fitTo, paths]);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<[number, number]>([0, 0]);
  const [drag, setDrag] = useState<{ i: number } | null>(null);
  const [panning, setPanning] = useState<[number, number] | null>(null);
  const tf = useMemo(() => plotTransform(bounds, W, H), [bounds, H]);
  // zoom/pan are applied as an SVG transform on top of the fitted transform
  const toScreen = (x: number, y: number) => {
    const [px, py] = tf.toPx(x, y);
    return [px * zoom + pan[0], py * zoom + pan[1]];
  };
  const svgPoint = (e: { clientX: number; clientY: number }) => {
    const svg = svgRef.current!;
    const p = svg.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const q = p.matrixTransform(svg.getScreenCTM()!.inverse());
    return [q.x, q.y];
  };
  const toMetres = (sx: number, sy: number) => tf.toM((sx - pan[0]) / zoom, (sy - pan[1]) / zoom);

  const handles = useMemo(() => {
    if (!editable) return [];
    const every = Math.max(1, editable.every ?? 1);
    const idx: number[] = [];
    for (let i = 0; i < editable.points.length; i += every) idx.push(i);
    if (idx[idx.length - 1] !== editable.points.length - 1) idx.push(editable.points.length - 1);
    return idx;
  }, [editable]);

  const moveHandle = (i: number, mx: number, my: number) => {
    if (!editable || !onEdit) return;
    const pts = editable.points;
    const dx = mx - pts[i][1];
    const dy = my - pts[i][2];
    const every = Math.max(1, editable.every ?? 1);
    // A handle that stands for `every` points moves its neighbours with a triangular falloff, so a long path is
    // reshaped smoothly rather than kinked at one sample.
    onEdit(pts.map((p, j) => {
      const w = every === 1 ? (j === i ? 1 : 0) : Math.max(0, 1 - Math.abs(j - i) / every);
      return w > 0 ? [p[0], +(p[1] + dx * w).toFixed(3), +(p[2] + dy * w).toFixed(3)] : p;
    }));
  };

  const scaleBar = useMemo(() => {
    const target = 120 / (tf.scale * zoom);
    const nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find((v) => v >= target) ?? 1000;
    return { m: nice, px: nice * tf.scale * zoom };
  }, [tf.scale, zoom]);

  return (
    <svg ref={svgRef} viewBox={`0 0 ${W} ${H}`} className="w-full bg-bg rounded border border-line select-none touch-none"
      style={{ height: H }} role="application" aria-label="Top-down paths"
      onWheel={(e) => {
        const [sx, sy] = svgPoint(e);
        const f = e.deltaY < 0 ? 1.15 : 1 / 1.15;
        const nz = Math.max(0.2, Math.min(40, zoom * f));
        setPan([sx - ((sx - pan[0]) * nz) / zoom, sy - ((sy - pan[1]) * nz) / zoom]);
        setZoom(nz);
      }}
      onPointerDown={(e) => { if (!drag) { const [sx, sy] = svgPoint(e); setPanning([sx - pan[0], sy - pan[1]]); } }}
      onPointerMove={(e) => {
        const [sx, sy] = svgPoint(e);
        if (drag) { const [mx, my] = toMetres(sx, sy); moveHandle(drag.i, mx, my); }
        else if (panning) setPan([sx - panning[0], sy - panning[1]]);
      }}
      onPointerUp={() => { setDrag(null); setPanning(null); }}
      onPointerLeave={() => { setDrag(null); setPanning(null); }}
      onDoubleClick={() => { setZoom(1); setPan([0, 0]); }}>
      {paths.map((p) => {
        const s = STYLE[p.kind];
        if (p.points.length < 2) return null;
        const d = p.points.map((q, i) => { const [x, y] = toScreen(q[1], q[2]); return `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`; }).join(" ");
        const pickable = (p.kind === "track" || p.kind === "track-selected") && onPick;
        return (
          <g key={`${p.kind}-${p.id}`}>
            {pickable && <path d={d} fill="none" stroke="transparent" strokeWidth={12} style={{ cursor: "pointer" }}
              onPointerDown={(e) => e.stopPropagation()} onClick={() => onPick!(p.id)} />}
            <path d={d} fill="none" stroke={s.stroke} strokeWidth={s.width} strokeDasharray={s.dash} opacity={s.opacity ?? 1}
              pointerEvents="none" />
            {(p.kind === "track-selected" || p.kind === "ego") && (() => {
              const [x, y] = toScreen(p.points[p.points.length - 1][1], p.points[p.points.length - 1][2]);
              return <circle cx={x} cy={y} r={3.5} fill={s.stroke} pointerEvents="none" />;
            })()}
          </g>
        );
      })}
      {marker && (() => {
        const [x, y] = toScreen(marker[1], marker[2]);
        return <circle cx={x} cy={y} r={6} fill="none" stroke="rgb(var(--c-warn))" strokeWidth={2} pointerEvents="none" />;
      })()}
      {editable && handles.map((i) => {
        const p = editable.points[i];
        const [x, y] = toScreen(p[1], p[2]);
        return (
          <circle key={i} cx={x} cy={y} r={drag?.i === i ? 7 : 5} fill="rgb(var(--c-warn))" stroke="rgb(var(--c-bg))" strokeWidth={1.5}
            style={{ cursor: "grab" }}
            onPointerDown={(e) => { e.stopPropagation(); (e.target as Element).setPointerCapture?.(e.pointerId); setDrag({ i }); }}>
            <title>{`t = ${p[0].toFixed(2)} s`}</title>
          </circle>
        );
      })}
      <g transform={`translate(16 ${H - 18})`} pointerEvents="none">
        <line x1={0} y1={0} x2={scaleBar.px} y2={0} stroke="rgb(var(--c-ink-2))" strokeWidth={2} />
        <text x={0} y={-5} fill="rgb(var(--c-ink-2))" fontSize={11}>{scaleBar.m} m</text>
      </g>
    </svg>
  );
}
