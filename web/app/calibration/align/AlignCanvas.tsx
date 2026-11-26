"use client";

// The camera image with the LiDAR scan projected onto it, re-projected in the browser on every change so the
// sliders and the drag are live. Drag to turn the camera (yaw and pitch), Shift-drag to roll, Alt-drag to slide
// it sideways and up. In pair mode, click a projected point and then the pixel where that point really is.

import { useEffect, useMemo, useRef, useState } from "react";
import { depthColor, project, type Extrinsic, type Intrinsics } from "@/lib/tools/drivingGeom";

export type Pair = { point: number[]; uv: number[] };

type Props = {
  image: HTMLImageElement | null;
  edgeField: HTMLImageElement | null;
  points: Float32Array;
  stride: number;
  k: Intrinsics;
  ex: Extrinsic;
  width: number;
  height: number;
  show: "depth" | "edges" | "edgemap";
  pairMode: boolean;
  pairs: Pair[];
  onPair: (p: Pair) => void;
  onDrag: (d: { dyaw: number; dpitch: number; droll: number; dy: number; dz: number }) => void;
};

export default function AlignCanvas({ image, edgeField, points, stride, k, ex, width, height, show, pairMode, pairs, onPair,
  onDrag }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [pending, setPending] = useState<number | null>(null);
  const drag = useRef<{ x: number; y: number; shift: boolean; alt: boolean } | null>(null);
  const proj = useMemo(() => project(points, stride, k, ex, width, height), [points, stride, k, ex, width, height]);
  // 48 colour steps from 3 m to 50 m, so drawing 40,000 points does not build 40,000 colour strings.
  const palette = useMemo(() => Array.from({ length: 48 }, (_, i) => depthColor(3 + (i / 47) * 47)), []);

  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    if (show === "edgemap" && edgeField) ctx.drawImage(edgeField, 0, 0, width, height);
    else if (image) ctx.drawImage(image, 0, 0, width, height);
    const n = proj.inside.length;
    for (let i = 0; i < n; i++) {
      if (!proj.inside[i]) continue;
      const w = points[i * stride + 4];
      if (show !== "depth" && !(w > 0)) continue;
      ctx.fillStyle = show === "depth"
        ? palette[Math.max(0, Math.min(47, Math.round(((proj.depth[i] - 3) / 47) * 47)))] : "rgb(255 60 200)";
      ctx.fillRect(proj.uv[i * 2] - 1, proj.uv[i * 2 + 1] - 1, show === "depth" ? 2 : 3, show === "depth" ? 2 : 3);
    }
    for (const p of pairs) {
      const one = project(new Float32Array(p.point), 3, k, ex, width, height);
      const [u, v] = [one.uv[0], one.uv[1]];
      ctx.strokeStyle = "rgb(255 210 0)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(u, v);
      ctx.lineTo(p.uv[0], p.uv[1]);
      ctx.stroke();
      ctx.strokeRect(p.uv[0] - 4, p.uv[1] - 4, 8, 8);
    }
    if (pending !== null) {
      ctx.strokeStyle = "rgb(255 210 0)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(proj.uv[pending * 2], proj.uv[pending * 2 + 1], 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [proj, image, edgeField, show, width, height, points, stride, pairs, pending, k, ex, palette]);

  const toPx = (e: React.PointerEvent | React.MouseEvent) => {
    const r = ref.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * width, ((e.clientY - r.top) / r.height) * height];
  };

  const nearestPoint = (u: number, v: number): number | null => {
    let best: number | null = null;
    let bd = 10;
    for (let i = 0; i < proj.inside.length; i++) {
      if (!proj.inside[i]) continue;
      const d = Math.hypot(proj.uv[i * 2] - u, proj.uv[i * 2 + 1] - v);
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  };

  return (
    <canvas ref={ref} width={width} height={height} className={`w-full rounded border border-line ${pairMode ? "cursor-crosshair" : "cursor-grab"}`}
      aria-label="LiDAR points projected onto the camera image"
      onPointerDown={(e) => {
        if (pairMode) return;
        (e.target as Element).setPointerCapture?.(e.pointerId);
        const [x, y] = toPx(e);
        drag.current = { x, y, shift: e.shiftKey, alt: e.altKey };
      }}
      onPointerMove={(e) => {
        const d = drag.current;
        if (!d) return;
        const [x, y] = toPx(e);
        const dx = x - d.x;
        const dy = y - d.y;
        drag.current = { ...d, x, y };
        // Signs measured against the projection: +yaw moves the scan left, +pitch moves it down.
        if (d.alt) onDrag({ dyaw: 0, dpitch: 0, droll: 0, dy: dx * 0.004, dz: dy * 0.004 });
        else if (d.shift) onDrag({ dyaw: 0, dpitch: 0, droll: dx * 0.05, dy: 0, dz: 0 });
        else onDrag({ dyaw: -(Math.atan(dx / k.fx) * 180) / Math.PI, dpitch: (Math.atan(dy / k.fy) * 180) / Math.PI, droll: 0, dy: 0, dz: 0 });
      }}
      onPointerUp={() => { drag.current = null; }}
      onClick={(e) => {
        if (!pairMode) return;
        const [u, v] = toPx(e);
        if (pending === null) { setPending(nearestPoint(u, v)); return; }
        const i = pending;
        setPending(null);
        onPair({ point: [points[i * stride], points[i * stride + 1], points[i * stride + 2]], uv: [u, v] });
      }} />
  );
}
