"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ProjectionPoint } from "@/lib/types";

// The embeddings map: every object or frame as one point, laid out so visually similar things sit together.
// Canvas rather than SVG because a real corpus is tens of thousands of points and one DOM node each would
// crawl. Drag to lasso a region; the lasso is the whole point of the map, since "these all look alike" is the
// selection a curator actually wants and no filter expresses it.

export type ColorBy = "cluster" | "class" | "state" | "source" | "conf" | "tag";

// Distinct hues for categorical colouring, dark-theme friendly. Index by a stable hash so a class keeps its
// colour between renders.
const PALETTE = [
  "#4c8dff", "#f2994a", "#27ae60", "#eb5757", "#bb6bd9", "#2d9cdb",
  "#f2c94c", "#56ccf2", "#6fcf97", "#ff8fa3", "#9b8cff", "#e0a458",
];

function hashColor(key: string | number | undefined | null): string {
  if (key === undefined || key === null || key === "") return "#5a6472";
  const s = String(key);
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
  return PALETTE[Math.abs(h) % PALETTE.length];
}

function confColor(c: number | null | undefined): string {
  if (c == null) return "#5a6472";
  // red (low) to green (high), the same reading as the gate bands
  const t = Math.max(0, Math.min(1, c));
  const r = Math.round(235 - 190 * t);
  const g = Math.round(87 + 88 * t);
  return `rgb(${r},${g},96)`;
}

function pointColor(p: ProjectionPoint, mode: ColorBy, tag: string | null): string {
  switch (mode) {
    case "cluster": return p.cluster === -1 ? "#4b5563" : hashColor(p.cluster);
    case "class": return hashColor(p.class_id);
    case "state": return hashColor(p.state);
    case "source": return hashColor(p.source);
    case "conf": return confColor(p.conf ?? null);
    case "tag": return tag && (p.tags ?? []).includes(tag) ? "#4c8dff" : "#3a4150";
    default: return "#4c8dff";
  }
}

type Props = {
  points: ProjectionPoint[];
  colorBy: ColorBy;
  tagFilter: string | null;
  selected: Set<string>;
  onSelect: (ids: string[], additive: boolean) => void;
  onHover?: (p: ProjectionPoint | null) => void;
};

export default function EmbeddingMap({ points, colorBy, tagFilter, selected, onSelect, onHover }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const [size, setSize] = useState({ w: 800, h: 560 });
  const [lasso, setLasso] = useState<{ x: number; y: number }[]>([]);
  const drawing = useRef(false);
  const additive = useRef(false);

  // Data bounds, so the layout fills the canvas whatever scale UMAP happened to produce.
  const bounds = useMemo(() => {
    if (!points.length) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of points) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    if (maxX - minX < 1e-9) maxX = minX + 1;
    if (maxY - minY < 1e-9) maxY = minY + 1;
    return { minX, maxX, minY, maxY };
  }, [points]);

  const PAD = 18;
  const toScreen = useCallback((p: { x: number; y: number }) => {
    const { minX, maxX, minY, maxY } = bounds;
    return {
      sx: PAD + ((p.x - minX) / (maxX - minX)) * (size.w - 2 * PAD),
      // flip y so the picture reads the usual way up
      sy: size.h - PAD - ((p.y - minY) / (maxY - minY)) * (size.h - 2 * PAD),
    };
  }, [bounds, size]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(320, Math.floor(r.width)), h: Math.max(320, Math.floor(r.height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // Draw: points first, then the in-progress lasso on top.
  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = size.w * dpr;
    cv.height = size.h * dpr;
    cv.style.width = `${size.w}px`;
    cv.style.height = `${size.h}px`;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);

    const hasSel = selected.size > 0;
    for (const p of points) {
      const { sx, sy } = toScreen(p);
      const isSel = selected.has(p.id);
      ctx.globalAlpha = hasSel && !isSel ? 0.18 : 0.85;
      ctx.fillStyle = isSel ? "#ffffff" : pointColor(p, colorBy, tagFilter);
      ctx.beginPath();
      ctx.arc(sx, sy, isSel ? 3.2 : 2.1, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    if (lasso.length > 1) {
      ctx.strokeStyle = "#4c8dff";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(lasso[0].x, lasso[0].y);
      for (const pt of lasso.slice(1)) ctx.lineTo(pt.x, pt.y);
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(76,141,255,0.10)";
      ctx.fill();
    }
  }, [points, size, colorBy, tagFilter, selected, lasso, toScreen]);

  // Even-odd ray cast: is a screen point inside the lasso polygon.
  const inside = useCallback((poly: { x: number; y: number }[], x: number, y: number) => {
    let hit = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-12) + xi) hit = !hit;
    }
    return hit;
  }, []);

  const localPos = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  };

  return (
    <div ref={wrapRef} className="relative w-full h-full min-h-[320px]">
      <canvas
        ref={canvasRef}
        className="block cursor-crosshair"
        onMouseDown={(e) => { drawing.current = true; additive.current = e.shiftKey; setLasso([localPos(e)]); }}
        onMouseMove={(e) => {
          if (drawing.current) { setLasso((l) => [...l, localPos(e)]); return; }
          if (!onHover) return;
          // cheapest useful hover: nearest point within a few px
          const { x, y } = localPos(e);
          let best: ProjectionPoint | null = null;
          let bestD = 36;
          for (const p of points) {
            const { sx, sy } = toScreen(p);
            const d = (sx - x) ** 2 + (sy - y) ** 2;
            if (d < bestD) { bestD = d; best = p; }
          }
          onHover(best);
        }}
        onMouseUp={() => {
          drawing.current = false;
          setLasso((l) => {
            if (l.length > 2) {
              const hits = points.filter((p) => {
                const { sx, sy } = toScreen(p);
                return inside(l, sx, sy);
              }).map((p) => p.id);
              onSelect(hits, additive.current);
            }
            return [];
          });
        }}
        onMouseLeave={() => { drawing.current = false; setLasso([]); onHover?.(null); }}
      />
      {!points.length && (
        <div className="absolute inset-0 flex items-center justify-center font-mono text-[11px] text-ink-3">
          no projection loaded. fit one to see the corpus laid out by similarity.
        </div>
      )}
    </div>
  );
}
