"use client";

// One rendered view of a reconstruction with the cuboid being placed drawn into it, through the same camera the
// server rendered the view with (so the box sits where the reconstruction says it is). On the top-down view a
// click places the box and a drag moves it.

import { useMemo, useRef, useState } from "react";

import { CUBOID_EDGES, bevPixelToWorld, cuboidCorners, invertRigid, projectToView } from "@/lib/tools/sensorsGeom";
import type { ReconView } from "@/lib/tools/sensorsApi";
import type { DraftBox } from "./ReconCloud";

export default function ViewWithBox({ src, view, draft, onPlace, alt }: {
  src: string; view: ReconView; draft: DraftBox | null; alt: string;
  onPlace?: (x: number, y: number) => void;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState(false);
  const inv = useMemo(() => invertRigid(view.T_world_cam), [view]);
  const lines = useMemo(() => {
    if (!draft) return [];
    const px = cuboidCorners(draft.center, draft.dims, draft.yaw).map((p) => projectToView(view, p, inv));
    return CUBOID_EDGES.flatMap(([a, b], i) => {
      const pa = px[a], pb = px[b];
      return pa && pb ? [{ x1: pa.u, y1: pa.v, x2: pb.u, y2: pb.v, front: i === 0 || i === 4 }] : [];
    });
  }, [draft, view, inv]);

  const toWorld = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r || !onPlace) return;
    const u = ((e.clientX - r.left) / r.width) * view.width;
    const v = ((e.clientY - r.top) / r.height) * view.height;
    const w = bevPixelToWorld(view, u, v);
    onPlace(w.x, w.y);
  };

  return (
    <div className="relative w-full" style={{ aspectRatio: `${view.width} / ${view.height}` }}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className="absolute inset-0 w-full h-full rounded select-none" draggable={false} />
      <svg ref={ref} viewBox={`0 0 ${view.width} ${view.height}`} preserveAspectRatio="none"
        className={`absolute inset-0 w-full h-full ${onPlace ? "cursor-crosshair" : "pointer-events-none"}`}
        onPointerDown={(e) => { if (!onPlace) return; (e.target as Element).setPointerCapture?.(e.pointerId); setDrag(true); toWorld(e); }}
        onPointerMove={(e) => { if (drag) toWorld(e); }}
        onPointerUp={() => setDrag(false)}>
        {lines.map((l, i) => (
          <line key={i} x1={l.x1} y1={l.y1} x2={l.x2} y2={l.y2} stroke={l.front ? "#f87171" : "#fbbf24"}
            strokeWidth={2} vectorEffect="non-scaling-stroke" />
        ))}
      </svg>
    </div>
  );
}
