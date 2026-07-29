"use client";

import { Circle, Image as KImage, Layer, Line, Stage } from "react-konva";
import type { LaneRow } from "@/lib/types";

// react-konva renders a Stage's children through its own reconciler, which cannot resolve lazy
// (next/dynamic) element types: a per-primitive dynamic import throws "Lazy element type must
// resolve to a class or function". So the whole Konva tree is this single statically-imported
// component, and the page loads it once with next/dynamic(ssr:false).

type Lane = LaneRow & { dirty?: boolean };

const COLOR: Record<string, string> = { proposed: "#58A6FF", human: "#FF7A2F", propagated: "#E3B341" };

type Props = {
  img: HTMLImageElement;
  meta: { width: number; height: number };
  scale: number;
  lanes: Lane[];
  sel: string | null;
  drivable: Record<string, number[][]> | null;
  adding: number[][] | null;
  // A drivable region being drawn, and which surface class it will become. Distinct from `adding`, which is
  // a lane: a lane is an open polyline and a region is a closed area, so they cannot share a buffer without
  // one of them rendering wrongly while the other is in progress.
  addingArea?: number[][] | null;
  addingAreaClass?: string | null;
  // Which drivable polygon is selected for editing, as "class:index". Regions are stored per class rather
  // than with ids of their own, so the position in its class list is the only handle there is.
  areaSel?: string | null;
  onSelectArea?: (key: string) => void;
  onDragAreaPoint?: (key: string, i: number, x: number, y: number) => void;
  onStageClick: (e: { evt: MouseEvent }) => void;
  onSelect: (laneId: string) => void;
  onDragPoint: (laneId: string, i: number, x: number, y: number) => void;
};

// One place deciding what a surface class looks like, so the overlay, the in-progress outline and the
// selected handles cannot drift apart.
const SURFACE: Record<string, { fill: string; stroke: string }> = {
  drivable: { fill: "rgba(86,211,100,0.22)", stroke: "#56D364" },
  fallback: { fill: "rgba(227,179,65,0.22)", stroke: "#E3B341" },
  non_drivable: { fill: "rgba(248,81,73,0.16)", stroke: "#F85149" },
};
const surfaceStyle = (cls: string) => SURFACE[cls] ?? SURFACE.non_drivable;

export default function LaneCanvas(p: Props) {
  const { img, meta, scale, lanes, sel, drivable, adding, addingArea, areaSel } = p;
  // Guard the screen-constant divisor: a zero/non-finite scale (a not-yet-fitted frame) would make every
  // strokeWidth/radius Infinity and flood Konva with warnings.
  const s = scale > 0 && Number.isFinite(scale) ? scale : 1;
  const selLane = lanes.find((l) => l.lane_id === sel);
  return (
    <Stage width={meta.width * s} height={meta.height * s} scaleX={s} scaleY={s} onMouseDown={p.onStageClick}>
      <Layer>
        <KImage image={img} width={meta.width} height={meta.height} listening={false} />
        {drivable && Object.entries(drivable).flatMap(([cls, polys]) =>
          polys.map((poly, i) => {
            const key = `${cls}:${i}`;
            const st = surfaceStyle(cls);
            // Clickable only when the page offers a way to act on the click. Left inert otherwise, so the
            // read-only surfaces this canvas also serves do not gain a selection nobody can use.
            const pickable = Boolean(p.onSelectArea);
            return (
              <Line key={`dr-${key}`} points={poly} closed listening={pickable}
                fill={st.fill} stroke={st.stroke}
                strokeWidth={(areaSel === key ? 3 : 1) / s}
                onClick={pickable ? () => p.onSelectArea?.(key) : undefined}
                onTap={pickable ? () => p.onSelectArea?.(key) : undefined} />
            );
          }))}
        {lanes.map((l) => (
          <Line key={l.lane_id} points={l.control_points.flat()} stroke={l.is_ego ? "#56D364" : (COLOR[l.source] || "#A0A6AD")}
            strokeWidth={(l.lane_id === sel ? 4 : 2.5) / s} dash={l.lane_type === "dashed" ? [10 / s, 8 / s] : undefined}
            tension={0.3} onClick={() => p.onSelect(l.lane_id)} hitStrokeWidth={14 / s} />
        ))}
        {selLane?.control_points.map((pt, i) => (
          <Circle key={i} x={pt[0]} y={pt[1]} radius={6 / s} fill="#FF7A2F" draggable
            onDragMove={(e) => p.onDragPoint(selLane.lane_id, i, e.target.x(), e.target.y())} />
        ))}
        {/* Handles on the selected region, so an approximate machine boundary can be pulled onto the kerb. */}
        {areaSel && drivable && (() => {
          const [cls, idx] = areaSel.split(":");
          const flat = drivable[cls]?.[Number(idx)];
          if (!flat) return null;
          const pts: number[][] = [];
          for (let i = 0; i + 1 < flat.length; i += 2) pts.push([flat[i], flat[i + 1]]);
          return pts.map((pt, i) => (
            <Circle key={`ah${i}`} x={pt[0]} y={pt[1]} radius={5 / s} fill={surfaceStyle(cls).stroke}
              stroke="#1d1d1d" strokeWidth={1 / s} draggable
              onDragMove={(e) => p.onDragAreaPoint?.(areaSel, i, e.target.x(), e.target.y())} />
          ));
        })()}

        {adding?.length ? <Line points={adding.flat()} stroke="#FF7A2F" strokeWidth={2 / s} dash={[6 / s, 4 / s]} /> : null}

        {/* A region being drawn: closed and tinted in its class colour, so it is obvious which surface the
            next click is adding to rather than having to remember which tool is armed. */}
        {addingArea?.length ? (
          <>
            <Line points={addingArea.flat()} closed
              fill={surfaceStyle(p.addingAreaClass || "drivable").fill}
              stroke={surfaceStyle(p.addingAreaClass || "drivable").stroke}
              strokeWidth={2 / s} dash={[6 / s, 4 / s]} />
            {addingArea.map((pt, i) => (
              <Circle key={`ap${i}`} x={pt[0]} y={pt[1]} radius={4 / s}
                fill={surfaceStyle(p.addingAreaClass || "drivable").stroke} />
            ))}
          </>
        ) : null}
      </Layer>
    </Stage>
  );
}
