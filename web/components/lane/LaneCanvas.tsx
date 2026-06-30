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
  onStageClick: (e: { evt: MouseEvent }) => void;
  onSelect: (laneId: string) => void;
  onDragPoint: (laneId: string, i: number, x: number, y: number) => void;
};

export default function LaneCanvas(p: Props) {
  const { img, meta, scale, lanes, sel, drivable, adding } = p;
  // Guard the screen-constant divisor: a zero/non-finite scale (a not-yet-fitted frame) would make every
  // strokeWidth/radius Infinity and flood Konva with warnings.
  const s = scale > 0 && Number.isFinite(scale) ? scale : 1;
  const selLane = lanes.find((l) => l.lane_id === sel);
  return (
    <Stage width={meta.width * s} height={meta.height * s} scaleX={s} scaleY={s} onMouseDown={p.onStageClick}>
      <Layer>
        <KImage image={img} width={meta.width} height={meta.height} listening={false} />
        {drivable && Object.entries(drivable).flatMap(([cls, polys]) =>
          polys.map((poly, i) => (
            <Line key={`dr-${cls}-${i}`} points={poly} closed listening={false}
              fill={cls === "drivable" ? "rgba(86,211,100,0.22)" : cls === "fallback" ? "rgba(227,179,65,0.22)" : "rgba(248,81,73,0.16)"}
              stroke={cls === "drivable" ? "#56D364" : cls === "fallback" ? "#E3B341" : "#F85149"} strokeWidth={1 / s} />
          )))}
        {lanes.map((l) => (
          <Line key={l.lane_id} points={l.control_points.flat()} stroke={l.is_ego ? "#56D364" : (COLOR[l.source] || "#A0A6AD")}
            strokeWidth={(l.lane_id === sel ? 4 : 2.5) / s} dash={l.lane_type === "dashed" ? [10 / s, 8 / s] : undefined}
            tension={0.3} onClick={() => p.onSelect(l.lane_id)} hitStrokeWidth={14 / s} />
        ))}
        {selLane?.control_points.map((pt, i) => (
          <Circle key={i} x={pt[0]} y={pt[1]} radius={6 / s} fill="#FF7A2F" draggable
            onDragMove={(e) => p.onDragPoint(selLane.lane_id, i, e.target.x(), e.target.y())} />
        ))}
        {adding?.length ? <Line points={adding.flat()} stroke="#FF7A2F" strokeWidth={2 / s} dash={[6 / s, 4 / s]} /> : null}
      </Layer>
    </Stage>
  );
}
