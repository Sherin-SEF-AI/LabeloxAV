"use client";

// The interactive annotation surface. A fixed-size konva Stage with a pan/zoom viewport (the stage
// transform), rendering the frame image plus every object's box and mask in image space. Tools: select
// (move + resize via Transformer, drag mask vertices), draw box, SAM point, SAM box. All geometry is in
// image pixels; the stage transform maps to screen. getRelativePointerPosition() returns image coords.

import { useEffect, useRef, useState } from "react";
import { Circle, Image as KImage, Layer, Line, Rect, Stage, Transformer } from "react-konva";
import type Konva from "konva";
import { classColor, classFill } from "@/lib/colors";
import type { EdObject, Tool, Viewport } from "./useEditor";

type LaneOverlay = { lane_id: string; control_points: number[][]; lane_type: string; is_ego: boolean; source: string };
export type LayerFlags = { boxes: boolean; masks: boolean; lanes: boolean; drivable: boolean };

type Props = {
  imageUrl: string;
  imgW: number;
  imgH: number;
  objects: EdObject[];
  selectedId: string | null;
  tool: Tool;
  candidate: number[][] | null;
  viewport: Viewport;
  panning: boolean;
  lanes?: LaneOverlay[];
  drivable?: Record<string, number[][]> | null;
  layers?: LayerFlags;
  onViewport: (v: Viewport) => void;
  onSelect: (id: string | null) => void;
  onUpdateBbox: (id: string, bbox: number[]) => void;
  onDrawBox: (bbox: number[]) => void;
  onSamPoint: (pt: number[], label: number) => void;
  onSamBox: (box: number[]) => void;
  onUpdateMask: (id: string, polys: number[][]) => void;
  onCursor: (pt: number[] | null) => void;
};

const MIN_SCALE = 0.05;
const MAX_SCALE = 20;
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

export default function EditorCanvas(p: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [draw, setDraw] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const trRef = useRef<Konva.Transformer>(null);
  const selRectRef = useRef<Konva.Rect>(null);

  // measure container
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  // load image
  useEffect(() => {
    const im = new window.Image();
    im.crossOrigin = "anonymous";
    im.src = p.imageUrl;
    im.onload = () => setImg(im);
  }, [p.imageUrl]);

  // fit on first ready (viewport.scale === 0 is the "fit pending" sentinel)
  useEffect(() => {
    if (img && p.viewport.scale === 0 && size.w > 1) {
      const s = Math.min(size.w / p.imgW, size.h / p.imgH) * 0.96;
      p.onViewport({ scale: s, ox: (size.w - p.imgW * s) / 2, oy: (size.h - p.imgH * s) / 2 });
    }
  }, [img, size, p]);

  // attach transformer to the selected box
  useEffect(() => {
    const tr = trRef.current;
    if (!tr) return;
    if (p.tool === "select" && selRectRef.current) tr.nodes([selRectRef.current]);
    else tr.nodes([]);
    tr.getLayer()?.batchDraw();
  }, [p.selectedId, p.tool, p.objects]);

  const v = p.viewport;
  const L = p.layers ?? { boxes: true, masks: true, lanes: true, drivable: true };
  const toImg = (): number[] => {
    const pt = stageRef.current?.getRelativePointerPosition();
    return pt ? [pt.x, pt.y] : [0, 0];
  };

  function onWheel(e: Konva.KonvaEventObject<WheelEvent>) {
    e.evt.preventDefault();
    const stage = stageRef.current;
    if (!stage) return;
    const pointer = stage.getPointerPosition();
    if (!pointer) return;
    const mx = (pointer.x - v.ox) / v.scale;
    const my = (pointer.y - v.oy) / v.scale;
    const newScale = clamp(v.scale * (e.evt.deltaY > 0 ? 0.9 : 1.1), MIN_SCALE, MAX_SCALE);
    p.onViewport({ scale: newScale, ox: pointer.x - mx * newScale, oy: pointer.y - my * newScale });
  }

  function onDown(e: Konva.KonvaEventObject<MouseEvent>) {
    if (p.panning) return; // space-pan handled by Stage drag
    const [x, y] = toImg();
    if (p.tool === "box" || p.tool === "sam-box") {
      setDraw({ x0: x, y0: y, x1: x, y1: y });
    } else if (p.tool === "sam-point") {
      p.onSamPoint([x, y], e.evt.shiftKey ? 0 : 1);
    } else if (p.tool === "select" && e.target === e.target.getStage()) {
      p.onSelect(null); // clicked empty canvas
    }
  }

  function onMove() {
    const [x, y] = toImg();
    p.onCursor([x, y]);
    if (draw) setDraw((d) => (d ? { ...d, x1: x, y1: y } : d));
  }

  function onUp() {
    if (!draw) return;
    const box = [Math.min(draw.x0, draw.x1), Math.min(draw.y0, draw.y1), Math.max(draw.x0, draw.x1), Math.max(draw.y0, draw.y1)];
    setDraw(null);
    if (box[2] - box[0] < 3 || box[3] - box[1] < 3) return; // ignore tiny
    if (p.tool === "box") p.onDrawBox(box);
    else if (p.tool === "sam-box") p.onSamBox(box);
  }

  const sel = p.objects.find((o) => o.id === p.selectedId) || null;

  return (
    <div ref={wrapRef} className="w-full h-full bg-bg-2 overflow-hidden"
      style={{ cursor: p.panning ? "grab" : p.tool === "select" ? "default" : "crosshair" }}>
      <Stage
        ref={stageRef}
        width={size.w}
        height={size.h}
        scaleX={v.scale}
        scaleY={v.scale}
        x={v.ox}
        y={v.oy}
        draggable={p.panning}
        onWheel={onWheel}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={() => p.onCursor(null)}
        onDragEnd={(e) => {
          if (p.panning) p.onViewport({ ...v, ox: e.target.x(), oy: e.target.y() });
        }}
      >
        <Layer>
          {img && <KImage image={img} width={p.imgW} height={p.imgH} listening={false} />}

          {/* drivable-area segmentation (M2.2), drawn behind everything */}
          {L.drivable && p.drivable && Object.entries(p.drivable).flatMap(([cls, polys]) =>
            polys.map((poly, i) => (
              <Line key={`dr-${cls}-${i}`} points={poly} closed listening={false}
                fill={cls === "drivable" ? "rgba(86,211,100,0.18)" : cls === "fallback" ? "rgba(227,179,65,0.18)" : "rgba(248,81,73,0.14)"}
                stroke={cls === "drivable" ? "#56D364" : cls === "fallback" ? "#E3B341" : "#F85149"} strokeWidth={1 / v.scale} />
            )),
          )}

          {/* masks */}
          {L.masks && p.objects.filter((o) => o.visible).flatMap((o) =>
            o.mask.map((poly, i) => (
              <Line key={`m${o.id}-${i}`} points={poly} closed listening={false}
                stroke={classColor(o.class_id)} strokeWidth={1.5 / v.scale}
                fill={classFill(o.class_id, o.id === p.selectedId ? 0.3 : 0.16)} />
            )),
          )}

          {/* lane splines (M2.1), drawn above masks */}
          {L.lanes && p.lanes?.map((ln) => (
            <Line key={ln.lane_id} points={ln.control_points.flat()} listening={false} tension={0.3}
              stroke={ln.is_ego ? "#56D364" : ln.source === "human" ? "#FF7A2F" : ln.source === "propagated" ? "#E3B341" : "#58A6FF"}
              strokeWidth={2.5 / v.scale} dash={ln.lane_type === "dashed" ? [10 / v.scale, 8 / v.scale] : undefined} />
          ))}

          {/* boxes */}
          {L.boxes && p.objects.filter((o) => o.visible).map((o) => {
            const w = o.bbox[2] - o.bbox[0];
            const h = o.bbox[3] - o.bbox[1];
            const isSel = o.id === p.selectedId;
            return (
              <Rect
                key={o.id}
                ref={isSel ? selRectRef : undefined}
                x={o.bbox[0]} y={o.bbox[1]} width={w} height={h}
                stroke={classColor(o.class_id)} strokeWidth={(isSel ? 2.5 : 1.5) / v.scale}
                dash={o.isNew ? [6 / v.scale, 4 / v.scale] : undefined}
                draggable={isSel && p.tool === "select"}
                onMouseDown={(e) => {
                  if (p.tool === "select") {
                    e.cancelBubble = true;
                    p.onSelect(o.id);
                  }
                }}
                onDragEnd={(e) => {
                  const nx = e.target.x();
                  const ny = e.target.y();
                  p.onUpdateBbox(o.id, [nx, ny, nx + w, ny + h]);
                }}
                onTransformEnd={(e) => {
                  const node = e.target as Konva.Rect;
                  const sx = node.scaleX();
                  const sy = node.scaleY();
                  node.scaleX(1);
                  node.scaleY(1);
                  const nx = node.x();
                  const ny = node.y();
                  p.onUpdateBbox(o.id, [nx, ny, nx + w * sx, ny + h * sy]);
                }}
              />
            );
          })}

          {/* selected object's mask vertices (drag to edit) */}
          {sel && p.tool === "select" && sel.mask.map((poly, pi) =>
            poly.map((_, k) =>
              k % 2 === 0 ? (
                <Circle key={`v${pi}-${k}`} x={poly[k]} y={poly[k + 1]} radius={3.5 / v.scale}
                  fill="#0B0C0E" stroke={classColor(sel.class_id)} strokeWidth={1.5 / v.scale} draggable
                  onDragEnd={(e) => {
                    const next = sel.mask.map((pp) => pp.slice());
                    next[pi][k] = e.target.x();
                    next[pi][k + 1] = e.target.y();
                    p.onUpdateMask(sel.id, next);
                  }} />
              ) : null,
            ),
          )}

          {/* SAM candidate */}
          {p.candidate?.map((poly, i) => (
            <Line key={`cand${i}`} points={poly} closed listening={false}
              stroke="#56D364" strokeWidth={2 / v.scale} fill="rgba(86,211,100,0.25)" />
          ))}

          {/* rubber-band while drawing */}
          {draw && (
            <Rect x={Math.min(draw.x0, draw.x1)} y={Math.min(draw.y0, draw.y1)}
              width={Math.abs(draw.x1 - draw.x0)} height={Math.abs(draw.y1 - draw.y0)} listening={false}
              stroke={p.tool === "sam-box" ? "#56D364" : "#FF7A2F"} strokeWidth={1.5 / v.scale}
              dash={[6 / v.scale, 4 / v.scale]} />
          )}

          {p.tool === "select" && (
            <Transformer ref={trRef} rotateEnabled={false} ignoreStroke borderStroke="#FF7A2F"
              anchorStroke="#FF7A2F" anchorFill="#0B0C0E" anchorSize={8 / v.scale}
              boundBoxFunc={(oldB, newB) => (newB.width < 4 || newB.height < 4 ? oldB : newB)} />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
