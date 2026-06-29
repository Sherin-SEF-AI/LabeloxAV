"use client";

// The interactive annotation surface. A fixed-size konva Stage with a pan/zoom viewport (the stage
// transform), rendering the frame image plus every object's box and mask in image space. Tools: select
// (move + resize via Transformer, drag mask vertices), draw box, SAM point, SAM box. All geometry is in
// image pixels; the stage transform maps to screen. getRelativePointerPosition() returns image coords.

import { useEffect, useRef, useState } from "react";
import { Circle, Group, Image as KImage, Layer, Line, Rect, Stage, Text as KText, Transformer } from "react-konva";
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
  relationships?: { from_object_id: string; to_object_id: string; kind: string }[];
  layers?: LayerFlags;
  onViewport: (v: Viewport) => void;
  onSelect: (id: string | null) => void;
  onUpdateBbox: (id: string, bbox: number[], rot?: number) => void;
  onDrawBox: (bbox: number[]) => void;
  onDrawPolygon: (points: number[]) => void;   // manual polygon: flattened [x,y,...], no GPU/SAM needed
  onDrawPolyline: (points: number[]) => void;  // open polyline (curb/road_edge/barrier), not closed
  keypointDraft?: number[][] | null;           // in-progress pose points [[x,y,v],...] (placed so far)
  skeletonEdges?: number[][];                  // index pairs connecting keypoints, for rendering
  onPlaceKeypoint: (pt: number[]) => void;     // keypoint tool: drop the next skeleton point
  onUpdateKeypoints: (id: string, points: number[][]) => void;  // drag a committed pose point
  mPerPx?: number;                             // metres per pixel for the measure tool (LiDAR BEV)
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
  const [poly, setPoly] = useState<number[]>([]); // in-progress manual polygon, flattened [x,y,...]
  const [measure, setMeasure] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null);
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
    im.onerror = () => setImg(null); // a missing frame image (404) must not leave the viewport unfit
  }, [p.imageUrl]);

  // fit from the known frame dimensions (viewport.scale === 0 is the "fit pending" sentinel). This must not
  // wait for the image to load, or a missing image leaves scale at 0 and every stroke/radius becomes Infinity.
  useEffect(() => {
    if (p.viewport.scale === 0 && size.w > 1 && p.imgW > 0 && p.imgH > 0) {
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
    } else if (p.tool === "measure") {
      setMeasure({ x0: x, y0: y, x1: x, y1: y });
    } else if (p.tool === "polygon" || p.tool === "polyline") {
      setPoly((pp) => [...pp, x, y]); // each click drops a vertex; double-click closes/commits
    } else if (p.tool === "keypoint") {
      p.onPlaceKeypoint([x, y]); // each click drops the next skeleton point
    } else if (p.tool === "sam-point") {
      p.onSamPoint([x, y], e.evt.shiftKey ? 0 : 1);
    } else if (p.tool === "select" && e.target === e.target.getStage()) {
      p.onSelect(null); // clicked empty canvas
    }
  }

  function onDblClick() {
    if (p.tool === "polygon") {
      if (poly.length >= 6) p.onDrawPolygon(poly); // at least 3 vertices, closed
      setPoly([]);
    } else if (p.tool === "polyline") {
      if (poly.length >= 4) p.onDrawPolyline(poly); // at least 2 vertices, open
      setPoly([]);
    }
  }

  // abandon a half-drawn polygon/polyline when the tool changes
  useEffect(() => { if (p.tool !== "polygon" && p.tool !== "polyline") setPoly([]); }, [p.tool]);
  useEffect(() => { if (p.tool !== "measure") setMeasure(null); }, [p.tool]);

  function onMove() {
    const [x, y] = toImg();
    p.onCursor([x, y]);
    if (draw) setDraw((d) => (d ? { ...d, x1: x, y1: y } : d));
    if (measure) setMeasure((m) => (m ? { ...m, x1: x, y1: y } : m));
  }

  function onUp() {
    if (measure) {
      // ruler is ephemeral: keep the last segment on screen until the next drag or tool change
      const tiny = Math.hypot(measure.x1 - measure.x0, measure.y1 - measure.y0) < 2;
      if (tiny) setMeasure(null);
    }
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
        onDblClick={onDblClick}
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

          {/* open polylines (curb/road_edge/barrier): an open line, no fill, no AABB box */}
          {L.boxes && p.objects.filter((o) => o.visible && o.polyline && o.polyline.length >= 2).map((o) => (
            <Line key={`pl${o.id}`} points={o.polyline!.flat()} listening={p.tool === "select"}
              stroke={classColor(o.class_id)} strokeWidth={(o.id === p.selectedId ? 2.5 : 1.5) / v.scale}
              hitStrokeWidth={10 / v.scale}
              onMouseDown={(e) => { if (p.tool === "select") { e.cancelBubble = true; p.onSelect(o.id); } }} />
          ))}

          {/* boxes (rendered around their centre so rotation is about the centre; bbox stays the AABB).
              Polyline objects render as the open line above, not as a box. */}
          {L.boxes && p.objects.filter((o) => o.visible && !(o.polyline && o.polyline.length >= 2)).map((o) => {
            const w = o.bbox[2] - o.bbox[0];
            const h = o.bbox[3] - o.bbox[1];
            const cx = o.bbox[0] + w / 2;
            const cy = o.bbox[1] + h / 2;
            const isSel = o.id === p.selectedId;
            return (
              <Rect
                key={o.id}
                ref={isSel ? selRectRef : undefined}
                x={cx} y={cy} offsetX={w / 2} offsetY={h / 2} width={w} height={h} rotation={o.rot ?? 0}
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
                  const ncx = e.target.x();
                  const ncy = e.target.y();
                  p.onUpdateBbox(o.id, [ncx - w / 2, ncy - h / 2, ncx + w / 2, ncy + h / 2]);
                }}
                onTransformEnd={(e) => {
                  const node = e.target as Konva.Rect;
                  const sx = node.scaleX();
                  const sy = node.scaleY();
                  node.scaleX(1);
                  node.scaleY(1);
                  const nw = w * sx;
                  const nh = h * sy;
                  const ncx = node.x();
                  const ncy = node.y();
                  const rot = ((node.rotation() % 360) + 360) % 360;
                  p.onUpdateBbox(o.id, [ncx - nw / 2, ncy - nh / 2, ncx + nw / 2, ncy + nh / 2], rot);
                }}
              />
            );
          })}

          {/* relationship connectors: a thin dashed line between the centres of two related objects */}
          {p.relationships?.map((r) => {
            const a = p.objects.find((o) => o.id === r.from_object_id);
            const b = p.objects.find((o) => o.id === r.to_object_id);
            if (!a || !b) return null;
            const ca = [(a.bbox[0] + a.bbox[2]) / 2, (a.bbox[1] + a.bbox[3]) / 2];
            const cb = [(b.bbox[0] + b.bbox[2]) / 2, (b.bbox[1] + b.bbox[3]) / 2];
            return <Line key={`rel-${r.from_object_id}-${r.to_object_id}-${r.kind}`} listening={false}
              points={[ca[0], ca[1], cb[0], cb[1]]} stroke="#E3B341" strokeWidth={1 / v.scale}
              dash={[4 / v.scale, 3 / v.scale]} />;
          })}

          {/* selected object's mask vertices: drag to move, right-click to delete */}
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
                  }}
                  onContextMenu={(e) => {
                    e.evt.preventDefault();
                    if (poly.length <= 6) return; // never below a triangle
                    const next = sel.mask.map((pp) => pp.slice());
                    next[pi].splice(k, 2);
                    p.onUpdateMask(sel.id, next);
                  }} />
              ) : null,
            ),
          )}

          {/* edge midpoints: click to insert a new vertex on that edge */}
          {sel && p.tool === "select" && sel.mask.map((poly, pi) => {
            const n = poly.length / 2;
            return Array.from({ length: n }, (_, j) => {
              const ax = poly[2 * j], ay = poly[2 * j + 1];
              const bx = poly[2 * ((j + 1) % n)], by = poly[2 * ((j + 1) % n) + 1];
              const mx = (ax + bx) / 2, my = (ay + by) / 2;
              return (
                <Circle key={`mid${pi}-${j}`} x={mx} y={my} radius={2.5 / v.scale}
                  fill="rgba(255,122,47,0.55)" stroke="#0B0C0E" strokeWidth={1 / v.scale}
                  onClick={() => {
                    const next = sel.mask.map((pp) => pp.slice());
                    next[pi].splice(2 * j + 2, 0, mx, my);
                    p.onUpdateMask(sel.id, next);
                  }} />
              );
            });
          })}

          {/* SAM candidate */}
          {p.candidate?.map((poly, i) => (
            <Line key={`cand${i}`} points={poly} closed listening={false}
              stroke="#56D364" strokeWidth={2 / v.scale} fill="rgba(86,211,100,0.25)" />
          ))}

          {/* in-progress manual polygon/polyline: open line + vertex dots; double-click commits */}
          {(p.tool === "polygon" || p.tool === "polyline") && poly.length >= 2 && (
            <>
              <Line points={poly} listening={false} stroke="#FF7A2F" strokeWidth={1.5 / v.scale}
                dash={[6 / v.scale, 4 / v.scale]} />
              {poly.map((_, k) => (k % 2 === 0 ? (
                <Circle key={`pp${k}`} x={poly[k]} y={poly[k + 1]} radius={3 / v.scale}
                  fill="#0B0C0E" stroke="#FF7A2F" strokeWidth={1.5 / v.scale} listening={false} />
              ) : null))}
            </>
          )}

          {/* committed keypoints: skeleton edges + dots (dots draggable when the object is selected) */}
          {L.boxes && p.objects.filter((o) => o.visible && o.keypoints?.points?.length).map((o) => {
            const pts = o.keypoints!.points;
            const isSel = o.id === p.selectedId;
            const col = classColor(o.class_id);
            return (
              <Group key={`kp-${o.id}`}>
                {(p.skeletonEdges ?? []).map(([a, b], ei) => {
                  const pa = pts[a], pb = pts[b];
                  if (!pa || !pb || pa[2] <= 0 || pb[2] <= 0) return null;
                  return <Line key={`e${ei}`} points={[pa[0], pa[1], pb[0], pb[1]]} listening={false}
                    stroke={col} strokeWidth={1.5 / v.scale} opacity={0.85} />;
                })}
                {pts.map((pt, ki) => (pt[2] <= 0 ? null : (
                  <Circle key={`k${ki}`} x={pt[0]} y={pt[1]} radius={3 / v.scale}
                    fill={pt[2] === 2 ? "#56D364" : "#E3B341"} stroke="#0B0C0E" strokeWidth={1 / v.scale}
                    draggable={isSel && p.tool === "select"}
                    onDragEnd={(e) => {
                      const next = pts.map((q) => q.slice());
                      next[ki] = [e.target.x(), e.target.y(), next[ki][2] || 2];
                      p.onUpdateKeypoints(o.id, next);
                    }} />
                )))}
              </Group>
            );
          })}

          {/* in-progress pose: placed points + the skeleton edges that connect them */}
          {p.tool === "keypoint" && p.keypointDraft && p.keypointDraft.length > 0 && (
            <Group>
              {(p.skeletonEdges ?? []).map(([a, b], ei) => {
                const pa = p.keypointDraft![a], pb = p.keypointDraft![b];
                if (!pa || !pb) return null;
                return <Line key={`de${ei}`} points={[pa[0], pa[1], pb[0], pb[1]]} listening={false}
                  stroke="#FF7A2F" strokeWidth={1.5 / v.scale} opacity={0.7} />;
              })}
              {p.keypointDraft.map((pt, ki) => (
                <Circle key={`dk${ki}`} x={pt[0]} y={pt[1]} radius={3.5 / v.scale} listening={false}
                  fill="#FF7A2F" stroke="#0B0C0E" strokeWidth={1 / v.scale} />
              ))}
            </Group>
          )}

          {/* measure / ruler: line + distance (px, and metres on a LiDAR BEV) */}
          {measure && (() => {
            const dpx = Math.hypot(measure.x1 - measure.x0, measure.y1 - measure.y0);
            const label = p.mPerPx ? `${(dpx * p.mPerPx).toFixed(2)} m` : `${Math.round(dpx)} px`;
            return (
              <Group listening={false}>
                <Line points={[measure.x0, measure.y0, measure.x1, measure.y1]}
                  stroke="#58A6FF" strokeWidth={1.5 / v.scale} dash={[5 / v.scale, 4 / v.scale]} />
                <Circle x={measure.x0} y={measure.y0} radius={2.5 / v.scale} fill="#58A6FF" />
                <Circle x={measure.x1} y={measure.y1} radius={2.5 / v.scale} fill="#58A6FF" />
                <KText x={(measure.x0 + measure.x1) / 2 + 6 / v.scale} y={(measure.y0 + measure.y1) / 2 - 8 / v.scale}
                  text={label} fontSize={13 / v.scale} fill="#58A6FF" />
              </Group>
            );
          })()}

          {/* rubber-band while drawing */}
          {draw && (
            <Rect x={Math.min(draw.x0, draw.x1)} y={Math.min(draw.y0, draw.y1)}
              width={Math.abs(draw.x1 - draw.x0)} height={Math.abs(draw.y1 - draw.y0)} listening={false}
              stroke={p.tool === "sam-box" ? "#56D364" : "#FF7A2F"} strokeWidth={1.5 / v.scale}
              dash={[6 / v.scale, 4 / v.scale]} />
          )}

          {p.tool === "select" && (
            <Transformer ref={trRef} rotateEnabled rotationSnaps={[0, 90, 180, 270]} ignoreStroke
              borderStroke="#FF7A2F" anchorStroke="#FF7A2F" anchorFill="#0B0C0E" anchorSize={8 / v.scale}
              boundBoxFunc={(oldB, newB) => (newB.width < 4 || newB.height < 4 ? oldB : newB)} />
          )}
        </Layer>
      </Stage>
    </div>
  );
}
