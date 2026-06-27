"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { FrameMeta, LaneRow } from "@/lib/types";
import BackButton from "@/components/BackButton";

// M2.1 lane spline editor: propose lanes (CLRerNet on pod / classical local), edit as draggable control
// points, draw implicit lanes on unmarked roads, pick lane type, mark ego, propagate across frames.

const Stage = dynamic(() => import("react-konva").then((m) => m.Stage), { ssr: false });
const Layer = dynamic(() => import("react-konva").then((m) => m.Layer), { ssr: false });
const KImage = dynamic(() => import("react-konva").then((m) => m.Image), { ssr: false });
const Line = dynamic(() => import("react-konva").then((m) => m.Line), { ssr: false });
const Circle = dynamic(() => import("react-konva").then((m) => m.Circle), { ssr: false });

const TYPES = ["solid", "dashed", "double", "road_edge", "implicit", "fallback"];
const COLOR: Record<string, string> = { proposed: "#58A6FF", human: "#FF7A2F", propagated: "#E3B341" };

type Lane = LaneRow & { dirty?: boolean };

export default function LaneEditor() {
  const router = useRouter();
  const frameId = String(useParams().frameId);
  const [meta, setMeta] = useState<FrameMeta | null>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [lanes, setLanes] = useState<Lane[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [adding, setAdding] = useState<number[][] | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [drivable, setDrivable] = useState<Record<string, number[][]> | null>(null);
  const [mapHint, setMapHint] = useState<{ road_class?: string; lane_count?: number | null; speed_limit?: number | null } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);

  const load = useCallback(async () => {
    const [m, ls] = await Promise.all([api.frame(frameId), api.framesLanes(frameId)]);
    setMeta(m);
    setLanes(ls);
    const im = new window.Image();
    im.src = m.image_url;
    im.onload = () => setImg(im);
  }, [frameId]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    api.framePriors(frameId).then((p) => setMapHint(p.has_map ? { road_class: p.road_class, lane_count: p.lane_count, speed_limit: p.speed_limit } : null)).catch(() => {});
  }, [frameId]);
  useEffect(() => {
    if (!meta || !wrapRef.current) return;
    setScale(wrapRef.current.clientWidth / meta.width);
  }, [meta, img]);

  const toImg = (e: { evt: MouseEvent }) => {
    const r = wrapRef.current!.getBoundingClientRect();
    return [(e.evt.clientX - r.left) / scale, (e.evt.clientY - r.top) / scale];
  };

  const onStageClick = (e: { evt: MouseEvent }) => {
    if (adding) setAdding([...adding, toImg(e)]);
  };

  const dragPoint = (laneId: string, i: number, x: number, y: number) => {
    setLanes((ls) => ls.map((l) => l.lane_id === laneId
      ? { ...l, dirty: true, control_points: l.control_points.map((p, j) => (j === i ? [x, y] : p)) } : l));
  };

  const save = async () => {
    for (const l of lanes.filter((x) => x.dirty)) {
      await api.updateLane(l.lane_id, { control_points: l.control_points, lane_type: l.lane_type, is_ego: l.is_ego });
    }
    setMsg("saved");
    await load();
  };

  const segDrivable = async () => {
    await api.segmentDrivable(frameId);
    const d = await api.getDrivable(frameId);
    setDrivable(d.found ? d.classes! : null);
    setMsg(d.found ? `drivable ${Math.round((d.coverage!.drivable || 0) * 100)}%` : "no surface");
  };
  const propose = async () => { const r = await api.proposeLanes(frameId); setMsg(`proposed ${r.proposed} (${r.model})`); await load(); };
  const propagate = async () => { const r = await api.propagateLanes(frameId, 8); setMsg(`propagated to ${r.created} lane-frames`); };
  const finishAdd = async (type: string) => {
    if (!adding || adding.length < 2) { setAdding(null); return; }
    await api.createLane(frameId, { control_points: adding, lane_type: type, is_ego: false });
    setAdding(null);
    await load();
  };
  const setType = (t: string) => { if (sel) setLanes((ls) => ls.map((l) => l.lane_id === sel ? { ...l, lane_type: t, dirty: true } : l)); };
  const toggleEgo = () => { if (sel) setLanes((ls) => ls.map((l) => ({ ...l, is_ego: l.lane_id === sel ? !l.is_ego : l.is_ego, dirty: l.lane_id === sel ? true : l.dirty }))); };
  const del = async () => { if (sel) { await api.deleteLane(sel); setSel(null); await load(); } };

  const selLane = lanes.find((l) => l.lane_id === sel);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex items-center gap-3 px-3 h-11 border-b hairline shrink-0 font-mono text-[11px]">
        <BackButton />
        <span className="text-ink-3">/ LANES <span className="text-ink-2">{frameId.slice(0, 8)}</span></span>
        <button onClick={propose} className="border border-line px-2 py-1 hover:border-accent">propose lanes</button>
        <button onClick={segDrivable} className="border border-line px-2 py-1 hover:border-accent">drivable</button>
        <button onClick={() => setAdding([])} className={`border px-2 py-1 ${adding ? "border-accent text-accent" : "border-line hover:border-accent"}`}>+ lane</button>
        {adding && (
          <>
            <span className="text-ink-3">{adding.length} pts, finish as:</span>
            {["solid", "implicit"].map((t) => <button key={t} onClick={() => finishAdd(t)} className="border border-line px-2 py-1 hover:border-accent">{t}</button>)}
            <button onClick={() => setAdding(null)} className="text-ink-3 hover:text-block">cancel</button>
          </>
        )}
        <button onClick={propagate} className="border border-line px-2 py-1 hover:border-accent">propagate →</button>
        {mapHint && (
          <span className="text-info border border-line px-2 py-0.5" title="OSM map prior (a hint, confirm against the markings)">
            map: {mapHint.road_class}{mapHint.lane_count ? ` · ${mapHint.lane_count} lanes` : ""}{mapHint.speed_limit ? ` · ${mapHint.speed_limit}` : ""}
          </span>
        )}
        <button onClick={save} className="border border-pass text-pass px-2 py-1 hover:bg-pass/10 ml-auto">save</button>
        {msg && <span className="text-warn">{msg}</span>}
      </header>

      <div className="flex flex-1 min-h-0">
        <div ref={wrapRef} className="flex-1 overflow-hidden bg-bg-2">
          {img && meta && (
            <Stage width={meta.width * scale} height={meta.height * scale} scaleX={scale} scaleY={scale} onMouseDown={onStageClick}>
              <Layer>
                <KImage image={img} width={meta.width} height={meta.height} listening={false} />
                {drivable && Object.entries(drivable).flatMap(([cls, polys]) =>
                  polys.map((poly, i) => (
                    <Line key={`dr-${cls}-${i}`} points={poly} closed listening={false}
                      fill={cls === "drivable" ? "rgba(86,211,100,0.22)" : cls === "fallback" ? "rgba(227,179,65,0.22)" : "rgba(248,81,73,0.16)"}
                      stroke={cls === "drivable" ? "#56D364" : cls === "fallback" ? "#E3B341" : "#F85149"} strokeWidth={1 / scale} />
                  )))}
                {lanes.map((l) => (
                  <Line key={l.lane_id} points={l.control_points.flat()} stroke={l.is_ego ? "#56D364" : (COLOR[l.source] || "#A0A6AD")}
                    strokeWidth={(l.lane_id === sel ? 4 : 2.5) / scale} dash={l.lane_type === "dashed" ? [10 / scale, 8 / scale] : undefined}
                    tension={0.3} onClick={() => setSel(l.lane_id)} hitStrokeWidth={14 / scale} />
                ))}
                {selLane?.control_points.map((p, i) => (
                  <Circle key={i} x={p[0]} y={p[1]} radius={6 / scale} fill="#FF7A2F" draggable
                    onDragMove={(e) => dragPoint(selLane.lane_id, i, e.target.x(), e.target.y())} />
                ))}
                {adding?.length ? <Line points={adding.flat()} stroke="#FF7A2F" strokeWidth={2 / scale} dash={[6 / scale, 4 / scale]} /> : null}
              </Layer>
            </Stage>
          )}
        </div>

        <aside className="w-56 border-l hairline p-3 space-y-3 font-mono text-[11px]">
          <div className="text-ink-3 uppercase text-[10px]">lanes ({lanes.length})</div>
          {lanes.map((l) => (
            <div key={l.lane_id} onClick={() => setSel(l.lane_id)} className={`flex items-center gap-1.5 cursor-pointer ${l.lane_id === sel ? "text-ink" : "text-ink-3"}`}>
              <span className="w-2.5 h-2.5 inline-block" style={{ background: l.is_ego ? "#56D364" : (COLOR[l.source] || "#A0A6AD") }} />
              <span className="truncate flex-1">{l.lane_type}{l.is_ego ? " (ego)" : ""}</span>
              <span className="text-ink-3">{l.source[0]}</span>
            </div>
          ))}
          {selLane && (
            <div className="border-t hairline pt-2 space-y-2">
              <div className="text-ink-3 uppercase text-[10px]">selected lane</div>
              <select value={selLane.lane_type} onChange={(e) => setType(e.target.value)} className="w-full bg-bg border border-line px-1 py-0.5 text-ink">
                {TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
              <button onClick={toggleEgo} className={`w-full border px-2 py-1 ${selLane.is_ego ? "border-pass text-pass" : "border-line text-ink-3"}`}>{selLane.is_ego ? "ego lane ✓" : "mark ego"}</button>
              <button onClick={del} className="w-full border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block">delete</button>
              <button onClick={() => router.push(`/frame/${frameId}`)} className="w-full border border-line text-ink-3 px-2 py-1 hover:border-accent">open frame editor</button>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
