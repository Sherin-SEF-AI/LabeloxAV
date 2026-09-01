"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useParams, useRouter } from "next/navigation";
import { useConfirm } from "@/components/ConfirmProvider";
import { api, humanizeError } from "@/lib/api";
import type { FrameMeta, LaneRow } from "@/lib/types";
import BackButton from "@/components/BackButton";

// M2.1 lane spline editor: propose lanes (CLRerNet on pod / classical local), edit as draggable control
// points, draw implicit lanes on unmarked roads, pick lane type, mark ego, propagate across frames.
//
// Two views over the same lanes. The camera view is where the paint is, and it is where a lane is hardest
// to judge: the road runs to a vanishing point, so the far half of a lane is a handful of pixels, two
// parallel lanes converge, and a curve and a lane change look the same. The bird's-eye view flattens the
// road, and on it parallel is parallel, a constant-width lane has constant width, and a curve is a curve.
//
// It could not be built before because half the mathematics was missing: lifting a pixel to the ground has
// existed since the georeferencing work, and turning a ground point back into a pixel appeared nowhere in
// the repo. Without that inverse a BEV can be looked at and not drawn on. Lanes drawn here are stored in
// image space like every other lane, so nothing downstream knows the difference.

// react-konva's reconciler cannot render lazy element types, so the whole Konva tree is one component
// loaded via next/dynamic(ssr:false), not per-primitive dynamic imports.
const LaneCanvas = dynamic(() => import("@/components/lane/LaneCanvas"), { ssr: false });

const TYPES = ["solid", "dashed", "double", "road_edge", "implicit", "fallback"];
const COLOR: Record<string, string> = { proposed: "#58A6FF", human: "#FF7A2F", propagated: "#E3B341" };

type Lane = LaneRow & { dirty?: boolean };

export default function LaneEditor() {
  const confirm = useConfirm();
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

  // Bird's-eye view state. Loaded on demand: the warp is a server round trip and most lane work does not
  // need it.
  const [bev, setBev] = useState<Awaited<ReturnType<typeof api.frameBev>> | null>(null);
  const [bevLanes, setBevLanes] = useState<(LaneRow & { bev_points: number[][]; dropped: number })[]>([]);
  const [bevMode, setBevMode] = useState(false);
  const [bevDraw, setBevDraw] = useState<number[][] | null>(null);
  const bevRef = useRef<HTMLDivElement>(null);

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
  const del = async () => {
    if (!sel) return;
    // A lane is a hand-drawn spline that took real work and there is no undo on this surface, so the
    // deletion asks first. It used to go straight through on a single click.
    const lane = lanes.find((l) => l.lane_id === sel);
    if (!(await confirm({ title: "Delete this lane?",
                          body: lane ? `${lane.lane_type}${lane.is_ego ? ", ego lane" : ""}` : undefined,
                          danger: true, confirmLabel: "Delete" }))) return;
    await api.deleteLane(sel); setSel(null); await load();
  };

  const openBev = useCallback(async () => {
    setMsg("flattening the road...");
    try {
      const [v, ls] = await Promise.all([api.frameBev(frameId), api.lanesInBev(frameId)]);
      setBev(v);
      setBevLanes(ls.lanes);
      setBevMode(true);
      const dropped = ls.lanes.reduce((n, l) => n + l.dropped, 0);
      setMsg(dropped
        // Dropped rather than clamped, and said out loud: a control point above the horizon was never on
        // the road plane, and silently pinning it to the edge would invent a position.
        ? `${dropped} control point${dropped === 1 ? "" : "s"} sit above the horizon and are not on this view`
        : v.caveat);
    } catch (e) {
      setBevMode(false);
      setMsg(`no bird's-eye view: ${humanizeError(e)}`);
    }
  }, [frameId]);

  const finishBevDraw = async (type: string) => {
    if (!bevDraw || bevDraw.length < 2) { setBevDraw(null); return; }
    try {
      await api.createLaneFromBev(frameId, { points: bevDraw, lane_type: type, is_ego: false });
      setBevDraw(null);
      await load();
      await openBev();
    } catch (e) {
      setMsg(humanizeError(e));
    }
  };

  const selLane = lanes.find((l) => l.lane_id === sel);

  return (
    <div className="min-h-screen flex flex-col">
      <h1 className="sr-only">Lane and drivable annotation</h1>
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
        <button onClick={() => (bevMode ? setBevMode(false) : void openBev())}
          title="flatten the road: parallel lanes look parallel and a curve looks like a curve"
          className={`border px-2 py-1 ${bevMode ? "border-accent text-accent" : "border-line hover:border-accent"}`}>
          {bevMode ? "camera view" : "bird's-eye"}
        </button>
        <button onClick={propagate} className="border border-line px-2 py-1 hover:border-accent">propagate →</button>
        {mapHint && (
          <span className="text-info border border-line px-2 py-0.5" title="OSM map prior (a hint, confirm against the markings)">
            map: {mapHint.road_class}{mapHint.lane_count ? ` · ${mapHint.lane_count} lanes` : ""}{mapHint.speed_limit ? ` · ${mapHint.speed_limit}` : ""}
          </span>
        )}
        <button onClick={save} className="border border-pass text-pass px-2 py-1 hover:bg-pass/10 ml-auto">save</button>
        {msg && <span className="text-warn">{msg}</span>}
      </header>

      {/* The BEV drawing strip. Plain DOM rather than Konva: it is one image with an overlaid SVG, and
          pulling in a second canvas stack for that would be a lot of machinery for a polyline. */}
      {bevMode && bev && (
        <div className="border-b hairline bg-bg-2 p-2 flex gap-3 items-start overflow-auto">
          <div ref={bevRef} className="relative shrink-0 border border-line"
            style={{ width: bev.view.width, height: bev.view.height }}
            onClick={(e) => {
              if (!bevDraw) return;
              const r = e.currentTarget.getBoundingClientRect();
              setBevDraw([...bevDraw, [e.clientX - r.left, e.clientY - r.top]]);
            }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={bev.image} alt="the road, flattened" className="block select-none"
              style={{ width: bev.view.width, height: bev.view.height }} draggable={false} />
            <svg className="absolute inset-0 pointer-events-none"
              width={bev.view.width} height={bev.view.height}>
              {bevLanes.map((l) => (
                l.bev_points.length >= 2 ? (
                  <polyline key={l.lane_id} fill="none" strokeWidth={2}
                    stroke={l.is_ego ? "#56D364" : (COLOR[l.source] || "#A0A6AD")}
                    strokeDasharray={l.lane_type === "dashed" ? "8 6" : undefined}
                    points={l.bev_points.map((p) => `${p[0]},${p[1]}`).join(" ")} />
                ) : null
              ))}
              {bevDraw && bevDraw.length > 0 && (
                <>
                  <polyline fill="none" stroke="#FF7A2F" strokeWidth={2} strokeDasharray="4 3"
                    points={bevDraw.map((p) => `${p[0]},${p[1]}`).join(" ")} />
                  {bevDraw.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={3} fill="#FF7A2F" />)}
                </>
              )}
              {/* Distance rules, because the whole point of this view is that it is metric. */}
              {[10, 20, 30, 40, 50].filter((m) => m > bev.view.near_m && m < bev.view.far_m).map((m) => {
                const y = (bev.view.far_m - m) * bev.view.px_per_m;
                return (
                  <g key={m}>
                    <line x1={0} y1={y} x2={bev.view.width} y2={y} stroke="#3D444D" strokeWidth={1} />
                    <text x={4} y={y - 3} fill="#6C727A" fontSize={10} fontFamily="monospace">{m} m</text>
                  </g>
                );
              })}
            </svg>
          </div>
          <div className="font-mono text-[11px] space-y-2 min-w-[16rem]">
            <div className="text-ink-3 uppercase text-[10px]">bird&apos;s-eye</div>
            <div className="text-ink-3">
              {bev.view.near_m}&ndash;{bev.view.far_m} m ahead, &plusmn;{bev.view.half_width_m} m across,
              at {bev.view.px_per_m} px/m
            </div>
            {/* A nominal calibration and a measured one produce views that look identical and mean
                different things. An annotator drawing metric lanes should know which they have. */}
            <div className={bev.calibration.source === "nominal" ? "text-warn" : "text-ink-3"}>
              calibration: {bev.calibration.source} ({bev.calibration.model})
            </div>
            <div className="text-ink-3">{bev.caveat}</div>
            {bevDraw ? (
              <div className="space-y-1.5">
                <div className="text-accent">{bevDraw.length} point{bevDraw.length === 1 ? "" : "s"} - click to add</div>
                <div className="flex gap-1.5 flex-wrap">
                  {["solid", "dashed", "implicit"].map((t) => (
                    <button key={t} onClick={() => void finishBevDraw(t)} disabled={bevDraw.length < 2}
                      className="border border-line px-2 py-1 hover:border-accent disabled:opacity-40">
                      finish as {t}
                    </button>
                  ))}
                  <button onClick={() => setBevDraw(null)} className="text-ink-3 hover:text-block px-1">cancel</button>
                </div>
              </div>
            ) : (
              <button onClick={() => setBevDraw([])}
                className="border border-line px-2 py-1 hover:border-accent">draw a lane here</button>
            )}
          </div>
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        <div ref={wrapRef} className="flex-1 overflow-hidden bg-bg-2">
          {img && meta && (
            <LaneCanvas img={img} meta={meta} scale={scale} lanes={lanes} sel={sel} drivable={drivable}
              adding={adding} onStageClick={onStageClick} onSelect={setSel} onDragPoint={dragPoint} />
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
