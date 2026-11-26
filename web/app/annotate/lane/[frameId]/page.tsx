"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { useParams, useRouter } from "next/navigation";
import { useConfirm } from "@/components/ConfirmProvider";
import { api, humanizeError } from "@/lib/api";
import type { FrameMeta, LaneRow } from "@/lib/types";
import BackButton from "@/components/BackButton";
import { pressable } from "@/components/ui/pressable";
import { T } from "@/lib/i18n";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, shortId } from "@/lib/format";

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
// The top-down editor (#7): splines, types, connections, signals, pose propagation and map export.
const BevLaneEditor = dynamic(() => import("@/components/lane/BevLaneEditor"), { ssr: false });

const TYPES = ["solid", "dashed", "double", "road_edge", "implicit", "fallback", "stop_line", "unknown"];
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
  const [busy, setBusy] = useState<string | null>(null);
  const [drivable, setDrivable] = useState<Record<string, number[][]> | null>(null);
  const [mapHint, setMapHint] = useState<{ road_class?: string; lane_count?: number | null; speed_limit?: number | null } | null>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);

  // Bird's-eye view state. Loaded on demand: the warp is a server round trip and most lane work does not
  // need it.
  const [bev, setBev] = useState<Awaited<ReturnType<typeof api.frameBev>> | null>(null);
  const [bevMode, setBevMode] = useState(false);

  const load = useCallback(async () => {
    const [m, ls] = await Promise.all([api.frame(frameId), api.framesLanes(frameId)]);
    setMeta(m);
    setLanes(ls);
    const im = new window.Image();
    im.src = m.image_url;
    im.onload = () => setImg(im);
  }, [frameId]);

  useEffect(() => { load().catch((e) => setMsg(`Could not load this frame: ${humanizeError(e)}`)); }, [load]);
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
    setMsg("Saved");
    await load();
  };

  // Each of these is a server round trip of a second or more (a model runs for the first two), so the button
  // says it is working, and a failure is reported rather than left as an unhandled rejection.
  const run = async (what: string, fn: () => Promise<void>) => {
    setBusy(what);
    try { await fn(); } catch (e) { setMsg(`${what} failed: ${humanizeError(e)}`); } finally { setBusy(null); }
  };
  const segDrivable = () => run("Drivable area", async () => {
    await api.segmentDrivable(frameId);
    const d = await api.getDrivable(frameId);
    setDrivable(d.found ? d.classes! : null);
    setMsg(d.found ? `Drivable area: ${Math.round((d.coverage!.drivable || 0) * 100)}% of the frame` : "No drivable surface found");
  });
  const propose = () => run("Propose lanes", async () => { const r = await api.proposeLanes(frameId); setMsg(`Proposed ${r.proposed} lane${r.proposed === 1 ? "" : "s"} (${r.model})`); await load(); });
  const propagate = () => run("Propagate", async () => { const r = await api.propagateLanes(frameId, 8); setMsg(`Propagated to ${r.created} lanes on the following frames`); });
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
    setMsg("Flattening the road...");
    try {
      const v = await api.frameBev(frameId);
      setBev(v);
      setBevMode(true);
      setMsg(null);
    } catch (e) {
      setBevMode(false);
      setMsg(`No bird's-eye view: ${humanizeError(e)}`);
    }
  }, [frameId]);

  const selLane = lanes.find((l) => l.lane_id === sel);
  const msgIsBad = !!msg && /failed|could not|^no /i.test(msg);

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex items-center gap-2 px-3 h-12 border-b hairline shrink-0 text-[12px]">
        <BackButton label="Back" />
        <h1 className="text-[15px] font-semibold tracking-tight text-ink ml-1">
          Lanes <span className="sr-only">and drivable annotation</span>
        </h1>
        <span className="text-[12.5px] text-ink-3 mr-2 font-mono" title={frameId}>{shortId(frameId)}</span>
        <button type="button" onClick={propose} disabled={!!busy} className="btn"
          title="detect the lane markings on this frame and add them as proposed lanes">
          {busy === "Propose lanes" ? "Proposing..." : "Propose lanes"}
        </button>
        <button type="button" onClick={segDrivable} disabled={!!busy} className="btn"
          title="segment the drivable road surface and overlay it">
          {busy === "Drivable area" ? "Segmenting..." : "Drivable area"}
        </button>
        <button type="button" onClick={() => setAdding([])} aria-pressed={!!adding}
          title="click points along a lane on the image, then pick its type"
          className={`btn ${adding ? "btn-on" : ""}`}>+ Add lane</button>
        {adding && (
          <>
            <span className="text-ink-3">{adding.length} point{adding.length === 1 ? "" : "s"}, finish as:</span>
            {["solid", "implicit"].map((t) => <button key={t} type="button" onClick={() => finishAdd(t)} className="btn">{humanize(t)}</button>)}
            <button type="button" onClick={() => setAdding(null)} className="btn"><T k="action.cancel">Cancel</T></button>
          </>
        )}
        <span className="w-px h-5 bg-line mx-1" aria-hidden />
        <button type="button" onClick={() => (bevMode ? setBevMode(false) : void openBev())}
          title="flatten the road: parallel lanes look parallel and a curve looks like a curve"
          aria-pressed={bevMode}
          className={`btn ${bevMode ? "btn-on" : ""}`}>
          {bevMode ? "Camera view" : "Bird's-eye view"}
        </button>
        <button type="button" onClick={propagate} disabled={!!busy} className="btn"
          title="copy these lanes forward onto the next 8 frames">
          {busy === "Propagate" ? "Propagating..." : "Propagate →"}
        </button>
        {mapHint && (
          <span className="text-info border border-line rounded px-2 py-1" title="OSM map prior (a hint, confirm against the markings)">
            Map: {humanize(mapHint.road_class)}{mapHint.lane_count ? ` · ${mapHint.lane_count} lanes` : ""}{mapHint.speed_limit ? ` · ${mapHint.speed_limit} km/h` : ""}
          </span>
        )}
        {msg && <span role="status" className={`ml-auto truncate max-w-[480px] text-[12px] ${msgIsBad ? "text-warn" : "text-ink-2"}`} title={msg}>{msg}</span>}
        <button type="button" onClick={save} className={`btn btn-primary ${msg ? "" : "ml-auto"}`}><T k="action.save">Save</T></button>
      </header>

      {bevMode && bev && (
        <div className="border-b hairline bg-bg-2 p-2 overflow-auto">
          <BevLaneEditor frameId={frameId} image={bev.image} view={bev.view} calibrationSource={bev.calibration.source}
            caveat={bev.caveat} onChanged={() => { void load(); }} />
        </div>
      )}

      <div className="flex flex-1 min-h-0">
        <div ref={wrapRef} className="flex-1 overflow-hidden bg-bg-2 relative">
          {!img && (
            <div className="absolute inset-0 flex items-center justify-center text-[12.5px] text-ink-3" role="status">
              {msgIsBad ? msg : "Loading frame..."}
            </div>
          )}
          {img && meta && (
            <LaneCanvas img={img} meta={meta} scale={scale} lanes={lanes} sel={sel} drivable={drivable}
              adding={adding} onStageClick={onStageClick} onSelect={setSel} onDragPoint={dragPoint} />
          )}
        </div>

        <aside className="w-64 border-l hairline p-3 space-y-3 text-[12.5px] overflow-y-auto">
          <h2 className="section-title">Lanes <span className="font-normal text-ink-3">({lanes.length})</span></h2>
          {img && lanes.length === 0 && !adding && (
            <EmptyState compact icon="info" title="No lanes on this frame"
              action={<button type="button" onClick={propose} disabled={!!busy} className="btn">Propose lanes</button>}>
              Propose them from the lane markings, or use Add lane to click points along one by hand.
            </EmptyState>
          )}
          <div className="space-y-0.5">
          {lanes.map((l) => (
            <div key={l.lane_id} {...pressable(() => setSel(l.lane_id))} aria-pressed={l.lane_id === sel}
              className={`flex items-center gap-2 px-1.5 py-1 rounded cursor-pointer ${l.lane_id === sel ? "text-ink bg-accent/10" : "text-ink-2 hover:bg-head"}`}>
              <span className="w-2.5 h-2.5 inline-block rounded-sm" aria-hidden style={{ background: l.is_ego ? "#56D364" : (COLOR[l.source] || "#A0A6AD") }} />
              <span className="truncate flex-1">{humanize(l.lane_type)}{l.is_ego ? " (ego)" : ""}</span>
              <span className="text-ink-3 text-[11px]">{humanize(l.source)}</span>
            </div>
          ))}
          </div>
          {selLane && (
            <div className="border-t hairline pt-3 space-y-2">
              <div className="section-title">Selected lane</div>
              <select value={selLane.lane_type} onChange={(e) => setType(e.target.value)} aria-label="lane type" className="field w-full">
                {TYPES.map((t) => <option key={t} value={t}>{humanize(t)}</option>)}
              </select>
              <button type="button" onClick={toggleEgo} aria-pressed={!!selLane.is_ego} className={`btn w-full ${selLane.is_ego ? "btn-on" : ""}`}>{selLane.is_ego ? "Ego lane ✓" : "Mark as ego lane"}</button>
              <button type="button" onClick={() => router.push(`/frame/${frameId}`)} className="btn w-full">Open frame editor</button>
              <button type="button" onClick={del} className="btn btn-danger w-full mt-2"><T k="action.delete">Delete</T></button>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
}
