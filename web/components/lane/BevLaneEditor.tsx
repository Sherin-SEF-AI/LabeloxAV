"use client";

// The top-down lane editor (#7). Lanes are drawn and reshaped as splines on the flattened road, typed, connected
// to one another (successor / predecessor / left and right neighbour), tied to a stop line and a signal, carried
// along the route by the ego poses, and exported as Lanelet2 and OpenDRIVE.
//
// Everything is stored the way lanes always were: image-space control points on the `lane` row. The spline drawn
// here is the one the export writes (lib/tools/drivingGeom.ts catmullRom mirrors services/hdmap/bev_lanes.py).
//
// Keys: D draw a lane, Enter finish it, Esc cancel or deselect, Delete remove the selected point (or lane),
// R add a relation from the selected lane (then click the target lane). Double-click a lane to add a point.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import { humanize, plural } from "@/lib/format";
import { catmullRom, metricToBev, nearestSegment, type BevView } from "@/lib/tools/drivingGeom";
import {
  LANE_RELATIONS, LANE_TYPES, addLaneRelation, bevLanes, deleteLaneRelation, downloadText, exportLanes,
  propagateLanesByPose, saveBevLane, type BevLane, type BevState, type LaneExport, type LaneRelationKind,
} from "@/lib/tools/drivingApi";
import { useToolJob } from "@/lib/tools/toolJobs";
import { useConfirm } from "@/components/ConfirmProvider";

type Props = {
  frameId: string;
  image: string;
  view: BevView;
  calibrationSource: string;
  caveat: string;
  onChanged: () => void;
};

// Colour says where a lane came from (the camera view's convention: proposed blue, a person's orange, carried
// yellow); the line pattern says its type. A white line would vanish into the paint it is drawn over.
const SOURCE_STROKE: Record<string, string> = {
  proposed: "rgb(var(--c-accent))",
  human: "rgb(255 122 47)",
  propagated: "rgb(var(--c-warn))",
};
const DASH: Record<string, string | undefined> = { dashed: "10 8", implicit: "3 5", fallback: "3 5", unknown: "1 4" };

const REL_LABEL: Record<LaneRelationKind, string> = {
  successor: "continues into",
  predecessor: "continues from",
  left_neighbour: "has to its left",
  right_neighbour: "has to its right",
  stops_at: "stops at",
  controlled_by: "is controlled by",
};

function splinePath(points: number[][]): string {
  const s = catmullRom(points, 8);
  return s.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
}

export default function BevLaneEditor({ frameId, image, view, calibrationSource, caveat, onChanged }: Props) {
  const confirm = useConfirm();
  const svgRef = useRef<SVGSVGElement>(null);
  const [st, setSt] = useState<BevState | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [zoom, setZoom] = useState(0.75);
  const [sel, setSel] = useState<string | null>(null);
  const [selPt, setSelPt] = useState<number | null>(null);
  const [draft, setDraft] = useState<number[][] | null>(null);
  const [drawType, setDrawType] = useState<string>("solid");
  const [edits, setEdits] = useState<Record<string, number[][]>>({});
  const [drag, setDrag] = useState<{ lane: string; i: number } | null>(null);
  const [relKind, setRelKind] = useState<LaneRelationKind>("successor");
  const [picking, setPicking] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [nProp, setNProp] = useState(10);
  const [exp, setExp] = useState<LaneExport | null>(null);
  const [propJob, setPropJob] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setSt(await bevLanes(frameId));
      setEdits({});
      setErr(null);
    } catch (e) {
      setErr(humanizeError(e));
    }
  }, [frameId]);
  useEffect(() => { void load(); }, [load]);

  useToolJob(propJob, {
    onDone: (j) => {
      setPropJob(null);
      const r = (j.result || {}) as { created?: number; frames?: number };
      setMsg(j.status === "done" ? `Carried ${plural(r.created ?? 0, "lane")} onto ${plural(r.frames ?? 0, "frame")}` : `Propagation ${j.status}: ${j.error ?? ""}`);
    },
  });

  const lanes = useMemo(() => st?.lanes ?? [], [st]);
  const pts = useCallback((l: BevLane) => edits[l.lane_id] ?? l.bev_points, [edits]);
  const selLane = lanes.find((l) => l.lane_id === sel) ?? null;
  const rels = useMemo(() => (st?.relations ?? []).filter((r) => r.lane_id === sel), [st, sel]);
  const laneName = useCallback((id: string | null) => {
    const i = lanes.findIndex((l) => l.lane_id === id);
    return i < 0 ? "a lane on another frame" : `lane ${i + 1}, ${humanize(lanes[i].lane_type).toLowerCase()}`;
  }, [lanes]);

  const toBev = (e: { clientX: number; clientY: number }): number[] => {
    const svg = svgRef.current;
    if (!svg) return [0, 0];
    const p = svg.createSVGPoint();
    p.x = e.clientX;
    p.y = e.clientY;
    const q = p.matrixTransform(svg.getScreenCTM()!.inverse());
    return [Math.round(q.x * 10) / 10, Math.round(q.y * 10) / 10];
  };

  const run = async (what: string, fn: () => Promise<void>) => {
    setBusy(what);
    try { await fn(); } catch (e) { setMsg(`${what} failed: ${humanizeError(e)}`); } finally { setBusy(null); }
  };

  const saveLane = (l: BevLane, points: number[][], patch: Partial<Pick<BevLane, "lane_type" | "is_ego">> = {}) =>
    run("Save", async () => {
      await saveBevLane(frameId, { lane_id: l.lane_id, bev_points: points, lane_type: patch.lane_type ?? l.lane_type,
                                   is_ego: patch.is_ego ?? l.is_ego });
      await load();
      onChanged();
      setMsg("Saved");
    });

  const finishDraft = useCallback(() => {
    if (!draft || draft.length < 2) { setDraft(null); return; }
    const d = draft;
    setDraft(null);
    void run("Draw", async () => {
      const r = await saveBevLane(frameId, { bev_points: d, lane_type: drawType, is_ego: false });
      await load();
      onChanged();
      setSel(r.lane_id);
      setMsg(`Added a ${humanize(drawType).toLowerCase()} lane`);
    });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, drawType, frameId, load, onChanged]);

  const removePoint = useCallback((l: BevLane, i: number) => {
    const cur = pts(l);
    if (cur.length <= 2) { setMsg("A lane keeps at least two points; delete the lane instead"); return; }
    const next = cur.filter((_, j) => j !== i);
    setEdits((e) => ({ ...e, [l.lane_id]: next }));
    setSelPt(null);
    void saveLane(l, next);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pts]);

  const deleteLane = useCallback(async (l: BevLane) => {
    if (!(await confirm({ title: "Delete this lane?", body: `${humanize(l.lane_type)}, ${humanize(l.source)}`, danger: true,
                          confirmLabel: "Delete" }))) return;
    await run("Delete", async () => { await api.deleteLane(l.lane_id); setSel(null); await load(); onChanged(); });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [confirm, load, onChanged]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "SELECT" || t.tagName === "TEXTAREA")) return;
      if (e.key === "d" || e.key === "D") { setDraft([]); setSel(null); setPicking(false); e.preventDefault(); }
      else if (e.key === "Enter" && draft) { finishDraft(); e.preventDefault(); }
      else if (e.key === "Escape") { setDraft(null); setPicking(false); if (!draft) { setSel(null); setSelPt(null); } }
      else if ((e.key === "Delete" || e.key === "Backspace") && selLane) {
        e.preventDefault();
        if (selPt !== null) removePoint(selLane, selPt); else void deleteLane(selLane);
      } else if ((e.key === "r" || e.key === "R") && selLane) { setPicking(true); e.preventDefault(); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [draft, finishDraft, selLane, selPt, removePoint, deleteLane]);

  const pickTarget = (target: BevLane) => {
    if (!selLane || target.lane_id === selLane.lane_id) return;
    setPicking(false);
    void run("Relation", async () => {
      await addLaneRelation(selLane.lane_id, { relation: relKind, target_lane_id: target.lane_id });
      await load();
      setMsg(`Lane ${REL_LABEL[relKind]} ${laneName(target.lane_id)}`);
    });
  };

  const onLaneClick = (l: BevLane, e: React.MouseEvent) => {
    e.stopPropagation();
    if (draft) return;
    if (picking) { pickTarget(l); return; }
    setSel(l.lane_id);
    setSelPt(null);
  };

  const onLaneDouble = (l: BevLane, e: React.MouseEvent) => {
    e.stopPropagation();
    const [x, y] = toBev(e);
    const cur = pts(l);
    const seg = nearestSegment(cur, x, y);
    if (seg.index < 0) return;
    const next = [...cur.slice(0, seg.index + 1), [x, y], ...cur.slice(seg.index + 1)];
    setEdits((ed) => ({ ...ed, [l.lane_id]: next }));
    void saveLane(l, next);
  };

  const onMove = (e: React.PointerEvent) => {
    if (!drag) return;
    const [x, y] = toBev(e);
    setEdits((ed) => {
      const l = lanes.find((q) => q.lane_id === drag.lane);
      if (!l) return ed;
      const cur = ed[drag.lane] ?? l.bev_points;
      return { ...ed, [drag.lane]: cur.map((p, j) => (j === drag.i ? [x, y] : p)) };
    });
  };
  const onUp = () => {
    if (!drag) return;
    const l = lanes.find((q) => q.lane_id === drag.lane);
    setDrag(null);
    if (l && edits[l.lane_id]) void saveLane(l, edits[l.lane_id]);
  };

  // Lanelets the person has paired: shade between the two splines so the lane reads as a lane.
  const pairs = useMemo(() => (st?.relations ?? []).filter((r) => r.relation === "right_neighbour" || r.relation === "left_neighbour"),
    [st]);

  const doExport = (fmt: "lanelet2" | "opendrive") => run("Export", async () => {
    const r = await exportLanes([frameId]);
    setExp(r);
    downloadText(fmt === "lanelet2" ? `lanes_${frameId.slice(0, 8)}.osm` : `lanes_${frameId.slice(0, 8)}.xodr`,
      fmt === "lanelet2" ? r.lanelet2 : r.opendrive);
  });

  const doPropagate = () => run("Propagate", async () => {
    const r = await propagateLanesByPose(frameId, nProp);
    if (r.queued) { setPropJob(r.job_id); setMsg(`Carrying lanes onto ${nProp} frames in the background...`); return; }
    setMsg(r.reason ?? `Carried ${plural(r.created, "lane")} onto ${plural(r.frames_with_lanes, "frame")}`
      + (r.out_of_view ? `; ${r.out_of_view} left the view` : "")
      + (r.skipped_human ? `; ${r.skipped_human} kept a person's lane already there` : "")
      + (r.skipped_no_pose ? `; ${r.skipped_no_pose} frames had no pose` : ""));
  });

  if (err) return <div className="p-3 text-[12px] text-warn" role="alert">Top-down editor unavailable: {err}</div>;
  if (!st) return <div className="p-3 text-[12px] text-ink-3" role="status">Loading lanes...</div>;

  const W = view.width;
  const H = view.height;
  const rules = [10, 20, 30, 40, 50].filter((m) => m > view.near_m && m < view.far_m);

  return (
    <div className="flex gap-3 items-start">
      <div className="shrink-0 border border-line overflow-auto max-h-[78vh] bg-bg" style={{ width: W * zoom + 2 }}>
        <svg ref={svgRef} width={W * zoom} height={H * zoom} viewBox={`0 0 ${W} ${H}`}
          role="application" aria-label="Top-down lane editor"
          className={draft || picking ? "cursor-crosshair" : "cursor-default"}
          onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}
          onClick={(e) => { if (draft) setDraft([...draft, toBev(e)]); else if (!picking) { setSel(null); setSelPt(null); } }}>
          <image href={image} x={0} y={0} width={W} height={H} preserveAspectRatio="none" />
          {rules.map((m) => {
            const y = metricToBev(view, m, 0)[1];
            return (
              <g key={m} pointerEvents="none">
                <line x1={0} y1={y} x2={W} y2={y} stroke="rgb(var(--c-line-2))" strokeWidth={1} />
                <text x={4} y={y - 3} fill="rgb(var(--c-ink-3))" fontSize={11} fontFamily="monospace">{m} m</text>
              </g>
            );
          })}
          {pairs.map((r) => {
            const a = lanes.find((l) => l.lane_id === r.lane_id);
            const b = lanes.find((l) => l.lane_id === r.target_lane_id);
            if (!a || !b) return null;
            const pa = catmullRom(pts(a), 6);
            const pb = catmullRom(pts(b), 6);
            const pbr = pb[0][1] < pb[pb.length - 1][1] === pa[0][1] < pa[pa.length - 1][1] ? [...pb].reverse() : pb;
            const poly = [...pa, ...pbr].map((p) => `${p[0]},${p[1]}`).join(" ");
            return <polygon key={r.relation_id} points={poly} fill="rgb(var(--c-accent) / 0.16)" stroke="none" pointerEvents="none" />;
          })}
          {lanes.map((l) => {
            const p = pts(l);
            if (p.length < 2) return null;
            const d = splinePath(p);
            const on = l.lane_id === sel;
            const col = l.lane_type === "stop_line" ? "rgb(var(--c-block))"
              : l.is_ego ? "rgb(var(--c-pass))" : SOURCE_STROKE[l.source] ?? "rgb(var(--c-ink-3))";
            return (
              <g key={l.lane_id}>
                <path d={d} fill="none" stroke="transparent" strokeWidth={16} style={{ cursor: picking ? "copy" : "pointer" }}
                  onClick={(e) => onLaneClick(l, e)} onDoubleClick={(e) => onLaneDouble(l, e)} />
                <path d={d} fill="none" stroke={col}
                  strokeWidth={on ? 4.5 : l.lane_type === "stop_line" || l.lane_type === "road_edge" ? 4 : 2.5}
                  strokeDasharray={DASH[l.lane_type]}
                  opacity={l.source === "propagated" ? 0.75 : 1} pointerEvents="none" />
                {l.lane_type === "double" && (
                  <path d={d} fill="none" stroke={col} strokeWidth={1.5} transform="translate(5,0)" pointerEvents="none" />
                )}
                {on && p.map((q, i) => (
                  <circle key={i} cx={q[0]} cy={q[1]} r={i === selPt ? 7 : 5.5}
                    fill={i === selPt ? "rgb(var(--c-warn))" : "rgb(var(--c-accent))"} stroke="rgb(var(--c-bg))" strokeWidth={1.5}
                    style={{ cursor: "grab" }}
                    onPointerDown={(e) => { e.stopPropagation(); (e.target as Element).setPointerCapture?.(e.pointerId); setDrag({ lane: l.lane_id, i }); setSelPt(i); }}
                    onClick={(e) => e.stopPropagation()} />
                ))}
                {(st.relations.filter((r) => r.lane_id === l.lane_id && r.relation === "successor")).map((r) => {
                  const t = lanes.find((x) => x.lane_id === r.target_lane_id);
                  if (!t) return null;
                  const a = p[p.length - 1];
                  const b = pts(t)[0];
                  return <line key={r.relation_id} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke="rgb(var(--c-pass))"
                    strokeWidth={2} strokeDasharray="4 3" markerEnd="url(#bev-arrow)" pointerEvents="none" />;
                })}
              </g>
            );
          })}
          <defs>
            <marker id="bev-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgb(var(--c-pass))" />
            </marker>
          </defs>
          {draft && draft.length > 0 && (
            <g pointerEvents="none">
              <path d={splinePath(draft)} fill="none" stroke="rgb(var(--c-warn))" strokeWidth={2.5} strokeDasharray="5 4" />
              {draft.map((q, i) => <circle key={i} cx={q[0]} cy={q[1]} r={4} fill="rgb(var(--c-warn))" />)}
            </g>
          )}
        </svg>
      </div>

      <div className="text-[12px] space-y-3 w-72 shrink-0">
        <div>
          <div className="section-title">Top-down lanes</div>
          <div className="text-ink-3">
            {view.near_m}&ndash;{view.far_m} m ahead, &plusmn;{view.half_width_m} m across.{" "}
            <span className={calibrationSource === "nominal" ? "text-warn" : ""}>Calibration: {calibrationSource}.</span>
          </div>
          <div className="text-ink-3 mt-1">{caveat}</div>
        </div>
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="label-caps">Zoom</span>
          {[0.5, 0.75, 1].map((z) => (
            <button key={z} type="button" className={`chip ${zoom === z ? "chip-on" : ""}`} onClick={() => setZoom(z)}>{Math.round(z * 100)}%</button>
          ))}
        </div>

        {draft ? (
          <div className="space-y-1.5 panel p-2">
            <div className="text-accent">{plural(draft.length, "point")}. Click the road to add more; <span className="kbd">Enter</span> finishes.</div>
            <label className="flex items-center gap-2">
              <span className="text-ink-3">Type</span>
              <select className="field flex-1" value={drawType} onChange={(e) => setDrawType(e.target.value)} aria-label="New lane type">
                {LANE_TYPES.map((t) => <option key={t} value={t}>{humanize(t)}</option>)}
              </select>
            </label>
            <div className="flex gap-1.5">
              <button type="button" className="btn btn-primary" disabled={draft.length < 2 || !!busy} onClick={finishDraft}>Finish lane</button>
              <button type="button" className="btn" onClick={() => setDraft(null)}>Cancel</button>
            </div>
          </div>
        ) : (
          <button type="button" className="btn w-full" onClick={() => { setDraft([]); setSel(null); }}>
            Draw a lane <span className="kbd ml-1">D</span>
          </button>
        )}

        {selLane ? (
          <div className="panel p-2 space-y-2">
            <div className="flex items-center justify-between">
              <span className="section-title">{laneName(selLane.lane_id).replace(/^lane/, "Lane")}</span>
              <span className="text-ink-3">{humanize(selLane.source)}{selLane.method === "ego_pose" ? " by pose" : ""}</span>
            </div>
            {selLane.dropped > 0 && (
              <div className="text-warn">{plural(selLane.dropped, "control point")} {selLane.dropped === 1 ? "lies" : "lie"} above the horizon, off this view; saving here drops {selLane.dropped === 1 ? "it" : "them"}.</div>
            )}
            <label className="flex items-center gap-2">
              <span className="text-ink-3 w-10">Type</span>
              <select className="field flex-1" value={selLane.lane_type} aria-label="Lane type"
                onChange={(e) => void saveLane(selLane, pts(selLane), { lane_type: e.target.value })}>
                {LANE_TYPES.map((t) => <option key={t} value={t}>{humanize(t)}</option>)}
              </select>
            </label>
            <button type="button" className={`btn w-full ${selLane.is_ego ? "btn-on" : ""}`} aria-pressed={selLane.is_ego}
              onClick={() => void saveLane(selLane, pts(selLane), { is_ego: !selLane.is_ego })}>
              {selLane.is_ego ? "Ego lane" : "Mark as ego lane"}
            </button>
            <div className="text-ink-3">Drag a point to move it, double-click the lane to add one, <span className="kbd">Delete</span> removes the selected point.</div>

            <div className="border-t hairline pt-2 space-y-1.5">
              <div className="label-caps">Connections</div>
              {rels.length === 0 && <div className="text-ink-3">None yet.</div>}
              {rels.map((r) => (
                <div key={r.relation_id} className="flex items-center gap-1.5">
                  <span className="flex-1 truncate">
                    {REL_LABEL[r.relation]}{" "}
                    {r.target_lane_id ? laneName(r.target_lane_id)
                      : r.target_object_id ? `signal ${r.target_object_id.slice(0, 8)}` : `map element ${String(r.target_element_id).slice(0, 8)}`}
                  </span>
                  <button type="button" className="btn" aria-label="Remove connection"
                    onClick={() => void run("Remove", async () => { await deleteLaneRelation(r.relation_id); await load(); })}>Remove</button>
                </div>
              ))}
              <div className="flex gap-1.5">
                <select className="field flex-1" value={relKind} aria-label="Connection kind"
                  onChange={(e) => setRelKind(e.target.value as LaneRelationKind)}>
                  {LANE_RELATIONS.map((k) => <option key={k} value={k}>{humanize(k)}</option>)}
                </select>
                {relKind !== "controlled_by" && (
                  <button type="button" className={`btn ${picking ? "btn-on" : ""}`} onClick={() => setPicking((p) => !p)}>
                    {picking ? "Click a lane..." : "Pick lane"} <span className="kbd ml-1">R</span>
                  </button>
                )}
              </div>
              {relKind === "stops_at" && <div className="text-ink-3">Pick a lane typed stop line.</div>}
              {relKind === "controlled_by" && (
                st.signals.length ? (
                  <div className="flex flex-wrap gap-1">
                    {st.signals.map((s, i) => (
                      <button key={s.object_id} type="button" className="chip"
                        onClick={() => void run("Relation", async () => {
                          await addLaneRelation(selLane.lane_id, { relation: "controlled_by", target_object_id: s.object_id });
                          await load();
                        })}>
                        {humanize(s.class_name)} {i + 1}
                      </button>
                    ))}
                  </div>
                ) : <div className="text-ink-3">No traffic signal is labelled on this frame. Label it in the frame editor first.</div>
              )}
            </div>
            <button type="button" className="btn btn-danger w-full" onClick={() => void deleteLane(selLane)}>Delete lane</button>
          </div>
        ) : (
          <div className="text-ink-3">{lanes.length ? "Click a lane to edit its shape, type and connections." : "No lanes on this frame yet."}</div>
        )}

        <div className="panel p-2 space-y-1.5">
          <div className="label-caps">Along the route</div>
          {st.has_pose ? (
            <>
              <div className="text-ink-3">Carry these lanes onto the next frames through the ego poses ({st.pose_source}). The road does not
                move, so the lanes land where they are, as far as the flat-road assumption and the poses allow.</div>
              <div className="flex gap-1.5 items-center">
                <input className="field w-16" type="number" min={1} max={Math.min(500, st.next_frames || 1)} value={nProp}
                  aria-label="Frames to carry onto" onChange={(e) => setNProp(Math.max(1, Number(e.target.value) || 1))} />
                <span className="text-ink-3">of {st.next_frames} frames</span>
                <button type="button" className="btn ml-auto" disabled={!!busy || !!propJob || !lanes.length} onClick={doPropagate}>
                  {busy === "Propagate" || propJob ? "Carrying..." : "Propagate"}
                </button>
              </div>
            </>
          ) : (
            <div className="text-ink-3">This frame has no ego pose, so lanes can only follow the paint: use Propagate in the header (optical flow).</div>
          )}
        </div>

        <div className="panel p-2 space-y-1.5">
          <div className="label-caps">Export this frame</div>
          <div className="flex gap-1.5">
            <button type="button" className="btn flex-1" disabled={!!busy || !lanes.length} onClick={() => doExport("lanelet2")}>Lanelet2</button>
            <button type="button" className="btn flex-1" disabled={!!busy || !lanes.length} onClick={() => doExport("opendrive")}>OpenDRIVE</button>
          </div>
          {exp && (
            <div className="text-ink-3">
              {plural(exp.lanelets, "lanelet")} from {plural(exp.boundaries, "boundary", "boundaries")}, {plural(exp.regulatory, "regulatory element")}
              {exp.unpaired.length ? `; ${exp.unpaired.length} unpaired (${exp.unpaired[0].reason})` : ""}.
              {exp.note && <div className="text-warn mt-1">{exp.note}</div>}
            </div>
          )}
        </div>
        {msg && <div role="status" className={/failed/i.test(msg) ? "text-warn" : "text-ink-2"}>{msg}</div>}
      </div>
    </div>
  );
}
