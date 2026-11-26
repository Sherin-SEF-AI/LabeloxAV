"use client";

// Camera-LiDAR alignment (#9). The scan is projected onto the camera image in the browser, so the six sliders and
// the drag move it live, with the edge-alignment and mutual-information scores the automatic search optimises
// recomputed on every change. Point pairs (a LiDAR point and the pixel where it really is) give a reprojection
// error in pixels and can be solved directly. Auto-align refines from wherever the sliders are and checks the
// result on frames it did not fit. Saving writes a `manual` calibration beside the dataset's, which is never
// overwritten; Revert removes it.

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { Badge, Button, KV, Panel, Segmented } from "@/components/ui/kit";
import { api, humanizeError } from "@/lib/api";
import { ago, humanize, plural, shortId } from "@/lib/format";
import { useQuery } from "@/lib/useQuery";
import { useToolJob } from "@/lib/tools/toolJobs";
import { edgeScore, mutualInformation, pairsError, project, sampleField, type Extrinsic } from "@/lib/tools/drivingGeom";
import {
  alignFrames, alignHistory, alignRefine, alignRevert, alignSave, alignSessions, alignSolvePairs, alignView, decodePoints,
  type AlignSession, type RefineResult,
} from "@/lib/tools/drivingApi";
import type { Pair } from "./AlignCanvas";

const AlignCanvas = dynamic(() => import("./AlignCanvas"), { ssr: false });
const NO_SESSIONS: AlignSession[] = [];
const AXES = [
  { key: "roll", label: "Roll", unit: "deg", rot: 0, range: 3, step: 0.01 },
  { key: "pitch", label: "Pitch", unit: "deg", rot: 1, range: 3, step: 0.01 },
  { key: "yaw", label: "Yaw", unit: "deg", rot: 2, range: 3, step: 0.01 },
  { key: "x", label: "Forward", unit: "m", tr: 0, range: 0.3, step: 0.005 },
  { key: "y", label: "Left", unit: "m", tr: 1, range: 0.3, step: 0.005 },
  { key: "z", label: "Up", unit: "m", tr: 2, range: 0.3, step: 0.005 },
] as const;

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((res, rej) => {
    const im = new window.Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("image failed to load"));
    im.src = src;
  });
}

function channel(im: HTMLImageElement, w: number, h: number, gray: boolean): Float32Array {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  ctx.drawImage(im, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    out[i] = gray ? 0.114 * d[i * 4 + 2] + 0.587 * d[i * 4 + 1] + 0.299 * d[i * 4] : d[i * 4] / 255;
  }
  return out;
}

const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? "n/a" : v.toFixed(d));

export default function AlignPage() {
  const sessionsQ = useQuery(["drivingAlignSessions"], () => alignSessions());
  const sessions = sessionsQ.data ?? NO_SESSIONS;
  const [sid, setSid] = useState("");
  const [cam, setCam] = useState("");
  useEffect(() => { if (!sid && sessions.length) setSid(sessions[0].session_id); }, [sessions, sid]);
  const sess = sessions.find((s) => s.session_id === sid);
  useEffect(() => { if (sess) setCam((c) => (sess.cameras.some((x) => x.cam_id === c) ? c : sess.cameras[0]?.cam_id ?? "")); }, [sess]);
  const framesQ = useQuery(sid && cam ? ["drivingAlignFrames", sid, cam] : null, () => alignFrames(sid, cam));
  const frames = useMemo(() => framesQ.data?.frames ?? [], [framesQ.data]);
  const [fi, setFi] = useState(0);
  useEffect(() => { setFi(Math.floor(frames.length / 3)); }, [frames]);
  const frame = frames[Math.min(fi, Math.max(0, frames.length - 1))];

  const viewQ = useQuery(frame ? ["drivingAlignView", frame.frame_id] : null, () => alignView(frame.frame_id), { keepPrevious: true });
  const view = viewQ.data && frame && viewQ.data.frame_id === frame.frame_id ? viewQ.data : null;
  const points = useMemo(() => (view ? decodePoints(view.points_b64) : new Float32Array()), [view]);

  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [edgeImg, setEdgeImg] = useState<HTMLImageElement | null>(null);
  const [fields, setFields] = useState<{ edge: Float32Array; gray: Float32Array } | null>(null);
  useEffect(() => {
    setImg(null); setEdgeImg(null); setFields(null);
    if (!view) return;
    let live = true;
    (async () => {
      const meta = await api.frame(view.frame_id);
      const [im, em] = await Promise.all([loadImage(meta.image_url), view.edge_png ? loadImage(view.edge_png) : Promise.resolve(null)]);
      if (!live) return;
      setImg(im);
      setEdgeImg(em);
      setFields({ edge: em ? channel(em, view.width, view.height, false) : new Float32Array(view.width * view.height),
                  gray: channel(im, view.width, view.height, true) });
    })().catch((e) => { if (live) setMsg(`Could not load this view: ${humanizeError(e)}`); });
    return () => { live = false; };
  }, [view]);

  // The extrinsic on the sliders. Starts at the calibration in force; survives a frame change within a session.
  const [ex, setEx] = useState<Extrinsic | null>(null);
  const viewKey = view ? `${view.session_id}/${view.cam_id}/${view.current.source}/${view.current.rpy_deg.join()}` : "";
  useEffect(() => { if (view) setEx({ rpy_deg: [...view.current.rpy_deg], xyz_m: [...view.current.xyz_m] }); }, [viewKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const k = useMemo(() => (view ? { fx: view.current.fx, fy: view.current.fy, cx: view.current.cx, cy: view.current.cy } : null), [view]);

  const [show, setShow] = useState<"depth" | "edges" | "edgemap">("depth");
  const [pairMode, setPairMode] = useState(false);
  const [pairs, setPairs] = useState<Pair[]>([]);
  useEffect(() => { setPairs([]); }, [frame?.frame_id]);
  const [msg, setMsg] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [job, setJob] = useState<string | null>(null);
  const [refined, setRefined] = useState<RefineResult | null>(null);
  const historyQ = useQuery(sid ? ["drivingAlignHistory", sid, cam] : null, () => alignHistory(sid, cam));

  const { job: jobRow } = useToolJob(job, {
    onDone: (j) => {
      setJob(null);
      if (j.status !== "done") { setMsg(`Auto-align ${j.status}: ${j.error ?? ""}`); return; }
      const r = j.result as unknown as RefineResult;
      setRefined(r);
      void historyQ.reload();
      setMsg(r.improved ? "Auto-align found a better fit; check it on the image before saving" : "Auto-align found nothing better than where it started");
    },
  });

  // Live scores, the same formulas the server optimises.
  const live = useMemo(() => {
    if (!view || !ex || !k || !fields || !points.length) return null;
    const { uv, inside } = project(points, 5, k, ex, view.width, view.height);
    const n = inside.length;
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = points[i * 5 + 4];
    const edge = edgeScore(uv, inside, w, fields.edge, view.width, view.height);
    const a: number[] = [];
    const b: number[] = [];
    for (let i = 0; i < n; i++) {
      if (!inside[i]) continue;
      a.push(points[i * 5 + 3]);
      b.push(sampleField(fields.gray, view.width, view.height, uv[i * 2], uv[i * 2 + 1]));
    }
    const mi = mutualInformation(a, b);
    const cur = project(points, 5, k, { rpy_deg: view.current.rpy_deg, xyz_m: view.current.xyz_m }, view.width, view.height);
    let s = 0;
    let m = 0;
    for (let i = 0; i < n; i++) if (inside[i] && cur.inside[i]) { s += Math.hypot(uv[i * 2] - cur.uv[i * 2], uv[i * 2 + 1] - cur.uv[i * 2 + 1]); m++; }
    const pe = pairsError(pairs.map((p) => p.point), pairs.map((p) => p.uv), k, ex, view.width, view.height);
    return { edge, mi, objective: (edge ?? 0) + (mi ?? 0), shift: m ? s / m : null, inView: a.length, pairs: pe };
  }, [view, ex, k, fields, points, pairs]);

  const setAxis = useCallback((axis: (typeof AXES)[number], value: number) => {
    setEx((e) => {
      if (!e) return e;
      const next = { rpy_deg: [...e.rpy_deg], xyz_m: [...e.xyz_m] };
      if ("rot" in axis) next.rpy_deg[axis.rot] = value; else next.xyz_m[axis.tr] = value;
      return next;
    });
  }, []);

  const run = async (what: string, fn: () => Promise<void>) => {
    setBusy(what);
    try { await fn(); } catch (e) { setMsg(`${what} failed: ${humanizeError(e)}`); } finally { setBusy(null); }
  };

  const fitFrames = () => {
    if (!frame) return [];
    const others = [Math.floor(frames.length * 0.15), Math.floor(frames.length * 0.85)].map((i) => frames[i]?.frame_id)
      .filter((f): f is string => !!f && f !== frame.frame_id);
    return [frame.frame_id, ...others];
  };

  const base = view?.base;
  const scoreCur = view?.score_current;

  return (
    <PageShell active="Camera-LiDAR alignment" filters={(
      <>
        <label htmlFor="align-session" className="text-ink-3">Session</label>
        <select id="align-session" className="field text-[12.5px]" value={sid} onChange={(e) => { setSid(e.target.value); setRefined(null); }}>
          {sessions.map((s) => (
            <option key={s.session_id} value={s.session_id}>{s.route || s.vehicle_id || shortId(s.session_id)} · {plural(s.scans, "scan")}</option>
          ))}
        </select>
        {sess && sess.cameras.length > 1 && (
          <select className="field text-[12.5px]" value={cam} onChange={(e) => setCam(e.target.value)} aria-label="Camera">
            {sess.cameras.map((c) => <option key={c.cam_id} value={c.cam_id}>{c.cam_id}</option>)}
          </select>
        )}
        {frames.length > 0 && (
          <label className="flex items-center gap-2 text-ink-3">
            Frame {fi + 1} of {frames.length}
            <input type="range" min={0} max={frames.length - 1} value={fi} onChange={(e) => setFi(Number(e.target.value))} aria-label="Frame" className="w-48" />
          </label>
        )}
        {view && <Badge state={view.current.source === "manual" ? "info" : "pass"}>{humanize(view.current.source)} calibration in force</Badge>}
      </>
    )}>
      <div className="page-wrap">
        <LoadState loading={sessionsQ.loading && !sessions.length} error={sessionsQ.error} empty={!sessionsQ.loading && !sessions.length}
          emptyTitle="No LiDAR sessions" emptyText="Alignment needs a session with LiDAR scans and camera frames at the same instants." emptyIcon="ruler">
          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_22rem]">
            <div className="space-y-3 min-w-0">
              <Panel title="Projection" desc="Drag to turn the camera, Shift-drag to roll, Alt-drag to slide it. Colour is distance (warm is near)."
                right={<Segmented options={[{ value: "depth", label: "All points" }, { value: "edges", label: "Depth edges" }, { value: "edgemap", label: "Image edges" }]}
                  value={show} onChange={(v) => setShow(v as typeof show)} />}>
                <LoadState loading={(!view && (viewQ.loading || framesQ.loading))} error={viewQ.error || framesQ.error}
                  empty={!framesQ.loading && !!sid && !frames.length} emptyText="No camera frame of this session has a LiDAR scan within 60 ms.">
                  {view && ex && k && (
                    <AlignCanvas image={img} edgeField={edgeImg} points={points} stride={5} k={k} ex={ex} width={view.width} height={view.height}
                      show={show} pairMode={pairMode} pairs={pairs}
                      onPair={(p) => setPairs((ps) => [...ps, p])}
                      onDrag={(d) => setEx((e) => e && ({
                        rpy_deg: [e.rpy_deg[0] + d.droll, e.rpy_deg[1] + d.dpitch, e.rpy_deg[2] + d.dyaw],
                        xyz_m: [e.xyz_m[0], e.xyz_m[1] + d.dy, e.xyz_m[2] + d.dz],
                      }))} />
                  )}
                </LoadState>
              </Panel>

              {view && ex && base && (
                <Panel title="Extrinsic" desc={`Camera pose in the ${view.current.source === "manual" ? "" : "LiDAR "}ego frame. Sliders span ${AXES[0].range} degrees and ${AXES[3].range} m either side of the ${base.source} calibration.`}>
                  <div className="grid sm:grid-cols-2 gap-x-6 gap-y-2 text-[12px]">
                    {AXES.map((a) => {
                      const val = "rot" in a ? ex.rpy_deg[a.rot] : ex.xyz_m[a.tr];
                      const b0 = "rot" in a ? base.rpy_deg[a.rot] : base.xyz_m[a.tr];
                      return (
                        <label key={a.key} className="block">
                          <span className="flex justify-between"><span className="label-caps">{a.label}</span>
                            <span className="font-mono">{val.toFixed(a.unit === "deg" ? 3 : 3)} {a.unit} <span className="text-ink-3">({val - b0 >= 0 ? "+" : ""}{(val - b0).toFixed(3)})</span></span></span>
                          <input type="range" className="w-full" min={b0 - a.range} max={b0 + a.range} step={a.step} value={val}
                            aria-label={`${a.label} (${a.unit})`} onChange={(e) => setAxis(a, Number(e.target.value))} />
                        </label>
                      );
                    })}
                  </div>
                  <div className="flex gap-1.5 mt-2 flex-wrap">
                    <Button onClick={() => setEx({ rpy_deg: [...view.current.rpy_deg], xyz_m: [...view.current.xyz_m] })}>Reset to {view.current.source}</Button>
                    {view.has_manual && <Button onClick={() => setEx({ rpy_deg: [...base.rpy_deg], xyz_m: [...base.xyz_m] })}>Load {base.source}</Button>}
                  </div>
                </Panel>
              )}
            </div>

            <div className="space-y-3 min-w-0">
              <Panel title="Scores" desc="Edge: how much of the scan's depth edges land on image edges (0 to 1). Information: how well LiDAR reflectivity predicts image brightness (0 to 1).">
                {live ? (
                  <div className="text-[12px] space-y-1">
                    <KV k="Edge alignment" v={`${fmt(live.edge)} (in force ${fmt(scoreCur?.edge)})`}
                      state={live.edge !== null && scoreCur?.edge != null ? (live.edge >= scoreCur.edge ? "pass" : "warn") : undefined} />
                    <KV k="Mutual information" v={`${fmt(live.mi)} (in force ${fmt(scoreCur?.mi)})`} />
                    <KV k="Objective" v={fmt(live.objective)} />
                    <KV k="Moved from the calibration in force" v={live.shift === null ? "n/a" : `${live.shift.toFixed(1)} px mean`} />
                    <KV k="Points in view" v={String(live.inView)} />
                    <KV k="Pair reprojection error" v={live.pairs.rms === null ? "no pairs" : `${live.pairs.rms.toFixed(2)} px RMS over ${live.pairs.n}`} />
                  </div>
                ) : <div className="text-[12px] text-ink-3">{view ? "Reading the image..." : "Pick a frame."}</div>}
              </Panel>

              <Panel title="Point pairs" desc="Click a projected point, then the pixel where that point really is. Four or more pairs can be solved for the extrinsic.">
                <div className="flex gap-1.5 flex-wrap text-[12px]">
                  <Button active={pairMode} onClick={() => setPairMode((m) => !m)}>{pairMode ? "Picking pairs" : "Pick pairs"}</Button>
                  <Button disabled={pairs.length < 4 || !!busy || !ex || !frame} onClick={() => run("Solve", async () => {
                    const r = await alignSolvePairs(frame!.frame_id, pairs.map((p) => p.point), pairs.map((p) => p.uv), ex!.rpy_deg, ex!.xyz_m);
                    setEx({ rpy_deg: r.rpy_deg, xyz_m: r.xyz_m });
                    setMsg(`Solved from ${r.pairs} pairs: ${fmt(r.before.rms, 2)} px to ${fmt(r.after.rms, 2)} px RMS`);
                  })}>Solve from pairs</Button>
                  <Button disabled={!pairs.length} onClick={() => setPairs([])}>Clear ({pairs.length})</Button>
                </div>
              </Panel>

              <Panel title="Auto-align" desc="Searches rotation over a grid, then all six parameters, from where the sliders are now. Scored on this frame and two others; checked on frames it did not see.">
                <div className="space-y-2 text-[12px]">
                  <Button variant="primary" disabled={!!job || !!busy || !ex || !frame} onClick={() => run("Auto-align", async () => {
                    const q = await alignRefine(fitFrames(), ex!.rpy_deg, ex!.xyz_m);
                    setJob(q.job_id);
                    setRefined(null);
                    setMsg(null);
                  })}>{job ? `Aligning${jobRow ? ` (${Math.round((jobRow.progress || 0) * 100)}%)` : "..."}` : "Auto-align"}</Button>
                  {refined && (
                    <div className="space-y-1">
                      <KV k="Fit frames" v={`${fmt(refined.before.objective)} to ${fmt(refined.after.objective)}`} state={refined.improved ? "pass" : "warn"} />
                      <KV k="Held-out frames" v={refined.held_out ? `${fmt(refined.held_out.before.objective)} to ${fmt(refined.held_out.after.objective)}` : "none available"}
                        state={refined.held_out ? (refined.held_out.improved ? "pass" : "warn") : undefined} />
                      <KV k="Moved from the start" v={refined.shift_from_start.mean_px === null ? "n/a" : `${refined.shift_from_start.mean_px} px mean`} />
                      <div className={refined.recommend_save ? "text-pass" : "text-warn"}>
                        {refined.recommend_save ? "Better on the fit frames and on frames it did not see." :
                          refined.improved ? "Better on the fit frames only: it may be fitting their noise. Not recommended for saving." :
                            "No improvement over the start."}
                        {refined.at_bound ? " It stopped at the edge of the search range; start closer." : ""}
                      </div>
                      <Button disabled={!refined.improved} onClick={() => setEx({ rpy_deg: refined.rpy_deg, xyz_m: refined.xyz_m })}>Apply to sliders</Button>
                    </div>
                  )}
                </div>
              </Panel>

              <Panel title="Save" desc="Stored as this camera's manual calibration beside the dataset's, which is never overwritten. Every 3D projection of the session reads it from then on. Reviewer.">
                <div className="flex gap-1.5 flex-wrap text-[12px]">
                  <Button variant="primary" disabled={!!busy || !ex || !frame} onClick={() => run("Save", async () => {
                    const r = await alignSave({ frame_ids: fitFrames(), rpy_deg: ex!.rpy_deg, xyz_m: ex!.xyz_m,
                      method: pairs.length >= 4 ? "pairs" : refined && refined.rpy_deg.join() === ex!.rpy_deg.join() ? "auto" : "manual",
                      pairs_points: pairs.map((p) => p.point), pairs_uv: pairs.map((p) => p.uv) });
                    setMsg(`Saved (${r.improved ? "scores better" : "scores not better"} than the calibration it replaced: ${fmt(r.before.objective)} to ${fmt(r.after.objective)})`);
                    await Promise.all([viewQ.reload(), historyQ.reload(), sessionsQ.reload()]);
                  })}>Save as manual calibration</Button>
                  {view?.has_manual && (
                    <Button variant="ghost" disabled={!!busy} onClick={() => run("Revert", async () => {
                      await alignRevert(view.session_id, view.cam_id);
                      setMsg(`Manual calibration removed; the ${base?.source} calibration is in force again`);
                      await Promise.all([viewQ.reload(), historyQ.reload()]);
                    })}>Revert to {base?.source}</Button>
                  )}
                </div>
                {msg && <div role="status" className={`text-[12px] mt-2 ${/failed/i.test(msg) ? "text-warn" : "text-ink-2"}`}>{msg}</div>}
              </Panel>

              <Panel title="History" collapsible defaultOpen={false}>
                <LoadState loading={historyQ.loading && !historyQ.data} error={historyQ.error}
                  empty={!!historyQ.data && !historyQ.data.alignments.length} emptyText="No alignments yet.">
                  <table className="data-table w-full text-[11.5px]">
                    <thead><tr><th>When</th><th>Method</th><th className="text-right">Before</th><th className="text-right">After</th><th>Saved</th></tr></thead>
                    <tbody>
                      {(historyQ.data?.alignments ?? []).map((h) => (
                        <tr key={h.alignment_id}>
                          <td>{h.created_at ? ago(h.created_at) : ""}</td>
                          <td>{humanize(h.method)}</td>
                          <td className="text-right">{fmt(h.score_before.objective)}</td>
                          <td className="text-right">{fmt(h.score_after.objective)}</td>
                          <td>{h.saved ? "yes" : ""}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </LoadState>
              </Panel>
            </div>
          </div>
        </LoadState>
      </div>
    </PageShell>
  );
}
