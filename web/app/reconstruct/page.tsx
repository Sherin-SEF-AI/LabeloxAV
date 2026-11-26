"use client";

// Scene reconstruction (#15). A stretch of a drive is rebuilt in 3D from its camera frames, ego poses and LiDAR:
// either trained Gaussian splats (gsplat, on the GPU, held to after 18:00 IST during office hours) or a
// colourised LiDAR point map (CPU, no training). The page shows how well it fits (PSNR on frames the training
// never saw), the views it rendered, including poses the car never drove (1.5 m to each side, 3 m up), and a
// free 3D view. A person places a cuboid on the top-down view, checks it in the other views, and sends it to
// every scan and camera frame of the session through the label-once propagation, where it lands in review.

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import { api, humanizeError } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";
import { Badge, Bar, Button, ErrLine, Field, HintBar, NumberInput, Panel, Segmented, Stat } from "@/components/ui/kit";
import { duration, fromNs, num, shortId } from "@/lib/format";
import { useToolJob } from "@/lib/tools/toolJobs";
import { bevUpYaw, groundAt } from "@/lib/tools/sensorsGeom";
import {
  deleteReconstruction, placeInRecon, reconPlaced, reconPlyUrl, reconPreview, reconSessions, reconViewUrl,
  reconstruct, reconstruction, reconstructions, type PlacedRow, type ReconSession, type ReconSessions,
  type Reconstruction,
} from "@/lib/tools/sensorsApi";
import ViewWithBox from "./ViewWithBox";
import type { DraftBox } from "./ReconCloud";

const ReconCloud = dynamic(() => import("./ReconCloud"), { ssr: false });

const DIMS: Record<string, number[]> = {
  sedan: [4.4, 1.8, 1.5], hatchback: [3.9, 1.7, 1.5], suv: [4.7, 1.9, 1.8], vehicle_fallback: [4.4, 1.8, 1.6],
  truck: [7.5, 2.5, 3.2], bus: [11, 2.6, 3.2], motorcycle: [2, 0.8, 1.4], pedestrian: [0.6, 0.6, 1.7],
  autorickshaw: [2.7, 1.4, 1.8], pole: [0.4, 0.4, 6], traffic_sign: [0.2, 0.8, 2.5], cone: [0.4, 0.4, 0.7],
};
const VIEW_LABEL: Record<string, string> = {
  camera: "Camera pose", "left_1.5m": "1.5 m left", "right_1.5m": "1.5 m right", raised_3m: "3 m up",
};
const cname = (n: string | undefined) => (n ?? "").replace(/_/g, " ");

function istHour(): number {
  const d = new Date();
  return ((d.getUTCHours() * 60 + d.getUTCMinutes() + 330) % 1440) / 60;
}

export default function ReconstructPage() {
  const [sess, setSess] = useState<ReconSessions | null>(null);
  const [sessionId, setSessionId] = useState("");
  const [camId, setCamId] = useState("");
  const [startS, setStartS] = useState(0);
  const [nFrames, setNFrames] = useState(50);
  const [stride, setStride] = useState(1);
  const [iters, setIters] = useState(7000);
  const [method, setMethod] = useState<"gsplat" | "pointmap">("gsplat");
  const [list, setList] = useState<Reconstruction[]>([]);
  const [rid, setRid] = useState<string | null>(null);
  const [rec, setRec] = useState<Reconstruction | null>(null);
  const [cloud, setCloud] = useState<{ points: Float32Array; count: number; anchor: number[] } | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [classes, setClasses] = useState<OntologyClass[]>([]);
  const [classId, setClassId] = useState<number | null>(null);
  const [draft, setDraft] = useState<DraftBox | null>(null);
  const [kf, setKf] = useState(0);
  const [kind, setKind] = useState("camera");
  const [placed, setPlaced] = useState<PlacedRow[]>([]);
  const [placeJob, setPlaceJob] = useState<string | null>(null);
  const [snap, setSnap] = useState(true);

  useEffect(() => {
    reconSessions().then((s) => {
      setSess(s);
      if (s.sessions.length) setSessionId((cur) => cur || s.sessions[0].session_id);
      if (!s.gsplat.installed) setMethod("pointmap");
    }).catch((e) => setLoadErr(humanizeError(e)));
    api.ontology().then((o) => {
      setClasses(o.classes);
      setClassId((cur) => cur ?? (o.classes.find((c) => c.name === "sedan") ?? o.classes[0])?.id ?? null);
    }).catch(() => {});
  }, []);

  const session: ReconSession | undefined = sess?.sessions.find((s) => s.session_id === sessionId);
  const cam = session?.cameras.find((c) => c.cam_id === camId) ?? session?.cameras[0];
  const spanS = cam ? (cam.last_ts_ns - cam.first_ts_ns) / 1e9 : 0;

  const loadList = useCallback(async () => {
    if (!sessionId) return;
    try {
      const r = await reconstructions(sessionId);
      setList(r.reconstructions);
      setRid((cur) => cur && r.reconstructions.some((x) => x.recon_id === cur) ? cur
        : (r.reconstructions.find((x) => x.status === "done") ?? r.reconstructions[0])?.recon_id ?? null);
    } catch (e) { setErr(humanizeError(e)); }
  }, [sessionId]);

  useEffect(() => { setCamId(""); setStartS(0); setRid(null); setRec(null); void loadList(); }, [sessionId, loadList]);

  const loadRec = useCallback(async (id: string | null) => {
    setRec(null); setCloud(null); setDraft(null); setPlaced([]);
    if (!id) return;
    try {
      const r = await reconstruction(id);
      setRec(r);
      setKf(0); setKind("camera");
      if (r.status === "done") {
        const [c, p] = await Promise.all([reconPreview(id), reconPlaced(id)]);
        setCloud(c);
        setPlaced(p.placed);
      }
    } catch (e) { setErr(humanizeError(e)); }
  }, []);
  useEffect(() => { void loadRec(rid); }, [rid, loadRec]);

  const { job } = useToolJob(jobId, { onDone: () => { void loadList(); if (rid) void loadRec(rid); } });
  const { job: pjob } = useToolJob(placeJob, { onDone: () => { if (rid) reconPlaced(rid).then((p) => setPlaced(p.placed)).catch(() => {}); } });

  const start = async () => {
    if (!session || !cam) return;
    setErr(null);
    try {
      const r = await reconstruct({
        session_id: session.session_id, method, cam_id: cam.cam_id, n_frames: nFrames, stride, iters,
        start_ts_ns: Math.round(cam.first_ts_ns + startS * 1e9),
      });
      setJobId(r.job.job_id);
      setRid(r.reconstruction.recon_id);
      await loadList();
    } catch (e) { setErr(humanizeError(e)); }
  };

  const views = useMemo(() => rec?.views ?? [], [rec]);
  const bev = views.find((v) => v.kind === "bev");
  const keyframes = useMemo(() => Array.from(new Set(views.filter((v) => v.kind === "camera").map((v) => v.frame_index ?? 0))), [views]);
  const kfIndex = keyframes[Math.min(kf, Math.max(0, keyframes.length - 1))];
  const shown = views.find((v) => v.frame_index === kfIndex && v.name === kind);
  const camView = views.find((v) => v.frame_index === kfIndex && v.kind === "camera");
  const cameraPath = useMemo(() => views.filter((v) => v.kind === "camera").map((v) => [v.T_world_cam[0][3], v.T_world_cam[1][3], v.T_world_cam[2][3]]), [views]);
  const cls = classes.find((c) => c.id === classId);

  const placeAt = (x: number, y: number) => {
    const dims = draft?.dims ?? DIMS[cls?.name ?? ""] ?? [4.4, 1.8, 1.5];
    const g = cloud ? groundAt(cloud.points, cloud.count, cloud.anchor, x, y) : null;
    const z = g != null ? g + dims[2] / 2 : (draft?.center[2] ?? 0);
    setDraft({ center: [x, y, z], dims, yaw: draft?.yaw ?? (bev ? bevUpYaw(bev) : 0) });
  };

  const sendPlace = async () => {
    if (!rid || !draft || classId == null) return;
    setErr(null);
    try {
      const j = await placeInRecon(rid, { center: draft.center, dims: draft.dims, yaw: draft.yaw, class_id: classId,
        ground_snap: snap, max_range_m: 60 });
      setPlaceJob(j.job_id);
      setPlaced((await reconPlaced(rid)).placed);
    } catch (e) { setErr(humanizeError(e)); }
  };

  const quiet = istHour() >= 10 && istHour() < 18;
  const m = rec?.metrics ?? {};

  return (
    <PageShell active="Scene reconstruction">
      <div className="page-wrap grid grid-cols-1 xl:grid-cols-[340px_1fr] gap-4">
        <div className="flex flex-col gap-4 min-w-0">
          <Panel title="Rebuild a stretch of road" desc="Frames, ego poses and LiDAR of one camera's window. Splats train on the GPU; the point map needs no training.">
            {loadErr && <ErrLine err={loadErr} />}
            {!sess && !loadErr && <p className="text-[12px] text-ink-3">Loading sessions...</p>}
            {sess && !sess.sessions.length && (
              <EmptyState compact icon="box" title="No session can be rebuilt">A reconstruction needs a session with ego poses and LiDAR scans (KITTI imports have both).</EmptyState>
            )}
            {sess && sess.sessions.length > 0 && (
              <div className="flex flex-col gap-1">
                <Field label="Session">
                  <select className="field w-48" value={sessionId} onChange={(e) => setSessionId(e.target.value)}>
                    {sess.sessions.map((s) => <option key={s.session_id} value={s.session_id}>{s.name ?? shortId(s.session_id)} ({s.clouds} scans)</option>)}
                  </select>
                </Field>
                <Field label="Camera">
                  <select className="field w-48" value={cam?.cam_id ?? ""} onChange={(e) => setCamId(e.target.value)}>
                    {session?.cameras.map((c) => <option key={c.cam_id} value={c.cam_id}>{c.cam_id} ({num(c.frames)} frames)</option>)}
                  </select>
                </Field>
                <Field label={`Start at ${startS.toFixed(1)} s of ${spanS.toFixed(1)}`} tip="Where the window starts in the drive">
                  <input type="range" min={0} max={Math.max(0, spanS)} step={0.1} value={startS}
                    onChange={(e) => setStartS(Number(e.target.value))} className="w-40" aria-label="Window start" />
                </Field>
                <Field label="Frames" tip="Posed frames in the window"><NumberInput value={nFrames} onChange={(n) => setNFrames(Math.max(3, Math.min(200, Math.round(n || 3))))} w="w-20" /></Field>
                <Field label="Every nth frame"><NumberInput value={stride} onChange={(n) => setStride(Math.max(1, Math.min(20, Math.round(n || 1))))} w="w-20" /></Field>
                <Field label="Method">
                  <Segmented value={method} onChange={setMethod} options={[
                    { value: "gsplat", label: "Gaussian splats", tip: sess.gsplat.installed ? `gsplat ${sess.gsplat.version}, trained on the GPU` : "gsplat is not installed on this server" },
                    { value: "pointmap", label: "Point map", tip: "Coloured LiDAR, no training" }]} />
                </Field>
                {method === "gsplat" && (
                  <Field label="Training steps" tip="7000 steps on 50 KITTI frames take about two minutes on the RTX 5080">
                    <NumberInput value={iters} onChange={(n) => setIters(Math.max(100, Math.min(30000, Math.round(n || 100))))} w="w-24" />
                  </Field>
                )}
                {method === "gsplat" && quiet && (
                  <p className="text-[12px] text-warn mt-1">Office hours (10:00 to 18:00 IST): training is queued and starts after 18:00. The point map runs now.</p>
                )}
                {method === "gsplat" && !sess.gsplat.installed && <p className="text-[12px] text-block">gsplat is not installed here; use the point map.</p>}
                <div className="flex items-center gap-2 mt-2">
                  <Button variant="primary" onClick={start} disabled={!cam || (method === "gsplat" && !sess.gsplat.installed)}
                    tip={method === "gsplat" ? "Needs the reviewer role" : undefined}>Rebuild</Button>
                  {job && job.status !== "done" && (
                    <span className="flex-1 flex items-center gap-2 text-[12px] text-ink-2">
                      <Bar frac={job.progress} /> {job.quiet_hours ? "waiting for 18:00 IST" : job.stage ?? job.status}
                    </span>
                  )}
                </div>
                <ErrLine err={err} />
              </div>
            )}
          </Panel>

          <Panel title="Reconstructions" desc="Newest first, for this session.">
            {!list.length ? <p className="text-[12px] text-ink-3">None yet.</p> : (
              <ul className="flex flex-col gap-1">
                {list.map((r) => (
                  <li key={r.recon_id}>
                    <button type="button" onClick={() => setRid(r.recon_id)}
                      className={`w-full text-left px-2 py-1.5 rounded border text-[12px] ${rid === r.recon_id ? "border-accent bg-accent/10" : "border-line hover:bg-head"}`}>
                      <div className="flex items-center gap-2">
                        <span className="text-ink">{r.method === "gsplat" ? "Splats" : "Point map"}</span>
                        <Badge state={r.status === "done" ? "pass" : r.status === "error" ? "block" : "info"}>{r.status}</Badge>
                        <span className="ml-auto text-ink-3">{r.metrics.psnr_held_out != null ? `${r.metrics.psnr_held_out} dB` : ""}</span>
                      </div>
                      <div className="text-ink-3 mt-0.5">{r.cam_id ?? "camera"} · {r.metrics.window?.frames_posed ?? r.params.n_frames as number} frames · {r.first_ts_ns ? fromNs(r.first_ts_ns) : "not started"}</div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>

        <div className="flex flex-col gap-4 min-w-0">
          {!rec && (
            <div className="panel"><EmptyState icon="cube" title="Choose or rebuild a window">Its rendered views, fit and 3D view open here.</EmptyState></div>
          )}
          {rec && rec.status !== "done" && (
            <div className="panel p-4">
              <div className="flex items-center gap-2 text-[13px]">
                <Badge state={rec.status === "error" ? "block" : "info"}>{rec.status}</Badge>
                <span className="text-ink-2">{rec.method === "gsplat" ? "Gaussian splats" : "Point map"} of {rec.params.n_frames as number} frames</span>
              </div>
              {rec.error && <ErrLine err={rec.error} />}
              {job && job.job_id === rec.job_id && job.status !== "done" && (
                <div className="flex items-center gap-2 mt-3 text-[12px]"><Bar frac={job.progress} /><span>{job.quiet_hours ? "Waiting for 18:00 IST" : job.stage}</span></div>
              )}
            </div>
          )}
          {rec && rec.status === "done" && (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 2xl:grid-cols-6 gap-3">
                <Stat label="Method" value={rec.method === "gsplat" ? "Splats" : "Point map"} sub={rec.cam_id ?? undefined} />
                <Stat label="Frames" value={num(m.window?.frames_posed)} sub={`${num(m.window?.scans)} scans`} />
                {rec.method === "gsplat" ? (
                  <>
                    <Stat label="PSNR held out" value={m.psnr_held_out != null ? `${m.psnr_held_out} dB` : "n/a"} sub={`${m.held_out_frames ?? 0} frames never trained on`} tone="info" />
                    <Stat label="PSNR trained" value={m.psnr_train != null ? `${m.psnr_train} dB` : "n/a"} sub={`${m.train_frames ?? 0} frames`} />
                    <Stat label="Splats" value={num(m.gaussians)} sub={`from ${num(m.initial_points)} LiDAR points`} />
                    <Stat label="Training" value={duration(m.seconds ?? null)} sub={`${num(m.iterations)} steps`} />
                  </>
                ) : (
                  <>
                    <Stat label="PSNR at cameras" value={m.psnr_camera_views != null ? `${m.psnr_camera_views} dB` : "n/a"} sub="rendered points vs the frames" />
                    <Stat label="Points" value={num(m.points ?? m.initial_points)} sub="coloured from the frames" />
                  </>
                )}
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,420px)_1fr] gap-4">
                <Panel title="Top-down view" desc="Click to place a cuboid, drag to move it. Only what is below 1 m over the cameras is drawn, so the road is not hidden by trees."
                  right={rec.has_ply ? <a className="link text-[12px]" href={reconPlyUrl(rec.recon_id)}>Download .ply</a> : undefined}>
                  {bev ? (
                    <div className="max-h-[640px] overflow-auto">
                      <ViewWithBox src={reconViewUrl(rec.recon_id, bev.index)} view={bev} draft={draft} onPlace={placeAt} alt="Top-down render" />
                    </div>
                  ) : <p className="text-[12px] text-ink-3">No top-down view.</p>}
                </Panel>

                <div className="flex flex-col gap-4 min-w-0">
                  <Panel title="Rendered views" desc="Each keyframe from its own pose and from poses the car never took. A held-out frame was never trained on, so its render is a genuine novel view."
                    right={<Segmented value={kind} onChange={setKind} options={["camera", "left_1.5m", "right_1.5m", "raised_3m"].map((k) => ({ value: k, label: VIEW_LABEL[k] }))} />}>
                    {keyframes.length > 0 && (
                      <div className="flex items-center gap-3 mb-2 text-[12px]">
                        <input type="range" min={0} max={keyframes.length - 1} value={Math.min(kf, keyframes.length - 1)}
                          onChange={(e) => setKf(Number(e.target.value))} className="flex-1" aria-label="Keyframe" />
                        <span className="text-ink-2 tabular-nums">frame {kfIndex}</span>
                        {camView?.held_out && <Badge state="info">held out</Badge>}
                      </div>
                    )}
                    {shown ? (
                      <div className="grid grid-cols-1 gap-2">
                        <ViewWithBox src={reconViewUrl(rec.recon_id, shown.index)} view={shown} draft={draft} alt={`${VIEW_LABEL[kind]} render`} />
                        {kind === "camera" && camView?.frame_id && (
                          <div>
                            <div className="label-caps mb-1">The real frame</div>
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img src={`/api/frames/${camView.frame_id}/image`} alt="Real camera frame" className="w-full rounded" />
                          </div>
                        )}
                      </div>
                    ) : <p className="text-[12px] text-ink-3">No view.</p>}
                  </Panel>
                </div>
              </div>

              <div className="grid grid-cols-1 2xl:grid-cols-[1fr_380px] gap-4">
                <Panel title="3D view" desc={`${num(cloud?.count)} ${rec.method === "gsplat" ? "splat centres" : "map points"} in their colours; the blue line is the camera path. Drag to orbit.`}>
                  <div className="h-[460px]">
                    {cloud ? <ReconCloud points={cloud.points} count={cloud.count} anchor={cloud.anchor} cameras={cameraPath} draft={draft} />
                      : <p className="text-[12px] text-ink-3">Loading the cloud...</p>}
                  </div>
                </Panel>

                <Panel title="Cuboid" desc="Placed in the reconstruction, then put on every scan and camera frame of the session, linked and in review.">
                  {!draft ? <HintBar>Click the top-down view where the object stands.</HintBar> : (
                    <div className="flex flex-col gap-1">
                      <Field label="Class">
                        <select className="field w-44" value={classId ?? ""} onChange={(e) => {
                          const id = Number(e.target.value); setClassId(id);
                          const d = DIMS[classes.find((c) => c.id === id)?.name ?? ""];
                          if (d && draft) setDraft({ ...draft, dims: d, center: [draft.center[0], draft.center[1], draft.center[2] - draft.dims[2] / 2 + d[2] / 2] });
                        }}>
                          {classes.map((c) => <option key={c.id} value={c.id}>{cname(c.name)}</option>)}
                        </select>
                      </Field>
                      {(["Length", "Width", "Height"] as const).map((lab, i) => (
                        <Field key={lab} label={`${lab} (m)`}>
                          <NumberInput value={draft.dims[i]} w="w-20" onChange={(n) => {
                            if (!Number.isFinite(n) || n <= 0) return;
                            const dims = [...draft.dims]; dims[i] = n; setDraft({ ...draft, dims });
                          }} />
                        </Field>
                      ))}
                      <Field label={`Heading ${((draft.yaw * 180) / Math.PI).toFixed(0)} deg`}>
                        <input type="range" min={-180} max={180} value={Math.round((draft.yaw * 180) / Math.PI)}
                          onChange={(e) => setDraft({ ...draft, yaw: (Number(e.target.value) * Math.PI) / 180 })} className="w-40" aria-label="Heading" />
                      </Field>
                      <Field label="Centre height (m)">
                        <NumberInput value={Number(draft.center[2].toFixed(2))} w="w-20" onChange={(n) => Number.isFinite(n) && setDraft({ ...draft, center: [draft.center[0], draft.center[1], n] })} />
                      </Field>
                      <Field label="Snap to each scan's road" tip="Per scan, lower the box onto the LiDAR ground under it">
                        <input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} />
                      </Field>
                      <div className="text-[11px] text-ink-3 font-mono mt-1">x {draft.center[0].toFixed(2)}, y {draft.center[1].toFixed(2)} (session frame)</div>
                      <div className="flex items-center gap-2 mt-2">
                        <Button variant="primary" onClick={sendPlace} disabled={classId == null}>Put on every frame</Button>
                        <Button variant="ghost" onClick={() => setDraft(null)}>Clear</Button>
                      </div>
                      {pjob && pjob.status !== "done" && pjob.status !== "error" && (
                        <div className="flex items-center gap-2 mt-2 text-[12px]"><Bar frac={pjob.progress} /><span>{pjob.stage ?? pjob.status}</span></div>
                      )}
                    </div>
                  )}
                  <div className="mt-4">
                    <div className="label-caps mb-1">Placed from this reconstruction</div>
                    {!placed.length ? <p className="text-[12px] text-ink-3">Nothing yet.</p> : (
                      <ul className="flex flex-col gap-1.5">
                        {placed.map((p) => (
                          <li key={p.job_id} className="text-[12px] border border-line rounded px-2 py-1.5">
                            <div className="flex items-center gap-2">
                              <span className="text-ink">{cname(classes.find((c) => c.id === p.class_id)?.name)}</span>
                              <Badge state={p.status === "done" ? "pass" : p.status === "error" ? "block" : "info"}>{p.status}</Badge>
                            </div>
                            {p.result && (
                              <div className="text-ink-3 mt-0.5">
                                {num(p.result.scans)} scans, {num(p.result.boxes_2d)} camera boxes
                                {Object.values(p.result.tracks_2d ?? {}).slice(0, 2).map((t) => (
                                  <a key={t} href={`/track/${t}`} className="link ml-2">track {shortId(t)}</a>
                                ))}
                              </div>
                            )}
                            {p.error && <div className="text-block mt-0.5">{p.error}</div>}
                          </li>
                        ))}
                      </ul>
                    )}
                    {placed.some((p) => p.status === "done") && (
                      <a href="/lidar/annotate" className="link text-[12px] mt-2 inline-block">Review the 3D tracks in LiDAR annotation</a>
                    )}
                  </div>
                  <ErrLine err={err} />
                </Panel>
              </div>
              <div className="flex justify-end">
                <Button variant="ghost" onClick={async () => {
                  if (!rec || !window.confirm("Delete this reconstruction and its renders? Labels placed from it stay.")) return;
                  try { await deleteReconstruction(rec.recon_id); setRid(null); await loadList(); } catch (e) { setErr(humanizeError(e)); }
                }} tip="Needs the reviewer role">Delete reconstruction</Button>
              </div>
            </>
          )}
        </div>
      </div>
    </PageShell>
  );
}
