"use client";

// Label once in 3D, get it everywhere (#3). The session's scans are placed in one frame by their ego poses and
// drawn as one static scene. A person places one cuboid there (drag it in the bird's-eye view, set its size,
// heading and class), and one job puts it on every scan within range (snapped to that scan's road surface)
// and projects it into every camera frame captured with those scans, as 2D boxes linked to the 3D track.
// Everything lands in review; the review view below shows each scan's 2D box on its frame with the LiDAR
// support behind it, so a person accepts the track or rejects the frames where the object had moved.

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import { humanizeError } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import Inspector from "@/components/shell/Inspector";
import EmptyState from "@/components/ui/EmptyState";
import type { ViewerCuboid } from "@/components/lidar/PointCloudViewer";
import { StateBadge } from "@/components/StateBadge";
import { fromNs, num } from "@/lib/format";
import { useToolJob } from "@/lib/tools/toolJobs";
import { groundNear } from "@/lib/tools/scene3dGeom";
import {
  labelOnce, labelOnceTracks, scenePoints, scenePoses, trackReview, trackReviewApply,
  type LabelOnceTrack, type ScenePoints, type ScenePose, type TrackReview,
} from "@/lib/tools/scene3dApi";

const PointCloudViewer = dynamic(() => import("@/components/lidar/PointCloudViewer"), { ssr: false });
const classLabel = (n: string | null | undefined) => (n ?? "").replace(/_/g, " ");

const DIMS: Record<string, number[]> = {
  sedan: [4.4, 1.8, 1.5], hatchback: [3.9, 1.7, 1.5], suv: [4.7, 1.9, 1.8], vehicle_fallback: [4.4, 1.8, 1.6],
  truck: [7.5, 2.5, 3.2], bus: [11, 2.6, 3.2], motorcycle: [2, 0.8, 1.4], pedestrian: [0.6, 0.6, 1.7],
  autorickshaw: [2.7, 1.4, 1.8], pole: [0.4, 0.4, 6], traffic_sign: [0.2, 0.8, 2.5], barrier: [2, 0.4, 1],
  cone: [0.4, 0.4, 0.7],
};

export default function LabelOnceWorkspace({ sessionId, classes }: { sessionId: string; classes: OntologyClass[] }) {
  const [scene, setScene] = useState<ScenePoints | null>(null);
  const [poses, setPoses] = useState<ScenePose[]>([]);
  const [every, setEvery] = useState(10);
  const [draft, setDraft] = useState<{ center: number[]; dims: number[]; yaw: number } | null>(null);
  const [classId, setClassId] = useState<number | null>(null);
  const [snap, setSnap] = useState(true);
  const [range, setRange] = useState(60);
  const [tracks, setTracks] = useState<LabelOnceTrack[]>([]);
  const [review, setReview] = useState<TrackReview | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (classId == null && classes.length) setClassId((classes.find((c) => c.name === "vehicle_fallback") ?? classes[0]).id);
  }, [classes, classId]);

  const loadTracks = useCallback(async () => {
    if (!sessionId) return;
    try { setTracks((await labelOnceTracks(sessionId)).tracks); } catch { setTracks([]); }
  }, [sessionId]);

  const loadScene = useCallback(async () => {
    if (!sessionId) return;
    setBusy("Placing scans"); setErr(null);
    try {
      const [s, p] = await Promise.all([scenePoints(sessionId, { every }), scenePoses(sessionId)]);
      setScene(s);
      setPoses(p.poses);
      setNote(`${num(s.count)} points from ${s.scans} of ${p.poses.length} scans, placed by their ego poses.`);
      if (!draft && p.poses.length) {
        const mid = p.poses[Math.floor(p.poses.length / 3)];
        const c = [mid.x + 12 * Math.cos(mid.yaw), mid.y + 12 * Math.sin(mid.yaw)];
        const g = groundNear(s.points, s.count, c[0], c[1]) ?? mid.z - 1.7;
        setDraft({ center: [c[0], c[1], g + 0.8], dims: [4.4, 1.8, 1.6], yaw: mid.yaw });
      }
      await loadTracks();
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(null); }
  }, [sessionId, every, draft, loadTracks]);

  useEffect(() => { setScene(null); setReview(null); setDraft(null); void loadTracks(); }, [sessionId, loadTracks]);

  const cuboids: ViewerCuboid[] = useMemo(() => (draft ? [{
    object_3d_id: "draft", center: draft.center, dims: draft.dims, yaw: draft.yaw, state: "review",
  }] : []), [draft]);

  const onMove = useCallback((_id: string, x: number, y: number) => {
    setDraft((d) => {
      if (!d) return d;
      const g = scene ? groundNear(scene.points, scene.count, x, y) : null;
      return { ...d, center: [x, y, g != null ? g + d.dims[2] / 2 : d.center[2]] };
    });
  }, [scene]);

  const setDim = (i: number, v: number) => setDraft((d) => {
    if (!d) return d;
    const dims = [...d.dims]; dims[i] = v;
    const center = [...d.center];
    if (i === 2) center[2] = d.center[2] - d.dims[2] / 2 + v / 2;      // keep the base where it is
    return { ...d, dims, center };
  });

  const onClass = (id: number) => {
    setClassId(id);
    const n = classes.find((c) => c.id === id)?.name ?? "";
    if (DIMS[n]) setDraft((d) => (d ? { ...d, dims: DIMS[n], center: [d.center[0], d.center[1], d.center[2] - d.dims[2] / 2 + DIMS[n][2] / 2] } : d));
  };

  const { job } = useToolJob(jobId, {
    onDone: async (j) => {
      setJobId(null);
      if (j.status !== "done") { setErr(j.error || `The job ${j.status}.`); return; }
      const r = j.result as Record<string, unknown>;
      setNote(`Placed on ${r.scans} scans (${r.with_points} with LiDAR points), ${r.boxes_2d} camera boxes. All in review.`);
      await loadTracks();
      if (r.track_3d_id) await openReview(String(r.track_3d_id));
    },
  });

  const propagate = async () => {
    if (!draft || classId == null) return;
    setErr(null);
    try {
      const j = await labelOnce(sessionId, { center: draft.center, dims: draft.dims, yaw: draft.yaw, class_id: classId,
        ground_snap: snap, max_range_m: range });
      setJobId(j.job_id);
    } catch (e) { setErr(humanizeError(e)); }
  };

  const openReview = async (tid: string) => {
    setBusy("Loading review"); setErr(null);
    try { setReview(await trackReview(tid)); setPicked(new Set()); } catch (e) { setErr(humanizeError(e)); } finally { setBusy(null); }
  };

  const decide = async (action: "accept" | "reject", ids?: string[]) => {
    if (!review) return;
    setBusy("Saving"); setErr(null);
    try {
      const r = await trackReviewApply(review.track_3d_id, { action, object_3d_ids: ids });
      setNote(`${action === "accept" ? "Accepted" : "Rejected"} ${r.updated_3d} cuboids and ${r.updated_2d} camera boxes.`);
      setReview(await trackReview(review.track_3d_id));
      setPicked(new Set());
      await loadTracks();
    } catch (e) { setErr(humanizeError(e)); } finally { setBusy(null); }
  };

  const trajectory = useMemo(() => poses.map((p) => ({ x: p.x, y: p.y })), [poses]);
  const inView = review ? review.scans.filter((s) => s.boxes.length) : [];

  return (
    <div className="flex h-full flex-1">
      <Inspector title="Label once" side="left" width="w-80">
        <div className="flex flex-col gap-3 p-4 text-[12px]">
          {!sessionId && <div className="text-ink-3">Choose a session first (switch back to scan cuboids to pick one).</div>}
          {sessionId && (
            <div className="flex items-end gap-2">
              <label className="flex flex-1 flex-col gap-1">Every
                <select className="field" value={every} onChange={(e) => setEvery(Number(e.target.value))} aria-label="scan step">
                  {[1, 2, 5, 10, 20].map((n) => <option key={n} value={n}>{n === 1 ? "every scan" : `every ${n}th scan`}</option>)}
                </select>
              </label>
              <button type="button" className="btn btn-primary" onClick={loadScene} disabled={!!busy}>{scene ? "Reload scene" : "Load scene"}</button>
            </div>
          )}
          {draft && (
            <div className="flex flex-col gap-2 rounded border border-line bg-bg-2 p-2.5">
              <div className="font-medium text-ink-2">The cuboid <span className="font-normal text-ink-3">(drag it in the bird&apos;s-eye view)</span></div>
              <label className="flex items-center gap-2">Class
                <select className="field min-w-0 flex-1 py-1" value={classId ?? ""} onChange={(e) => onClass(Number(e.target.value))}>
                  {classes.map((c) => <option key={c.id} value={c.id}>{classLabel(c.name)}</option>)}
                </select>
              </label>
              {["Length", "Width", "Height"].map((lab, i) => (
                <label key={lab} className="flex items-center gap-2"><span className="w-12">{lab}</span>
                  <input type="range" min={0.2} max={14} step={0.05} value={draft.dims[i]} className="flex-1"
                    onChange={(e) => setDim(i, Number(e.target.value))} />
                  <span className="w-12 text-right tabular-nums">{draft.dims[i].toFixed(2)} m</span></label>
              ))}
              <label className="flex items-center gap-2"><span className="w-12">Heading</span>
                <input type="range" min={-Math.PI} max={Math.PI} step={0.005} value={draft.yaw} className="flex-1"
                  onChange={(e) => setDraft({ ...draft, yaw: Number(e.target.value) })} />
                <span className="w-12 text-right tabular-nums">{Math.round((draft.yaw * 180) / Math.PI)}°</span></label>
              <label className="flex items-center gap-2"><input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} />
                Snap to each scan&apos;s road surface</label>
              <label className="flex items-center gap-2">Only scans within
                <input type="number" className="field w-16" min={5} max={150} value={range} onChange={(e) => setRange(Number(e.target.value) || 60)} /> m</label>
              <button type="button" className="btn btn-primary" onClick={propagate} disabled={!!jobId || classId == null}>
                {jobId ? `Placing… ${Math.round((job?.progress ?? 0) * 100)}%` : "Put it on every scan and camera"}
              </button>
              <div className="text-ink-3">Right for things that do not move (parked vehicles, poles, signs, barriers). The review below is where a moving one shows.</div>
            </div>
          )}
          {tracks.length > 0 && (
            <div>
              <div className="label-caps mb-1">Labelled once in this session</div>
              <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
                {tracks.map((t) => (
                  <button key={t.track_3d_id} type="button" onClick={() => openReview(t.track_3d_id)} aria-pressed={review?.track_3d_id === t.track_3d_id}
                    className={`rounded border px-2 py-1 text-left ${review?.track_3d_id === t.track_3d_id ? "border-accent bg-head" : "border-line bg-bg-2"}`}>
                    <div className="flex justify-between"><span>{classLabel(t.class_name)}</span><span className="text-ink-3">{t.scans} scans</span></div>
                    <div className="text-[11px] text-ink-3">{Object.entries(t.states).map(([k, v]) => `${v} ${k}`).join(" · ")}</div>
                  </button>
                ))}
              </div>
            </div>
          )}
          {note && <div role="status" className="text-ink-2">{note}</div>}
          {err && <div role="alert" className="rounded border border-block/40 bg-block/10 p-2 text-block">{err}</div>}
        </div>
      </Inspector>

      <main className="flex min-w-0 flex-1 flex-col">
        {review ? (
          <div className="flex min-h-0 flex-1 flex-col">
            <div className="flex flex-wrap items-center gap-2 border-b border-line px-3 py-2 text-[12px]">
              <span className="font-medium">Review: {classLabel(classes.find((c) => c.id === review.class_id)?.name)}</span>
              <span className="text-ink-3">{review.n} scans · {inView.length} with the object in a camera</span>
              <span className="ml-auto flex gap-2">
                <button type="button" className="btn" onClick={() => setReview(null)}>Back to the scene</button>
                <button type="button" className="btn btn-danger" disabled={!picked.size || !!busy}
                  onClick={() => decide("reject", Array.from(picked))}>Reject {picked.size || ""} picked</button>
                <button type="button" className="btn btn-primary" disabled={!!busy} onClick={() => decide("accept")}
                  data-tip="accept every cuboid and camera box of this track (reviewer)">Accept all</button>
              </span>
            </div>
            <div className="grid min-h-0 flex-1 auto-rows-min grid-cols-1 gap-2 overflow-y-auto p-3 xl:grid-cols-2">
              {inView.map((s) => s.boxes.map((b) => (
                <label key={b.object_id} className={`panel cursor-pointer p-2 ${picked.has(s.object_3d_id) ? "ring-2 ring-block" : ""}`}>
                  <div className="mb-1 flex items-center gap-2 text-[11px]">
                    <input type="checkbox" checked={picked.has(s.object_3d_id)} aria-label="pick to reject"
                      onChange={(e) => setPicked((p) => { const n = new Set(p); if (e.target.checked) n.add(s.object_3d_id); else n.delete(s.object_3d_id); return n; })} />
                    <span>{fromNs(s.ts_ns)}</span><span className="text-ink-3">{b.cam_id}</span>
                    <span className="text-ink-3">{s.n_points ?? 0} LiDAR points · {s.range_m?.toFixed(0)} m</span>
                    <span className="ml-auto"><StateBadge state={b.state} /></span>
                  </div>
                  <div className="relative w-full" style={{ aspectRatio: `${b.width} / ${b.height}` }}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={`/api/frames/${b.frame_id}/image`} alt={`${b.cam_id} frame`} className="absolute inset-0 h-full w-full" loading="lazy" />
                    <svg viewBox={`0 0 ${b.width} ${b.height}`} className="absolute inset-0 h-full w-full" preserveAspectRatio="none">
                      <rect x={b.bbox[0]} y={b.bbox[1]} width={b.bbox[2] - b.bbox[0]} height={b.bbox[3] - b.bbox[1]}
                        fill="none" stroke={b.state === "rejected" ? "#f87171" : b.state === "accepted" ? "#22d3ee" : "#fbbf24"} strokeWidth={3} />
                    </svg>
                  </div>
                </label>
              )))}
              {!inView.length && <EmptyState icon="cuboid" title="No camera saw it">The cuboid is on {review.n} scans but outside every camera frame.</EmptyState>}
            </div>
          </div>
        ) : (
          <>
            <div className="relative flex-1 border-b border-line">
              <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">Scene (all placed scans)</div>
              {scene ? (
                <PointCloudViewer points={scene.points} count={scene.count} colorBy="height" intensityRange={[0, 1]}
                  source="lidar" mode="perspective" cuboids={cuboids} selectedId="draft" trajectory={trajectory} />
              ) : (
                <div className="flex h-full items-center justify-center">
                  {busy ? <span role="status" className="text-ink-3">{busy}...</span> : (
                    <EmptyState icon="cuboid" title="Load the scene">
                      Every scan of the session placed by its ego pose, so a static object is drawn once for all of them.
                    </EmptyState>
                  )}
                </div>
              )}
            </div>
            <div className="relative h-1/2 min-h-[240px]">
              <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">Bird&apos;s-eye view <span className="font-normal">(drag the box onto the object)</span></div>
              {scene && (
                <PointCloudViewer points={scene.points} count={scene.count} colorBy="height" intensityRange={[0, 1]}
                  source="lidar" mode="bev" pointSize={0.25} cuboids={cuboids} selectedId="draft" trajectory={trajectory}
                  onSelectCuboid={() => {}} onMoveCuboid={onMove} />
              )}
            </div>
          </>
        )}
      </main>
    </div>
  );
}
