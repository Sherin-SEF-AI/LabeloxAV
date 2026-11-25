"use client";

// The 3D-to-2D linked annotation workspace (M-L2.4). One physical object across the cloud and the camera:
// select a cuboid in the 3D view and its projection highlights on the synchronized camera image, and click a
// projected box on the image to select the cuboid. Each object carries auto-computed distance, velocity, and
// occlusion, and a correction can be propagated to similar objects in a batch.

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, lidarCloudPoints, type Cuboid3D, type LidarCloud, type LidarPoints , humanizeError } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import Inspector from "@/components/shell/Inspector";
import SessionPicker from "@/components/lidar/SessionPicker";
import EmptyState from "@/components/ui/EmptyState";
import { fromNs, humanize, num, shortId } from "@/lib/format";

// Ontology class names read as words ("two wheeler") but keep their case: "suv" is not "Suv".
const classLabel = (n: string | null | undefined) => (n ?? "").replace(/_/g, " ");
const fmt = (v: unknown, digits = 1) =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : v == null ? "-" : String(v);

const PointCloudViewer = dynamic(() => import("@/components/lidar/PointCloudViewer"), { ssr: false });

// The camera overlay's viewBox must be the frame's own pixel size. It was hardcoded to 1280x960, which
// is right for the Tigor rig and wrong for every dashcam and imported session in the corpus: a 1920x1080
// frame drew every projected box at two thirds scale and offset, and the boxes still looked like boxes.
const FALLBACK_W = 1280, FALLBACK_H = 960;

export default function LinkedWorkspacePage() {
  const [sessionId, setSessionId] = useState("");
  const [clouds, setClouds] = useState<LidarCloud[]>([]);
  const [cloud, setCloud] = useState<LidarCloud | null>(null);
  const [data, setData] = useState<LidarPoints | null>(null);
  const [cuboids, setCuboids] = useState<Cuboid3D[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [projections, setProjections] = useState<Record<string, number[]>>({}); // cam_f bbox per cuboid
  const [props, setProps] = useState<Record<string, number | null> | null>(null);
  const [linked, setLinked] = useState<Record<string, number[]> | null>(null);
  const [similar, setSimilar] = useState<{ object_3d_id: string; dims_dist: number }[]>([]);
  const [classes, setClasses] = useState<OntologyClass[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [frameSize, setFrameSize] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => { api.ontology().then((o) => setClasses(o.classes)).catch(() => {}); }, []);

  const frameId = useMemo(() => cuboids.find((c) => c.frame_id)?.frame_id || null, [cuboids]);
  const selected = useMemo(() => cuboids.find((c) => c.object_3d_id === selectedId) || null, [cuboids, selectedId]);

  // The frame's real pixel size, for the overlay's viewBox. Falls back to the rig default only when the
  // frame cannot be read, and a wrong size here silently misplaces every projected box.
  useEffect(() => {
    if (!frameId) { setFrameSize(null); return; }
    let live = true;
    api.frame(frameId)
      .then((f) => { if (live && f.width && f.height) setFrameSize({ w: f.width, h: f.height }); })
      .catch(() => { if (live) setFrameSize(null); });
    return () => { live = false; };
  }, [frameId]);

  const openCloud = useCallback(async (c: LidarCloud) => {
    setCloud(c);
    setSelectedId(null);
    setProps(null);
    setSimilar([]);
    try {
      const [pts, objs] = await Promise.all([
        lidarCloudPoints(c.cloud_id, { variant: "raw", max: 250000 }),
        api.lidarObjects3d(c.cloud_id),
      ]);
      setData(pts);
      setCuboids(objs.objects);
      // project every cuboid onto the front camera for the clickable overlay (cloud -> camera link)
      const entries = await Promise.all(objs.objects.map(async (o) => {
        try {
          const lv = await api.lidarObject3dLinked(o.object_3d_id);
          return [o.object_3d_id, lv.projections["cam_f"]] as const;
        } catch {
          return [o.object_3d_id, undefined] as const;
        }
      }));
      const map: Record<string, number[]> = {};
      for (const [id, bbox] of entries) if (bbox) map[id] = bbox;
      setProjections(map);
    } catch (e) {
      setErr(humanizeError(e));
    }
  }, []);

  // selecting a cuboid pulls its linked 2D object, per-camera projections, and auto-computed properties
  useEffect(() => {
    if (!selectedId) { setProps(null); setLinked(null); setSimilar([]); return; }
    api.lidarObject3dLinked(selectedId).then((lv) => setLinked(lv.projections)).catch(() => {});
    api.lidarObject3dProperties(selectedId).then((r) => setProps(r.properties)).catch(() => {});
    api.lidarSimilar3d(selectedId, 8).then((r) => setSimilar(r.similar)).catch(() => {});
  }, [selectedId]);

  const link = async () => {
    if (!cloud) return;
    try {
      await api.lidarLinkCloud(cloud.cloud_id);
      await openCloud(cloud);
    } catch (e) {
      setErr(humanizeError(e));
    }
  };

  const batchCorrect = async (classId: number) => {
    if (!selected || !similar.length) return;
    const ids = [selected.object_3d_id, ...similar.map((s) => s.object_3d_id)];
    try {
      await api.lidarBatchCorrect(ids, classId);
      if (cloud) await openCloud(cloud);
    } catch (e) {
      setErr(humanizeError(e));
    }
  };

  return (
    <PageShell active="LINKED 2D-3D" title="Linked 2D-3D">
      <div data-theme="dark" className="flex h-full bg-[#0a0e14] text-ink">
        <Inspector title="3D to camera" side="left" width="w-80">
          <div className="flex flex-col gap-3 p-4">
            <SessionPicker value={sessionId} onChange={setSessionId} onLoad={(id) => {
              if (!id.trim()) { setErr("Choose a session first."); return; }
              setErr(null);
              api.lidarClouds(id.trim()).then((r) => {
                setClouds(r.clouds);
                if (!r.clouds.length) setErr("No point clouds in this session. Build pseudo-LiDAR for it on the Point clouds page.");
              }).catch((x) => setErr(humanizeError(x)));
            }} />

            {clouds.length > 0 && (
              <div className="flex max-h-28 flex-col gap-1 overflow-y-auto">
                {clouds.map((c) => (
                  <button key={c.cloud_id} type="button" onClick={() => openCloud(c)} aria-pressed={cloud?.cloud_id === c.cloud_id}
                    title={`${c.cloud_id} · ts_ns ${c.ts_ns}`}
                    className={`rounded border px-2.5 py-1.5 text-left text-[12px] ${cloud?.cloud_id === c.cloud_id ? "border-accent/60 bg-accent/15" : "border-line bg-bg-2 hover:border-line-3"}`}>
                    <div className="flex justify-between">
                      <span className="font-medium">{c.source === "lidar" ? "LiDAR" : humanize(c.source)}</span>
                      <span className="text-ink-3 tabular-nums">{num(c.point_count)} points</span>
                    </div>
                    <div className="truncate text-[11px] text-ink-3">{fromNs(c.ts_ns)} · <span className="font-mono">{shortId(c.cloud_id)}</span></div>
                  </button>
                ))}
              </div>
            )}

            {cloud && (
              <button type="button" onClick={link} className="btn btn-primary"
                title="match each 3D box to the 2D box it projects onto in the camera image">
                Link cuboids to 2D objects
              </button>
            )}

            {selected && (
              <div className="flex flex-col gap-2 rounded border border-line bg-bg-2 p-2.5 text-[12px]">
                <div className="font-medium text-ink">Selected: {classLabel(selected.class_name)}</div>
                <div className="text-ink-3">2D object: {selected.object_id ? <span className="font-mono">{shortId(selected.object_id)}</span> : "not linked yet"}</div>
                {props && (
                  <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-ink-2">
                    <span>Distance</span><span className="text-right tabular-nums">{fmt(props.distance_m)} m</span>
                    <span>Heading</span><span className="text-right tabular-nums">{fmt(props.heading_deg, 0)}°</span>
                    <span>Speed</span><span className="text-right tabular-nums">{fmt(props.velocity_mps)} m/s</span>
                    <span>Occlusion</span><span className="text-right tabular-nums">{fmt(props.occlusion, 2)}</span>
                  </div>
                )}
                {linked && (
                  <div className="text-ink-3">Visible in: {Object.keys(linked).join(", ") || "no camera"}</div>
                )}
                {similar.length > 0 && (
                  <div>
                    <div className="mb-1.5 text-ink-2">{similar.length} similar boxes. Correct them all to:</div>
                    <div className="flex flex-wrap gap-1">
                      {classes.filter((c) => ["sedan", "suv", "truck", "bus", "autorickshaw"].includes(c.name)).map((c) => (
                        <button key={c.id} type="button" onClick={() => batchCorrect(c.id)}
                          className="chip">{classLabel(c.name)}</button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            {err && <div role="alert" className="rounded border border-block/40 bg-block/10 p-2.5 text-[12px] text-block">{err}</div>}
          </div>
        </Inspector>

        <div className="flex flex-1 min-w-0">
          <div className="relative flex-1 border-r border-line">
            <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">Point cloud</div>
            {!data && (
              <div className="flex h-full items-center justify-center">
                <EmptyState icon="link" title={clouds.length ? "Pick a cloud" : "No cloud loaded"}>
                  {clouds.length
                    ? "Choose a cloud on the left. Select a box here or on the camera image and the other view follows."
                    : "Choose a session on the left, then a cloud. Each 3D box is shown with its projection on the camera image."}
                </EmptyState>
              </div>
            )}
            {data && (
              <PointCloudViewer points={data.points} count={data.count} colorBy="height"
                intensityRange={[data.intensityMin, data.intensityMax]} source={data.source} mode="perspective"
                cuboids={cuboids} selectedId={selectedId} onSelectCuboid={setSelectedId} />
            )}
          </div>
          <div className="relative w-1/2 bg-black">
            <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">Front camera <span className="font-mono font-normal">cam_f</span></div>
            {frameId && (
              <div className="relative h-full w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`/api/frames/${frameId}/image`} alt="cam_f" className="h-full w-full object-contain" />
                <svg viewBox={`0 0 ${frameSize?.w ?? FALLBACK_W} ${frameSize?.h ?? FALLBACK_H}`}
                  className="absolute inset-0 h-full w-full"
                  preserveAspectRatio="xMidYMid meet">
                  {cuboids.map((c) => {
                    const b = projections[c.object_3d_id];
                    if (!b) return null;
                    const sel = c.object_3d_id === selectedId;
                    return (
                      <rect key={c.object_3d_id} x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]}
                        fill={sel ? "rgba(255,255,255,0.12)" : "none"} stroke={sel ? "#ffffff" : "#22d3ee"}
                        strokeWidth={sel ? 4 : 2} className="cursor-pointer"
                        onClick={() => setSelectedId(c.object_3d_id)} />
                    );
                  })}
                </svg>
              </div>
            )}
            {!frameId && (
              <div className="flex h-full items-center justify-center px-6 text-center text-[12.5px] text-ink-3">
                {cloud ? "This cloud has no 3D boxes yet. Create them on the 3D cuboids page, then link them here." : "The camera image appears here once a cloud with 3D boxes is open."}
              </div>
            )}
          </div>
        </div>
      </div>
    </PageShell>
  );
}
