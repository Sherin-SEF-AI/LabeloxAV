"use client";

// The 3D-to-2D linked annotation workspace (M-L2.4). One physical object across the cloud and the camera:
// select a cuboid in the 3D view and its projection highlights on the synchronized camera image, and click a
// projected box on the image to select the cuboid. Each object carries auto-computed distance, velocity, and
// occlusion, and a correction can be propagated to similar objects in a batch.

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, lidarCloudPoints, type Cuboid3D, type LidarCloud, type LidarPoints } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import Inspector from "@/components/shell/Inspector";

const PointCloudViewer = dynamic(() => import("@/components/lidar/PointCloudViewer"), { ssr: false });

const CAM_W = 1280, CAM_H = 960;

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

  useEffect(() => { api.ontology().then((o) => setClasses(o.classes)).catch(() => {}); }, []);

  const frameId = useMemo(() => cuboids.find((c) => c.frame_id)?.frame_id || null, [cuboids]);
  const selected = useMemo(() => cuboids.find((c) => c.object_3d_id === selectedId) || null, [cuboids, selectedId]);

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
      setErr(String(e));
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
      setErr(String(e));
    }
  };

  const batchCorrect = async (classId: number) => {
    if (!selected || !similar.length) return;
    const ids = [selected.object_3d_id, ...similar.map((s) => s.object_3d_id)];
    try {
      await api.lidarBatchCorrect(ids, classId);
      if (cloud) await openCloud(cloud);
    } catch (e) {
      setErr(String(e));
    }
  };

  return (
    <PageShell active="LINKED 2D-3D" title="Linked 2D-3D"
      meta={err ? <span className="text-block">{err}</span> : undefined}>
      <div className="flex h-full bg-[#0a0e14] text-neutral-200">
        <Inspector title="Controls" side="left" width="w-80">
          <div className="flex flex-col gap-3 p-4">
            <form onSubmit={(e) => { e.preventDefault(); api.lidarClouds(sessionId.trim()).then((r) => setClouds(r.clouds)).catch((x) => setErr(String(x))); }}
              className="flex gap-2">
              <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} placeholder="session id"
                className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs" />
              <button className="rounded bg-cyan-700 px-3 py-1 text-xs hover:bg-cyan-600">Load</button>
            </form>

            {clouds.length > 0 && (
              <div className="flex max-h-28 flex-col gap-1 overflow-y-auto">
                {clouds.map((c) => (
                  <button key={c.cloud_id} onClick={() => openCloud(c)}
                    className={`rounded border px-2 py-1 text-left text-xs ${cloud?.cloud_id === c.cloud_id ? "border-cyan-600 bg-cyan-950" : "border-neutral-800 bg-neutral-900"}`}>
                    {c.source} | {c.point_count.toLocaleString()} pts
                  </button>
                ))}
              </div>
            )}

            {cloud && (
              <button onClick={link} className="rounded bg-emerald-800 px-2 py-1.5 text-xs hover:bg-emerald-700">
                Link cuboids to 2D objects
              </button>
            )}

            {selected && (
              <div className="flex flex-col gap-2 rounded border border-neutral-800 bg-neutral-900 p-2 text-xs">
                <div className="text-neutral-400">Selected {selected.class_name}</div>
                <div className="text-neutral-500">2D object: {selected.object_id ? selected.object_id.slice(0, 8) : "unlinked"}</div>
                {props && (
                  <div className="grid grid-cols-2 gap-x-2 gap-y-0.5 text-neutral-300">
                    <span>distance</span><span className="text-right">{props.distance_m} m</span>
                    <span>heading</span><span className="text-right">{props.heading_deg} deg</span>
                    <span>velocity</span><span className="text-right">{props.velocity_mps} m/s</span>
                    <span>occlusion</span><span className="text-right">{props.occlusion}</span>
                  </div>
                )}
                {linked && (
                  <div className="text-neutral-500">visible in: {Object.keys(linked).join(", ") || "none"}</div>
                )}
                {similar.length > 0 && (
                  <div>
                    <div className="mb-1 text-neutral-400">{similar.length} similar. Batch correct to:</div>
                    <div className="flex flex-wrap gap-1">
                      {classes.filter((c) => ["sedan", "suv", "truck", "bus", "autorickshaw"].includes(c.name)).map((c) => (
                        <button key={c.id} onClick={() => batchCorrect(c.id)}
                          className="rounded bg-neutral-800 px-2 py-0.5 hover:bg-cyan-800">{c.name}</button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
            {err && <div className="rounded border border-red-900 bg-red-950 p-2 text-xs text-red-300">{err}</div>}
          </div>
        </Inspector>

        <div className="flex flex-1 min-w-0">
          <div className="relative flex-1 border-r border-neutral-800">
            <div className="absolute left-3 top-3 z-10 text-xs uppercase tracking-wider text-neutral-500">Cloud</div>
            {data && (
              <PointCloudViewer points={data.points} count={data.count} colorBy="height"
                intensityRange={[data.intensityMin, data.intensityMax]} source={data.source} mode="perspective"
                cuboids={cuboids} selectedId={selectedId} onSelectCuboid={setSelectedId} />
            )}
          </div>
          <div className="relative w-1/2 bg-black">
            <div className="absolute left-3 top-3 z-10 text-xs uppercase tracking-wider text-neutral-500">Camera (cam_f)</div>
            {frameId && (
              <div className="relative h-full w-full">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={`/api/frames/${frameId}/image`} alt="cam_f" className="h-full w-full object-contain" />
                <svg viewBox={`0 0 ${CAM_W} ${CAM_H}`} className="absolute inset-0 h-full w-full"
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
            {!frameId && <div className="flex h-full items-center justify-center text-sm text-neutral-600">Pick a cloud with cuboids.</div>}
          </div>
        </div>
      </div>
    </PageShell>
  );
}
