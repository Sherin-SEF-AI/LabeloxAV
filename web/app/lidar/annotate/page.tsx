"use client";

// The 3D cuboid annotation workspace (M-L2.1), BEV-first. Load a cloud, AI-lift its 2D objects into
// ground-snapped cuboids, then refine: select a box (click in either view), drag it in the BEV, set its
// dimensions and yaw, pick an ontology class, ground-snap, and save. Cuboids render in the 3D and BEV views
// and project onto the camera image. Human edits write object_3d with source=human.

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, lidarCloudPoints, type Cuboid3D, type LidarCloud, type LidarPoints } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";
import type { ColorBy } from "@/components/lidar/PointCloudViewer";

const PointCloudViewer = dynamic(() => import("@/components/lidar/PointCloudViewer"), { ssr: false });

const DEFAULT_DIMS: Record<string, number[]> = {
  sedan: [4.2, 1.8, 1.5], suv: [4.6, 1.9, 1.7], truck: [7.0, 2.5, 3.0], bus: [11.0, 2.6, 3.2],
  motorcycle: [2.0, 0.8, 1.4], pedestrian: [0.6, 0.6, 1.7], autorickshaw: [2.6, 1.4, 1.8],
};

export default function CuboidAnnotatePage() {
  const [sessionId, setSessionId] = useState("");
  const [clouds, setClouds] = useState<LidarCloud[]>([]);
  const [cloud, setCloud] = useState<LidarCloud | null>(null);
  const [data, setData] = useState<LidarPoints | null>(null);
  const [colorBy, setColorBy] = useState<ColorBy>("height");
  const [cuboids, setCuboids] = useState<Cuboid3D[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [classes, setClasses] = useState<OntologyClass[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.ontology().then((o) => setClasses(o.classes)).catch(() => {});
  }, []);

  const selected = useMemo(
    () => cuboids.find((c) => c.object_3d_id === selectedId) || null,
    [cuboids, selectedId],
  );

  const loadClouds = useCallback(async (sid: string) => {
    setErr(null);
    try {
      const r = await api.lidarClouds(sid.trim());
      setClouds(r.clouds);
      if (!r.clouds.length) setErr("No clouds in this session.");
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  const openCloud = useCallback(async (c: LidarCloud) => {
    setCloud(c);
    setSelectedId(null);
    setBusy(true);
    try {
      const [pts, objs] = await Promise.all([
        lidarCloudPoints(c.cloud_id, { variant: "raw", max: 300000 }),
        api.lidarObjects3d(c.cloud_id),
      ]);
      setData(pts);
      setCuboids(objs.objects);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const reloadCuboids = useCallback(async () => {
    if (!cloud) return;
    const objs = await api.lidarObjects3d(cloud.cloud_id);
    setCuboids(objs.objects);
  }, [cloud]);

  const aiLift = async () => {
    if (!cloud) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.lidarLiftCloud(cloud.cloud_id);
      if (!r.cuboids) setErr("No 2D objects on the synchronized frame to lift.");
      await reloadCuboids();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  // local geometry edits (optimistic) then persist
  const patchLocal = (id: string, patch: Partial<Cuboid3D>) =>
    setCuboids((cs) => cs.map((c) => (c.object_3d_id === id ? { ...c, ...patch } : c)));

  const onMoveCuboid = (id: string, x: number, y: number, commit: boolean) => {
    const cur = cuboids.find((c) => c.object_3d_id === id);
    if (!cur) return;
    const center = [x, y, cur.center[2]];
    patchLocal(id, { center });
    if (commit) saveCuboid(id, { center });
  };

  const saveCuboid = async (id: string, fields: Partial<Cuboid3D>) => {
    const cur = cuboids.find((c) => c.object_3d_id === id);
    if (!cur) return;
    try {
      const saved = await api.lidarPatchCuboid(id, {
        class_id: (fields.class_id as number) ?? cur.class_id,
        center: (fields.center as number[]) ?? cur.center,
        dims: (fields.dims as number[]) ?? cur.dims,
        yaw: (fields.yaw as number) ?? cur.yaw,
        ground_snap: Boolean(fields.attrs && (fields.attrs as Record<string, unknown>).ground_snap),
        expected_version: cur.version,
      });
      patchLocal(id, saved);
    } catch (e) {
      setErr(String(e));
      reloadCuboids();
    }
  };

  const groundSnap = (id: string) => {
    const cur = cuboids.find((c) => c.object_3d_id === id);
    if (cur) saveCuboid(id, { attrs: { ground_snap: true } });
  };

  const addCuboid = async () => {
    if (!cloud) return;
    const cls = classes.find((c) => c.name === "sedan") || classes[0];
    if (!cls) return;
    try {
      const created = await api.lidarCreateCuboid(cloud.cloud_id, {
        class_id: cls.id, center: [12, 0, 1], dims: DEFAULT_DIMS[cls.name] || [4, 1.8, 1.5],
        yaw: 0, ground_snap: true,
      });
      setCuboids((cs) => [...cs, created]);
      setSelectedId(created.object_3d_id);
    } catch (e) {
      setErr(String(e));
    }
  };

  const removeCuboid = async (id: string) => {
    try {
      await api.lidarDeleteCuboid(id);
      setCuboids((cs) => cs.filter((c) => c.object_3d_id !== id));
      setSelectedId(null);
    } catch (e) {
      setErr(String(e));
    }
  };

  const setDim = (i: number, v: number) => {
    if (!selected) return;
    const dims = [...selected.dims];
    dims[i] = v;
    patchLocal(selected.object_3d_id, { dims });
  };

  return (
    <div className="flex h-[calc(100vh-3.5rem)] bg-[#0a0e14] text-neutral-200">
      <aside className="flex w-80 shrink-0 flex-col gap-3 overflow-y-auto border-r border-neutral-800 p-4">
        <div className="text-xs uppercase tracking-wider text-neutral-500">Cuboid annotation</div>
        <form onSubmit={(e) => { e.preventDefault(); loadClouds(sessionId); }} className="flex gap-2">
          <input value={sessionId} onChange={(e) => setSessionId(e.target.value)} placeholder="session id"
            className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-900 px-2 py-1 text-xs" />
          <button className="rounded bg-cyan-700 px-3 py-1 text-xs hover:bg-cyan-600">Load</button>
        </form>

        {clouds.length > 0 && (
          <div className="flex max-h-32 flex-col gap-1 overflow-y-auto">
            {clouds.map((c) => (
              <button key={c.cloud_id} onClick={() => openCloud(c)}
                className={`rounded border px-2 py-1 text-left text-xs ${
                  cloud?.cloud_id === c.cloud_id ? "border-cyan-600 bg-cyan-950" : "border-neutral-800 bg-neutral-900"}`}>
                {c.source} | {c.point_count.toLocaleString()} pts | {c.cloud_id.slice(0, 8)}
              </button>
            ))}
          </div>
        )}

        {cloud && (
          <div className="flex gap-2">
            <button onClick={aiLift} disabled={busy}
              className="flex-1 rounded bg-emerald-800 px-2 py-1.5 text-xs hover:bg-emerald-700 disabled:opacity-40">
              AI lift 2D to 3D
            </button>
            <button onClick={addCuboid}
              className="rounded border border-neutral-700 px-2 py-1.5 text-xs hover:border-cyan-700">+ box</button>
          </div>
        )}

        {cloud && (
          <div>
            <div className="mb-1 text-xs uppercase tracking-wider text-neutral-500">Cuboids ({cuboids.length})</div>
            <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
              {cuboids.map((c) => (
                <button key={c.object_3d_id} onClick={() => setSelectedId(c.object_3d_id)}
                  className={`flex justify-between rounded border px-2 py-1 text-left text-xs ${
                    selectedId === c.object_3d_id ? "border-white bg-neutral-800" : "border-neutral-800 bg-neutral-900"}`}>
                  <span>{c.class_name}</span>
                  <span className="text-neutral-500">{c.state} | {c.box_source}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {selected && (
          <div className="flex flex-col gap-2 rounded border border-neutral-800 bg-neutral-900 p-2 text-xs">
            <div className="text-neutral-400">Selected ({selected.source})</div>
            <label className="flex items-center justify-between gap-2">
              class
              <select value={selected.class_id}
                onChange={(e) => { const id = Number(e.target.value); patchLocal(selected.object_3d_id, { class_id: id, class_name: classes.find((x) => x.id === id)?.name }); saveCuboid(selected.object_3d_id, { class_id: id }); }}
                className="min-w-0 flex-1 rounded border border-neutral-700 bg-neutral-950 px-1 py-0.5">
                {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
              </select>
            </label>
            {["L", "W", "H"].map((lab, i) => (
              <label key={lab} className="flex items-center gap-2">
                <span className="w-3">{lab}</span>
                <input type="range" min={0.3} max={14} step={0.1} value={selected.dims[i]}
                  onChange={(e) => setDim(i, Number(e.target.value))}
                  onMouseUp={() => saveCuboid(selected.object_3d_id, { dims: selected.dims })}
                  className="flex-1" />
                <span className="w-10 text-right text-neutral-400">{selected.dims[i].toFixed(1)}</span>
              </label>
            ))}
            <label className="flex items-center gap-2">
              <span className="w-3">yaw</span>
              <input type="range" min={-3.14159} max={3.14159} step={0.01} value={selected.yaw}
                onChange={(e) => patchLocal(selected.object_3d_id, { yaw: Number(e.target.value) })}
                onMouseUp={() => saveCuboid(selected.object_3d_id, { yaw: selected.yaw })}
                className="flex-1" />
              <span className="w-10 text-right text-neutral-400">{(selected.yaw * 57.3).toFixed(0)}</span>
            </label>
            <div className="flex gap-2">
              <button onClick={() => groundSnap(selected.object_3d_id)}
                className="flex-1 rounded bg-neutral-800 px-2 py-1 hover:bg-neutral-700">Ground snap</button>
              <button onClick={() => removeCuboid(selected.object_3d_id)}
                className="rounded bg-red-900 px-2 py-1 hover:bg-red-800">Delete</button>
            </div>
            <div className="text-neutral-600">Drag the box in the BEV view to move it.</div>
          </div>
        )}

        {err && <div className="rounded border border-red-900 bg-red-950 p-2 text-xs text-red-300">{err}</div>}
      </aside>

      <main className="flex flex-1 flex-col">
        <div className="relative flex-1 border-b border-neutral-800">
          <div className="absolute left-3 top-3 z-10 text-xs uppercase tracking-wider text-neutral-500">3D</div>
          {data && (
            <PointCloudViewer points={data.points} count={data.count} colorBy={colorBy}
              intensityRange={[data.intensityMin, data.intensityMax]} source={data.source} mode="perspective"
              cuboids={cuboids} selectedId={selectedId} onSelectCuboid={setSelectedId} />
          )}
          {!data && <div className="flex h-full items-center justify-center text-sm text-neutral-600">Load a session and pick a cloud.</div>}
        </div>
        <div className="relative h-2/5 min-h-[220px]">
          <div className="absolute left-3 top-3 z-10 text-xs uppercase tracking-wider text-neutral-500">BEV (drag to move)</div>
          <div className="absolute right-3 top-3 z-10 flex gap-1">
            {(["height", "intensity"] as ColorBy[]).map((k) => (
              <button key={k} onClick={() => setColorBy(k)}
                className={`rounded px-2 py-0.5 text-xs ${colorBy === k ? "bg-cyan-700" : "bg-neutral-800"}`}>{k}</button>
            ))}
          </div>
          {data && (
            <PointCloudViewer points={data.points} count={data.count} colorBy={colorBy}
              intensityRange={[data.intensityMin, data.intensityMax]} source={data.source} mode="bev" pointSize={0.4}
              cuboids={cuboids} selectedId={selectedId} onSelectCuboid={setSelectedId} onMoveCuboid={onMoveCuboid} />
          )}
        </div>
      </main>
    </div>
  );
}
