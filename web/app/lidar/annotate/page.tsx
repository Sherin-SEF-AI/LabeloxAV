"use client";

import { T } from "@/lib/i18n";

// The 3D cuboid annotation workspace (M-L2.1), BEV-first. Load a cloud, AI-lift its 2D objects into
// ground-snapped cuboids, then refine: select a box (click in either view), drag it in the BEV, set its
// dimensions and yaw, pick an ontology class, ground-snap, and save. Cuboids render in the 3D and BEV views
// and project onto the camera image. Human edits write object_3d with source=human.

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { api, lidarCloudPoints, type Cuboid3D, type LidarCloud, type LidarPoints , humanizeError } from "@/lib/api";
import type { EgoTrajectory, OccupancyGridRow, OccupancyVoxels, OntologyClass } from "@/lib/types";
import type { ColorBy } from "@/components/lidar/PointCloudViewer";
import PageShell from "@/components/shell/PageShell";
import { useConfirm } from "@/components/ConfirmProvider";
import Inspector from "@/components/shell/Inspector";
import { StateBadge } from "@/components/StateBadge";
import SessionPicker from "@/components/lidar/SessionPicker";
import EmptyState from "@/components/ui/EmptyState";
import { Segmented } from "@/components/ui/kit";
import { fromNs, humanize, num, shortId } from "@/lib/format";

// Ontology class names read as words ("two wheeler") but keep their case: "suv" is not "Suv".
const classLabel = (n: string | null | undefined) => (n ?? "").replace(/_/g, " ");

const PointCloudViewer = dynamic(() => import("@/components/lidar/PointCloudViewer"), { ssr: false });
const EgoTrajectoryPlot = dynamic(() => import("@/components/lidar/EgoTrajectory"), { ssr: false });
// #3 label once in 3D: a cuboid drawn once in the session scene, put on every scan and camera (own workspace).
const LabelOnceWorkspace = dynamic(() => import("@/components/lidar/LabelOnceWorkspace"), { ssr: false });

const DEFAULT_DIMS: Record<string, number[]> = {
  sedan: [4.2, 1.8, 1.5], suv: [4.6, 1.9, 1.7], truck: [7.0, 2.5, 3.0], bus: [11.0, 2.6, 3.2],
  motorcycle: [2.0, 0.8, 1.4], pedestrian: [0.6, 0.6, 1.7], autorickshaw: [2.6, 1.4, 1.8],
};

export default function CuboidAnnotatePage() {
  const confirm = useConfirm();
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
  const [trajectory, setTrajectory] = useState<EgoTrajectory | null>(null);
  const [grids, setGrids] = useState<OccupancyGridRow[]>([]);
  const [occ, setOcc] = useState<OccupancyVoxels | null>(null);
  const [gridIdx, setGridIdx] = useState(0);
  const [mode, setMode] = useState<"scan" | "once">("scan");

  useEffect(() => {
    api.ontology().then((o) => setClasses(o.classes)).catch(() => {});
  }, []);

  const selected = useMemo(
    () => cuboids.find((c) => c.object_3d_id === selectedId) || null,
    [cuboids, selectedId],
  );

  const loadClouds = useCallback(async (sid: string) => {
    if (!sid.trim()) { setErr("Choose a session first."); return; }
    setErr(null);
    try {
      const r = await api.lidarClouds(sid.trim());
      setClouds(r.clouds);
      if (!r.clouds.length) setErr("No point clouds in this session. Build pseudo-LiDAR for it on the Point clouds page.");
      // Loaded beside the clouds rather than on demand: the trajectory is what makes two clouds
      // comparable, so its absence is something to see before annotating, not after.
      try {
        setTrajectory(await api.egoSessionPose(sid.trim()));
      } catch {
        setTrajectory(null);
      }
      try {
        const g = await api.occupancyList(sid.trim());
        setGrids(g.grids);
        setGridIdx(0);
        setOcc(null);
      } catch {
        setGrids([]);
        setOcc(null);
      }
    } catch (e) {
      setErr(humanizeError(e));
    }
  }, []);

  const buildPose = useCallback(async () => {
    if (!sessionId.trim()) return;
    setBusy(true);
    try {
      await api.egoBuildPose(sessionId.trim());
      setTrajectory(await api.egoSessionPose(sessionId.trim()));
    } catch (e) {
      setErr(humanizeError(e));
    } finally {
      setBusy(false);
    }
  }, [sessionId]);

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
      setErr(humanizeError(e));
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
      setErr(humanizeError(e));
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
      setErr(humanizeError(e));
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
      setErr(humanizeError(e));
    }
  };

  const removeCuboid = async (id: string) => {
    if (!(await confirm({ title: "Remove this 3D box?", danger: true, confirmLabel: "Remove" }))) return;
    try {
      await api.lidarDeleteCuboid(id);
      setCuboids((cs) => cs.filter((c) => c.object_3d_id !== id));
      setSelectedId(null);
    } catch (e) {
      setErr(humanizeError(e));
    }
  };

  const setDim = (i: number, v: number) => {
    if (!selected) return;
    const dims = [...selected.dims];
    dims[i] = v;
    patchLocal(selected.object_3d_id, { dims });
  };

  return (
    <PageShell active="CUBOID ANNOTATE" title="Cuboid annotation"
      right={<Segmented options={[{ value: "scan", label: "Cuboids on a scan" },
        { value: "once", label: "Label once in 3D", tip: "draw one cuboid in the whole session's scene; it lands on every scan and camera" }]}
        value={mode} onChange={(v) => setMode(v as "scan" | "once")} />}>
      {mode === "once" ? (
        <div data-theme="dark" className="flex h-full bg-[#0a0e14] text-ink">
          <LabelOnceWorkspace sessionId={sessionId} classes={classes} />
        </div>
      ) : (
      <div data-theme="dark" className="flex h-full bg-[#0a0e14] text-ink">
        <Inspector title="Cuboids" side="left" width="w-80">
          <div className="flex flex-col gap-3 p-4">
<SessionPicker value={sessionId} onChange={setSessionId} onLoad={loadClouds} />

        {clouds.length > 0 && (
          <div className="flex max-h-32 flex-col gap-1 overflow-y-auto">
            {clouds.map((c) => (
              <button key={c.cloud_id} type="button" onClick={() => openCloud(c)} aria-pressed={cloud?.cloud_id === c.cloud_id}
                title={`${c.cloud_id} · ts_ns ${c.ts_ns}`}
                className={`rounded border px-2.5 py-1.5 text-left text-[12px] ${
                  cloud?.cloud_id === c.cloud_id ? "border-accent/60 bg-accent/15" : "border-line bg-bg-2 hover:border-line-3"}`}>
                <div className="flex justify-between">
                  <span className="font-medium">{c.source === "lidar" ? "LiDAR" : humanize(c.source)}</span>
                  <span className="text-ink-3 tabular-nums">{num(c.point_count)} points</span>
                </div>
                <div className="truncate text-[11px] text-ink-3">{fromNs(c.ts_ns)} · <span className="font-mono">{shortId(c.cloud_id)}</span></div>
              </button>
            ))}
          </div>
        )}

        {/* The trajectory the cuboids are placed against. A session without one is a session where every
            cuboid sits in the coordinate frame of its own instant and cannot be compared to the next. */}
        {sessionId && (
          <div className="rounded border border-line bg-bg-2">
            <div className="border-b border-line px-2.5 py-1.5 text-[12px] font-medium text-ink-2">
              Ego trajectory
            </div>
            <EgoTrajectoryPlot trajectory={trajectory} currentTs={cloud?.ts_ns ?? null} height={160} />
            {trajectory && trajectory.poses === 0 && (
              <button type="button" onClick={buildPose} disabled={busy}
                title="estimate the vehicle's path from this session's GNSS and frames"
                className="btn m-2">
                Recover trajectory
              </button>
            )}
          </div>
        )}

        {/* The occupancy scrubber. A grid per instant, stepped in time, with the share of its flow field
            that a track actually spoke for printed beside it: the rest is an assumed zero and an arrow
            drawn from it would be confidence nobody has. */}
        {grids.length > 0 && (
          <div className="rounded border border-line bg-bg-2 p-2">
            <div className="mb-1 text-[12px] font-medium text-ink-2">
              Occupancy <span className="font-normal text-ink-3">· {grids.length} grids</span>
            </div>
            <input type="range" min={0} max={grids.length - 1} value={gridIdx} className="w-full"
              aria-label="occupancy time"
              onChange={(e) => {
                const i = Number(e.target.value);
                setGridIdx(i);
                api.occupancyVoxels(grids[i].grid_id).then(setOcc).catch(() => setOcc(null));
              }} />
            <div className="text-[11px] text-ink-2">
              {num(grids[gridIdx]?.occupied)} voxels ·{" "}
              {grids[gridIdx]?.flow_share != null
                ? `${((grids[gridIdx].flow_share ?? 0) * 100).toFixed(1)}% with a track velocity`
                : "no flow recorded"}
              {grids[gridIdx] && !grids[gridIdx].placed_by_pose && " · not placed by a pose"}
            </div>
            {occ?.truncated && (
              <div className="text-[11px] text-warn">
                Showing {num(occ.n)} of {num(occ.total)} voxels
              </div>
            )}
          </div>
        )}

        {cloud && (
          <div className="flex gap-2">
            <button type="button" onClick={aiLift} disabled={busy}
              title="turn the 2D boxes on the synchronised camera frame into ground-snapped 3D cuboids"
              className="btn btn-primary flex-1">
              AI lift 2D to 3D
            </button>
            <button type="button" onClick={addCuboid} title="add a box 12 m ahead of the vehicle"
              className="btn">+ Add box</button>
          </div>
        )}

        {cloud && (
          <div>
            <div className="mb-1.5 text-[12px] font-medium text-ink-2">Cuboids <span className="font-normal text-ink-3">({cuboids.length})</span></div>
            {cuboids.length === 0 && (
              <div className="text-[12px] text-ink-3">None yet. AI lift creates them from the camera&apos;s 2D boxes, or add one by hand.</div>
            )}
            <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
              {cuboids.map((c) => (
                <button key={c.object_3d_id} type="button" onClick={() => setSelectedId(c.object_3d_id)} aria-pressed={selectedId === c.object_3d_id}
                  className={`flex justify-between rounded border px-2 py-1 text-left text-[12px] ${
                    selectedId === c.object_3d_id ? "border-accent bg-head" : "border-line bg-bg-2"}`}>
                  <span>{classLabel(c.class_name)}</span>
                  <span className="flex items-center gap-1.5">
                    <StateBadge state={c.state} />
                    <span className="text-ink-3">{humanize(c.box_source)}</span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}

        {selected && (
          <div className="flex flex-col gap-2 rounded border border-line bg-bg-2 p-2.5 text-[12px]">
            <div className="flex items-center gap-2 text-ink-2">
              <span className="font-medium">Selected box</span>
              <StateBadge state={selected.state} />
            </div>
            <label className="flex items-center justify-between gap-2">
              Class
              <select value={selected.class_id}
                onChange={(e) => { const id = Number(e.target.value); patchLocal(selected.object_3d_id, { class_id: id, class_name: classes.find((x) => x.id === id)?.name }); saveCuboid(selected.object_3d_id, { class_id: id }); }}
                className="field min-w-0 flex-1 py-1">
                {classes.map((c) => <option key={c.id} value={c.id}>{classLabel(c.name)}</option>)}
              </select>
            </label>
            {["Length", "Width", "Height"].map((lab, i) => (
              <label key={lab} className="flex items-center gap-2">
                <span className="w-12">{lab}</span>
                <input type="range" min={0.3} max={14} step={0.1} value={selected.dims[i]}
                  onChange={(e) => setDim(i, Number(e.target.value))}
                  onMouseUp={() => saveCuboid(selected.object_3d_id, { dims: selected.dims })}
                  className="flex-1" />
                <span className="w-12 text-right text-ink-2 tabular-nums">{selected.dims[i].toFixed(1)} m</span>
              </label>
            ))}
            <label className="flex items-center gap-2">
              <span className="w-12">Heading</span>
              <input type="range" min={-3.14159} max={3.14159} step={0.01} value={selected.yaw}
                onChange={(e) => patchLocal(selected.object_3d_id, { yaw: Number(e.target.value) })}
                onMouseUp={() => saveCuboid(selected.object_3d_id, { yaw: selected.yaw })}
                className="flex-1" />
              <span className="w-12 text-right text-ink-2 tabular-nums">{(selected.yaw * 57.3).toFixed(0)}°</span>
            </label>
            <div className="flex gap-2">
              <button type="button" onClick={() => groundSnap(selected.object_3d_id)}
                title="drop the box so its base sits on the ground"
                className="btn flex-1">Ground snap</button>
              <button type="button" onClick={() => removeCuboid(selected.object_3d_id)}
                className="btn btn-danger"><T k="common.delete">Delete</T></button>
            </div>
            <div className="text-ink-3">Drag the box in the bird&apos;s-eye view to move it.</div>
          </div>
        )}

        {err && <div role="alert" className="rounded border border-block/40 bg-block/10 p-2.5 text-[12px] text-block">{err}</div>}
          </div>
        </Inspector>

      <main className="flex flex-1 flex-col">
        <div className="relative flex-1 border-b border-line">
          <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">3D view</div>
          {data && (
            <PointCloudViewer points={data.points} count={data.count} colorBy={colorBy}
              occupancy={occ ? { voxels: occ.voxels, flow: occ.flow, origin: occ.origin,
                                 voxelM: occ.voxel_m } : null}
              flowShare={grids[gridIdx]?.flow_share ?? null}
              intensityRange={[data.intensityMin, data.intensityMax]} source={data.source} mode="perspective"
              cuboids={cuboids} selectedId={selectedId} onSelectCuboid={setSelectedId} />
          )}
          {!data && (
            <div className="flex h-full items-center justify-center">
              {busy ? <span role="status" className="text-[12.5px] text-ink-3">Loading points...</span> : (
                <EmptyState icon="cuboid" title={clouds.length ? "Pick a cloud" : "No cloud loaded"}>
                  {clouds.length
                    ? "Choose one of the clouds on the left to place and refine 3D boxes on it."
                    : "Choose a session on the left. Its point clouds are listed under it."}
                </EmptyState>
              )}
            </div>
          )}
        </div>
        <div className="relative h-2/5 min-h-[220px]">
          <div className="absolute left-3 top-3 z-10 text-[12px] font-medium text-ink-3">Bird&apos;s-eye view <span className="font-normal">(drag a box to move it)</span></div>
          <div className="absolute right-3 top-3 z-10 flex items-center gap-1" role="group" aria-label="colour points by">
            <span className="text-[12px] text-ink-3 mr-1">Colour by</span>
            {(["height", "intensity"] as ColorBy[]).map((k) => (
              <button key={k} type="button" onClick={() => setColorBy(k)} aria-pressed={colorBy === k}
                className={`btn ${colorBy === k ? "btn-on" : ""}`}>{humanize(k)}</button>
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
      )}
    </PageShell>
  );
}
