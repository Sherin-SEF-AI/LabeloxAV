"use client";

import { useEffect, useState } from "react";

import type { ColorBy } from "@/components/lidar/PointCloudViewer";
import { api, humanizeError, lidarCloudPoints, type Cuboid3D, type LidarCloud, type LidarPoints } from "@/lib/api";
import type { FrameMeta, Ontology } from "@/lib/types";

// 3D and LiDAR mode for one frame in the editor: the session's point cloud nearest the frame, its points,
// the 3D cuboids on it, and every edit on them. Moved out of app/frame/[id]/page.tsx unchanged; the page
// renders the viewer and the cuboid panel from what this returns.

type Confirm = (opts: { title: string; body?: string; danger?: boolean; confirmLabel?: string }) => Promise<boolean>;

const CUBOID_DIMS: Record<string, number[]> = {
  sedan: [4.2, 1.8, 1.5], suv: [4.6, 1.9, 1.7], truck: [7.0, 2.5, 3.0], bus: [11.0, 2.6, 3.2],
  motorcycle: [2.0, 0.8, 1.4], pedestrian: [0.6, 0.6, 1.7], autorickshaw: [2.6, 1.4, 1.8],
};

export function useCuboidLayer({ mode, meta, onto, confirm }: {
  mode: string;
  meta: FrameMeta | null;
  onto: Ontology | null;
  confirm: Confirm;
}) {
  const [cloud3d, setCloud3d] = useState<LidarCloud | null>(null);
  const [pts3d, setPts3d] = useState<LidarPoints | null>(null);
  const [cub3d, setCub3d] = useState<Cuboid3D[]>([]);
  const [cubSel, setCubSel] = useState<string | null>(null);
  const [colorBy3d, setColorBy3d] = useState<ColorBy>("height");
  const [lidarMsg, setLidarMsg] = useState<string | null>(null);

  // 3D mode: load the session cloud nearest this frame's timestamp, then its points and 3D cuboids.
  useEffect(() => {
    if (mode !== "lidar3d" || !meta || cloud3d) return;
    let cancelled = false;
    (async () => {
      setLidarMsg("loading point cloud...");
      try {
        const r = await api.lidarClouds(meta.session_id);
        if (!r.clouds.length) { if (!cancelled) setLidarMsg("no point cloud for this session"); return; }
        const near = r.clouds.reduce((a, b) => (Math.abs(b.ts_ns - meta.ts_ns) < Math.abs(a.ts_ns - meta.ts_ns) ? b : a));
        const [pts, objs] = await Promise.all([lidarCloudPoints(near.cloud_id, { variant: "raw", max: 300000 }), api.lidarObjects3d(near.cloud_id)]);
        if (cancelled) return;
        setCloud3d(near); setPts3d(pts); setCub3d(objs.objects); setLidarMsg(null);
      } catch (e) { if (!cancelled) setLidarMsg("cloud load failed: " + humanizeError(e)); }
    })();
    return () => { cancelled = true; };
  }, [mode, meta, cloud3d]);

  const cubSelected = cub3d.find((c) => c.object_3d_id === cubSel) || null;
  const patchCub = (cid: string, patch: Partial<Cuboid3D>) => setCub3d((cs) => cs.map((c) => (c.object_3d_id === cid ? { ...c, ...patch } : c)));
  const saveCub = async (cid: string, fields: Partial<Cuboid3D>) => {
    const cur = cub3d.find((c) => c.object_3d_id === cid); if (!cur) return;
    try {
      const saved = await api.lidarPatchCuboid(cid, {
        class_id: (fields.class_id as number) ?? cur.class_id, center: (fields.center as number[]) ?? cur.center,
        dims: (fields.dims as number[]) ?? cur.dims, yaw: (fields.yaw as number) ?? cur.yaw,
        pitch: (fields.pitch as number) ?? cur.pitch, roll: (fields.roll as number) ?? cur.roll,
        ground_snap: Boolean(fields.attrs && (fields.attrs as Record<string, unknown>).ground_snap), expected_version: cur.version,
      });
      patchCub(cid, saved);
    } catch (e) { setLidarMsg("save failed: " + humanizeError(e)); }
  };
  const moveCub = (cid: string, x: number, y: number, commit: boolean) => {
    const cur = cub3d.find((c) => c.object_3d_id === cid); if (!cur) return;
    const center = [x, y, cur.center[2]]; patchCub(cid, { center }); if (commit) saveCub(cid, { center });
  };
  const addCub = async () => {
    if (!cloud3d || !onto) return;
    const cls = onto.classes.find((c) => c.name === "sedan") || onto.classes[0]; if (!cls) return;
    try {
      const created = await api.lidarCreateCuboid(cloud3d.cloud_id, { class_id: cls.id, center: [12, 0, 1], dims: CUBOID_DIMS[cls.name] || [4, 1.8, 1.5], yaw: 0, ground_snap: true });
      setCub3d((cs) => [...cs, created]); setCubSel(created.object_3d_id);
    } catch (e) { setLidarMsg("add failed: " + humanizeError(e)); }
  };
  const delCub = async (cid: string) => {
    // A 3D cuboid is not covered by the frame's checkpoints, which snapshot 2D objects, so this deletion is
    // genuinely irreversible and asks first.
    if (!(await confirm({ title: "Delete this 3D box?", danger: true, confirmLabel: "Delete" }))) return;
    try { await api.lidarDeleteCuboid(cid); setCub3d((cs) => cs.filter((c) => c.object_3d_id !== cid)); setCubSel(null); }
    catch (e) { setLidarMsg("delete failed: " + humanizeError(e)); }
  };
  const aiLift3d = async () => {
    if (!cloud3d) return;
    setLidarMsg("lifting 2D objects to 3D...");
    try { const r = await api.lidarLiftCloud(cloud3d.cloud_id); const objs = await api.lidarObjects3d(cloud3d.cloud_id); setCub3d(objs.objects); setLidarMsg(r.cuboids ? `lifted ${r.cuboids} cuboids` : "no 2D objects to lift"); }
    catch (e) { setLidarMsg("lift failed: " + humanizeError(e)); }
  };

  return {
    cloud3d, pts3d, cub3d, cubSel, setCubSel, colorBy3d, setColorBy3d, lidarMsg,
    cubSelected, patchCub, saveCub, moveCub, addCub, delCub, aiLift3d,
  };
}
