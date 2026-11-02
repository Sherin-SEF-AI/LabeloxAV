"use client";

import { useState } from "react";

import { api, humanizeError } from "@/lib/api";
import { trackOp } from "@/lib/canvasOps";
import type { FrameMeta, LaneRow } from "@/lib/types";

// Lanes and the drivable surface for one frame in the editor: their state, and every edit that does not
// need the canvas (saving, adding, typing, deleting, propagating lanes; saving and editing surface
// regions). The page keeps what does need the canvas, turning a pointer into image coordinates, and
// passes `reload`, which refetches all of the frame's overlay layers after an edit lands.
//
// Moved out of app/frame/[id]/page.tsx unchanged: the handlers are the same code with the page's
// loadLayers() now this hook's reload().

type Confirm = (opts: { title: string; body?: string; danger?: boolean; confirmLabel?: string }) => Promise<boolean>;

export function useLaneLayer({ frameId: id, meta, flash, confirm, reload }: {
  frameId: string;
  meta: FrameMeta | null;
  flash: (message: string) => void;
  confirm: Confirm;
  reload: () => Promise<void>;
}) {
  const [lanes, setLanes] = useState<(LaneRow & { dirty?: boolean })[]>([]);
  const [drivable, setDrivable] = useState<Record<string, number[][]> | null>(null);
  // provenance of the machine-produced overlays (drivable/seg), so the layers panel can show who made each
  const [drivableMeta, setDrivableMeta] = useState<{ source: string; model?: string | null } | null>(null);
  // Lanes-mode editing (canvas swap): selected lane, the in-progress add path
  const [laneSel, setLaneSel] = useState<string | null>(null);
  const [laneAdding, setLaneAdding] = useState<number[][] | null>(null);
  // Drivable region editing. Separate buffers from the lane ones: a lane is an open polyline and a surface
  // region is a closed area, and sharing a buffer would render one of them wrongly mid-draw.
  const [areaAdding, setAreaAdding] = useState<number[][] | null>(null);
  const [areaClass, setAreaClass] = useState<string>("drivable");
  const [areaSel, setAreaSel] = useState<string | null>(null);

  const laneDragPoint = (laneId: string, i: number, x: number, y: number) =>
    setLanes((ls) => ls.map((l) => l.lane_id === laneId
      ? { ...l, dirty: true, control_points: l.control_points.map((p, j) => (j === i ? [x, y] : p)) } : l));
  const saveLanes = async () => {
    const dirty = lanes.filter((x) => x.dirty);
    if (!dirty.length) { flash("no lane edits"); return; }
    for (const l of dirty) await api.updateLane(l.lane_id, { control_points: l.control_points, lane_type: l.lane_type, is_ego: l.is_ego });
    await reload(); flash(`saved ${dirty.length} lane${dirty.length === 1 ? "" : "s"}`);
  };
  const finishAddLane = async (type: string) => {
    if (!laneAdding || laneAdding.length < 2) { setLaneAdding(null); return; }
    await api.createLane(id, { control_points: laneAdding, lane_type: type, is_ego: false });
    setLaneAdding(null); await reload(); flash("lane added");
  };
  const setLaneType = (t: string) => { if (laneSel) setLanes((ls) => ls.map((l) => l.lane_id === laneSel ? { ...l, lane_type: t, dirty: true } : l)); };
  const toggleLaneEgo = () => { if (laneSel) setLanes((ls) => ls.map((l) => ({ ...l, is_ego: l.lane_id === laneSel ? !l.is_ego : l.is_ego, dirty: l.lane_id === laneSel ? true : l.dirty }))); };
  const delLane = async () => {
    if (!laneSel) return;
    if (!(await confirm({ title: "Delete this lane?", danger: true, confirmLabel: "Delete" }))) return;
    await api.deleteLane(laneSel); setLaneSel(null); await reload(); flash("lane deleted");
  };
  const propagateLanes = async () => {
    const r = await trackOp("propagate", "propagate lanes", () => api.propagateLanes(id, 8),
      (res) => `${res.created} frames`);
    flash(`propagated to ${r.created} lane-frames`);
  };

  // ---- drivable surface editing --------------------------------------------------------------------
  // The mask is stored as flat [x,y,x,y,...] rings per class, which is what the API takes and returns, so
  // the editor speaks that shape rather than converting back and forth and risking a mismatch on save.
  const saveDrivable = async (next: Record<string, number[][]>) => {
    if (!meta) return;
    try {
      const r = await api.refineDrivable(id, next, meta.width, meta.height);
      setDrivable(next);
      setDrivableMeta({ source: "human", model: "human" });
      const pct = Math.round((r.coverage.drivable ?? 0) * 100);
      flash(`surface saved, ${pct}% drivable`);
    } catch (e) { flash("surface save failed: " + humanizeError(e)); }
  };

  const finishArea = async () => {
    // Three points is the minimum that encloses anything. Fewer is a stray click, and closing it would
    // write a degenerate sliver into the coverage statistics.
    if (!areaAdding || areaAdding.length < 3) { setAreaAdding(null); return; }
    const flat = areaAdding.flat();
    const next = { ...(drivable ?? {}) };
    next[areaClass] = [...(next[areaClass] ?? []), flat];
    setAreaAdding(null);
    await saveDrivable(next);
  };

  const dragAreaPoint = (key: string, i: number, x: number, y: number) => {
    const [cls, idx] = key.split(":");
    setDrivable((d) => {
      if (!d?.[cls]?.[Number(idx)]) return d;
      const polys = d[cls].map((p2, j) => {
        if (j !== Number(idx)) return p2;
        const copy = [...p2];
        copy[i * 2] = x; copy[i * 2 + 1] = y;
        return copy;
      });
      return { ...d, [cls]: polys };
    });
  };

  const deleteArea = async () => {
    if (!areaSel || !drivable) return;
    const [cls, idx] = areaSel.split(":");
    if (!(await confirm({ title: `Delete this ${cls.replace("_", " ")} region?`, danger: true,
                          confirmLabel: "Delete" }))) return;
    const next = { ...drivable, [cls]: drivable[cls].filter((_p, j) => j !== Number(idx)) };
    setAreaSel(null);
    await saveDrivable(next);
  };

  return {
    lanes, setLanes, drivable, setDrivable, drivableMeta, setDrivableMeta,
    laneSel, setLaneSel, laneAdding, setLaneAdding, areaAdding, setAreaAdding, areaClass, setAreaClass,
    areaSel, setAreaSel,
    laneDragPoint, saveLanes, finishAddLane, setLaneType, toggleLaneEgo, delLane, propagateLanes,
    saveDrivable, finishArea, dragAreaPoint, deleteArea,
  };
}
