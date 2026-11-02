"use client";

import type { ColorBy } from "@/components/lidar/PointCloudViewer";
import type { Cuboid3D } from "@/lib/api";
import type { OntologyClass } from "@/lib/types";

// The editor's right panel in 3D/LiDAR mode: the cuboid list and the selected cuboid's class, size and
// orientation. Presentational: the frame editor owns the cuboids and passes the actions in.

export default function CuboidPanel({
  cuboids, selectedId, sel, hasCloud, message, classes, allClasses, colorBy,
  onAdd, onAiLift, onSelect, onPatch, onSave, onDelete, onColorBy,
}: {
  cuboids: Cuboid3D[];
  selectedId: string | null;
  sel: Cuboid3D | null;
  hasCloud: boolean;
  message: string | null;
  classes: OntologyClass[];      // what the project labels: the class picker offers these
  allClasses: OntologyClass[];   // every class, to name one outside the project's set
  colorBy: ColorBy;
  onAdd: () => void;
  onAiLift: () => void;
  onSelect: (cuboidId: string) => void;
  onPatch: (cuboidId: string, patch: Partial<Cuboid3D>) => void;
  onSave: (cuboidId: string, fields: Partial<Cuboid3D>) => void;
  onDelete: (cuboidId: string) => void;
  onColorBy: (by: ColorBy) => void;
}) {
  return (
    <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2 text-[12px]">
      <div className="flex items-center justify-between">
        <span className="text-ink-3 uppercase text-[10px]">cuboids ({cuboids.length})</span>
        <button onClick={onAdd} disabled={!hasCloud} className="border border-line text-ink-3 px-1.5 py-0.5 hover:border-accent disabled:opacity-40">+ box</button>
      </div>
      <button onClick={onAiLift} disabled={!hasCloud} className="w-full border border-line text-ink-2 px-2 py-1 hover:border-accent disabled:opacity-40">AI lift 2D to 3D</button>
      {cuboids.map((c) => (
        <div key={c.object_3d_id} onClick={() => onSelect(c.object_3d_id)}
          className={`flex items-center justify-between gap-1.5 cursor-pointer ${c.object_3d_id === selectedId ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
          <span className="truncate flex-1">{c.class_name}</span>
          <span className="text-ink-3">{c.state} {c.box_source}</span>
        </div>
      ))}
      {!cuboids.length && <div className="text-ink-3 py-4 text-center">{message ?? "no cuboids. lift or + box."}</div>}
      {sel && (
        <div className="border-t hairline pt-2 space-y-2">
          <div className="text-ink-3 uppercase text-[10px]">selected ({sel.source})</div>
          <select value={sel.class_id}
            onChange={(e) => { const cid = Number(e.target.value); onPatch(sel.object_3d_id, { class_id: cid, class_name: allClasses.find((x) => x.id === cid)?.name }); onSave(sel.object_3d_id, { class_id: cid }); }}
            className="w-full bg-bg border border-line px-1 py-0.5 text-ink">
            {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
          {["L", "W", "H"].map((lab, i) => (
            <label key={lab} className="flex items-center gap-2">
              <span className="w-3 text-ink-3">{lab}</span>
              <input type="range" min={0.3} max={14} step={0.1} value={sel.dims[i]}
                onChange={(e) => onPatch(sel.object_3d_id, { dims: sel.dims.map((d, j) => (j === i ? Number(e.target.value) : d)) })}
                onMouseUp={() => onSave(sel.object_3d_id, { dims: sel.dims })} className="flex-1" />
              <span className="w-9 text-right text-ink-3">{sel.dims[i].toFixed(1)}</span>
            </label>
          ))}
          <label className="flex items-center gap-2">
            <span className="w-3 text-ink-3">yaw</span>
            <input type="range" min={-3.14159} max={3.14159} step={0.01} value={sel.yaw}
              onChange={(e) => onPatch(sel.object_3d_id, { yaw: Number(e.target.value) })}
              onMouseUp={() => onSave(sel.object_3d_id, { yaw: sel.yaw })} className="flex-1" />
            <span className="w-9 text-right text-ink-3">{(sel.yaw * 57.3).toFixed(0)}</span>
          </label>
          {/* 9-DOF: pitch (up/down a ramp) and roll (side bank). Auto-estimated from the ground on
              lift; editable here for slopes the estimator clamped or missed. */}
          {([["pitch", sel.pitch ?? 0], ["roll", sel.roll ?? 0]] as const).map(([axis, val]) => (
            <label key={axis} className="flex items-center gap-2">
              <span className="w-3 text-ink-3" title={axis === "pitch" ? "tilt up/down a ramp" : "side bank"}>{axis === "pitch" ? "pit" : "rol"}</span>
              <input type="range" min={-0.6} max={0.6} step={0.01} value={val}
                onChange={(e) => onPatch(sel.object_3d_id, { [axis]: Number(e.target.value) })}
                onMouseUp={() => onSave(sel.object_3d_id, { [axis]: axis === "pitch" ? sel.pitch : sel.roll })} className="flex-1" />
              <span className="w-9 text-right text-ink-3">{(val * 57.3).toFixed(0)}</span>
            </label>
          ))}
          <div className="flex items-center gap-1 pt-0.5">
            {(["height", "intensity", "source", "segment"] as ColorBy[]).map((kb) => (
              <button key={kb} onClick={() => onColorBy(kb)} className={`px-1.5 py-0.5 border ${colorBy === kb ? "border-accent text-accent" : "border-line text-ink-3"}`}>{kb}</button>
            ))}
          </div>
          <div className="flex gap-2">
            <button onClick={() => onSave(sel.object_3d_id, { attrs: { ground_snap: true } })} className="flex-1 border border-line text-ink-2 px-2 py-1 hover:border-accent">ground snap</button>
            <button onClick={() => onDelete(sel.object_3d_id)} className="border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block">delete</button>
          </div>
          <div className="text-ink-3">drag the box in the BEV view to move it.</div>
        </div>
      )}
    </div>
  );
}
