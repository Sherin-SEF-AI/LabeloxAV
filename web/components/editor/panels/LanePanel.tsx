"use client";

import type { LaneRow } from "@/lib/types";

// The editor's right panel in lanes mode: the lane list, the selected lane's type and ego flag, and the
// drivable-surface regions. Presentational: the frame editor owns the state and passes the actions in.

const LANE_TYPES = ["solid", "dashed", "double", "road_edge", "implicit", "fallback"];
const LANE_COLOR: Record<string, string> = { proposed: "#58A6FF", human: "#FF7A2F", propagated: "#E3B341" };

// The three surface classes the ternary drivable mask carries. Fallback is the unpaved shoulder India
// actually drives on, which is why it is a first-class surface rather than a kind of non-drivable.
const SURFACE_CLASSES = [
  { key: "drivable", label: "drivable", tone: "border-pass text-pass" },
  { key: "fallback", label: "fallback", tone: "border-warn text-warn" },
  { key: "non_drivable", label: "non-drv", tone: "border-block text-block" },
];

export default function LanePanel({
  lanes, laneSel, drivable, areaAdding, areaSel,
  onSelectLane, onSetLaneType, onToggleEgo, onDeleteLane,
  onStartArea, onFinishArea, onCancelArea, onRerunModel, onSaveDrivable, onDeleteArea,
}: {
  lanes: (LaneRow & { dirty?: boolean })[];
  laneSel: string | null;
  drivable: Record<string, number[][]> | null;
  areaAdding: number[][] | null;
  areaSel: string | null;
  onSelectLane: (laneId: string) => void;
  onSetLaneType: (laneType: string) => void;
  onToggleEgo: () => void;
  onDeleteLane: () => void;
  onStartArea: (surface: string) => void;
  onFinishArea: () => void;
  onCancelArea: () => void;
  onRerunModel: () => void;
  onSaveDrivable: () => void;
  onDeleteArea: () => void;
}) {
  return (
    <div className="flex-1 min-h-0 overflow-y-auto p-2 space-y-2 text-[12px]">
      <div className="text-ink-3 uppercase text-[10px]">lanes ({lanes.length})</div>
      {lanes.map((l) => (
        <div key={l.lane_id} onClick={() => onSelectLane(l.lane_id)}
          className={`flex items-center gap-1.5 cursor-pointer ${l.lane_id === laneSel ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}>
          <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: l.is_ego ? "#56D364" : (LANE_COLOR[l.source] || "#A0A6AD") }} />
          <span className="truncate flex-1">{l.lane_type}{l.is_ego ? " (ego)" : ""}</span>
          <span className="text-ink-3">{l.source[0]}</span>
        </div>
      ))}
      {!lanes.length && <div className="text-ink-3 py-4 text-center">no lanes. propose, or + lane to draw.</div>}
      {(() => {
        const sl = lanes.find((l) => l.lane_id === laneSel);
        if (!sl) return null;
        return (
          <div className="border-t hairline pt-2 space-y-2">
            <div className="text-ink-3 uppercase text-[10px]">selected lane</div>
            <select value={sl.lane_type} onChange={(e) => onSetLaneType(e.target.value)} className="w-full bg-bg border border-line px-1 py-0.5 text-ink">
              {LANE_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <button onClick={onToggleEgo} className={`w-full border px-2 py-1 ${sl.is_ego ? "border-pass text-pass" : "border-line text-ink-3"}`}>{sl.is_ego ? "ego lane set" : "mark ego"}</button>
            <button onClick={onDeleteLane} className="w-full border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block">delete lane</button>
          </div>
        );
      })()}

      {/* Drivable surface. The PUT behind this has existed since M2.2 and nothing could reach it,
          so `source` was `proposed` on 2,478 of 2,479 masks: the layer was machine output a person
          could look at and not correct. */}
      <div className="border-t hairline pt-2 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-ink-3 uppercase text-[10px]">drivable surface</span>
          <span className="text-ink-3">
            {drivable
              ? `${Object.values(drivable).reduce((n, ps) => n + ps.length, 0)} regions`
              : "none"}
          </span>
        </div>

        {areaAdding ? (
          <>
            <div className="text-ink-2">
              click to add points ({areaAdding.length}), then close the region
            </div>
            <div className="grid grid-cols-2 gap-1">
              <button onClick={onFinishArea} disabled={areaAdding.length < 3}
                className="border border-pass text-pass px-2 py-1 disabled:opacity-40">close region</button>
              <button onClick={onCancelArea}
                className="border border-line text-ink-3 px-2 py-1 hover:border-block">cancel</button>
            </div>
          </>
        ) : (
          <>
            <div className="grid grid-cols-3 gap-1">
              {SURFACE_CLASSES.map((c) => (
                <button key={c.key}
                  onClick={() => onStartArea(c.key)}
                  title={`draw a ${c.label} region`}
                  className={`border px-1 py-1 text-[10px] ${c.tone}`}>
                  + {c.label}
                </button>
              ))}
            </div>
            <button onClick={onRerunModel}
              className="w-full border border-line text-ink-3 px-2 py-1 hover:border-accent">
              re-run the model
            </button>
          </>
        )}

        {areaSel && (
          <div className="border-t hairline pt-2 space-y-1">
            <div className="text-ink-3 uppercase text-[10px]">
              selected {areaSel.split(":")[0].replace("_", " ")} region
            </div>
            <div className="text-ink-3 text-[10px]">
              drag a handle to pull the boundary onto the kerb, then save
            </div>
            <div className="grid grid-cols-2 gap-1">
              <button onClick={onSaveDrivable}
                className="border border-pass text-pass px-2 py-1">save edits</button>
              <button onClick={onDeleteArea}
                className="border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block">delete</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
