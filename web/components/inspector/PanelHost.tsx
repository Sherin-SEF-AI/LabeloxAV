"use client";

import dynamic from "next/dynamic";

import { T } from "@/lib/i18n";

import type { InspectorPanel } from "@/lib/api";
import RawPanel from "@/components/inspector/RawPanel";
import ImagePanel from "@/components/inspector/ImagePanel";
import CanPanel from "@/components/inspector/CanPanel";
import AudioPanel from "@/components/inspector/AudioPanel";

// The three panels that each bring a rendering library with them: maplibre for the map, three for the 3D
// view, uplot for the plots. Imported statically, all three shipped with /inspect/[session] whether or not
// the layout on screen had a map or a 3D view in it, which for the default camera-and-plot layout meant
// downloading and parsing two WebGL engines to draw neither. Loaded on demand, a layout pays for the
// panels it shows. ssr: false because all three need a canvas and none has anything to render without one.
function PanelLoading() {
  return <div className="p-3 text-[12px] text-ink-3" role="status"><T k="load.panel">loading panel</T></div>;
}
const PlotPanel = dynamic(() => import("@/components/inspector/PlotPanel"), { ssr: false, loading: PanelLoading });
const MapPanel = dynamic(() => import("@/components/inspector/MapPanel"), { ssr: false, loading: PanelLoading });
const Scene3DPanel = dynamic(() => import("@/components/inspector/Scene3DPanel"), { ssr: false, loading: PanelLoading });

// Renders one panel by type inside a shared chrome. Every panel subscribes to the one clock and reads via
// the shared MCAP reader. Adding a panel type is one case here plus its component.

const TITLE: Record<string, string> = {
  image: "Camera", imu_plot: "Time series", can_plot: "CAN signals", map: "GPS map",
  raw: "Raw message", can_table: "CAN table", audio: "Audio", scene3d: "3D point cloud",
};

export default function PanelHost({ panel, onRemove, onFrame, onCycleSpan, onGrow, onShrink,
                                   onDragStart, onDragOver, onDrop, dragging }: {
  panel: InspectorPanel;
  onRemove: () => void;
  onFrame?: (frameId: string | null, tsNs: string) => void;
  onCycleSpan?: () => void;
  onGrow?: () => void;
  onShrink?: () => void;
  onDragStart?: () => void;
  onDragOver?: (e: React.DragEvent) => void;
  onDrop?: () => void;
  dragging?: boolean;
}) {
  const title = `${TITLE[panel.type] ?? panel.type}${panel.topic ? " · " + panel.topic : ""}`;
  const known = ["raw", "imu_plot", "can_plot", "image", "map", "can_table", "audio", "scene3d"];
  return (
    <div
      onDragOver={onDragOver}
      onDrop={(e) => { e.preventDefault(); onDrop?.(); }}
      className={`flex flex-col h-full min-h-0 border rounded bg-panel overflow-hidden ${dragging ? "border-accent opacity-60" : "hairline"}`}>
      {/* The header is the drag handle, not the whole panel: a 3D view and a plot both want the pointer for
          their own gestures, and making the body draggable would take orbit and zoom away from them. */}
      <div draggable onDragStart={onDragStart}
        className="flex items-center gap-2 px-2 py-1 border-b hairline cursor-grab active:cursor-grabbing">
        <span className="text-[12px] font-medium text-ink-2 truncate">{title}</span>
        <span className="ml-auto flex items-center gap-0.5">
          <button onClick={onShrink} className="text-[13px] leading-none text-ink-3 hover:text-ink px-1.5 py-0.5 rounded hover:bg-head" data-tip="Shorter" aria-label={`make the ${title} panel shorter`}>&minus;</button>
          <button onClick={onGrow} className="text-[13px] leading-none text-ink-3 hover:text-ink px-1.5 py-0.5 rounded hover:bg-head" data-tip="Taller" aria-label={`make the ${title} panel taller`}>+</button>
          <button onClick={onCycleSpan} className="text-[13px] leading-none text-ink-3 hover:text-ink px-1.5 py-0.5 rounded hover:bg-head" data-tip="Toggle full width" aria-label={`toggle full width for the ${title} panel`}>&harr;</button>
          <button onClick={onRemove} className="text-[13px] leading-none text-ink-3 hover:text-block px-1.5 py-0.5 rounded hover:bg-head" data-tip="Remove panel" aria-label={`remove the ${title} panel`}>&times;</button>
        </span>
      </div>
      <div className="flex-1 min-h-0">
        {panel.type === "raw" && panel.topic && <RawPanel topic={panel.topic} />}
        {(panel.type === "imu_plot" || panel.type === "can_plot") && <PlotPanel panel={panel} />}
        {panel.type === "image" && <ImagePanel panel={panel} onFrame={onFrame} />}
        {panel.type === "map" && <MapPanel />}
        {panel.type === "can_table" && <CanPanel />}
        {panel.type === "audio" && panel.topic && <AudioPanel topic={panel.topic} />}
        {panel.type === "scene3d" && <Scene3DPanel />}
        {!known.includes(panel.type) && <div className="p-3 text-[12px] text-ink-3">Unknown panel type: {panel.type}</div>}
      </div>
    </div>
  );
}
