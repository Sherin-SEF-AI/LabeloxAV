"use client";

import type { InspectorPanel } from "@/lib/api";
import RawPanel from "@/components/inspector/RawPanel";
import PlotPanel from "@/components/inspector/PlotPanel";
import ImagePanel from "@/components/inspector/ImagePanel";
import MapPanel from "@/components/inspector/MapPanel";
import CanPanel from "@/components/inspector/CanPanel";
import AudioPanel from "@/components/inspector/AudioPanel";

// Renders one panel by type inside a shared chrome. Every panel subscribes to the one clock and reads via
// the shared MCAP reader. Adding a panel type is one case here plus its component.

const TITLE: Record<string, string> = {
  image: "camera", imu_plot: "time-series", can_plot: "CAN signals", map: "GPS map",
  raw: "raw message", can_table: "CAN table", audio: "audio",
};

export default function PanelHost({ panel, onRemove, onFrame }: {
  panel: InspectorPanel;
  onRemove: () => void;
  onFrame?: (frameId: string | null, tsNs: string) => void;
}) {
  const title = `${TITLE[panel.type] ?? panel.type}${panel.topic ? " · " + panel.topic : ""}`;
  const known = ["raw", "imu_plot", "can_plot", "image", "map", "can_table", "audio"];
  return (
    <div className="flex flex-col h-full min-h-0 border hairline rounded bg-panel overflow-hidden">
      <div className="flex items-center gap-2 px-2 py-1 border-b hairline">
        <span className="font-mono text-[10px] uppercase tracking-wider text-ink-3 truncate">{title}</span>
        <button onClick={onRemove} className="ml-auto font-mono text-[10px] text-ink-3 hover:text-block px-1" title="remove panel">x</button>
      </div>
      <div className="flex-1 min-h-0">
        {panel.type === "raw" && panel.topic && <RawPanel topic={panel.topic} />}
        {(panel.type === "imu_plot" || panel.type === "can_plot") && <PlotPanel panel={panel} />}
        {panel.type === "image" && <ImagePanel panel={panel} onFrame={onFrame} />}
        {panel.type === "map" && <MapPanel />}
        {panel.type === "can_table" && <CanPanel />}
        {panel.type === "audio" && panel.topic && <AudioPanel topic={panel.topic} />}
        {!known.includes(panel.type) && <div className="p-3 font-mono text-[11px] text-ink-3">unknown panel type: {panel.type}</div>}
      </div>
    </div>
  );
}
