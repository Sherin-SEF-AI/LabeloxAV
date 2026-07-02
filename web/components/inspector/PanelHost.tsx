"use client";

import type { InspectorPanel } from "@/lib/api";
import RawPanel from "@/components/inspector/RawPanel";

// Renders one panel by type inside a shared chrome. Every panel subscribes to the one clock and reads via
// the shared MCAP reader. The raw panel lands with the shell (M-I.3); the plot, image, map, CAN, and audio
// panels are added in M-I.4, each as one case here plus its component.

const TITLE: Record<string, string> = {
  image: "camera", imu_plot: "time-series", can_plot: "CAN signals", map: "GPS map",
  raw: "raw message", can_table: "CAN table", audio: "audio",
};

export default function PanelHost({ panel, onRemove }: { panel: InspectorPanel; onRemove: () => void }) {
  const title = `${TITLE[panel.type] ?? panel.type}${panel.topic ? " · " + panel.topic : ""}`;
  return (
    <div className="flex flex-col h-full min-h-0 border hairline rounded bg-panel overflow-hidden">
      <div className="flex items-center gap-2 px-2 py-1 border-b hairline">
        <span className="font-mono text-[10px] uppercase tracking-wider text-ink-3 truncate">{title}</span>
        <button onClick={onRemove} className="ml-auto font-mono text-[10px] text-ink-3 hover:text-block px-1" title="remove panel">x</button>
      </div>
      <div className="flex-1 min-h-0">
        {panel.type === "raw" && panel.topic ? (
          <RawPanel topic={panel.topic} />
        ) : (
          <div className="p-3 font-mono text-[11px] text-ink-3">{title} panel renders in M-I.4</div>
        )}
      </div>
    </div>
  );
}
