"use client";

import type { InspectorTopic } from "@/lib/api";

// The topic browser, from session_index: every topic with its schema, rate, and count. Clicking a topic adds
// the panel best suited to it (image for a CompressedImage topic, a plot for numeric sensors, raw otherwise).

function panelTypeFor(t: InspectorTopic): string {
  const s = t.schema.toLowerCase();
  const n = t.name.toLowerCase();
  if (s.includes("compressedimage") || s.includes("image") || n.includes("camera") || n.includes("cam")) return "image";
  if (s.includes("locationfix") || n.includes("gnss") || n.includes("gps")) return "map";
  if (n.includes("imu")) return "imu_plot";
  if (n.includes("can")) return "can_plot";
  return "raw";
}

export default function TopicBrowser({ topics, onAdd }: {
  topics: InspectorTopic[];
  onAdd: (type: string, topic: string) => void;
}) {
  return (
    <div className="h-full overflow-auto no-scrollbar">
      <div className="font-mono text-[10px] uppercase tracking-wider text-ink-3 px-2 py-1.5 sticky top-0 bg-panel">topics ({topics.length})</div>
      {topics.map((t) => (
        <button key={t.name} onClick={() => onAdd(panelTypeFor(t), t.name)}
          title={`add a ${panelTypeFor(t)} panel for ${t.name}`}
          className="w-full text-left px-2 py-1 hover:bg-bg-2 border-b hairline/50">
          <div className="font-mono text-[11px] text-ink-2 truncate">{t.name}</div>
          <div className="font-mono text-[9px] text-ink-3 flex gap-2">
            <span className="truncate">{t.schema || "unknown"}</span>
            <span className="ml-auto shrink-0">{t.rate}Hz · {t.count}</span>
          </div>
        </button>
      ))}
    </div>
  );
}
