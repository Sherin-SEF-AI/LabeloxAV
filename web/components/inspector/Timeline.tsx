"use client";

import { useEffect, useRef } from "react";
import { useClock, useClockTick } from "@/lib/inspector/clock";
import type { InspectorEvent, InspectorTopic } from "@/lib/api";

// The session timeline: the full ts_ns range with a scrubber that moves the one shared clock, a per-sensor
// health strip (gaps rendered as colored bands from the index), and typed event markers. The scrubber and
// bands are positioned imperatively off the clock so playback is smooth without re-rendering the tree.

const MARKER_COLOR: Record<string, string> = {
  hard_brake: "#F85149", regen_spike: "#58A6FF", scenario: "#E3B341", quality: "#FF7A2F",
  gold: "#56D364", canary: "#A371F7",
};

function fmt(offsetSec: number): string {
  const s = Math.max(0, offsetSec);
  const mm = Math.floor(s / 60).toString().padStart(2, "0");
  const ss = (s % 60).toFixed(2).padStart(5, "0");
  return `${mm}:${ss}`;
}

export default function Timeline({ topics, gaps, events, onMarkerClick }: {
  topics: InspectorTopic[];
  gaps: Record<string, [number, number][]>;
  events: InspectorEvent[];
  onMarkerClick?: (ts_ns: number) => void;
}) {
  const clock = useClock();
  const { playing } = useClockTick(8);
  const trackRef = useRef<HTMLDivElement>(null);
  const scrubRef = useRef<HTMLDivElement>(null);
  const hudRef = useRef<HTMLSpanElement>(null);
  const dragging = useRef(false);

  const dur = clock.durationSec || 1;
  const startNs = clock.startNs;

  // move the scrubber + HUD imperatively on every clock tick (no React re-render)
  useEffect(() => {
    return clock.subscribe((offsetSec) => {
      const pct = Math.min(1, Math.max(0, offsetSec / dur)) * 100;
      if (scrubRef.current) scrubRef.current.style.left = `${pct}%`;
      if (hudRef.current) hudRef.current.textContent = fmt(offsetSec);
    });
  }, [clock, dur]);

  const seekFromEvent = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    clock.seek(((clientX - r.left) / r.width) * dur);
  };

  useEffect(() => {
    const move = (e: MouseEvent) => { if (dragging.current) seekFromEvent(e.clientX); };
    const up = () => { dragging.current = false; };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    return () => { window.removeEventListener("mousemove", move); window.removeEventListener("mouseup", up); };
  });

  const pctOfNs = (tsNs: number) => (Number(BigInt(Math.round(tsNs)) - startNs) / 1e9 / dur) * 100;

  return (
    <div className="border-t hairline bg-panel px-3 py-2 select-none">
      <div className="flex items-center gap-3 mb-2 font-mono text-[11px]">
        <button onClick={() => clock.step(-1 / 30)} className="border border-line px-1.5 rounded hover:border-accent" title="step back">{"|<"}</button>
        <button onClick={() => clock.toggle()} className="border border-accent/50 bg-accent/10 text-accent px-3 rounded hover:bg-accent/20">{playing ? "pause" : "play"}</button>
        <button onClick={() => clock.step(1 / 30)} className="border border-line px-1.5 rounded hover:border-accent" title="step forward">{">|"}</button>
        <select onChange={(e) => clock.setSpeed(Number(e.target.value))} defaultValue="1" className="bg-bg-2 border border-line rounded px-1 text-ink-2">
          {[0.25, 0.5, 1, 2, 4].map((x) => <option key={x} value={x}>{x}x</option>)}
        </select>
        <span ref={hudRef} className="text-accent tabular-nums">00:00.00</span>
        <span className="text-ink-3">/ {fmt(dur)}</span>
        <span className="ml-auto text-ink-3">ts_ns base {startNs.toString()}</span>
      </div>

      {/* scrub track with event markers */}
      <div ref={trackRef} onMouseDown={(e) => { dragging.current = true; seekFromEvent(e.clientX); }}
        className="relative h-6 bg-bg-2 rounded cursor-pointer">
        {events.map((ev, i) => (
          <div key={i} title={`${ev.label}${ev.detail ? " - " + ev.detail : ""}`}
            onClick={(e) => { e.stopPropagation(); if (onMarkerClick) onMarkerClick(ev.ts_ns); clock.seek(Number(BigInt(Math.round(ev.ts_ns)) - startNs) / 1e9); }}
            style={{ left: `${pctOfNs(ev.ts_ns)}%`, background: MARKER_COLOR[ev.kind] ?? "#6C727A" }}
            className="absolute top-0 bottom-0 w-[2px] hover:w-[3px]" />
        ))}
        <div ref={scrubRef} style={{ left: "0%" }} className="absolute top-0 bottom-0 w-[2px] bg-accent pointer-events-none">
          <div className="absolute -top-1 -left-[3px] w-2 h-2 rounded-full bg-accent" />
        </div>
      </div>

      {/* per-sensor health strip: one row per topic, gaps as bands */}
      <div className="mt-1.5 space-y-0.5">
        {topics.map((t) => (
          <div key={t.name} className="flex items-center gap-2">
            <span className="w-32 shrink-0 font-mono text-[9.5px] text-ink-3 truncate" title={`${t.name} @ ${t.rate}Hz`}>{t.name}</span>
            <div className="relative flex-1 h-1.5 rounded bg-pass/30">
              {(gaps[t.name] ?? []).map((g, i) => {
                const lo = pctOfNs(g[0]);
                const w = Math.max(0.3, pctOfNs(g[1]) - lo);
                return <div key={i} style={{ left: `${lo}%`, width: `${w}%` }} className="absolute top-0 bottom-0 bg-block rounded" title="gap / dropout" />;
              })}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
