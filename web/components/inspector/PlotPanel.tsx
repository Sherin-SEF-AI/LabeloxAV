"use client";

import { useEffect, useRef, useState } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { InspectorPanel } from "@/lib/api";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// Time-series plot (uPlot, built for high-frequency data): any numeric topic or field. IMU accelerometer and
// gyro; decoded CAN signals. Loads the whole topic once, renders smoothly at 200Hz, and pins a vertical
// cursor at the shared clock time (moved imperatively on each tick, no chart rebuild).

const COLORS = ["#FF7A2F", "#58A6FF", "#56D364", "#E3B341", "#A371F7", "#F85149"];

export default function PlotPanel({ panel }: { panel: InspectorPanel }) {
  const clock = useClock();
  const { mcap } = useMcap();
  const hostRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const uRef = useRef<uPlot | null>(null);
  const [status, setStatus] = useState("loading series...");

  useEffect(() => {
    if (!panel.topic) return;
    let alive = true;
    let ro: ResizeObserver | null = null;
    (async () => {
      try {
        const { t, values } = await mcap.collect(panel.topic!, clock.startNs);
        if (!alive || !hostRef.current) return;
        let fields = Object.keys(values).filter((k) => new Set(values[k]).size > 1); // drop constant fields
        if (fields.length === 0) fields = Object.keys(values);
        fields = fields.slice(0, 6);
        if (t.length === 0 || fields.length === 0) { setStatus("no numeric fields on this topic"); return; }
        setStatus("");
        const data: uPlot.AlignedData = [t, ...fields.map((f) => values[f])];
        const rect = hostRef.current.getBoundingClientRect();
        const opts: uPlot.Options = {
          width: Math.max(120, rect.width), height: Math.max(80, rect.height),
          padding: [8, 8, 0, 0], cursor: { show: false }, legend: { show: true },
          scales: { x: { time: false } },
          axes: [
            { stroke: "#6C727A", grid: { stroke: "#20242A" }, ticks: { stroke: "#20242A" }, font: "10px monospace" },
            { stroke: "#6C727A", grid: { stroke: "#20242A" }, ticks: { stroke: "#20242A" }, font: "10px monospace", size: 42 },
          ],
          series: [{ label: "t(s)" }, ...fields.map((f, i) => ({ label: f, stroke: COLORS[i % COLORS.length], width: 1 }))],
        };
        uRef.current?.destroy();
        uRef.current = new uPlot(opts, data, hostRef.current);
        ro = new ResizeObserver(() => {
          const r = hostRef.current?.getBoundingClientRect();
          if (r && uRef.current) uRef.current.setSize({ width: Math.max(120, r.width), height: Math.max(80, r.height) });
        });
        ro.observe(hostRef.current);
      } catch (e) {
        if (alive) setStatus("plot load error: " + String(e));
      }
    })();
    return () => { alive = false; ro?.disconnect(); uRef.current?.destroy(); uRef.current = null; };
  }, [mcap, panel.topic, clock.startNs]);

  // pin the cursor at the clock position
  useEffect(() => {
    return clock.subscribe((offsetSec) => {
      const u = uRef.current;
      if (!u || !cursorRef.current) return;
      const x = u.valToPos(offsetSec, "x");
      cursorRef.current.style.left = `${x}px`;
      cursorRef.current.style.display = x >= 0 ? "block" : "none";
    });
  }, [clock]);

  return (
    <div className="relative h-full w-full">
      <div ref={hostRef} className="absolute inset-0" />
      <div ref={cursorRef} className="absolute top-0 bottom-0 w-[1px] bg-accent/80 pointer-events-none" style={{ left: 0 }} />
      {status && <div className="absolute inset-0 flex items-center justify-center font-mono text-[10px] text-ink-3 pointer-events-none">{status}</div>}
    </div>
  );
}
