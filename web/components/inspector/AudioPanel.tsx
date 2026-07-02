"use client";

import { useEffect, useRef, useState } from "react";
import { useClock } from "@/lib/inspector/clock";
import { useMcap } from "@/lib/inspector/mcapContext";

// Audio panel (optional): a coarse amplitude waveform of a microphone channel across the session, with a
// playhead advancing on the clock. Sessions without an audio topic show a note; this stays lightweight and
// never blocks the other panels.

export default function AudioPanel({ topic }: { topic: string }) {
  const clock = useClock();
  const { mcap } = useMcap();
  const hostRef = useRef<HTMLCanvasElement>(null);
  const headRef = useRef<HTMLDivElement>(null);
  const ampsRef = useRef<{ t: number[]; amp: number[] } | null>(null);
  const [note, setNote] = useState("loading audio...");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        // amplitude proxy: the magnitude of any numeric field per message (real PCM decode is codec-specific)
        const { t, values } = await mcap.collect(topic, clock.startNs);
        if (!alive) return;
        const keys = Object.keys(values);
        if (t.length === 0 || keys.length === 0) { setNote("no audio samples"); return; }
        const amp = t.map((_, i) => Math.hypot(...keys.map((k) => values[k][i] ?? 0)));
        ampsRef.current = { t, amp };
        setNote("");
        draw();
      } catch (e) { if (alive) setNote("audio load error: " + String(e)); }
    })();
    return () => { alive = false; };
  }, [mcap, topic, clock.startNs]);

  const draw = () => {
    const c = hostRef.current, data = ampsRef.current;
    if (!c || !data) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const w = c.width = c.clientWidth, h = c.height = c.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = "#56D364";
    ctx.beginPath();
    const max = Math.max(1, ...data.amp);
    data.amp.forEach((a, i) => {
      const x = (data.t[i] / clock.durationSec) * w;
      const y = h - (a / max) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  };

  useEffect(() => {
    return clock.subscribe((offsetSec) => {
      if (headRef.current) headRef.current.style.left = `${(offsetSec / clock.durationSec) * 100}%`;
    });
  }, [clock]);

  return (
    <div className="relative h-full w-full">
      <canvas ref={hostRef} className="absolute inset-0 w-full h-full" />
      <div ref={headRef} className="absolute top-0 bottom-0 w-[1px] bg-accent pointer-events-none" style={{ left: 0 }} />
      {note && <div className="absolute inset-0 flex items-center justify-center font-mono text-[10px] text-ink-3 pointer-events-none">{note}</div>}
    </div>
  );
}
