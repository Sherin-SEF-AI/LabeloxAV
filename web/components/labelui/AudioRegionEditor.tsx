"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { AnnotationRow, LabelConfig } from "@/lib/types";

// Audio: a waveform you drag on to create a time region.
//
// The waveform is decoded once via WebAudio and reduced to per-pixel min/max peaks. Drawing every sample
// would be both unreadable and slow (a 60s clip at 48kHz is ~2.9M samples for maybe 900 pixels); min/max
// peaks per column is what makes a waveform look like a waveform rather than a smear.

type Props = {
  uri: string;
  annotations: AnnotationRow[];
  config: LabelConfig;
  activeLabel: string | null;
  onCreate: (tStart: number, tEnd: number) => void;
  onSelect: (id: string | null) => void;
  selectedId: string | null;
};

function colorFor(config: LabelConfig, label: string | null): string {
  return (config.labels ?? []).find((x) => x.name === label)?.color || "#4c8dff";
}

export default function AudioRegionEditor({
  uri, annotations, config, activeLabel, onCreate, onSelect, selectedId,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [peaks, setPeaks] = useState<Float32Array | null>(null);
  const [duration, setDuration] = useState(0);
  const [drag, setDrag] = useState<{ x0: number; x1: number } | null>(null);
  const [playhead, setPlayhead] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const W = 900, H = 140;

  // decode once
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(uri);
        if (!res.ok) throw new Error(`audio fetch ${res.status}`);
        const buf = await res.arrayBuffer();
        const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        const ctx = new Ctx();
        const decoded = await ctx.decodeAudioData(buf);
        if (cancelled) return;
        const ch = decoded.getChannelData(0);
        const per = Math.max(1, Math.floor(ch.length / W));
        const out = new Float32Array(W * 2);
        for (let i = 0; i < W; i++) {
          let lo = 1, hi = -1;
          for (let j = i * per; j < Math.min((i + 1) * per, ch.length); j++) {
            if (ch[j] < lo) lo = ch[j];
            if (ch[j] > hi) hi = ch[j];
          }
          out[i * 2] = lo; out[i * 2 + 1] = hi;
        }
        setPeaks(out);
        setDuration(decoded.duration);
        ctx.close();
      } catch (e) {
        if (!cancelled) setErr(String(e));
      }
    })();
    return () => { cancelled = true; };
  }, [uri]);

  const regions = annotations.filter((a) => a.kind === "region");

  useEffect(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr; cv.height = H * dpr;
    cv.style.width = `${W}px`; cv.style.height = `${H}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // waveform
    ctx.strokeStyle = "#5a6472";
    ctx.beginPath();
    if (peaks) {
      for (let i = 0; i < W; i++) {
        const lo = peaks[i * 2], hi = peaks[i * 2 + 1];
        ctx.moveTo(i + 0.5, H / 2 - (hi * H) / 2);
        ctx.lineTo(i + 0.5, H / 2 - (lo * H) / 2);
      }
    } else {
      ctx.moveTo(0, H / 2); ctx.lineTo(W, H / 2);
    }
    ctx.stroke();

    // regions
    for (const r of regions) {
      const t0 = Number(r.payload.t_start ?? 0), t1 = Number(r.payload.t_end ?? 0);
      if (!duration) continue;
      const x0 = (t0 / duration) * W, x1 = (t1 / duration) * W;
      const c = colorFor(config, r.label);
      ctx.fillStyle = `${c}33`;
      ctx.fillRect(x0, 0, Math.max(1, x1 - x0), H);
      ctx.strokeStyle = r.annotation_id === selectedId ? "#ffffff" : c;
      ctx.lineWidth = r.annotation_id === selectedId ? 2 : 1;
      ctx.strokeRect(x0, 0, Math.max(1, x1 - x0), H);
    }

    // in-progress drag
    if (drag) {
      ctx.fillStyle = "rgba(76,141,255,0.20)";
      ctx.fillRect(Math.min(drag.x0, drag.x1), 0, Math.abs(drag.x1 - drag.x0), H);
    }

    // playhead
    if (duration && playhead) {
      const x = (playhead / duration) * W;
      ctx.strokeStyle = "#f2c94c";
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
  }, [peaks, regions, duration, drag, playhead, selectedId, config]);

  const xToT = useCallback((x: number) => (duration ? (x / W) * duration : 0), [duration]);

  const localX = (e: React.MouseEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    return Math.max(0, Math.min(W, e.clientX - r.left));
  };

  return (
    <div className="p-4 space-y-2">
      {err && <div className="font-mono text-[11px] text-block">could not decode audio: {err}</div>}
      <canvas ref={canvasRef}
        className="block border border-line cursor-crosshair"
        onMouseDown={(e) => setDrag({ x0: localX(e), x1: localX(e) })}
        onMouseMove={(e) => setDrag((d) => (d ? { ...d, x1: localX(e) } : null))}
        onMouseUp={(e) => {
          const d = drag;
          setDrag(null);
          if (!d) return;
          const x1 = localX(e);
          const a = Math.min(d.x0, x1), b = Math.max(d.x0, x1);
          if (b - a < 3) {
            // a click, not a drag: seek instead of creating a zero-length region
            if (audioRef.current && duration) audioRef.current.currentTime = xToT(a);
            // also let a click select an existing region
            const t = xToT(a);
            const hit = regions.find((r) => t >= Number(r.payload.t_start) && t <= Number(r.payload.t_end));
            onSelect(hit ? hit.annotation_id : null);
            return;
          }
          onCreate(xToT(a), xToT(b));
        }}
        onMouseLeave={() => setDrag(null)} />

      <div className="flex items-center gap-3 font-mono text-[11px] text-ink-3">
        <audio ref={audioRef} src={uri} controls
          onTimeUpdate={(e) => setPlayhead((e.target as HTMLAudioElement).currentTime)}
          className="h-7" />
        <span>{duration ? `${duration.toFixed(2)}s` : "loading..."}</span>
        <span>{regions.length} regions</span>
        {!activeLabel && <span>pick a label, then drag on the waveform</span>}
      </div>
    </div>
  );
}
