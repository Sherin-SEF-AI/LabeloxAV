"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { humanizeError } from "@/lib/api";
import type { AnnotationRow, LabelConfig } from "@/lib/types";

// Audio: a waveform, or a spectrogram, that you drag on to create a time region.
//
// The waveform is decoded once via WebAudio and reduced to per-pixel min/max peaks. Drawing every sample
// would be both unreadable and slow (a 60s clip at 48kHz is ~2.9M samples for maybe 900 pixels); min/max
// peaks per column is what makes a waveform look like a waveform rather than a smear.
//
// The spectrogram exists because amplitude alone is the wrong view for most of what gets annotated here. A
// horn, a siren and a reversing alarm all look like "loud" on a waveform and are immediately distinct by
// frequency; so is speech against traffic noise at the same level. The transform is a plain DFT over a
// Hann-windowed frame, computed once per column at the resolution actually drawn rather than at full FFT
// size, which keeps it fast enough to compute in the browser without a worker.

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
  const [view, setView] = useState<"wave" | "spectrogram">("wave");
  const [spec, setSpec] = useState<Float32Array | null>(null);
  const W = 900, H = 140;
  // 64 frequency bins over the drawn height. More would be invisible at 140 pixels; fewer would merge a
  // siren into the band next to it.
  const BINS = 64;

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
        setSpec(computeSpectrogram(ch, W, BINS));
        setDuration(decoded.duration);
        ctx.close();
      } catch (e) {
        if (!cancelled) setErr(humanizeError(e));
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

    if (view === "spectrogram" && spec) {
      // Low frequencies at the bottom, which is how every spectrogram anyone has read is oriented; drawing
      // it the other way up makes an otherwise familiar picture unreadable.
      const rowH = H / BINS;
      for (let c = 0; c < W; c++) {
        for (let k = 0; k < BINS; k++) {
          const v = spec[c * BINS + k];
          if (v <= 0.02) continue;   // near-silence stays background rather than a grey wash
          // Single-hue ramp rather than a rainbow: a rainbow map invents boundaries where the data is
          // smooth, and people read those bands as real structure.
          const l = Math.round(8 + v * 62);
          ctx.fillStyle = `hsl(206 70% ${l}%)`;
          ctx.fillRect(c, H - (k + 1) * rowH, 1, Math.ceil(rowH));
        }
      }
    } else {
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
    }

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
  }, [peaks, regions, duration, drag, playhead, selectedId, config, view, spec]);

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
        <span className="flex items-center gap-1 ml-auto">
          {(["wave", "spectrogram"] as const).map((v) => (
            <button key={v} onClick={() => setView(v)} disabled={v === "spectrogram" && !spec}
              className={`border px-1.5 py-0.5 ${
                view === v ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}
                disabled:opacity-40`}>
              {v}
            </button>
          ))}
        </span>
        {!activeLabel && <span>pick a label, then drag on the {view}</span>}
      </div>
    </div>
  );
}


// A Hann-windowed DFT, one column per drawn pixel.
//
// Deliberately not a full FFT library. The output is 64 bins over 900 columns, which is 57,600 magnitudes;
// computing them directly at the resolution actually displayed is faster than transforming at 2048 points
// and throwing 97% of the result away, and it removes a dependency from the annotation path.
//
// Magnitudes are converted to dB and clamped to a 60 dB floor, because linear magnitude renders almost
// everything as black: real audio spans several orders of magnitude and the quiet detail that matters for
// labelling lives in the bottom decade.
export function computeSpectrogram(samples: Float32Array, columns: number, bins: number): Float32Array {
  const out = new Float32Array(columns * bins);
  const frame = Math.max(bins * 2, Math.floor(samples.length / columns));
  const hop = Math.max(1, Math.floor(samples.length / columns));

  // Precomputed so the trigonometry is not repeated per column.
  const cosTab = new Float32Array(bins * bins * 2);
  const sinTab = new Float32Array(bins * bins * 2);
  const n = bins * 2;
  for (let k = 0; k < bins; k++) {
    for (let i = 0; i < n; i++) {
      const a = (-2 * Math.PI * k * i) / n;
      cosTab[k * n + i] = Math.cos(a);
      sinTab[k * n + i] = Math.sin(a);
    }
  }
  const window = new Float32Array(n);
  for (let i = 0; i < n; i++) window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));

  const buf = new Float32Array(n);
  for (let c = 0; c < columns; c++) {
    const start = c * hop;
    // Decimate the frame down to n points rather than transforming the whole hop: the column is one pixel
    // wide, so resolving detail finer than the frame it represents would be discarded anyway.
    const step = Math.max(1, Math.floor(frame / n));
    for (let i = 0; i < n; i++) {
      const idx = start + i * step;
      buf[i] = (idx < samples.length ? samples[idx] : 0) * window[i];
    }
    for (let k = 0; k < bins; k++) {
      let re = 0, im = 0;
      for (let i = 0; i < n; i++) {
        re += buf[i] * cosTab[k * n + i];
        im += buf[i] * sinTab[k * n + i];
      }
      const mag = Math.sqrt(re * re + im * im) / n;
      // dB with a 60 dB floor, normalised to 0..1 for drawing.
      const db = 20 * Math.log10(mag + 1e-9);
      out[c * bins + k] = Math.min(1, Math.max(0, (db + 60) / 60));
    }
  }
  return out;
}
