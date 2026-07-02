"use client";

// The one synchronized playback clock. Its value IS ts_ns; every panel subscribes to it. Absolute times are
// bigint (ts_ns exceeds Number.MAX_SAFE_INTEGER); the position is driven in relative seconds so uPlot, the
// timeline, and arithmetic stay in safe numbers. Panels subscribe imperatively for smooth per-frame updates;
// the HUD re-renders at a throttled rate for the readable timestamp.

import { createContext, useContext, useEffect, useRef, useState } from "react";

export class ClockStore {
  readonly startNs: bigint;
  readonly endNs: bigint;
  readonly durationSec: number;
  offsetSec = 0;
  playing = false;
  speed = 1;
  private listeners = new Set<(offsetSec: number) => void>();
  private raf = 0;
  private lastPerf = 0;

  constructor(startNs: bigint, endNs: bigint) {
    this.startNs = startNs;
    this.endNs = endNs > startNs ? endNs : startNs + 1n;
    this.durationSec = Number(this.endNs - this.startNs) / 1e9;
  }

  nowNs(): bigint {
    return this.startNs + BigInt(Math.round(Math.max(0, this.offsetSec) * 1e9));
  }

  subscribe(fn: (offsetSec: number) => void): () => void {
    this.listeners.add(fn);
    fn(this.offsetSec);
    return () => this.listeners.delete(fn);
  }

  private emit() {
    for (const fn of this.listeners) fn(this.offsetSec);
  }

  seek(sec: number) {
    this.offsetSec = Math.min(this.durationSec, Math.max(0, sec));
    this.emit();
  }

  step(deltaSec: number) {
    this.seek(this.offsetSec + deltaSec);
  }

  setSpeed(x: number) {
    this.speed = x;
  }

  play() {
    if (this.playing) return;
    if (this.offsetSec >= this.durationSec) this.offsetSec = 0;
    this.playing = true;
    this.lastPerf = performance.now();
    const loop = () => {
      if (!this.playing) return;
      const t = performance.now();
      const dt = (t - this.lastPerf) / 1000;
      this.lastPerf = t;
      this.offsetSec += dt * this.speed;
      if (this.offsetSec >= this.durationSec) {
        this.offsetSec = this.durationSec;
        this.playing = false;
      }
      this.emit();
      if (this.playing) this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  pause() {
    this.playing = false;
    if (this.raf) cancelAnimationFrame(this.raf);
  }

  toggle() {
    this.playing ? this.pause() : this.play();
  }

  dispose() {
    this.pause();
    this.listeners.clear();
  }
}

const ClockContext = createContext<ClockStore | null>(null);

export function ClockProvider({ startNs, endNs, children }: { startNs: bigint; endNs: bigint; children: React.ReactNode }) {
  const ref = useRef<ClockStore | null>(null);
  if (ref.current === null || ref.current.startNs !== startNs || ref.current.endNs !== endNs) {
    ref.current?.dispose();
    ref.current = new ClockStore(startNs, endNs);
  }
  useEffect(() => () => ref.current?.dispose(), []);
  return <ClockContext.Provider value={ref.current}>{children}</ClockContext.Provider>;
}

export function useClock(): ClockStore {
  const c = useContext(ClockContext);
  if (!c) throw new Error("useClock must be used inside a ClockProvider");
  return c;
}

// Re-render the caller on clock ticks, throttled to `hz`, returning the current offset seconds. For the HUD
// and controls; smooth panels should subscribe() imperatively instead.
export function useClockTick(hz = 15): { offsetSec: number; playing: boolean } {
  const clock = useClock();
  const [state, setState] = useState({ offsetSec: clock.offsetSec, playing: clock.playing });
  const last = useRef(0);
  useEffect(() => {
    const minMs = 1000 / hz;
    return clock.subscribe((offsetSec) => {
      const t = performance.now();
      if (t - last.current >= minMs || !clock.playing) {
        last.current = t;
        setState({ offsetSec, playing: clock.playing });
      }
    });
  }, [clock, hz]);
  return state;
}
