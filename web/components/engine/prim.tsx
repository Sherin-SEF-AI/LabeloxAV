"use client";

// Shared primitives for the Data Engine (M10-M19) plane surfaces. Operational Materialism: matte panels,
// monospace data, color earned only by state (pass/warn/block). Kept tiny and local so each plane page reads
// as one focused file.

import { useState } from "react";

export function Panel({ title, hint, children, className = "" }: {
  title: string; hint?: string; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`border border-line p-3 ${className}`}>
      <div className="flex items-baseline justify-between mb-2">
        <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">{title}</div>
        {hint && <div className="font-mono text-[10px] text-ink-3">{hint}</div>}
      </div>
      {children}
    </section>
  );
}

export function KV({ k, v, tone = "ink" }: { k: string; v: React.ReactNode; tone?: Tone }) {
  return (
    <div className="flex items-center justify-between gap-3 font-mono text-[11px] py-0.5">
      <span className="text-ink-3">{k}</span>
      <span className={toneText(tone)}>{v}</span>
    </div>
  );
}

export type Tone = "ink" | "pass" | "warn" | "block" | "accent" | "info" | "ink-3";
export function toneText(t: Tone): string {
  return { ink: "text-ink", pass: "text-pass", warn: "text-warn", block: "text-block",
    accent: "text-accent", info: "text-info", "ink-3": "text-ink-3" }[t];
}
export function toneBg(t: Tone): string {
  return { ink: "bg-ink", pass: "bg-pass", warn: "bg-warn", block: "bg-block",
    accent: "bg-accent", info: "bg-info", "ink-3": "bg-ink-3" }[t];
}

export function Verdict({ ok, yes = "pass", no = "fail" }: { ok: boolean; yes?: string; no?: string }) {
  return (
    <span className={`font-mono text-[11px] px-1.5 py-0.5 border ${ok
      ? "border-pass/40 text-pass" : "border-block/40 text-block"}`}>
      {ok ? yes : no}
    </span>
  );
}

export function Bar({ frac, tone = "accent" }: { frac: number; tone?: Tone }) {
  return (
    <span className="flex-1 h-3 bg-line/40 relative block">
      <span className={`absolute left-0 top-0 h-full ${toneBg(tone)}`}
        style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%` }} />
    </span>
  );
}

export function NumField({ label, value, onChange, step = "any", w = "w-24" }: {
  label: string; value: number; onChange: (n: number) => void; step?: string; w?: string;
}) {
  return (
    <label className="flex items-center gap-2 font-mono text-[11px]">
      <span className="text-ink-3">{label}</span>
      <input type="number" step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className={`${w} bg-bg-2 border border-line px-1.5 py-0.5 text-ink focus:border-accent outline-none`} />
    </label>
  );
}

// A run button that awaits an async action, showing a running state and surfacing the error inline.
export function useRun<T>() {
  const [out, setOut] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const run = async (fn: () => Promise<T>) => {
    setBusy(true); setErr(null);
    try { setOut(await fn()); } catch (e) { setErr(String(e instanceof Error ? e.message : e)); setOut(null); }
    finally { setBusy(false); }
  };
  return { out, err, busy, run, setOut };
}

export function RunButton({ busy, onClick, label = "run" }: { busy: boolean; onClick: () => void; label?: string }) {
  return (
    <button onClick={onClick} disabled={busy}
      className="font-mono text-[11px] px-3 py-1 border border-line text-ink-2 hover:text-accent hover:border-accent
        disabled:opacity-40 disabled:hover:text-ink-2 disabled:hover:border-line">
      {busy ? "running..." : label}
    </button>
  );
}

export function ErrLine({ err }: { err: string | null }) {
  if (!err) return null;
  return <div className="font-mono text-[10px] text-block border border-block/30 px-2 py-1 mt-2 break-all">{err}</div>;
}
