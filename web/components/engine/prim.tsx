"use client";

// Shared primitives for the Data Engine (M10-M19) plane surfaces. Blender-style: panels with a raised named
// header, rounded tool buttons with the blue action state, recessed value fields, and status-colored bars.
// The export signatures are unchanged so the ten plane pages inherit the new look without edits.

import { useState } from "react";

export function Panel({ title, hint, children, className = "" }: {
  title: string; hint?: string; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`bg-panel border border-line rounded ${className}`}>
      <div className="panel-head">
        <span className="uppercase tracking-wide text-[10px] text-ink-2 font-medium">{title}</span>
        {hint && <span className="ml-auto text-[10px] text-ink-3">{hint}</span>}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

export function KV({ k, v, tone = "ink" }: { k: string; v: React.ReactNode; tone?: Tone }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[12px] py-0.5">
      <span className="text-ink-3">{k}</span>
      <span className={`font-mono ${toneText(tone)}`}>{v}</span>
    </div>
  );
}

export type Tone = "ink" | "pass" | "warn" | "block" | "accent" | "info" | "ink-3";
export function toneText(t: Tone): string {
  return { ink: "text-ink", pass: "text-pass", warn: "text-warn", block: "text-block",
    accent: "text-accent-2", info: "text-info", "ink-3": "text-ink-3" }[t];
}
export function toneBg(t: Tone): string {
  return { ink: "bg-ink", pass: "bg-pass", warn: "bg-warn", block: "bg-block",
    accent: "bg-accent", info: "bg-info", "ink-3": "bg-ink-3" }[t];
}

export function Verdict({ ok, yes = "pass", no = "fail" }: { ok: boolean; yes?: string; no?: string }) {
  return (
    <span className={`inline-flex items-center text-[11px] px-1.5 py-0.5 rounded border ${ok
      ? "border-pass/40 text-pass bg-pass/5" : "border-block/40 text-block bg-block/5"}`}>
      {ok ? yes : no}
    </span>
  );
}

export function Bar({ frac, tone = "accent" }: { frac: number; tone?: Tone }) {
  return (
    <span className="flex-1 h-2.5 bg-bg-2 rounded-sm relative block overflow-hidden">
      <span className={`absolute left-0 top-0 h-full rounded-sm ${toneBg(tone)}`}
        style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%` }} />
    </span>
  );
}

export function NumField({ label, value, onChange, step = "any", w = "w-24" }: {
  label: string; value: number; onChange: (n: number) => void; step?: string; w?: string;
}) {
  return (
    <label className="flex items-center justify-between gap-2 text-[12px]">
      <span className="text-ink-2">{label}</span>
      <input type="number" step={step} value={Number.isFinite(value) ? value : ""}
        onChange={(e) => onChange(parseFloat(e.target.value))} className={`field ${w}`} />
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
    <button onClick={onClick} disabled={busy} className="btn btn-primary">
      {busy ? "running..." : label}
    </button>
  );
}

export function ErrLine({ err }: { err: string | null }) {
  if (!err) return null;
  return <div className="text-[11px] text-block border border-block/30 rounded px-2 py-1 mt-2 break-all bg-block/5">{err}</div>;
}
