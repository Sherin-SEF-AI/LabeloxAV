"use client";

// The Blender-style UI kit. A small set of self-explanatory controls that read like Blender's interface:
// rounded tool buttons with subtle gradients, the blue active state, recessed input fields, panels with a
// named header and an optional one-line description, tooltips on every control, and status badges. Every page
// and the shared chrome build from these, so the look stays consistent and controls explain themselves.

import { useState } from "react";

type Variant = "default" | "primary" | "ghost";
type State = "pass" | "warn" | "block" | "info" | "accent" | "muted";

const STATE_TEXT: Record<State, string> = {
  pass: "text-pass", warn: "text-warn", block: "text-block", info: "text-info",
  accent: "text-accent-2", muted: "text-ink-3",
};
const STATE_BG: Record<State, string> = {
  pass: "bg-pass", warn: "bg-warn", block: "bg-block", info: "bg-info", accent: "bg-accent", muted: "bg-ink-3",
};

// A tool button. `tip` renders a hover tooltip so the button explains itself. `active` shows the blue toggle
// state. `variant="primary"` is the blue call-to-action.
export function Button({ children, onClick, tip, variant = "default", active = false, disabled = false,
  icon, className = "" }: {
  children?: React.ReactNode; onClick?: () => void; tip?: string; variant?: Variant; active?: boolean;
  disabled?: boolean; icon?: React.ReactNode; className?: string;
}) {
  const base = variant === "ghost"
    ? "inline-flex items-center justify-center gap-1.5 px-2.5 py-1 text-[12px] rounded text-ink-2 hover:text-ink hover:bg-head border border-transparent hover:border-line"
    : `btn ${active ? "btn-on" : ""} ${variant === "primary" ? "btn-primary" : ""}`;
  return (
    <button type="button" onClick={onClick} disabled={disabled} data-tip={tip}
      className={`${base} ${className}`}>
      {icon && <span className="text-[13px] leading-none">{icon}</span>}
      {children}
    </button>
  );
}

// A square icon-only button (toolbar). Always carries a tooltip so the glyph is never a mystery.
export function IconButton({ glyph, tip, onClick, active = false, className = "" }: {
  glyph: React.ReactNode; tip: string; onClick?: () => void; active?: boolean; className?: string;
}) {
  return (
    <button type="button" onClick={onClick} data-tip={tip}
      className={`inline-flex items-center justify-center w-7 h-7 rounded text-[14px] leading-none border transition-colors
        ${active ? "btn-on border-transparent" : "text-ink-2 border-line bg-head/60 hover:text-ink hover:bg-head"} ${className}`}>
      {glyph}
    </button>
  );
}

// A panel: a named region with a header, an optional one-line description (self-explanatory), an optional
// header-right slot, and collapse. Bodies read as Blender sub-panels.
export function Panel({ title, desc, right, children, collapsible = false, defaultOpen = true, className = "" }: {
  title: string; desc?: string; right?: React.ReactNode; children: React.ReactNode;
  collapsible?: boolean; defaultOpen?: boolean; className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`bg-panel border border-line rounded ${className}`}>
      <div className="panel-head">
        <button type="button" onClick={() => collapsible && setOpen(!open)}
          className={`flex items-center gap-1.5 ${collapsible ? "hover:text-ink" : "cursor-default"}`}>
          {collapsible && <span className="text-ink-3 text-[9px]">{open ? "▾" : "▸"}</span>}
          <span className="uppercase tracking-wide text-[10px] text-ink-2 font-medium">{title}</span>
        </button>
        {right && <div className="ml-auto flex items-center gap-2">{right}</div>}
      </div>
      {open && (
        <div className="p-3">
          {desc && <p className="text-[11px] text-ink-3 mb-2 leading-snug">{desc}</p>}
          {children}
        </div>
      )}
    </section>
  );
}

// A labeled field row. The label sits left, the control right, like a Blender properties row.
export function Field({ label, tip, children }: { label: string; tip?: string; children: React.ReactNode }) {
  return (
    <label className="flex items-center justify-between gap-3 py-1" data-tip={tip}>
      <span className="text-[12px] text-ink-2">{label}</span>
      {children}
    </label>
  );
}

// A number input styled as a Blender value field.
export function NumberInput({ value, onChange, step = "any", w = "w-24" }: {
  value: number; onChange: (n: number) => void; step?: string; w?: string;
}) {
  return (
    <input type="number" step={step} value={Number.isFinite(value) ? value : ""}
      onChange={(e) => onChange(parseFloat(e.target.value))} className={`field ${w}`} />
  );
}

export function TextInput({ value, onChange, placeholder, className = "flex-1" }: {
  value: string; onChange: (s: string) => void; placeholder?: string; className?: string;
}) {
  return (
    <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
      className={`field ${className}`} />
  );
}

export function TextArea({ value, onChange, rows = 5, className = "w-full" }: {
  value: string; onChange: (s: string) => void; rows?: number; className?: string;
}) {
  return (
    <textarea value={value} onChange={(e) => onChange(e.target.value)} rows={rows}
      className={`field font-mono text-[11px] ${className}`} />
  );
}

// A segmented control (Blender's row of joined toggle buttons). One option is active (blue).
export function Segmented<T extends string>({ options, value, onChange }: {
  options: { value: T; label: string; tip?: string }[]; value: T; onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex rounded overflow-hidden border border-[#2b2b2b]">
      {options.map((o, i) => (
        <button key={o.value} type="button" onClick={() => onChange(o.value)} data-tip={o.tip}
          className={`px-2.5 py-1 text-[12px] ${i > 0 ? "border-l border-[#2b2b2b]" : ""}
            ${value === o.value ? "btn-on" : "bg-head text-ink-2 hover:text-ink"}`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

// A status badge / verdict pill.
export function Badge({ children, state = "muted" }: { children: React.ReactNode; state?: State }) {
  return (
    <span className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[11px] border
      ${state === "muted" ? "border-line text-ink-3"
        : state === "pass" ? "border-pass/40 text-pass"
        : state === "warn" ? "border-warn/40 text-warn"
        : state === "block" ? "border-block/40 text-block"
        : "border-accent/50 text-accent-2"}`}>
      {children}
    </span>
  );
}

export function Verdict({ ok, yes = "pass", no = "fail" }: { ok: boolean; yes?: string; no?: string }) {
  return <Badge state={ok ? "pass" : "block"}>{ok ? yes : no}</Badge>;
}

// A key/value row for readouts.
export function KV({ k, v, state }: { k: string; v: React.ReactNode; state?: State }) {
  return (
    <div className="flex items-center justify-between gap-3 text-[12px] py-0.5">
      <span className="text-ink-3">{k}</span>
      <span className={`font-mono ${state ? STATE_TEXT[state] : "text-ink"}`}>{v}</span>
    </div>
  );
}

// A proportion bar.
export function Bar({ frac, state = "accent" }: { frac: number; state?: State }) {
  return (
    <span className="flex-1 h-2.5 bg-bg-2 rounded-sm relative block overflow-hidden">
      <span className={`absolute left-0 top-0 h-full rounded-sm ${STATE_BG[state]}`}
        style={{ width: `${Math.max(0, Math.min(1, frac)) * 100}%` }} />
    </span>
  );
}

// Async run helper shared by the interactive tool panels.
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

export function ErrLine({ err }: { err: string | null }) {
  if (!err) return null;
  return <div className="text-[11px] text-block border border-block/30 rounded px-2 py-1 mt-2 break-all bg-block/5">{err}</div>;
}

// A one-line hint bar used at the top of a page body to explain what the page is for.
export function HintBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-2 text-[12px] text-ink-2 bg-head/50 border border-line rounded px-3 py-2 mb-3">
      <span className="text-accent-2 text-[13px]">{"ℹ"}</span>
      <span>{children}</span>
    </div>
  );
}
