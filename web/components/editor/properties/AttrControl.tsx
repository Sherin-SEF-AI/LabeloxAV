"use client";

// One ontology attribute, rendered as the control its declared type calls for. Moved out of the frame page
// unchanged; it was the page's only other top-level component and had no reason to live there.

export default function AttrControl({ name, spec, value, onChange }: {
  name: string;
  spec: { type: string; values: unknown[] | null; range: number[] | null };
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = <span className="w-24 shrink-0 font-mono text-[11px] text-ink-3 truncate">{name}</span>;
  if (spec.type === "enum")
    return (
      <label className="flex items-center gap-2">{label}
        <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}
          className="flex-1 bg-panel border border-line px-1 py-0.5 font-mono text-[11px] text-ink">
          <option value="">-</option>
          {(spec.values || []).map((v) => <option key={String(v)} value={String(v)}>{String(v)}</option>)}
        </select>
      </label>
    );
  if (spec.type === "bool")
    return (
      <label className="flex items-center gap-2">{label}
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
      </label>
    );
  return (
    <label className="flex items-center gap-2">{label}
      <input type="number" step={spec.type === "float" ? 0.01 : 1} value={value == null ? "" : Number(value)}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        className="flex-1 bg-panel border border-line px-1 py-0.5 font-mono text-[11px] text-ink" />
    </label>
  );
}
