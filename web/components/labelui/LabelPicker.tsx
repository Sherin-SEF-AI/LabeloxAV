"use client";

import type { FieldDef, LabelConfig } from "@/lib/types";

// The config-driven control strip: the labels a project declares, and the typed fields each annotation
// carries. Rendered from label_config rather than hardcoded, which is what lets one editor serve a
// pedestrian-detection project and a speaker-diarization project without a code change.

export function LabelChips({ config, value, onChange, kind }: {
  config: LabelConfig; value: string | null; onChange: (v: string | null) => void; kind?: string;
}) {
  // Only offer labels this kind may actually carry, so the picker cannot compose a combination the server
  // will reject.
  const labels = (config.labels ?? []).filter((l) => !kind || !l.kinds.length || l.kinds.includes(kind));
  if (!labels.length) {
    return <span className="font-mono text-[11px] text-ink-3">no labels declared for this project</span>;
  }
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {labels.map((l) => (
        <button key={l.name} onClick={() => onChange(value === l.name ? null : l.name)}
          className={`px-1.5 py-0.5 font-mono text-[11px] border ${
            value === l.name ? "border-accent text-ink" : "border-line text-ink-3 hover:text-ink-2"}`}
          style={value === l.name && l.color ? { borderColor: l.color, color: l.color } : undefined}>
          {l.name}
        </button>
      ))}
    </div>
  );
}

export function FieldsForm({ fields, value, onChange }: {
  fields: FieldDef[]; value: Record<string, unknown>; onChange: (v: Record<string, unknown>) => void;
}) {
  if (!fields?.length) return null;
  const set = (k: string, v: unknown) => onChange({ ...value, [k]: v });
  return (
    <div className="space-y-1">
      {fields.map((f) => (
        <label key={f.name} className="flex items-center gap-2 font-mono text-[11px]">
          <span className="text-ink-3 min-w-[80px]">
            {f.name}{f.required && <span className="text-block">*</span>}
          </span>
          {f.type === "enum" ? (
            <select value={String(value[f.name] ?? "")} onChange={(e) => set(f.name, e.target.value)}
              className="bg-bg border border-line px-1.5 py-0.5 text-ink flex-1">
              <option value="">-</option>
              {f.values.map((v) => <option key={v} value={v}>{v}</option>)}
            </select>
          ) : f.type === "bool" ? (
            <input type="checkbox" checked={Boolean(value[f.name])}
              onChange={(e) => set(f.name, e.target.checked)} />
          ) : (
            <input
              type={f.type === "float" || f.type === "int" ? "number" : "text"}
              step={f.type === "float" ? "0.01" : f.type === "int" ? "1" : undefined}
              value={String(value[f.name] ?? "")}
              onChange={(e) => set(f.name, e.target.value)}
              className="bg-bg border border-line px-1.5 py-0.5 text-ink flex-1" />
          )}
        </label>
      ))}
    </div>
  );
}
