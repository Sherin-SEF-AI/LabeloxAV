"use client";

// One ontology attribute, rendered as the control its declared type calls for.
//
// Two of the four branches here were wrong before this. `bool_array` fell through to the number input, so
// `helmet` - the per-rider attribute, and the one that matters most on an Indian road - offered a spinner
// and wrote a Number into a field the server validates as a list of booleans, which is a 400 every time.
// And the number input ignored `range` entirely, so the client happily submitted values the server was
// about to reject with no indication of which end was wrong.

const DERIVED_HINT = "computed from another attribute; correct that one instead";

export default function AttrControl({ name, spec, value, onChange }: {
  name: string;
  spec: { type: string; values: unknown[] | null; range: number[] | null; derived_from?: string | null };
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = <span className="w-24 shrink-0 font-mono text-[11px] text-ink-3 truncate">{name}</span>;

  // Read-only rather than absent. An annotator who set occupant_count to three wants to see that
  // triple_riding followed; hiding it makes the derivation invisible and the number look unrecorded.
  if (spec.derived_from)
    return (
      <label className="flex items-center gap-2" title={`${DERIVED_HINT} (${spec.derived_from})`}>
        {label}
        <span className="flex-1 font-mono text-[11px] text-ink-3">
          {value == null ? "-" : String(value)}
          <span className="ml-1.5 text-[9px] uppercase tracking-wide text-ink-3/70">derived</span>
        </span>
      </label>
    );

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

  if (spec.type === "bool_array") {
    // One checkbox per occupant, plus and minus to change how many. The list length is the claim about how
    // many riders there are, so it is edited explicitly rather than inferred from another field: a
    // two-element helmet array on a scooter carrying three is a wrong statement, not a missing one.
    const arr: boolean[] = Array.isArray(value) ? value.map(Boolean) : [];
    const set = (i: number, on: boolean) => onChange(arr.map((v, j) => (j === i ? on : v)));
    return (
      <div className="flex items-center gap-2">{label}
        <div className="flex-1 flex items-center gap-1 flex-wrap">
          {arr.map((on, i) => (
            <label key={i} className="flex items-center gap-0.5" title={`occupant ${i + 1}`}>
              <input type="checkbox" checked={on} onChange={(e) => set(i, e.target.checked)}
                aria-label={`${name} occupant ${i + 1}`} />
              <span className="font-mono text-[10px] text-ink-3">{i + 1}</span>
            </label>
          ))}
          {!arr.length && <span className="font-mono text-[10px] text-ink-3">none</span>}
          <button type="button" aria-label={`add ${name} entry`} onClick={() => onChange([...arr, false])}
            className="ml-auto px-1 font-mono text-[11px] text-ink-3 hover:text-ink">+</button>
          <button type="button" aria-label={`remove ${name} entry`} disabled={!arr.length}
            onClick={() => onChange(arr.slice(0, -1))}
            className={`px-1 font-mono text-[11px] ${arr.length ? "text-ink-3 hover:text-ink" : "text-line cursor-not-allowed"}`}>
            &minus;
          </button>
        </div>
      </div>
    );
  }

  const [lo, hi] = spec.range ?? [];
  const num = value == null ? null : Number(value);
  // Out of range is shown, not silently corrected. Clamping would change what the annotator typed without
  // telling them, and typing 100 into a 0-6 field is more likely a wrong field than a wrong value.
  const bad = num != null && spec.range != null && (num < lo! || num > hi!);
  return (
    <label className="flex items-center gap-2">{label}
      <input type="number" step={spec.type === "float" ? 0.01 : 1}
        min={lo} max={hi} value={num == null ? "" : num}
        aria-invalid={bad || undefined}
        title={spec.range ? `${lo} to ${hi}` : undefined}
        onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        className={`flex-1 bg-panel border px-1 py-0.5 font-mono text-[11px] text-ink ${bad ? "border-block" : "border-line"}`} />
      {bad && <span className="font-mono text-[10px] text-block shrink-0">{lo}-{hi}</span>}
    </label>
  );
}
