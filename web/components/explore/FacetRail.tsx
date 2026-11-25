"use client";

import type { ExplorePredicate, FacetBucket, Facets } from "@/lib/types";
import { humanize, num } from "@/lib/format";

// The facet sidebar. Every bar is both a statistic and a control: the number is how many you would get if you
// picked it, and clicking toggles that value into the predicate. Counts come from the server with each facet's
// own clause dropped, which is what keeps a list from collapsing to the one value you already selected.

// The value shown for a bucket. Codes (`traffic_sign`, `auto_accept`) read as words; the value toggled into
// the predicate is still the code. Confidence buckets are ranges ("0.6-0.8") and are left alone.
const ACRONYMS = /\b(vlm|gpu|ocr|sam|yolo|dino|lidar)\b/gi;
const shown = (v: unknown, raw: boolean) => (raw ? String(v)
  : (humanize(String(v)) || String(v)).replace(ACRONYMS, (m) => (m.toLowerCase() === "lidar" ? "LiDAR" : m.toUpperCase())));

function Bar({ b, max, active, onClick, raw = false }: {
  b: FacetBucket; max: number; active: boolean; onClick: () => void; raw?: boolean;
}) {
  const pct = max > 0 ? Math.max(2, Math.round((b.count / max) * 100)) : 0;
  return (
    <button
      onClick={onClick}
      aria-pressed={active}
      title={`${b.value} (${b.count})`}
      className={`relative w-full text-left px-2 py-1 rounded text-[12.5px] overflow-hidden transition-colors
        ${active ? "text-ink" : "text-ink-2 hover:text-ink hover:bg-head/50"}`}
    >
      <div
        className={`absolute inset-y-0.5 left-0 rounded-sm ${active ? "bg-accent/30" : "bg-line/40"}`}
        style={{ width: `${pct}%` }}
      />
      <span className="relative flex justify-between gap-2">
        <span className="truncate">{shown(b.value, raw)}</span>
        <span className={`shrink-0 tabular-nums ${active ? "text-ink-2" : "text-ink-3"}`}>{num(b.count)}</span>
      </span>
    </button>
  );
}

function Group({ title, buckets, selected, onToggle, raw = false }: {
  title: string; buckets: FacetBucket[]; selected: string[]; onToggle: (v: string) => void; raw?: boolean;
}) {
  if (!buckets?.length) return null;
  const max = Math.max(...buckets.map((b) => b.count), 1);
  return (
    <div className="border-b hairline pb-2 mb-2">
      <div className="text-[12px] font-semibold text-ink-2 px-2 py-1.5">{title}</div>
      <div className="max-h-44 overflow-auto no-scrollbar">
        {buckets.map((b) => (
          <Bar key={String(b.value)} b={b} max={max} raw={raw}
            active={selected.includes(String(b.value))} onClick={() => onToggle(String(b.value))} />
        ))}
      </div>
    </div>
  );
}

export default function FacetRail({ facets, predicate, onChange }: {
  facets: Facets | null;
  predicate: ExplorePredicate;
  onChange: (p: ExplorePredicate) => void;
}) {
  // Toggle a value in a list-valued clause, dropping the clause entirely when it empties so the predicate
  // stays minimal (and so a saved view does not carry empty arrays).
  const toggle = (key: keyof ExplorePredicate, value: string) => {
    const cur = ((predicate[key] as string[] | undefined) ?? []).slice();
    const i = cur.indexOf(value);
    if (i >= 0) cur.splice(i, 1);
    else cur.push(value);
    const next: ExplorePredicate = { ...predicate };
    if (cur.length) (next[key] as unknown as string[]) = cur;
    else delete next[key];
    onChange(next);
  };

  const toggleConf = (b: FacetBucket) => {
    const on = predicate.min_conf === b.lo && predicate.max_conf === b.hi;
    const next = { ...predicate };
    if (on) { delete next.min_conf; delete next.max_conf; }
    else { next.min_conf = b.lo; next.max_conf = b.hi; }
    onChange(next);
  };

  if (!facets) {
    return (
      <div className="p-3 space-y-2.5" role="status" aria-label="loading facets">
        {Array.from({ length: 9 }, (_, i) => (
          <span key={i} className="skeleton h-3 block" style={{ width: `${[45, 80, 65, 72, 50, 85, 60, 40, 70][i]}%` }} />
        ))}
      </div>
    );
  }

  const filtered = Object.keys(predicate).length > 0;

  return (
    <div className="p-2">
      <div className="px-2 pt-1 pb-2.5 mb-2 border-b hairline">
        <div className="text-[20px] leading-none font-semibold text-ink tabular-nums">{num(facets.total)}</div>
        <div className="mt-1 flex items-center justify-between gap-2 text-[12px] text-ink-3">
          <span>objects match</span>
          {filtered && (
            <button type="button" onClick={() => onChange({})} className="link text-[12px]">Clear filters</button>
          )}
        </div>
      </div>

      <Group title="Class" buckets={facets.classes} selected={predicate.class_names ?? []}
        onToggle={(v) => toggle("class_names", v)} />
      <Group title="State" buckets={facets.states} selected={predicate.states ?? []}
        onToggle={(v) => toggle("states", v)} />
      <Group title="Source" buckets={facets.sources} selected={predicate.sources ?? []}
        onToggle={(v) => toggle("sources", v)} />
      <Group title="Tags" raw buckets={facets.tags} selected={predicate.tags ?? []}
        onToggle={(v) => toggle("tags", v)} />

      <div className="border-b hairline pb-2 mb-2">
        <div className="text-[12px] font-semibold text-ink-2 px-2 py-1.5">Confidence</div>
        {facets.conf.map((b) => (
          <Bar key={b.value} b={b} raw max={Math.max(...facets.conf.map((x) => x.count), 1)}
            active={predicate.min_conf === b.lo && predicate.max_conf === b.hi}
            onClick={() => toggleConf(b)} />
        ))}
      </div>

      {Object.entries(facets.scene ?? {}).map(([axis, buckets]) => (
        <Group key={axis} title={humanize(axis)} buckets={buckets}
          selected={(predicate[axis as keyof ExplorePredicate] as string[] | undefined) ?? []}
          onToggle={(v) => toggle(axis as keyof ExplorePredicate, v)} />
      ))}

      <Group title="City" raw buckets={facets.cities} selected={predicate.cities ?? []}
        onToggle={(v) => toggle("cities", v)} />
    </div>
  );
}
