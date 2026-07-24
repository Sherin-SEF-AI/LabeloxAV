"use client";

import type { ExplorePredicate, FacetBucket, Facets } from "@/lib/types";

// The facet sidebar. Every bar is both a statistic and a control: the number is how many you would get if you
// picked it, and clicking toggles that value into the predicate. Counts come from the server with each facet's
// own clause dropped, which is what keeps a list from collapsing to the one value you already selected.

function Bar({ b, max, active, onClick }: { b: FacetBucket; max: number; active: boolean; onClick: () => void }) {
  const pct = max > 0 ? Math.max(2, Math.round((b.count / max) * 100)) : 0;
  return (
    <button
      onClick={onClick}
      title={`${b.value} (${b.count})`}
      className={`relative w-full text-left px-1.5 py-[3px] font-mono text-[11px] overflow-hidden
        ${active ? "text-ink" : "text-ink-3 hover:text-ink-2"}`}
    >
      <div
        className={`absolute inset-y-0 left-0 ${active ? "bg-accent/30" : "bg-line/40"}`}
        style={{ width: `${pct}%` }}
      />
      <span className="relative flex justify-between gap-2">
        <span className="truncate">{b.value}</span>
        <span className="text-ink-3 shrink-0">{b.count}</span>
      </span>
    </button>
  );
}

function Group({ title, buckets, selected, onToggle }: {
  title: string; buckets: FacetBucket[]; selected: string[]; onToggle: (v: string) => void;
}) {
  if (!buckets?.length) return null;
  const max = Math.max(...buckets.map((b) => b.count), 1);
  return (
    <div className="border-b hairline pb-1.5 mb-1.5">
      <div className="font-mono text-[10px] uppercase text-ink-3 px-1.5 py-1">{title}</div>
      <div className="max-h-44 overflow-auto no-scrollbar">
        {buckets.map((b) => (
          <Bar key={String(b.value)} b={b} max={max}
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
    return <div className="p-3 font-mono text-[11px] text-ink-3">loading facets...</div>;
  }

  return (
    <div className="p-1.5">
      <div className="px-1.5 py-1 font-mono text-[11px] text-ink-2">
        {facets.total.toLocaleString()} <span className="text-ink-3">objects match</span>
      </div>

      <Group title="class" buckets={facets.classes} selected={predicate.class_names ?? []}
        onToggle={(v) => toggle("class_names", v)} />
      <Group title="state" buckets={facets.states} selected={predicate.states ?? []}
        onToggle={(v) => toggle("states", v)} />
      <Group title="source" buckets={facets.sources} selected={predicate.sources ?? []}
        onToggle={(v) => toggle("sources", v)} />
      <Group title="tags" buckets={facets.tags} selected={predicate.tags ?? []}
        onToggle={(v) => toggle("tags", v)} />

      <div className="border-b hairline pb-1.5 mb-1.5">
        <div className="font-mono text-[10px] uppercase text-ink-3 px-1.5 py-1">confidence</div>
        {facets.conf.map((b) => (
          <Bar key={b.value} b={b} max={Math.max(...facets.conf.map((x) => x.count), 1)}
            active={predicate.min_conf === b.lo && predicate.max_conf === b.hi}
            onClick={() => toggleConf(b)} />
        ))}
      </div>

      {Object.entries(facets.scene ?? {}).map(([axis, buckets]) => (
        <Group key={axis} title={axis.replace(/_/g, " ")} buckets={buckets}
          selected={(predicate[axis as keyof ExplorePredicate] as string[] | undefined) ?? []}
          onToggle={(v) => toggle(axis as keyof ExplorePredicate, v)} />
      ))}

      <Group title="city" buckets={facets.cities} selected={predicate.cities ?? []}
        onToggle={(v) => toggle("cities", v)} />
    </div>
  );
}
