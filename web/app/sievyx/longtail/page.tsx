"use client";

// M12 SIEVYX long-tail: find and close the tail. The ODD coverage-gap report shows scenario cells
// underrepresented against the target operational design domain; unsupervised discovery clusters object
// embeddings and surfaces the rarest groups to name; maneuver recognition classifies a track's trajectory.
// Operational Materialism: a gap bar is amber, a missing cell red, rarity is neutral-ranked.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Bar, RunButton, useRun, ErrLine } from "@/components/engine/prim";
import { getJSON, runJSON } from "@/lib/engine";

type Gaps = { total_samples: number; n_cells: number; n_gaps: number;
  gaps: { cell: string; target_frac: number; actual_frac: number; gap: number; count: number; missing: boolean }[] };
type Discover = { n: number; n_clusters?: number;
  clusters: { size: number; rarity: number; method: string; rep_ids: string[] }[] };
type Maneuver = Record<string, unknown>;

const DEMO_COUNTS = JSON.stringify({ day_clear: 8000, night_clear: 1500, night_rain: 200, dusk_fog: 40 }, null, 2);
const DEMO_ODD = JSON.stringify({ day_clear: 0.4, night_clear: 0.25, night_rain: 0.2, dusk_fog: 0.15 }, null, 2);
const DEMO_TRAJ = JSON.stringify(
  [0, 1, 2, 3, 4, 5].map((t) => ({ t, x: t * 2, y: t < 3 ? 0 : (t - 2) * 1.5 })), null, 0);

export default function LongTailPage() {
  const [counts, setCounts] = useState(DEMO_COUNTS);
  const [odd, setOdd] = useState(DEMO_ODD);
  const gaps = useRun<Gaps>();

  const [traj, setTraj] = useState(DEMO_TRAJ);
  const man = useRun<Maneuver>();

  const disc = useRun<Discover>();

  return (
    <PageShell active="SIEVYX" title="Long-tail: ODD gaps, discovery, maneuvers"
      subtitle="what the fleet is missing, the rarest clusters, and trajectory maneuvers">
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-6xl">
        {/* ODD gaps */}
        <Panel title="ODD coverage gaps" hint="fleet vs target domain" className="lg:col-span-2">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_1.4fr] gap-3">
            <div>
              <div className="font-mono text-[10px] text-ink-3 mb-1">fleet counts</div>
              <textarea value={counts} onChange={(e) => setCounts(e.target.value)} rows={6}
                className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
            </div>
            <div>
              <div className="font-mono text-[10px] text-ink-3 mb-1">target ODD fractions</div>
              <textarea value={odd} onChange={(e) => setOdd(e.target.value)} rows={6}
                className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
              <div className="mt-2"><RunButton busy={gaps.busy}
                onClick={() => gaps.run(() => runJSON<Gaps>("/api/sievyx/odd/gaps",
                  { fleet_counts: JSON.parse(counts), target_odd: JSON.parse(odd) }))} label="report gaps" /></div>
              <ErrLine err={gaps.err} />
            </div>
            <div>
              {gaps.out && (
                <div className="space-y-1">
                  <div className="font-mono text-[10px] text-ink-3 mb-1">{gaps.out.n_gaps} gaps of {gaps.out.n_cells} cells</div>
                  {gaps.out.gaps.map((g) => (
                    <div key={g.cell} className="flex items-center gap-2 font-mono text-[10px]">
                      <span className={`w-24 shrink-0 truncate ${g.missing ? "text-block" : "text-ink-2"}`}>{g.cell}</span>
                      <Bar frac={g.gap * 5} tone={g.missing ? "block" : "warn"} />
                      <span className="w-14 shrink-0 text-right text-ink-3">{(g.gap * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                  {!gaps.out.gaps.length && <div className="font-mono text-[11px] text-pass">fleet covers the target ODD</div>}
                </div>
              )}
            </div>
          </div>
        </Panel>

        {/* discovery */}
        <Panel title="rare-cluster discovery" hint="live object embeddings">
          <RunButton busy={disc.busy}
            onClick={() => disc.run(() => getJSON<Discover>("/api/sievyx/discover?limit=400&min_size=3"))} label="discover" />
          <ErrLine err={disc.err} />
          {disc.out && (
            <div className="mt-3">
              <KV k="objects sampled" v={disc.out.n} tone="ink-3" />
              <KV k="clusters" v={disc.out.n_clusters ?? disc.out.clusters.length} tone="ink" />
              <div className="space-y-1 mt-2">
                {disc.out.clusters.map((c, i) => (
                  <div key={i} className="flex items-center gap-2 font-mono text-[10px]">
                    <span className="w-16 shrink-0 text-ink-2">n={c.size}</span>
                    <Bar frac={c.rarity} tone="accent" />
                    <span className="w-16 shrink-0 text-right text-ink-3">{c.rarity.toFixed(2)}</span>
                  </div>
                ))}
                {!disc.out.clusters.length && <div className="font-mono text-[11px] text-ink-3">not enough embedded objects to cluster</div>}
              </div>
            </div>
          )}
        </Panel>

        {/* maneuver */}
        <Panel title="maneuver recognition" hint="trajectory in ego metres">
          <textarea value={traj} onChange={(e) => setTraj(e.target.value)} rows={4}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={man.busy}
            onClick={() => man.run(() => runJSON<Maneuver>("/api/sievyx/maneuver", { trajectory: JSON.parse(traj) }))}
            label="recognize" /></div>
          <ErrLine err={man.err} />
          {man.out && (
            <div className="mt-3 space-y-0.5">
              {Object.entries(man.out).map(([k, v]) => (
                <KV key={k} k={k} v={typeof v === "number" ? v.toFixed(3) : String(v)}
                  tone={k === "maneuver" || k === "label" ? "accent" : "ink-3"} />
              ))}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
