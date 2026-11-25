"use client";

// M12 SIEVYX long-tail: find and close the tail. The ODD coverage-gap report shows scenario cells
// underrepresented against the target operational design domain; unsupervised discovery clusters object
// embeddings and surfaces the rarest groups to name; maneuver recognition classifies a track's trajectory.
// Operational Materialism: a gap bar is amber, a missing cell red, rarity is neutral-ranked.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Bar, RunButton, useRun, ErrLine } from "@/components/engine/prim";
import { getJSON, runJSON, parseInput } from "@/lib/engine";
import { humanize, num, pct } from "@/lib/format";

type Gaps = { total_samples: number; n_cells: number; n_gaps: number;
  gaps: { cell: string; target_frac: number; actual_frac: number; gap: number; count: number; missing: boolean }[] };
type Discover = { n: number; n_clusters?: number;
  clusters: { size: number; rarity: number; method: string; rep_ids: string[] }[] };
type Maneuver = Record<string, unknown>;

const EXAMPLE_COUNTS = JSON.stringify({ day_clear: 8000, night_clear: 1500, night_rain: 200, dusk_fog: 40 }, null, 2);
const EXAMPLE_ODD = JSON.stringify({ day_clear: 0.4, night_clear: 0.25, night_rain: 0.2, dusk_fog: 0.15 }, null, 2);
const EXAMPLE_TRAJ = JSON.stringify(
  [0, 1, 2, 3, 4, 5].map((t) => ({ t, x: t * 2, y: t < 3 ? 0 : (t - 2) * 1.5 })), null, 0);

export default function LongTailPage() {
  const [counts, setCounts] = useState("");
  const [odd, setOdd] = useState("");
  const gaps = useRun<Gaps>();

  const [traj, setTraj] = useState("");
  const man = useRun<Maneuver>();

  const disc = useRun<Discover>();

  return (
    <PageShell active="SIEVYX" title="Long tail"
      subtitle="Find the driving conditions and object types the data is short of">
      <div className="page-wrap grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* ODD gaps */}
        <Panel title="Missing driving conditions" hint="What you have, against the mix you need" className="lg:col-span-2">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr_1.3fr] gap-4">
            <label className="block">
              <span className="block text-[12px] text-ink-2 mb-1">Frames you have, per condition</span>
              <textarea value={counts} placeholder={EXAMPLE_COUNTS} onChange={(e) => setCounts(e.target.value)} rows={6}
                className="field w-full font-mono text-[12px]" />
            </label>
            <div>
              <label className="block">
                <span className="block text-[12px] text-ink-2 mb-1">Share you need, per condition (0 to 1)</span>
                <textarea value={odd} placeholder={EXAMPLE_ODD} onChange={(e) => setOdd(e.target.value)} rows={6}
                  className="field w-full font-mono text-[12px]" />
              </label>
              <div className="mt-2.5"><RunButton busy={gaps.busy}
                onClick={() => gaps.run(() => runJSON<Gaps>("/api/sievyx/odd/gaps",
                  { fleet_counts: parseInput(counts, "the frame counts"), target_odd: parseInput(odd, "the target shares") }))} label="Find gaps" /></div>
              <ErrLine err={gaps.err} />
            </div>
            <div>
              {gaps.out ? (
                <div className="space-y-1.5">
                  <div className="text-[12px] text-ink-3 mb-1">{num(gaps.out.n_gaps)} of {num(gaps.out.n_cells)} conditions are short</div>
                  {gaps.out.gaps.map((g) => (
                    <div key={g.cell} className="flex items-center gap-2 text-[12px]"
                      data-tip={g.missing ? "No frames at all for this condition" : `Have ${pct(g.actual_frac, 1)}, need ${pct(g.target_frac, 1)}`}>
                      <span className={`w-28 shrink-0 truncate ${g.missing ? "text-block" : "text-ink-2"}`}>{humanize(g.cell)}</span>
                      <Bar frac={g.gap * 5} tone={g.missing ? "block" : "warn"} />
                      <span className="w-16 shrink-0 text-right text-ink-3 tabular-nums">{pct(g.gap, 1)} short</span>
                    </div>
                  ))}
                  {!gaps.out.gaps.length && <div className="text-[12.5px] text-pass">You have enough of every condition.</div>}
                </div>
              ) : (
                <p className="section-desc">Paste both lists as JSON (the grey text shows the shape) and the shortfall per condition appears here.</p>
              )}
            </div>
          </div>
        </Panel>

        {/* discovery */}
        <Panel title="Rare object groups" hint="Groups of similar objects that are seldom seen">
          <p className="section-desc mb-3">Groups labelled objects by how they look and lists the smallest groups: candidates for a new class or more data.</p>
          <RunButton busy={disc.busy}
            onClick={() => disc.run(() => getJSON<Discover>("/api/sievyx/discover?limit=400&min_size=3"))} label="Find rare groups" />
          <ErrLine err={disc.err} />
          {disc.out && (
            <div className="mt-3">
              <KV k="Objects looked at" v={num(disc.out.n)} tone="ink-3" />
              <KV k="Groups found" v={num(disc.out.n_clusters ?? disc.out.clusters.length)} tone="ink" />
              <div className="space-y-1.5 mt-2">
                {disc.out.clusters.map((c, i) => (
                  <div key={i} className="flex items-center gap-2 text-[12px]" data-tip={`Rarity ${c.rarity.toFixed(2)} (1 is rarest)`}>
                    <span className="w-24 shrink-0 text-ink-2">{num(c.size)} objects</span>
                    <Bar frac={c.rarity} tone="accent" />
                    <span className="w-12 shrink-0 text-right text-ink-3 tabular-nums">{c.rarity.toFixed(2)}</span>
                  </div>
                ))}
                {!disc.out.clusters.length && <div className="text-[12.5px] text-ink-3">Not enough compared objects to group yet.</div>}
              </div>
            </div>
          )}
        </Panel>

        {/* maneuver */}
        <Panel title="Name a manoeuvre" hint="From a track's path, in metres from the ego vehicle">
          <label className="block">
            <span className="block text-[12px] text-ink-2 mb-1">Path points (t, x, y) as JSON</span>
            <textarea value={traj} placeholder={EXAMPLE_TRAJ} onChange={(e) => setTraj(e.target.value)} rows={4}
              className="field w-full font-mono text-[12px]" />
          </label>
          <div className="mt-2.5"><RunButton busy={man.busy}
            onClick={() => man.run(() => runJSON<Maneuver>("/api/sievyx/maneuver", { trajectory: parseInput(traj, "the path") }))}
            label="Name it" /></div>
          <ErrLine err={man.err} />
          {man.out && (
            <div className="mt-3 space-y-0.5">
              {Object.entries(man.out).map(([k, v]) => (
                <KV key={k} k={humanize(k)} v={typeof v === "number" ? v.toFixed(3) : humanize(String(v))}
                  tone={k === "maneuver" || k === "label" ? "accent" : "ink-3"} />
              ))}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
