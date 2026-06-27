"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  analyticsApi,
  type ClassRow,
  type ClusterMap,
  type DedupRate,
  type GeoPoint,
  type GrowthPoint,
  type Overview,
  type PiiCoverage,
  type ReviewAgreement,
  type ScenarioCoverage,
} from "@/lib/analytics-api";
import TopNav from "@/components/TopNav";
import { api } from "@/lib/api";
import type { Confusions } from "@/lib/types";

// The sales sheet: corpus totals, class distribution and long-tail coverage, label-source mix
// (auto-accepted vs human-touched), scenario coverage, and the review-agreement loop signal.
// Color only encodes state; bars are plain divs (no chart library).

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="panel px-3 py-3">
      <div className="font-mono text-[11px] uppercase text-ink-3">{label}</div>
      <div className="font-mono text-2xl text-ink mt-1">{value}</div>
      {sub && <div className="font-mono text-[11px] text-ink-3 mt-0.5">{sub}</div>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        {title}
      </div>
      <div className="p-3">{children}</div>
    </section>
  );
}

export default function AnalyticsPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [classes, setClasses] = useState<ClassRow[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioCoverage[]>([]);
  const [agreement, setAgreement] = useState<ReviewAgreement | null>(null);
  const [geo, setGeo] = useState<GeoPoint[]>([]);
  const [pii, setPii] = useState<PiiCoverage | null>(null);
  const [confusions, setConfusions] = useState<Confusions | null>(null);
  const [confBy, setConfBy] = useState<"class" | "camera" | "city">("class");
  const [scenes, setScenes] = useState<Record<string, Record<string, number>> | null>(null);
  const [dedup, setDedup] = useState<DedupRate | null>(null);
  const [growth, setGrowth] = useState<GrowthPoint[]>([]);
  const [cluster, setCluster] = useState<ClusterMap | null>(null);

  useEffect(() => {
    api.confusions(confBy).then(setConfusions).catch(() => setConfusions(null));
  }, [confBy]);

  useEffect(() => {
    analyticsApi.sceneSplits().then(setScenes).catch(() => {});
    analyticsApi.dedupRate().then(setDedup).catch(() => {});
    analyticsApi.growth().then(setGrowth).catch(() => {});
    analyticsApi.clusterMap(1200).then(setCluster).catch(() => {});
  }, []);

  const exportReport = async () => {
    const r = await fetch("/api/analytics/report");
    const blob = new Blob([JSON.stringify(await r.json(), null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "labelox-quality-sheet.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const [ov, cls, scn, agr, gp, pi] = await Promise.all([
          analyticsApi.overview(),
          analyticsApi.classes(),
          analyticsApi.scenarios(),
          analyticsApi.reviewAgreement(),
          analyticsApi.geo(),
          analyticsApi.pii().catch(() => null),
        ]);
        setOverview(ov);
        setClasses(cls);
        setScenarios(scn);
        setAgreement(agr);
        setGeo(gp);
        setPii(pi);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const present = classes.filter((c) => c.count > 0);
  const maxCount = present.reduce((m, c) => Math.max(m, c.count), 1);
  const maxScenario = scenarios.reduce((m, s) => Math.max(m, s.count), 1);
  const mix = overview?.source_mix;

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="ANALYTICS" right={
        <>
          <button onClick={exportReport} className="border border-line px-2 py-0.5 hover:border-accent" title="export the quality sheet as JSON">export report</button>
          <span className="border border-line px-2 py-0.5">{overview ? `${overview.objects} objects` : "..."}</span>
          <span className={`w-2 h-2 rounded-full ${loading ? "bg-warn" : "bg-pass"}`} />
        </>
      } />

      <main className="flex-1 overflow-auto p-4 space-y-4">
        {/* Top stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <Stat label="frames" value={overview ? String(overview.frames) : "-"} />
          <Stat label="objects" value={overview ? String(overview.objects) : "-"} />
          <Stat label="tracks" value={overview ? String(overview.tracks) : "-"} />
          <Stat label="scenarios" value={overview ? String(overview.scenarios) : "-"} />
          <Stat
            label="auto-accepted"
            value={overview ? `${overview.auto_accepted_pct}%` : "-"}
            sub={overview ? `${overview.human_touched_pct}% human-touched` : undefined}
          />
          <Stat
            label="long-tail coverage"
            value={
              overview
                ? `${overview.long_tail.covered_classes}/${overview.long_tail.total_classes}`
                : "-"
            }
            sub={overview ? `${overview.long_tail.coverage_pct}% of ontology` : undefined}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Class distribution */}
          <Section title={`class distribution (${present.length} classes present)`}>
            <div className="space-y-1">
              {present.map((c) => (
                <div key={c.class_id} className="flex items-center gap-2">
                  <div
                    className={`w-32 shrink-0 font-mono text-[11px] truncate ${
                      c.india ? "text-accent" : "text-ink-2"
                    }`}
                    title={`${c.name}${c.india ? " (india)" : ""}`}
                  >
                    {c.india ? "* " : ""}
                    {c.name}
                  </div>
                  <div className="flex-1 h-3 bg-line relative">
                    <div
                      className={`absolute left-0 top-0 h-full ${
                        c.india ? "bg-accent" : "bg-info"
                      }`}
                      style={{ width: `${(c.count / maxCount) * 100}%` }}
                    />
                  </div>
                  <div className="w-12 text-right font-mono text-[11px] text-ink-2">{c.count}</div>
                </div>
              ))}
              {!present.length && (
                <div className="font-mono text-xs text-ink-3 py-4 text-center">no objects yet</div>
              )}
            </div>
          </Section>

          {/* Label-source mix */}
          <Section title="label-source mix">
            {mix && mix.total > 0 ? (
              <div className="space-y-3">
                <div className="flex h-6 w-full overflow-hidden border hairline">
                  <div
                    className="bg-pass flex items-center justify-center"
                    style={{ width: `${mix.auto_accepted_pct}%` }}
                    title={`auto ${mix.auto_accepted_pct}%`}
                  />
                  <div
                    className="bg-warn flex items-center justify-center"
                    style={{ width: `${mix.human_touched_pct}%` }}
                    title={`human ${mix.human_touched_pct}%`}
                  />
                </div>
                <div className="flex items-center gap-4 font-mono text-xs">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 bg-pass inline-block" /> auto-accepted{" "}
                    {mix.auto_accepted_pct}%
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 bg-warn inline-block" /> human-touched{" "}
                    {mix.human_touched_pct}%
                  </span>
                </div>
                <div className="font-mono text-[11px] text-ink-3 grid grid-cols-2 gap-x-6 gap-y-0.5">
                  {Object.entries(mix.by_state).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span>state: {k}</span>
                      <span className="text-ink-2">{v}</span>
                    </div>
                  ))}
                  {Object.entries(mix.by_source).map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span>source: {k}</span>
                      <span className="text-ink-2">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="font-mono text-xs text-ink-3 py-4 text-center">no objects yet</div>
            )}
          </Section>

          {/* Scenario coverage */}
          <Section title="scenario coverage">
            <div className="space-y-1">
              {scenarios.map((s) => (
                <div key={s.type} className="flex items-center gap-2">
                  <div className="w-32 shrink-0 font-mono text-[11px] text-ink-2 truncate">
                    {s.type}
                  </div>
                  <div className="flex-1 h-3 bg-line relative">
                    <div
                      className="absolute left-0 top-0 h-full bg-accent"
                      style={{ width: `${(s.count / maxScenario) * 100}%` }}
                    />
                  </div>
                  <div className="w-12 text-right font-mono text-[11px] text-ink-2">{s.count}</div>
                  <div className="w-16 text-right font-mono text-[11px] text-ink-3">
                    crit {s.mean_criticality.toFixed(2)}
                  </div>
                </div>
              ))}
              {!scenarios.length && (
                <div className="font-mono text-xs text-ink-3 py-4 text-center">
                  no scenarios mined
                </div>
              )}
            </div>
          </Section>

          {/* Review agreement (the loop signal) */}
          <Section title="review agreement (loop signal)">
            {agreement && agreement.total_reviews > 0 ? (
              <div className="space-y-3">
                <div className="flex h-6 w-full overflow-hidden border hairline">
                  <div
                    className="bg-pass"
                    style={{ width: `${agreement.confirmed_pct}%` }}
                    title={`confirmed ${agreement.confirmed_pct}%`}
                  />
                  <div
                    className="bg-block"
                    style={{ width: `${agreement.reclassified_pct}%` }}
                    title={`reclassified ${agreement.reclassified_pct}%`}
                  />
                </div>
                <div className="flex items-center gap-4 font-mono text-xs">
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 bg-pass inline-block" /> confirmed{" "}
                    {agreement.confirmed_pct}%
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-2.5 h-2.5 bg-block inline-block" /> reclassified{" "}
                    {agreement.reclassified_pct}%
                  </span>
                </div>
                <div className="font-mono text-[11px] text-ink-3">
                  {agreement.total_reviews} reviews, mean {agreement.mean_time_spent_ms}ms each
                </div>
                {agreement.per_class.length > 0 && (
                  <div className="font-mono text-[11px] text-ink-3 space-y-0.5 pt-1 border-t hairline">
                    {agreement.per_class.slice(0, 8).map((p) => (
                      <div key={p.class_id} className="flex justify-between">
                        <span className="text-ink-2">{p.name}</span>
                        <span>
                          {p.confirmed}/{p.reviews} confirmed ({p.confirmed_pct}%)
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="font-mono text-xs text-ink-3 py-4 text-center">no reviews yet</div>
            )}
          </Section>

          <Section title={`model confusions — learn from corrections (${confusions?.total_corrections ?? 0} total)`}>
            <div className="flex items-center gap-2 mb-2 font-mono text-[11px]">
              <span className="text-ink-3">group by:</span>
              {(["class", "camera", "city"] as const).map((b) => (
                <button key={b} onClick={() => setConfBy(b)}
                  className={`border px-2 py-0.5 ${confBy === b ? "border-accent text-accent" : "border-line text-ink-3 hover:text-ink-2"}`}>{b}</button>
              ))}
            </div>
            {confusions && confusions.confusions.length ? (
              <table className="w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">model said</th><th>human corrected to</th>
                    {confBy !== "class" && <th>{confBy}</th>}<th className="text-right">count</th>
                  </tr>
                </thead>
                <tbody>
                  {confusions.confusions.map((c, i) => (
                    <tr key={i} className="border-b hairline">
                      <td className="py-1 text-block">{c.old_class}</td>
                      <td className="text-pass">{c.new_class}</td>
                      {confBy !== "class" && <td className="text-ink-3">{c.group ?? "-"}</td>}
                      <td className="text-right text-ink">{c.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="font-mono text-xs text-ink-3 py-4 text-center">no corrections recorded yet</div>
            )}
          </Section>
        </div>

        {/* DPDPA / PII anonymization evidence (Gate A) */}
        <Section title="PII anonymization (DPDPA gate A)">
          {pii && pii.total_frames > 0 ? (
            <div className="space-y-3">
              <div className="flex h-6 w-full overflow-hidden border hairline">
                <div
                  className="bg-pass"
                  style={{ width: `${pii.coverage_pct}%` }}
                  title={`anonymized ${pii.coverage_pct}%`}
                />
                <div
                  className="bg-block"
                  style={{ width: `${100 - pii.coverage_pct}%` }}
                  title={`unprotected ${(100 - pii.coverage_pct).toFixed(1)}%`}
                />
              </div>
              <div className="flex items-center gap-4 font-mono text-xs">
                <span className="flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 bg-pass inline-block" /> anonymized{" "}
                  {pii.coverage_pct}% ({pii.frames_anonymized}/{pii.total_frames})
                </span>
                <span className="text-ink-3">
                  {pii.faces_blurred} faces, {pii.plates_blurred} plates blurred
                </span>
              </div>
              <div className="font-mono text-[11px] text-ink-3 space-y-0.5 pt-1 border-t hairline">
                {Object.entries(pii.method_versions).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-ink-2 truncate">{k}</span>
                    <span>{v} frames</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">
              no anonymized frames yet
            </div>
          )}
        </Section>

        {/* Geo capture density */}
        <Section title={`geo capture density (${geo.length} points)`}>
          {geo.length > 0 ? (
            <div className="font-mono text-[11px] text-ink-3 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-0.5 max-h-48 overflow-auto">
              {geo.slice(0, 200).map((p, i) => (
                <div key={i} className="flex justify-between">
                  <span className="text-ink-2">{p.lat.toFixed(5)}</span>
                  <span>{p.lon.toFixed(5)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="font-mono text-xs text-ink-3 py-4 text-center">no gnss fixes</div>
          )}
        </Section>

        {/* Data Intelligence Layer (M1.7) */}
        <Section title="scene splits (zero-shot SigLIP 2)">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {scenes && Object.entries(scenes).map(([axis, vals]) => {
              const tot = Object.values(vals).reduce((a, b) => a + b, 0) || 1;
              return (
                <div key={axis}>
                  <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">{axis.replace("_", " ")}</div>
                  {Object.entries(vals).map(([v, n]) => (
                    <div key={v} className="font-mono text-[11px] flex items-center gap-1.5 mb-0.5">
                      <span className="w-16 text-ink-2 truncate">{v}</span>
                      <div className="flex-1 h-2 bg-bg-2"><div className="h-2 bg-accent" style={{ width: `${(100 * n) / tot}%` }} /></div>
                      <span className="text-ink-3 w-8 text-right">{n}</span>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>
        </Section>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Section title="duplicate rate (M1.1 dedup)">
            <div className="p-2 font-mono">
              <div className="text-2xl text-ink">{dedup ? `${(dedup.rate * 100).toFixed(1)}%` : "-"}</div>
              <div className="text-[11px] text-ink-3 mt-1">{dedup ? `${dedup.redundant} redundant / ${dedup.total} frames, ${dedup.groups} groups` : ""}</div>
            </div>
          </Section>
          <Section title="dataset growth (frames/day)">
            <div className="flex items-end gap-0.5 h-24 p-2">
              {growth.slice(-40).map((g, i) => {
                const max = Math.max(...growth.map((x) => x.frames), 1);
                return <div key={i} title={`${g.date}: ${g.frames}`} className="flex-1 bg-info/70 min-w-[2px]" style={{ height: `${(100 * g.frames) / max}%` }} />;
              })}
            </div>
            <div className="font-mono text-[10px] text-ink-3 px-2">{growth.length ? `${growth[growth.length - 1].cumulative} frames total` : ""}</div>
          </Section>
          <Section title="long-tail (rare classes)">
            <div className="p-2 font-mono text-[11px] text-ink-3">
              {overview ? <span>{overview.long_tail.coverage_pct}% coverage ({overview.long_tail.covered_classes}/{overview.long_tail.total_classes} classes)</span> : "-"}
            </div>
          </Section>
        </div>

        <Section title={`embedding cluster map (UMAP of DINOv3, ${cluster?.n ?? 0} frames, ${cluster?.clusters ?? 0} clusters)`}>
          {cluster && cluster.points.length ? (
            <ClusterScatter cluster={cluster} onPick={(fid) => router.push(`/frame/${fid}`)} />
          ) : (
            <div className="font-mono text-xs text-ink-3 py-8 text-center">computing projection…</div>
          )}
        </Section>
      </main>
    </div>
  );
}

function ClusterScatter({ cluster, onPick }: { cluster: ClusterMap; onPick: (fid: string) => void }) {
  const xs = cluster.points.map((p) => p.x);
  const ys = cluster.points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const W = 900, H = 360, pad = 10;
  const sx = (x: number) => pad + ((x - minX) / (maxX - minX || 1)) * (W - 2 * pad);
  const sy = (y: number) => pad + ((y - minY) / (maxY - minY || 1)) * (H - 2 * pad);
  const palette = ["#FF7A2F", "#56D364", "#58A6FF", "#E3B341", "#F85149", "#A0A6AD", "#bb86fc", "#03dac6"];
  const color = (c: number) => (c < 0 ? "#3a3f46" : palette[c % palette.length]);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 380 }}>
      {cluster.points.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2.5} fill={color(p.cluster)} opacity={0.8}
          onClick={() => onPick(p.frame_id)} style={{ cursor: "pointer" }}>
          <title>{`${p.time_of_day ?? "?"} / ${p.road_type ?? "?"} (cluster ${p.cluster})`}</title>
        </circle>
      ))}
    </svg>
  );
}
