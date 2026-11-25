"use client";

import { useCallback, useEffect, useState } from "react";

import { type Placeholder, placeholder } from "@/lib/emptyState";
import { Segmented, Stat } from "@/components/ui/kit";
import { compact, duration, humanize, num, pct, plural } from "@/lib/format";
import { useRouter } from "next/navigation";
import {
  analyticsApi,
  type ClassRow,
  type ClusterMap,
  type DedupRate,
  type GeoRelease,
  type GrowthPoint,
  type Overview,
  type PiiCoverage,
  type ReviewAgreement,
  type ScenarioCoverage,
} from "@/lib/analytics-api";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import EvalDrilldown from "@/components/explore/EvalDrilldown";
import RecallHonesty from "@/components/explore/RecallHonesty";
import { api, humanizeError } from "@/lib/api";
import type { ProductivityReport } from "@/lib/types";
import type { Confusions } from "@/lib/types";
import { apiGet } from "@/lib/api";

// The sales sheet: corpus totals, class distribution and long-tail coverage, label-source mix
// (auto-accepted vs human-touched), scenario coverage, and the review-agreement loop signal.
// Color only encodes state; bars are plain divs (no chart library).

// Draws whichever of the three states `placeholder` chose. A wait shows placeholder rows, an emptiness is
// stated plainly, and data draws nothing here at all.
function Blank({ p }: { p: Placeholder }) {
  if (!p) return null;
  if (p.kind === "loading") {
    return (
      <div className="space-y-2.5 py-1" role="status" aria-label="loading">
        {[62, 48, 70, 40].map((w, i) => <span key={i} className="skeleton h-3 block" style={{ width: `${w}%` }} />)}
      </div>
    );
  }
  return <div className="text-[12.5px] py-4 text-center text-ink-3">{p.text}</div>;
}

// A tile value: a skeleton while the overview is in flight rather than "..." or "-", both of which read as
// a measured nothing.
function tile(loading: boolean, v: React.ReactNode | null | undefined): React.ReactNode {
  if (v !== null && v !== undefined) return v;
  return loading ? <span className="skeleton inline-block h-[22px] w-24 align-middle" aria-label="loading" /> : "-";
}

function Section({ title, right, children, className = "" }: {
  title: string; right?: React.ReactNode; children: React.ReactNode; className?: string;
}) {
  return (
    <section className={`panel min-w-0 ${className}`}>
      <div className="panel-head">
        <h2 className="section-title">{title}</h2>
        {right && <div className="ml-auto flex items-center gap-2 font-normal text-[12px] text-ink-3">{right}</div>}
      </div>
      <div className="p-4">{children}</div>
    </section>
  );
}

const CONFUSION_ROWS = 15;

export default function AnalyticsPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<unknown>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [classes, setClasses] = useState<ClassRow[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioCoverage[]>([]);
  const [agreement, setAgreement] = useState<ReviewAgreement | null>(null);
  const [geo, setGeo] = useState<GeoRelease | null>(null);
  const [geoBusy, setGeoBusy] = useState(false);
  const [geoErr, setGeoErr] = useState<string | null>(null);
  const [pii, setPii] = useState<PiiCoverage | null>(null);
  const [piiFailed, setPiiFailed] = useState(false);
  const [confusions, setConfusions] = useState<Confusions | null>(null);
  const [confBy, setConfBy] = useState<"class" | "camera" | "city">("class");
  const [scenes, setScenes] = useState<Record<string, Record<string, number>> | null>(null);
  const [dedup, setDedup] = useState<DedupRate | null>(null);
  const [growth, setGrowth] = useState<GrowthPoint[]>([]);
  const [cluster, setCluster] = useState<ClusterMap | null>(null);
  const [prod, setProd] = useState<ProductivityReport | null>(null);
  const [confLoading, setConfLoading] = useState(true);
  const [allConfusions, setAllConfusions] = useState(false);

  useEffect(() => {
    setConfLoading(true);
    api.confusions(confBy).then(setConfusions).catch(() => setConfusions(null)).finally(() => setConfLoading(false));
  }, [confBy]);

  useEffect(() => {
    analyticsApi.sceneSplits().then(setScenes).catch(() => {});
    analyticsApi.dedupRate().then(setDedup).catch(() => {});
    analyticsApi.growth().then(setGrowth).catch(() => {});
    analyticsApi.clusterMap(1200).then(setCluster).catch(() => {});
    api.analyticsProductivity().then(setProd).catch(() => {});
  }, []);

  const exportReport = async () => {
    const report = await apiGet<unknown>("/api/analytics/report");
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "labelox-quality-sheet.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      // Capture density is not fetched here. Every release of it is charged against the privacy budget, so
      // loading it with the page spent the whole budget on ten visits to a dashboard, after which the panel
      // said "no gnss fixes" over a corpus with tens of thousands of them. It is released on a click below.
      const [ov, cls, scn, agr, pi] = await Promise.all([
        analyticsApi.overview(),
        analyticsApi.classes(),
        analyticsApi.scenarios(),
        analyticsApi.reviewAgreement(),
        analyticsApi.pii().catch(() => { setPiiFailed(true); return null; }),
      ]);
      setOverview(ov);
      setClasses(cls);
      setScenarios(scn);
      setAgreement(agr);
      setPii(pi);
    } catch (e) {
      // Without this the page had no error state at all: if any of the five required calls rejected, none
      // of the setters ran, loading went false, and every panel flipped from "loading" to its empty copy.
      // A dead backend rendered "no objects yet" over a 570k-object corpus, indefinitely.
      setErr(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  const releaseGeo = async () => {
    setGeoBusy(true);
    setGeoErr(null);
    try {
      setGeo(await analyticsApi.geo());
    } catch (e) {
      // A spent budget arrives as a 429 whose detail says so; that sentence is the panel's content.
      setGeoErr(humanizeError(e));
    } finally {
      setGeoBusy(false);
    }
  };

  const present = classes.filter((c) => c.count > 0);
  const maxCount = present.reduce((m, c) => Math.max(m, c.count), 1);
  const maxScenario = scenarios.reduce((m, s) => Math.max(m, s.count), 1);
  const mix = overview?.source_mix;

  const confRows = confusions?.confusions ?? [];
  const shownConf = allConfusions ? confRows : confRows.slice(0, CONFUSION_ROWS);

  return (
    <PageShell
      active="ANALYTICS"
      subtitle="What the corpus holds, how it was labelled, and how far the labels and the model can be trusted."
      primaryAction={
        <button onClick={exportReport} className="btn" data-tip="Download the quality sheet as a JSON file">Export report</button>
      }
      right={
        <span className="inline-flex items-center gap-2 text-[12px] text-ink-2">
          <span className={`w-2 h-2 rounded-full ${loading ? "bg-warn" : "bg-pass"}`} aria-hidden />
          {overview ? `${compact(overview.objects)} objects` : "Loading..."}
        </span>
      }
    >
      <div className="page-wrap space-y-4">
        {/* A failure says so and offers a retry. Every panel below renders its empty copy when a fetch
            fails, so without this banner a dead backend is indistinguishable from an empty corpus. */}
        {err != null && <LoadState error={err} onRetry={() => void load()} />}

        {/* Top stat cards */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          <Stat label="Frames" value={tile(loading, overview ? num(overview.frames) : null)} />
          <Stat label="Objects" value={tile(loading, overview ? num(overview.objects) : null)} />
          <Stat label="Tracks" value={tile(loading, overview ? num(overview.tracks) : null)} />
          <Stat label="Scenarios" value={tile(loading, overview ? num(overview.scenarios) : null)} />
          <Stat
            label="Auto-accepted"
            value={tile(loading, overview ? `${overview.auto_accepted_pct}%` : null)}
            sub={overview ? `${overview.human_touched_pct}% touched by a person` : undefined}
          />
          <Stat
            label="Long-tail coverage"
            value={tile(loading, overview
              ? `${overview.long_tail.covered_classes} / ${overview.long_tail.total_classes}`
              : null)}
            sub={overview ? `${overview.long_tail.coverage_pct}% of classes have examples` : undefined}
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* Class distribution */}
          <Section title="Class distribution"
            right={present.length ? <>{plural(present.length, "class", "classes")} present · <span className="text-accent-2">highlighted</span> = India-specific</> : null}>
            <div className="space-y-1.5 max-h-[420px] overflow-y-auto pr-1">
              {present.map((c) => (
                <div key={c.class_id} className="flex items-center gap-3">
                  <div
                    className={`w-36 shrink-0 text-[12.5px] truncate ${
                      c.india ? "text-accent-2" : "text-ink-2"
                    }`}
                    title={`${c.name}${c.india ? " (India-specific)" : ""}`}
                  >
                    {humanize(c.name)}
                  </div>
                  <div className="flex-1 h-2.5 bg-line rounded-sm relative overflow-hidden">
                    <div
                      className={`absolute left-0 top-0 h-full ${
                        c.india ? "bg-accent" : "bg-info"
                      }`}
                      style={{ width: `${(c.count / maxCount) * 100}%` }}
                    />
                  </div>
                  <div className="w-16 text-right text-[12px] text-ink-2 tabular-nums">{compact(c.count)}</div>
                </div>
              ))}
              <Blank p={placeholder(loading, present.length, "No objects yet.")} />
            </div>
          </Section>

          {/* Label-source mix */}
          <Section title="Where the labels came from">
            {mix && mix.total > 0 ? (
              <div className="space-y-4">
                <div className="flex h-5 w-full overflow-hidden rounded-sm bg-line">
                  <div
                    className="bg-pass"
                    style={{ width: `${mix.auto_accepted_pct}%` }}
                    title={`auto-accepted ${mix.auto_accepted_pct}%`}
                  />
                  <div
                    className="bg-warn"
                    style={{ width: `${mix.human_touched_pct}%` }}
                    title={`touched by a person ${mix.human_touched_pct}%`}
                  />
                </div>
                <div className="flex flex-wrap items-center gap-5 text-[12.5px]">
                  <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-pass" />Auto-accepted {mix.auto_accepted_pct}%</span>
                  <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-warn" />Touched by a person {mix.human_touched_pct}%</span>
                  <span className="inline-flex items-center gap-1.5 text-ink-3"><span className="w-2.5 h-2.5 rounded-sm bg-line" />Neither yet {Math.max(0, 100 - mix.auto_accepted_pct - mix.human_touched_pct).toFixed(2)}%</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-8 gap-y-3 text-[12.5px]">
                  <div>
                    <div className="label-caps mb-1">By review state</div>
                    {Object.entries(mix.by_state).map(([k, v]) => (
                      <div key={k} className="flex justify-between py-0.5">
                        <span className="text-ink-3">{humanize(k)}</span>
                        <span className="text-ink-2 tabular-nums">{num(v)}</span>
                      </div>
                    ))}
                  </div>
                  <div>
                    <div className="label-caps mb-1">By source</div>
                    {Object.entries(mix.by_source).map(([k, v]) => (
                      <div key={k} className="flex justify-between py-0.5">
                        <span className="text-ink-3">{humanize(k)}</span>
                        <span className="text-ink-2 tabular-nums">{num(v)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <Blank p={placeholder(loading, mix?.total ?? 0, "No objects yet.")} />
            )}
          </Section>

          {/* Scenario coverage */}
          <Section title="Scenario coverage" right={scenarios.length ? "count · mean criticality" : null}>
            <div className="space-y-1.5">
              {scenarios.map((s) => (
                <div key={s.type} className="flex items-center gap-3">
                  <div className="w-36 shrink-0 text-[12.5px] text-ink-2 truncate">
                    {humanize(s.type)}
                  </div>
                  <div className="flex-1 h-2.5 bg-line rounded-sm relative overflow-hidden">
                    <div
                      className="absolute left-0 top-0 h-full bg-accent"
                      style={{ width: `${(s.count / maxScenario) * 100}%` }}
                    />
                  </div>
                  <div className="w-14 text-right text-[12px] text-ink-2 tabular-nums">{compact(s.count)}</div>
                  <div className="w-12 text-right text-[12px] text-ink-3 tabular-nums" data-tip="Mean criticality">
                    {s.mean_criticality.toFixed(2)}
                  </div>
                </div>
              ))}
              <Blank p={placeholder(loading, scenarios.length, "No scenarios mined yet.")} />
            </div>
          </Section>

          {/* Review agreement (the loop signal) */}
          <Section title="Review agreement" right="Did reviewers keep the proposed class?">
            {agreement && agreement.total_reviews > 0 ? (
              <div className="space-y-4">
                <div className="flex h-5 w-full overflow-hidden rounded-sm">
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
                <div className="flex flex-wrap items-center gap-5 text-[12.5px]">
                  <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-pass" />Confirmed {agreement.confirmed_pct}%</span>
                  <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-block" />Reclassified {agreement.reclassified_pct}%</span>
                  <span className="text-ink-3">
                    {plural(agreement.total_reviews, "review")}
                    {agreement.mean_time_spent_ms > 0 ? `, ${duration(agreement.mean_time_spent_ms / 1000)} each on average` : ""}
                  </span>
                </div>
                {agreement.per_class.length > 0 && (
                  <table className="data-table">
                    <thead><tr><th>Class</th><th className="num">Confirmed</th><th className="num">Rate</th></tr></thead>
                    <tbody>
                      {agreement.per_class.slice(0, 8).map((p) => (
                        <tr key={p.class_id}>
                          <td className="text-ink">{humanize(p.name)}</td>
                          <td className="num">{num(p.confirmed)} of {num(p.reviews)}</td>
                          <td className="num">{p.confirmed_pct}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            ) : (
              <Blank p={placeholder(loading, agreement ? 1 : 0, "No reviews yet.")} />
            )}
          </Section>

          {/* M-F.4 productivity + QA analytics (DataOps view; non-punitive) */}
          {prod && (
            <Section title="Team productivity and cost" right={plural(prod.n_reviewers, "reviewer")}>
              <div className="grid grid-cols-2 xl:grid-cols-4 gap-2 mb-4">
                <Stat label="Human hours" value={num(prod.cost.human_hours)} sub={`$${num(prod.cost.human_cost_usd)}`} />
                <Stat label="GPU hours" value={num(prod.cost.gpu_hours)} />
                <Stat label="Cost per frame" value={prod.cost.cost_per_frame_usd != null ? `$${prod.cost.cost_per_frame_usd}` : "-"} sub={prod.cost.cost_per_object_usd != null ? `$${prod.cost.cost_per_object_usd} per object` : undefined} />
                <Stat label="Saved by auto-accept" value={`$${num(prod.cost.auto_accept_saved_usd)}`} sub={plural(prod.cost.n_auto_accept, "object")} />
              </div>
              <div className="text-[12.5px] text-ink-3 mb-3">
                Agreement between annotators: {prod.interannotator.agreement_rate != null
                  ? <span className="text-pass">{pct(prod.interannotator.agreement_rate)}</span>
                  : "-"} over {plural(prod.interannotator.shared_objects, "shared object")}
              </div>
              {/* throughput + agreement trend over time (one bar per active day) */}
              {prod.trend.length > 0 && (() => {
                const maxR = Math.max(...prod.trend.map((d) => d.reviews), 1);
                return (
                  <div className="mb-4">
                    <div className="text-[12px] text-ink-2 mb-1.5">Reviews per day <span className="text-ink-3">(greener means more agreement)</span></div>
                    <div className="flex items-end gap-0.5 h-12">
                      {prod.trend.map((d) => (
                        <div key={d.day} title={`${d.reviews} reviews, agreement ${d.agreement != null ? (d.agreement * 100).toFixed(0) + "%" : "-"}`}
                          className="flex-1 min-w-[3px] rounded-t" style={{ height: `${(d.reviews / maxR) * 100}%`, background: d.agreement != null ? `rgba(86,211,100,${0.35 + 0.6 * d.agreement})` : "#6C727A" }} />
                      ))}
                    </div>
                  </div>
                );
              })()}
              <table className="data-table">
                <thead><tr><th>Reviewer</th><th className="num">Reviews</th><th className="num">Corrected</th><th className="num">Agreement</th><th className="num">Objects / hour</th></tr></thead>
                <tbody>
                  {prod.reviewers.filter((r) => r.reviews >= 5).slice(0, 12).map((r) => (
                    <tr key={r.reviewer}>
                      <td className="text-ink">{r.reviewer}</td>
                      <td className="num">{num(r.reviews)}</td>
                      <td className={`num ${r.correction_rate != null && r.correction_rate > 0.5 ? "text-warn" : ""}`}>{r.correction_rate != null ? pct(r.correction_rate) : "-"}</td>
                      <td className={`num ${r.agreement != null && r.agreement >= 0.8 ? "text-pass" : ""}`}>{r.agreement != null ? pct(r.agreement) : "-"}</td>
                      <td className="num text-ink-3">{r.objects_per_hour ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="text-[12px] text-ink-3 mt-3 leading-snug">{prod.note}</div>
            </Section>
          )}

          <Section title="What reviewers corrected"
            right={confusions ? `${num(confusions.total_corrections)} corrections` : null}>
            <div className="flex items-center gap-2 mb-3 text-[12.5px]">
              <span className="text-ink-3">Group by</span>
              <Segmented value={confBy} onChange={(v) => setConfBy(v as "class" | "camera" | "city")}
                options={(["class", "camera", "city"] as const).map((b) => ({ value: b, label: humanize(b) }))} />
            </div>
            {confRows.length ? (
              <>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Model said</th><th>Corrected to</th>
                      {confBy !== "class" && <th>{humanize(confBy)}</th>}<th className="num">Count</th>
                    </tr>
                  </thead>
                  <tbody>
                    {shownConf.map((c, i) => (
                      <tr key={i}>
                        <td className="text-block">{humanize(c.old_class)}</td>
                        <td className="text-pass">{humanize(c.new_class)}</td>
                        {confBy !== "class" && <td className="text-ink-3">{c.group ?? "-"}</td>}
                        <td className="num text-ink">{num(c.count)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {confRows.length > CONFUSION_ROWS && (
                  <button onClick={() => setAllConfusions((v) => !v)} className="link text-[12.5px] mt-2">
                    {allConfusions ? "Show the top 15" : `Show all ${num(confRows.length)} pairs`}
                  </button>
                )}
              </>
            ) : (
              <Blank p={placeholder(confLoading, 0, "No corrections recorded yet.")} />
            )}
          </Section>

          {/* Model vs sealed gold: the same question from the other side, with the crops behind each cell. */}
          <Section title="Model against the gold set" right="Open a cell to see the crops" className="lg:col-span-2">
            <EvalDrilldown />
          </Section>
        </div>

        {/* Whether the gold recall above can be trusted, which nothing else on this page can tell you: the
            sealed set was built by people confirming machine boxes, so it leans toward what the model
            already sees. This is the only panel measured against a denominator the model did not help
            build, which is why it sits directly under the gold comparison rather than in a corner. */}
        <Section title="Is that recall real?" right="Blind audit, capture-recapture">
          <RecallHonesty />
        </Section>

        {/* DPDPA / PII anonymization evidence (Gate A) */}
        <Section title="Faces and plates anonymized" right="DPDPA gate A">
          {pii && pii.total_frames > 0 ? (
            <div className="space-y-4">
              <div className="flex h-5 w-full overflow-hidden rounded-sm">
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
              <div className="flex flex-wrap items-center gap-5 text-[12.5px]">
                <span className="inline-flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-sm bg-pass" />Anonymized {pii.coverage_pct}%</span>
                <span className="text-ink-3">
                  {num(pii.frames_anonymized)} of {num(pii.total_frames)} frames; {num(pii.faces_blurred)} faces and {num(pii.plates_blurred)} plates blurred
                </span>
              </div>
              <div className="text-[12.5px] space-y-0.5 pt-2 border-t hairline">
                <div className="label-caps mb-1">By redactor version</div>
                {Object.entries(pii.method_versions).map(([k, v]) => (
                  <div key={k} className="flex justify-between">
                    <span className="text-ink-2 truncate">{k}</span>
                    <span className="text-ink-3 tabular-nums">{plural(v, "frame")}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : loading ? (
            <Blank p={placeholder(true, 0, "")} />
          ) : (
            <div className="text-[12.5px] text-ink-3 py-4 text-center">
              {piiFailed ? "Could not read the anonymization audit." : "No anonymized frames yet."}
            </div>
          )}
        </Section>

        {/* Geo capture density: released as noised 250 m cells, on request, against a privacy budget */}
        <Section title="Where the data was captured" right={geo ? plural(geo.n_cells, "cell") : null}>
          {geo ? (
            <div className="space-y-3">
              <div className="text-[12.5px] text-ink-3">
                {num(geo.n_cells)} cells of {geo.cell_m} m released. {num(geo.suppressed_cells)} thin cells
                ({num(geo.suppressed_points)} fixes) were withheld to keep at least {geo.k_anonymity} per cell. This
                release used {geo.epsilon} of the privacy budget; {geo.budget.remaining} of {geo.budget.epsilon_total} is left.
              </div>
              <div className="text-[12px] text-ink-3 grid grid-cols-2 md:grid-cols-4 gap-x-6 gap-y-0.5 max-h-48 overflow-auto">
                {geo.cells.slice(0, 200).map((c) => (
                  <div key={c.cell_id} className="flex justify-between gap-2">
                    <span className="text-ink-2 tabular-nums">{c.lat.toFixed(3)}, {c.lon.toFixed(3)}</span>
                    <span className="tabular-nums">{Math.round(c.count)}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3 py-3 text-[12.5px] text-ink-3 text-center">
              <span className="max-w-xl">
                {geoErr ?? "Capture locations, released as noised 250 m cells. Each release spends part of the privacy budget, so it is shown only when asked for."}
              </span>
              {!geoErr && (
                <button onClick={releaseGeo} disabled={geoBusy} className="btn">
                  {geoBusy ? "Releasing..." : "Show capture density"}
                </button>
              )}
            </div>
          )}
        </Section>

        {/* Data Intelligence Layer (M1.7) */}
        <Section title="Scene conditions" right="Zero-shot SigLIP 2">
          {scenes ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {Object.entries(scenes).map(([axis, vals]) => {
                const tot = Object.values(vals).reduce((a, b) => a + b, 0) || 1;
                return (
                  <div key={axis}>
                    <div className="label-caps mb-1.5">{humanize(axis)}</div>
                    {Object.entries(vals).map(([v, n]) => (
                      <div key={v} className="text-[12px] flex items-center gap-2 mb-1">
                        <span className="w-20 text-ink-2 truncate" title={v}>{humanize(v)}</span>
                        <div className="flex-1 h-2 bg-line rounded-sm overflow-hidden"><div className="h-2 bg-accent" style={{ width: `${(100 * n) / tot}%` }} /></div>
                        <span className="text-ink-3 w-12 text-right tabular-nums">{compact(n)}</span>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          ) : <Blank p={placeholder(true, 0, "")} />}
        </Section>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <Section title="Duplicate frames">
            <div className="text-[22px] font-semibold text-ink tabular-nums">{dedup ? `${(dedup.rate * 100).toFixed(1)}%` : <span className="skeleton inline-block h-[22px] w-20" />}</div>
            <div className="text-[12px] text-ink-3 mt-1.5">{dedup ? `${num(dedup.redundant)} of ${num(dedup.total)} frames are near-copies, in ${plural(dedup.groups, "group")}` : ""}</div>
          </Section>
          <Section title="Frames added per day" right={growth.length ? `${num(growth[growth.length - 1].cumulative)} in total` : null}>
            <div className="flex items-end gap-0.5 h-24">
              {growth.slice(-40).map((g, i) => {
                const max = Math.max(...growth.map((x) => x.frames), 1);
                return <div key={i} title={`${g.date}: ${g.frames}`} className="flex-1 bg-info/70 min-w-[2px] rounded-t-sm" style={{ height: `${(100 * g.frames) / max}%` }} />;
              })}
            </div>
            {growth.length > 0 && (
              <div className="flex justify-between text-[11px] text-ink-3 mt-1.5">
                <span>{growth.slice(-40)[0].date}</span><span>{growth[growth.length - 1].date}</span>
              </div>
            )}
          </Section>
          <Section title="Rare classes covered">
            {overview ? (
              <>
                <div className="text-[22px] font-semibold text-ink tabular-nums">{overview.long_tail.coverage_pct}%</div>
                <div className="text-[12px] text-ink-3 mt-1.5">{overview.long_tail.covered_classes} of {overview.long_tail.total_classes} classes have at least one example</div>
              </>
            ) : <span className="skeleton inline-block h-[22px] w-20" />}
          </Section>
        </div>

        <Section title="Frames clustered by appearance"
          right={cluster ? `${num(cluster.n)} frames, ${plural(cluster.clusters, "cluster")} · click a dot to open the frame` : null}>
          {cluster && cluster.points.length ? (
            <ClusterScatter cluster={cluster} onPick={(fid) => router.push(`/frame/${fid}`)} />
          ) : (
            <div className="text-[12.5px] text-ink-3 py-8 text-center">{cluster ? "No frames have embeddings yet." : "Computing the projection..."}</div>
          )}
        </Section>
      </div>
    </PageShell>
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
