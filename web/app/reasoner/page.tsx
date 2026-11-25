"use client";

import { useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import SessionSelect from "@/components/shell/SessionSelect";
import Inspector from "@/components/shell/Inspector";
import EvidencePanel, { type EvidenceSubject } from "@/components/inspect/EvidencePanel";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
import EmptyState from "@/components/ui/EmptyState";
import { Stat } from "@/components/ui/kit";
import { humanize, num, pct, plural } from "@/lib/format";
import type { ReasonerAttribution, ReasonerRerun, ReasoningTrace } from "@/lib/types";

// The reasoning layer, and whether it is any good.
//
// The gate's only input worth the name was the detector's own confidence, which is the model grading its
// own homework and cannot catch the failures that actually hurt: the confident wrong ones. This page is
// where the layer that replaced that is inspected, tuned, and held to account.
//
// The accountability half is the point. Every weight in the evidence collectors is currently a guess, and
// the corpus holds the evidence to replace those guesses with measurements. A reasoning layer added on
// faith is one nobody can tune.

const CHECK_BLURB: Record<string, string> = {
  physics: "box size against the class's real-world height and the depth prior",
  geometry: "aspect ratio, the horizon, and the mask against the box",
  temporal: "whether the track agrees with itself across frames",
  scene: "whether this class belongs on this kind of road at all",
  cross_model: "whether the three detection paths agreed",
  corpus_memory: "what humans decided about the nearest-looking reviewed crops",
};

const CHECK_NAME: Record<string, string> = {
  physics: "Physical size", geometry: "Geometry", temporal: "Consistency over time", scene: "Scene fit",
  cross_model: "Model agreement", corpus_memory: "Past human decisions",
};
const checkName = (c: string) => CHECK_NAME[c] ?? humanize(c);

const DECISION_TONE: Record<string, string> = {
  accept: "text-pass",
  abstain: "text-ink-3",
  review: "text-warn",
  adjudicate: "text-accent",
  reject: "text-block",
};

export default function ReasonerPage() {
  // The rerun table says the reasoner disagreed with a label and names what it would suggest instead.
  // Without the object that is a claim you can only agree with.
  const [subject, setSubject] = useState<EvidenceSubject | null>(null);
  const attributionQ = useQuery(["reasonerAttribution"], () => api.reasonerAttribution());
  const outcomesQ = useQuery(["reasonerOutcomes"], () => api.reasonerOutcomes());
  const coverageQ = useQuery(["reasonerCoverage"], () => api.reasonerCoverage());
  const priorsQ = useQuery(["reasonerPriors"], () => api.reasonerPriors());
  const attribution: ReasonerAttribution | null = attributionQ.data ?? null;
  const outcomes = outcomesQ.data ?? null;
  const coverage = coverageQ.data ?? null;
  const priors = priorsQ.data ?? null;
  const failure = attributionQ.error ?? outcomesQ.error ?? coverageQ.error ?? priorsQ.error;
  useEffect(() => { if (failure) toast(humanizeError(failure), "error"); }, [failure]);
  const [busy, setBusy] = useState(false);

  // The tuning surface: try a hypothetical detection without running a session.
  const [cls, setCls] = useState("pedestrian");
  const [bbox, setBbox] = useState("100,300,110,304");
  const [conf, setConf] = useState(0.72);
  const [depth, setDepth] = useState("30");
  const [roadType, setRoadType] = useState("urban");
  const [explained, setExplained] = useState<(ReasoningTrace & { reasons: string[] }) | null>(null);

  const [sessionId, setSessionId] = useState("");
  const [rerun, setRerun] = useState<ReasonerRerun | null>(null);

  const explain = async () => {
    setBusy(true);
    try {
      const box = bbox.split(",").map((v) => Number(v.trim()));
      setExplained(await api.reasonerExplain({
        class_name: cls.trim(), bbox: box, conf,
        depth_m: depth ? Number(depth) : null,
        focal_px: depth ? 1000 : null,
        scene: roadType ? { road_type: roadType } : {},
      }));
    } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const doRerun = (apply: boolean) => async () => {
    if (!sessionId.trim()) { toast("Choose a session first", "error"); return; }
    setBusy(true);
    try {
      const r = await api.reasonerRerun(sessionId.trim(), apply);
      setRerun(r);
      if (apply) {
        toast(`Sent ${r.applied} labels back to review`, "success");
        // Applying demotes labels, which is what the outcome and coverage tables count.
        void invalidate("reasonerAttribution", "reasonerOutcomes", "reasonerCoverage");
      }
    } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const acceptError = outcomes?.by_decision?.accept?.error_rate;

  const fieldCls = "field";
  return (
    <PageShell active="Reasoning trace" title="Reasoning trace"
      subtitle="The checks that run before each auto-label, and how often they were right.">
      <div className="flex h-full min-h-0">
      <div className="flex-1 min-w-0 overflow-auto">
      <div className="page-wrap space-y-4">
        <div className="grid grid-cols-2 xl:grid-cols-4 gap-3">
          <Stat label="Corpus checked" value={coverage?.sampled ? pct(coverage.fraction ?? 0, 1) : "-"}
            sub={`${num(coverage?.reasoned_in_sample ?? 0)} of ${num(coverage?.sampled ?? 0)} sampled`} />
          <Stat label="Reasoning traces" value={num(attribution?.reasoned ?? 0)}
            sub={`${num(attribution?.reviewed ?? 0)} reviewed by a person since`} />
          <Stat label="Accepted, then corrected"
            value={acceptError == null ? "-" : pct(acceptError, 1)}
            tone={acceptError == null ? undefined : acceptError > 0.1 ? "block" : "pass"}
            sub="auto-accepted labels a person later fixed" />
          <Stat label="Size priors" value={num(priors?.classes_with_height.length ?? 0)}
            sub="classes with a known real-world height" />
        </div>

        {outcomes && Object.keys(outcomes.by_decision ?? {}).length > 0 && (
          <section className="panel">
            <div className="panel-head"><h2 className="section-title">Did its decisions hold up?</h2></div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Decision</th><th className="num">Objects</th><th className="num">Reviewed since</th>
                  <th className="num">Corrected</th><th className="num">Error rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(outcomes.by_decision).map(([d, b]) => (
                  <tr key={d}>
                    <td className={DECISION_TONE[d] ?? "text-ink-2"}>{humanize(d)}</td>
                    <td className="num">{num(b.total)}</td>
                    <td className="num">{num(b.reviewed)}</td>
                    <td className="num">{num(b.corrected)}</td>
                    <td className="num text-ink">{b.error_rate == null ? "-" : pct(b.error_rate, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {outcomes.headline && <p className="section-desc px-3.5 py-2.5 border-t hairline">{outcomes.headline}</p>}
          </section>
        )}

        <section className="panel">
          <div className="panel-head"><h2 className="section-title">Which checks catch real errors</h2></div>
          {!attribution || Object.keys(attribution.checks).length === 0 ? (
            <EmptyState compact icon="chart" title="Nothing measured yet">
              Each check is scored against what a reviewer later decided, so its precision appears once
              reasoned objects have been reviewed.
            </EmptyState>
          ) : (
            <>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Check</th><th>What it asks</th>
                    <th className="num">Objected</th><th className="num">Was right</th><th className="num">Precision</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(attribution.checks).map(([name, c]) => (
                    <tr key={name}>
                      <td className="text-ink">{checkName(name)}</td>
                      <td className="max-w-[22rem]">{CHECK_BLURB[name] ?? ""}</td>
                      <td className="num">{num(c.fired_against)}</td>
                      <td className="num">{num(c.correct_against)}</td>
                      <td className={`num ${
                        !c.measured ? "text-ink-3"
                          : (c.precision_against ?? 0) > 0.7 ? "text-pass" : "text-warn"}`}>
                        {c.precision_against == null ? "-" : pct(c.precision_against)}
                        {!c.measured && " (too few)"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {attribution.caveat && (
                // Said out loud rather than buried: review is not a random sample, so these numbers describe
                // the checks' behaviour on hard cases rather than on the corpus as a whole.
                <p className="section-desc px-3.5 py-2.5 border-t hairline">{attribution.caveat}</p>
              )}
            </>
          )}
        </section>

        <section className="panel">
          <div className="panel-head"><h2 className="section-title">Try a detection</h2></div>
          <div className="p-3.5 space-y-3">
            <p className="section-desc">
              Describe a detection and see which checks object to it and why. Nothing is saved.
            </p>
            <div className="flex items-end gap-3 text-[12.5px] flex-wrap">
              <label className="flex flex-col gap-1.5"><span className="text-ink-2 text-[12px]">Class</span>
                <input value={cls} onChange={(e) => setCls(e.target.value)} placeholder="pedestrian"
                  className={`${fieldCls} w-40`} />
              </label>
              <label className="flex flex-col gap-1.5"><span className="text-ink-2 text-[12px]">Box (x1, y1, x2, y2)</span>
                <input value={bbox} onChange={(e) => setBbox(e.target.value)} placeholder="x1,y1,x2,y2"
                  className={`${fieldCls} w-44 font-mono`} />
              </label>
              <label className="flex flex-col gap-1.5"><span className="text-ink-2 text-[12px]">Confidence</span>
                <input type="number" step="0.05" min="0" max="1" value={conf}
                  onChange={(e) => setConf(Number(e.target.value))}
                  className={`${fieldCls} w-24`} />
              </label>
              <label className="flex flex-col gap-1.5"><span className="text-ink-2 text-[12px]">Distance (m)</span>
                <input value={depth} onChange={(e) => setDepth(e.target.value)}
                  className={`${fieldCls} w-24`} />
              </label>
              <label className="flex flex-col gap-1.5"><span className="text-ink-2 text-[12px]">Road type</span>
                <select value={roadType} onChange={(e) => setRoadType(e.target.value)} className={fieldCls}>
                  {["urban", "highway", "rural", "service_road", ""].map((r) => (
                    <option key={r || "none"} value={r}>{r ? humanize(r) : "Unknown"}</option>
                  ))}
                </select>
              </label>
              <button type="button" onClick={explain} disabled={busy} className="btn btn-primary">
                Check it
              </button>
            </div>

            {explained && (
              <div className="rounded border border-line bg-bg-2/50 p-3 space-y-1.5">
                <div className="text-[13px]">
                  <span className={`font-medium ${DECISION_TONE[explained.decision] ?? "text-ink"}`}>
                    {humanize(explained.decision)}
                  </span>
                  <span className="text-ink-3"> · score {explained.score} · conflict {explained.conflict}</span>
                  {explained.suggested_class && (
                    <span className="text-warn"> · suggests {humanize(explained.suggested_class)}</span>
                  )}
                </div>
                {explained.findings.length === 0 ? (
                  <div className="text-[12.5px] text-ink-3">
                    No check could be applied, so the reasoner abstains and the gate decides on confidence
                    alone. That is different from having checked it and found nothing.
                  </div>
                ) : (
                  <ul className="space-y-1">
                    {explained.findings.map((f, i) => (
                      <li key={i} className="text-[12.5px]">
                        <span className={`tabular-nums ${f.weight < 0 ? "text-block" : "text-pass"}`}>
                          {f.weight > 0 ? "+" : ""}{f.weight.toFixed(2)}
                        </span>{" "}
                        <span className="text-ink-2">{checkName(f.check)}</span>{" "}
                        <span className="text-ink-3">{f.detail}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {explained.question && (
                  <div className="text-[12.5px] text-accent-2">
                    Would ask a reviewer: {explained.question}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head"><h2 className="section-title">Re-check a labelled session</h2></div>
          <div className="p-3.5 space-y-3">
            <p className="section-desc">
              Runs the checks over boxes that already exist, without detecting again. Decisions a person made
              are never overwritten.
            </p>
            <div className="flex items-center gap-2 text-[12.5px] flex-wrap">
              <SessionSelect value={sessionId} onChange={setSessionId} className="w-96" />
              <button type="button" onClick={doRerun(false)} disabled={busy} className="btn">
                {busy ? "Checking..." : "Preview"}
              </button>
              {rerun && !rerun.dry_run === false && rerun.would_demote > 0 && (
                <button type="button" onClick={doRerun(true)} disabled={busy} className="btn btn-primary"
                  data-tip="Moves these auto-accepted labels back into the review queue">
                  Send {plural(rerun.would_demote, "label")} back to review
                </button>
              )}
            </div>

            {rerun && (
              <div className="space-y-3">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <Stat label="Objects" value={num(rerun.objects)} sub={plural(rerun.frames, "frame")} />
                  <Stat label="Auto-accepted" value={num(rerun.auto_accepted)}
                    sub="were going into training unreviewed" />
                  <Stat label="Disagreed with" value={num(rerun.would_demote)}
                    tone={rerun.would_demote ? "warn" : "pass"}
                    sub={rerun.demote_rate_of_auto_accepted == null ? undefined
                      : `${pct(rerun.demote_rate_of_auto_accepted)} of the auto-accepted`} />
                </div>
                <div className="flex gap-1.5 flex-wrap">
                  {Object.entries(rerun.decisions).map(([d, n]) => (
                    <span key={d} className={`chip ${DECISION_TONE[d] ?? ""}`}>
                      {humanize(d)} <span className="text-ink-3 tabular-nums">{num(n)}</span>
                    </span>
                  ))}
                </div>
                {rerun.examples.length > 0 && (
                  <div className="panel overflow-hidden">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Class</th><th>Was</th><th>Verdict</th>
                          <th>Suggests</th><th>Why</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rerun.examples.map((e) => (
                          <tr key={e.object_id}
                            onClick={() => setSubject({
                              objectId: e.object_id,
                              suggestion: e.suggested_class
                                ? { class_id: 0, class_name: e.suggested_class } : null,
                              text: e.reasons.join(" · "),
                              kind: `reasoner: ${e.decision}`,
                            })}
                            title="show the object this verdict is about"
                            className={`cursor-pointer ${subject?.objectId === e.object_id ? "bg-accent/10" : ""}`}>
                            <td className="text-ink">{humanize(e.class_name)}</td>
                            <td>{humanize(e.state)}</td>
                            <td className={DECISION_TONE[e.decision] ?? "text-ink-2"}>{humanize(e.decision)}</td>
                            <td className="text-warn">{e.suggested_class ? humanize(e.suggested_class) : "-"}</td>
                            <td className="truncate max-w-[24rem]">{e.reasons[0]}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
      </div>
      <Inspector title="Evidence" side="right" width="w-[26rem]">
        <EvidencePanel subject={subject} />
      </Inspector>
      </div>
    </PageShell>
  );
}
