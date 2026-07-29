"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { toast } from "@/lib/toast";
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

const DECISION_TONE: Record<string, string> = {
  accept: "text-pass",
  abstain: "text-ink-3",
  review: "text-warn",
  adjudicate: "text-accent",
  reject: "text-block",
};

function Stat({ label, value, hint, tone }: {
  label: string; value: string | number; hint?: string; tone?: string;
}) {
  return (
    <div className="panel px-3 py-2 min-w-[130px]">
      <div className="font-mono text-[10px] uppercase text-ink-3">{label}</div>
      <div className={`font-mono text-[18px] tabular-nums ${tone ?? "text-ink"}`}>{value}</div>
      {hint && <div className="font-mono text-[10px] text-ink-3">{hint}</div>}
    </div>
  );
}

export default function ReasonerPage() {
  const [attribution, setAttribution] = useState<ReasonerAttribution | null>(null);
  const [outcomes, setOutcomes] = useState<Awaited<ReturnType<typeof api.reasonerOutcomes>> | null>(null);
  const [coverage, setCoverage] = useState<Awaited<ReturnType<typeof api.reasonerCoverage>> | null>(null);
  const [priors, setPriors] = useState<Awaited<ReturnType<typeof api.reasonerPriors>> | null>(null);
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

  const load = useCallback(async () => {
    try {
      const [a, o, c, p] = await Promise.all([
        api.reasonerAttribution(), api.reasonerOutcomes(),
        api.reasonerCoverage(), api.reasonerPriors(),
      ]);
      setAttribution(a); setOutcomes(o); setCoverage(c); setPriors(p);
    } catch (e) { toast(humanizeError(e), "error"); }
  }, []);

  useEffect(() => { load(); }, [load]);

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
    if (!sessionId.trim()) { toast("paste a session id", "error"); return; }
    setBusy(true);
    try {
      const r = await api.reasonerRerun(sessionId.trim(), apply);
      setRerun(r);
      if (apply) toast(`demoted ${r.applied} labels`, "success");
    } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const acceptError = outcomes?.by_decision?.accept?.error_rate;

  return (
    <PageShell active="REASONER" title="Reasoning layer"
      subtitle="what ran before each label, and whether it was right">
      <div className="p-4 space-y-4 max-w-6xl">
        <div className="flex gap-2 flex-wrap">
          <Stat label="corpus reasoned" value={`${((coverage?.fraction ?? 0) * 100).toFixed(1)}%`}
            hint={`${coverage?.reasoned_in_sample ?? 0} of ${coverage?.sampled ?? 0} sampled`} />
          <Stat label="traces" value={attribution?.reasoned ?? 0}
            hint={`${attribution?.reviewed ?? 0} since reviewed`} />
          <Stat label="accept error rate"
            value={acceptError == null ? "-" : `${(acceptError * 100).toFixed(1)}%`}
            tone={acceptError == null ? undefined : acceptError > 0.1 ? "text-block" : "text-pass"}
            hint="accepted, then corrected by a human" />
          <Stat label="priors" value={priors?.classes_with_height.length ?? 0}
            hint="classes with a height band" />
        </div>

        {outcomes && Object.keys(outcomes.by_decision ?? {}).length > 0 && (
          <section className="panel">
            <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
              did its decisions hold up
            </div>
            <div className="p-3 space-y-2">
              <table className="w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">decision</th><th>objects</th><th>since reviewed</th>
                    <th>corrected</th><th>error rate</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(outcomes.by_decision).map(([d, b]) => (
                    <tr key={d} className="border-b hairline">
                      <td className={`py-1 ${DECISION_TONE[d] ?? "text-ink-2"}`}>{d}</td>
                      <td className="text-ink-3 tabular-nums">{b.total}</td>
                      <td className="text-ink-3 tabular-nums">{b.reviewed}</td>
                      <td className="text-ink-3 tabular-nums">{b.corrected}</td>
                      <td className="text-ink tabular-nums">
                        {b.error_rate == null ? "-" : `${(b.error_rate * 100).toFixed(1)}%`}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="font-mono text-[10px] text-ink-3">{outcomes.headline}</div>
            </div>
          </section>
        )}

        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            which checks actually catch errors
          </div>
          <div className="p-3 space-y-2">
            {!attribution || Object.keys(attribution.checks).length === 0 ? (
              <div className="font-mono text-[11px] text-ink-3">
                Nothing measured yet. Precision is computed by joining each reasoning trace against what a
                human later decided, so it appears once reasoned objects have been reviewed.
              </div>
            ) : (
              <table className="w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">check</th><th>what it asks</th>
                    <th>objected</th><th>was right</th><th>precision</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(attribution.checks).map(([name, c]) => (
                    <tr key={name} className="border-b hairline">
                      <td className="py-1 text-ink">{name}</td>
                      <td className="text-ink-3 max-w-[22rem]">{CHECK_BLURB[name] ?? ""}</td>
                      <td className="text-ink-3 tabular-nums">{c.fired_against}</td>
                      <td className="text-ink-3 tabular-nums">{c.correct_against}</td>
                      <td className={`tabular-nums ${
                        !c.measured ? "text-ink-4"
                          : (c.precision_against ?? 0) > 0.7 ? "text-pass" : "text-warn"}`}>
                        {c.precision_against == null ? "-"
                          : `${(c.precision_against * 100).toFixed(0)}%`}
                        {!c.measured && " (thin)"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {attribution?.caveat && (
              // Said out loud rather than buried: review is not a random sample, so these numbers describe
              // the checks' behaviour on hard cases rather than on the corpus as a whole.
              <div className="font-mono text-[10px] text-ink-3 border-l-2 border-line pl-2">
                {attribution.caveat}
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            try a detection
          </div>
          <div className="p-3 space-y-2">
            <div className="font-mono text-[10px] text-ink-3">
              Adjusting a height band and seeing what it does to a known-bad detection is far more useful
              than reading the weights. Nothing is written; this runs Tier 1 and returns every finding.
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
              <input value={cls} onChange={(e) => setCls(e.target.value)} placeholder="class"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-40" />
              <input value={bbox} onChange={(e) => setBbox(e.target.value)} placeholder="x1,y1,x2,y2"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-44" />
              <label className="flex items-center gap-1 text-ink-3">conf
                <input type="number" step="0.05" min="0" max="1" value={conf}
                  onChange={(e) => setConf(Number(e.target.value))}
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink w-20" />
              </label>
              <label className="flex items-center gap-1 text-ink-3">depth m
                <input value={depth} onChange={(e) => setDepth(e.target.value)}
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink w-20" />
              </label>
              <select value={roadType} onChange={(e) => setRoadType(e.target.value)}
                className="bg-bg border border-line px-1.5 py-0.5 text-ink">
                {["urban", "highway", "rural", "service_road", ""].map((r) => (
                  <option key={r || "none"} value={r}>{r || "no scene"}</option>
                ))}
              </select>
              <button onClick={explain} disabled={busy}
                className="border border-accent px-2 py-0.5 text-accent hover:bg-accent/10 disabled:opacity-40">
                reason
              </button>
            </div>

            {explained && (
              <div className="space-y-1">
                <div className="font-mono text-[12px]">
                  <span className={DECISION_TONE[explained.decision] ?? "text-ink"}>
                    {explained.decision}
                  </span>
                  <span className="text-ink-3"> · score {explained.score} · conflict {explained.conflict}</span>
                  {explained.suggested_class && (
                    <span className="text-warn"> · suggests {explained.suggested_class}</span>
                  )}
                </div>
                {explained.findings.length === 0 ? (
                  <div className="font-mono text-[10.5px] text-ink-3">
                    No check could be applied, so the reasoner abstains and the gate decides on confidence
                    alone. That is deliberately different from having assessed it and found nothing.
                  </div>
                ) : (
                  <ul className="space-y-0.5">
                    {explained.findings.map((f, i) => (
                      <li key={i} className="font-mono text-[10.5px]">
                        <span className={f.weight < 0 ? "text-block" : "text-pass"}>
                          {f.weight > 0 ? "+" : ""}{f.weight.toFixed(2)}
                        </span>{" "}
                        <span className="text-ink-2">{f.check}</span>{" "}
                        <span className="text-ink-3">{f.detail}</span>
                      </li>
                    ))}
                  </ul>
                )}
                {explained.question && (
                  <div className="font-mono text-[10.5px] text-accent">
                    would ask the adjudicator: {explained.question}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            reason over a session already annotated
          </div>
          <div className="p-3 space-y-2">
            <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
              <input value={sessionId} onChange={(e) => setSessionId(e.target.value)}
                placeholder="session id"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-80" />
              <button onClick={doRerun(false)} disabled={busy}
                className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
                {busy ? "reasoning..." : "dry run"}
              </button>
              {rerun && !rerun.dry_run === false && rerun.would_demote > 0 && (
                <button onClick={doRerun(true)} disabled={busy}
                  className="border border-warn px-2 py-0.5 text-warn hover:bg-warn/10 disabled:opacity-40">
                  apply ({rerun.would_demote} demotions)
                </button>
              )}
            </div>
            <div className="font-mono text-[10px] text-ink-3">
              Applies the reasoning to boxes that already exist, without re-detecting. Human decisions are
              never overwritten: an object a person accepted or rejected is settled.
            </div>

            {rerun && (
              <div className="space-y-2">
                <div className="flex gap-2 flex-wrap">
                  <Stat label="objects" value={rerun.objects} hint={`${rerun.frames} frames`} />
                  <Stat label="auto-accepted" value={rerun.auto_accepted}
                    hint="were going into training unreviewed" />
                  <Stat label="disagreed with" value={rerun.would_demote}
                    tone={rerun.would_demote ? "text-warn" : "text-pass"}
                    hint={rerun.demote_rate_of_auto_accepted == null ? undefined
                      : `${(rerun.demote_rate_of_auto_accepted * 100).toFixed(0)}% of them`} />
                </div>
                <div className="flex gap-1 flex-wrap">
                  {Object.entries(rerun.decisions).map(([d, n]) => (
                    <span key={d}
                      className={`border border-line px-1.5 py-0.5 font-mono text-[10.5px] ${
                        DECISION_TONE[d] ?? "text-ink-2"}`}>
                      {d} <span className="text-ink-3">{n}</span>
                    </span>
                  ))}
                </div>
                {rerun.examples.length > 0 && (
                  <table className="w-full font-mono text-[11px]">
                    <thead>
                      <tr className="text-ink-3 text-left border-b hairline">
                        <th className="py-1">class</th><th>was</th><th>verdict</th>
                        <th>suggests</th><th>why</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rerun.examples.map((e) => (
                        <tr key={e.object_id} className="border-b hairline">
                          <td className="py-1 text-ink">{e.class_name}</td>
                          <td className="text-ink-3">{e.state}</td>
                          <td className={DECISION_TONE[e.decision] ?? "text-ink-2"}>{e.decision}</td>
                          <td className="text-warn">{e.suggested_class ?? "-"}</td>
                          <td className="text-ink-3 truncate max-w-[24rem]">{e.reasons[0]}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </div>
        </section>
      </div>
    </PageShell>
  );
}
