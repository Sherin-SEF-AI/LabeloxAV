"use client";

import { useState } from "react";
import { api, humanizeError, type LabelAgentReport, type LabelAgentStageReport } from "@/lib/api";
import { toast } from "@/lib/toast";
import { useQuery } from "@/lib/useQuery";
import { ago, num, pct } from "@/lib/format";
import EmptyState from "@/components/ui/EmptyState";

// The label agent: how each way of checking a machine label's class scores against what people ruled, and
// a way to run the agent on a session.
//
// The table exists because "the agent fixes labels" is a claim that needs two numbers, not one. "Fixed" is
// the share of wrong labels it corrected; "broke" is the share of right labels it changed to something wrong,
// with an upper bound because the right-label populations are small. A stage that fixes a lot and breaks a
// lot is worse than useless in a corpus where most labels are right, and the old crop classifier is exactly
// that. Rows are shown on the whole answer key: every stage here learns nothing from the item it is graded
// on (memory and the probe are cross-fitted), and the whole set is what makes the bound tight enough to read.

const ORDER = ["agent", "agent+ask", "memory@0.5", "probe@0.3", "family_zs@0.3", "zs_all@0.5", "classify_crop@0.5",
               "reasoner", "vlm_stored", "detector"];

const NAMES: Record<string, string> = {
  "agent": "Label agent (fixes only)",
  "agent+ask": "Label agent, if every question is taken",
  "memory@0.5": "Corpus memory alone",
  "probe@0.3": "Learned probe alone",
  "family_zs@0.3": "Family zero-shot alone",
  "zs_all@0.5": "Current crop classifier",
  "classify_crop@0.5": "Current crop classifier (fresh crops)",
  "reasoner": "Rule-based reasoner",
  "vlm_stored": "VLM judge (stored verdicts)",
  "detector": "Detector as is",
};

function rep(s: { test?: LabelAgentStageReport; all?: LabelAgentStageReport }) {
  return s.all ?? s.test;
}

export default function LabelAgentPanel() {
  const q = useQuery(["labelagentBenchmark"], () => api.labelagentBenchmark(), { keepPrevious: true });
  const report: LabelAgentReport | null = q.data?.report ?? null;
  const [sid, setSid] = useState("");
  const [busy, setBusy] = useState<"dry" | "apply" | null>(null);

  const run = async (apply: boolean) => {
    setBusy(apply ? "apply" : "dry");
    try {
      const r = await api.labelagentRun(sid.trim(), apply);
      toast(`${apply ? "Label agent queued" : "Dry run queued"} (job ${r.job_id.slice(0, 8)}). It runs after office hours on the GPU worker; follow it in Agent jobs.`);
    } catch (e) {
      toast("Could not queue the label agent: " + humanizeError(e), "warn");
    } finally {
      setBusy(null);
    }
  };

  const rows = report
    ? ORDER.filter((k) => report.stages[k]).map((k) => ({ key: k, r: rep(report.stages[k]) }))
        .filter((x): x is { key: string; r: LabelAgentStageReport } => !!x.r)
    : [];

  return (
    <section className="panel p-3 space-y-3">
      <div className="flex items-end gap-3">
        <div className="min-w-0">
          <h2 className="section-title">Label agent</h2>
          <p className="section-desc mt-0.5">
            Checks each machine label against what people called similar objects. It fixes a class only when
            two independent checks agree, keeps fixed labels in review, and never accepts anything.
          </p>
        </div>
      </div>

      {q.error ? <div className="text-[12px] text-block">{humanizeError(q.error)}</div> : null}
      {!q.error && !report && !q.loading && (
        <EmptyState compact icon="chart" title="No benchmark saved yet">
          Run <code>scripts/label_benchmark.py --stages all --save</code> to score every stage against human rulings.
        </EmptyState>
      )}

      {report && (
        <div className="space-y-1.5">
          <div className="text-[11.5px] text-ink-3">
            Answer key: {num(report.truth.items)} objects people ruled on, across {num(report.truth.sessions)} sessions
            ({num(report.truth.not_object)} were not objects at all). Scored {ago(report.created)}.
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="text-ink-3 text-left">
                  <th className="font-normal py-1 pr-2">Check</th>
                  <th className="font-normal py-1 px-2 text-right" title="Share of wrong labels changed to what the person ruled">Fixed</th>
                  <th className="font-normal py-1 px-2 text-right" title="Share of right labels changed to something wrong, with its 95% upper bound">Broke</th>
                  <th className="font-normal py-1 px-2 text-right" title="Of the labels it changed, how many it changed to the right answer">Changes right</th>
                  <th className="font-normal py-1 pl-2 text-right" title="Boxes people said were not objects, which it caught">Not-object caught</th>
                </tr>
              </thead>
              <tbody>
                {rows.map(({ key, r }) => {
                  const h = r.headline, p = r.pooled;
                  const ok = p.wrong_fix_upper95 != null && p.wrong_fix_upper95 <= 0.02;
                  return (
                    <tr key={key} className={`border-t hairline ${key === "agent" ? "text-ink" : "text-ink-2"}`}>
                      <td className="py-1 pr-2">{NAMES[key] ?? key}</td>
                      <td className="py-1 px-2 text-right tabular-nums">{pct(h.fix_rate, 1)}</td>
                      <td className={`py-1 px-2 text-right tabular-nums ${ok ? "text-pass" : (p.wrong_fix_rate ?? 0) > 0.05 ? "text-block" : ""}`}>
                        {pct(p.wrong_fix_rate, 1)} <span className="text-ink-3">&le; {pct(p.wrong_fix_upper95, 1)}</span>
                      </td>
                      <td className="py-1 px-2 text-right tabular-nums">{pct(h.change_precision, 1)}</td>
                      <td className="py-1 pl-2 text-right tabular-nums">{pct(h.not_object_catch, 1)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 border-t hairline pt-2.5">
        <input value={sid} onChange={(e) => setSid(e.target.value)} placeholder="session id"
          aria-label="session to check with the label agent"
          className="flex-1 min-w-[16rem] bg-bg border border-line rounded px-2 py-1 text-[12px] text-ink" />
        <button type="button" className="btn" disabled={!sid.trim() || !!busy} onClick={() => run(false)}
          title="Report what it would fix, flag and ask, without changing any label">
          {busy === "dry" ? "Queuing..." : "Dry run"}
        </button>
        <button type="button" className="btn btn-primary" disabled={!sid.trim() || !!busy} onClick={() => run(true)}
          title="Apply its fixes and questions as one run that Agent jobs can undo. Needs a reviewer.">
          {busy === "apply" ? "Queuing..." : "Run and apply"}
        </button>
      </div>
    </section>
  );
}
