"use client";

import { useCallback, useEffect, useState } from "react";

import { apiGet } from "@/lib/api";
import LoadState from "@/components/shell/LoadState";

// How far the reported recall can be trusted, which is a different question from what the recall is.
//
// Every recall figure on this page is recall against the sealed gold set, and that set was built by people
// confirming machine boxes far more readily than drawing new ones. So it is biased toward what the model
// already sees, and the number is an overestimate by an amount nothing on this page could tell you.
//
// A blind audit gives the same frames to a person with every prediction withheld server-side, and
// capture-recapture over the two independent observations estimates the objects NEITHER found. What comes
// back is recall against a denominator the model did not help build.
//
// Three states, and the difference between them is the entire point of the panel:
//
//   no audit    nothing has been checked. Not "fine": unknown, and it says so rather than showing nothing
//   unmeasured  an audit ran and could not conclude (no object was found by both observers)
//   scored      the two numbers, side by side, with the gap named
//
// The estimate is a lower bound on what was missed, because the two observers are not perfectly
// independent: a small, occluded, badly lit object is harder for both, the captures correlate, and the
// estimated population comes out low. The caveat is rendered, not left in a docstring.

type Slice = {
  stratum: string | null;
  class_id: number | null;
  class_name: string | null;
  measured: boolean;
  reason: string | null;
  population: number | null;
  lo: number | null;
  hi: number | null;
  model_recall: number | null;
  recall_lo: number | null;
  recall_hi: number | null;
  human_recall: number | null;
  gold_recall: number | null;
  overstatement: number | null;
  n_both: number;
  n_model_only: number;
  n_human_only: number;
};

type AuditRow = {
  audit_id: string;
  run_id: string;
  gold_id: string | null;
  job_id: string | null;
  status: string;
  n_frames: number;
  n_labeled: number;
  created_at: string | null;
};

type Estimate = { audit_id: string; status: string; caveat: string; slices: Slice[] };

const pct = (v: number | null | undefined) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);

function Bar({ label, value, tone }: { label: string; value: number | null; tone: string }) {
  return (
    <div className="space-y-0.5">
      <div className="flex justify-between font-mono text-[10px] text-ink-3">
        <span>{label}</span>
        <span className="text-ink tabular-nums">{pct(value)}</span>
      </div>
      <div className="h-2 w-full border hairline overflow-hidden">
        <div className={tone} style={{ width: `${Math.max(0, Math.min(100, (value ?? 0) * 100))}%` }} />
      </div>
    </div>
  );
}

export default function RecallHonesty() {
  const [audits, setAudits] = useState<AuditRow[] | null>(null);
  const [est, setEst] = useState<Estimate | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(async () => {
    setErr(null);
    try {
      const r = await apiGet<{ audits: AuditRow[] }>("/api/verdyx/blind-audits?limit=25");
      setAudits(r.audits);
      const scored = r.audits.find((a) => a.status === "scored");
      setSelected((s) => s ?? scored?.audit_id ?? null);
    } catch (e) {
      setErr(e);
    }
  }, []);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => {
    if (!selected) { setEst(null); return; }
    let live = true;
    void (async () => {
      try {
        const r = await apiGet<Estimate>(`/api/verdyx/blind-audit/${selected}/estimate`);
        if (live) setEst(r);
      } catch (e) {
        if (live) setErr(e);
      }
    })();
    return () => { live = false; };
  }, [selected]);

  if (err != null) return <LoadState error={err} onRetry={() => void load()} />;
  if (audits == null) {
    return <div className="font-mono text-xs text-ink-3/50 animate-pulse py-4 text-center">loading...</div>;
  }

  if (audits.length === 0) {
    return (
      <div className="space-y-2 font-mono text-[11px] text-ink-3">
        <div className="text-ink">gold recall on this page has never been checked against an independent observer.</div>
        <p className="leading-relaxed">
          Gold recall counts the objects somebody already found. Confirming a machine box takes one click and
          drawing a missed one takes half a minute, so the sealed set leans toward what the model already
          sees and the figure above is an overestimate by an unknown amount.
        </p>
        <p className="leading-relaxed">
          A blind audit closes that. Seed one against an inference run, label the frames with every
          prediction withheld, and the two independent observations give recall against a denominator the
          model did not help build.
        </p>
        <div className="text-ink-4">POST /api/verdyx/blind-audit/seed with a run_id to start one.</div>
      </div>
    );
  }

  const pooled = est?.slices.find((s) => s.stratum == null && s.class_id == null) ?? null;
  const strata = est?.slices.filter((s) => s.stratum != null && s.class_id == null) ?? [];
  const classes = est?.slices.filter((s) => s.class_id != null) ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-1.5 font-mono text-[11px]">
        {audits.map((a) => (
          <button
            key={a.audit_id}
            onClick={() => setSelected(a.audit_id)}
            className={`border px-2 py-0.5 ${a.audit_id === selected ? "border-accent text-ink" : "border-line text-ink-3 hover:border-accent"}`}
          >
            {a.audit_id.slice(0, 8)}
            <span className="text-ink-4 ml-1.5">
              {a.status === "scored" ? "scored" : `${a.n_labeled}/${a.n_frames} labelled`}
            </span>
          </button>
        ))}
      </div>

      {pooled == null ? (
        <div className="font-mono text-[11px] text-ink-3 py-3">
          {est == null
            ? "loading estimate..."
            : "this audit has not been scored yet, so there is no estimate to show. Nothing here is a "
              + "statement about the model until the frames are labelled and scored."}
        </div>
      ) : !pooled.measured ? (
        <div className="space-y-1.5 font-mono text-[11px]">
          <div className="text-warn">the audit ran and could not conclude</div>
          <div className="text-ink-3 leading-relaxed">{pooled.reason}</div>
          <div className="text-ink-4">
            Found by both {pooled.n_both}, by the model alone {pooled.n_model_only}, by the auditor alone{" "}
            {pooled.n_human_only}. With no overlap the population is unbounded above, so no number is
            reported rather than a number that would mean nothing.
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Bar label="gold recall (what this page reports)" value={pooled.gold_recall} tone="bg-ink-3" />
            <Bar label="recall against the estimated population" value={pooled.model_recall} tone="bg-accent" />
          </div>

          {pooled.overstatement != null && (
            <div className={`font-mono text-[11px] px-2 py-1.5 border hairline ${pooled.overstatement > 0.15 ? "text-fail" : "text-ink"}`}>
              gold recall overstates measured recall by {(pooled.overstatement * 100).toFixed(1)} points
              {pooled.overstatement > 0.15 && " - past the promotion tolerance, so the gate refuses on it"}
            </div>
          )}

          <div className="font-mono text-[11px] text-ink-3 grid grid-cols-2 sm:grid-cols-4 gap-2">
            <div>
              <div className="text-ink-4 uppercase text-[10px]">estimated population</div>
              <div className="text-ink tabular-nums">{pooled.population?.toFixed(1)}</div>
              <div className="text-ink-4 tabular-nums">
                {pooled.lo?.toFixed(0)} to {pooled.hi?.toFixed(0)} at 95%
              </div>
            </div>
            <div>
              <div className="text-ink-4 uppercase text-[10px]">found by both</div>
              <div className="text-ink tabular-nums">{pooled.n_both}</div>
            </div>
            <div>
              <div className="text-ink-4 uppercase text-[10px]">model only</div>
              <div className="text-ink tabular-nums">{pooled.n_model_only}</div>
            </div>
            <div>
              <div className="text-ink-4 uppercase text-[10px]">auditor only</div>
              <div className="text-ink tabular-nums">{pooled.n_human_only}</div>
            </div>
          </div>

          {strata.length > 0 && (
            <div className="space-y-1">
              <div className="font-mono text-[10px] uppercase text-ink-3">by scene density</div>
              <div className="overflow-x-auto">
                <table className="w-full font-mono text-[11px]">
                  <thead className="text-ink-4">
                    <tr className="text-left">
                      <th className="py-1 pr-3 font-normal">stratum</th>
                      <th className="py-1 pr-3 font-normal">recall</th>
                      <th className="py-1 pr-3 font-normal">population</th>
                      <th className="py-1 pr-3 font-normal">both / model / auditor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {strata.map((s) => (
                      <tr key={s.stratum} className="border-t hairline">
                        <td className="py-1 pr-3 text-ink">{s.stratum}</td>
                        <td className="py-1 pr-3 tabular-nums text-ink">
                          {s.measured ? `${pct(s.model_recall)} (${pct(s.recall_lo)} to ${pct(s.recall_hi)})` : "not measurable"}
                        </td>
                        <td className="py-1 pr-3 tabular-nums text-ink-3">
                          {s.measured ? s.population?.toFixed(1) : "-"}
                        </td>
                        <td className="py-1 pr-3 tabular-nums text-ink-3">
                          {s.n_both} / {s.n_model_only} / {s.n_human_only}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {classes.length > 0 && (
            <div className="space-y-1">
              <div className="font-mono text-[10px] uppercase text-ink-3">by class</div>
              <div className="flex flex-wrap gap-1">
                {classes.map((s) => (
                  <span
                    key={s.class_id}
                    title={s.measured
                      ? `${s.n_both} found by both, ${s.n_model_only} model only, ${s.n_human_only} auditor only`
                      : (s.reason ?? "not measurable")}
                    className={`border hairline px-1.5 py-0.5 font-mono text-[10px] ${s.measured ? "text-ink" : "text-ink-4"}`}
                  >
                    {s.class_name ?? s.class_id} {s.measured ? pct(s.model_recall) : "n/a"}
                  </span>
                ))}
              </div>
              <div className="font-mono text-[10px] text-ink-4">
                class rows are class-aware (a box found under the wrong name is a miss for that class), so
                they deliberately do not sum to the pooled figure above
              </div>
            </div>
          )}
        </div>
      )}

      {est?.caveat && (
        <div className="font-mono text-[10px] text-ink-4 leading-relaxed border-t hairline pt-2">
          {est.caveat}
        </div>
      )}
    </div>
  );
}
