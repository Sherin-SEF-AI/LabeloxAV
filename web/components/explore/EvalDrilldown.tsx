"use client";

import { useEffect, useState } from "react";
import { api, type GoldSetRow , humanizeError } from "@/lib/api";
import type { ConfusionCell, EvalPatchRow, EvalRun } from "@/lib/types";
import { useQuery } from "@/lib/useQuery";
import { ago, humanize, num, plural } from "@/lib/format";

// Model-versus-gold drill-down. The corrections table elsewhere on this page answers "what did humans fix";
// this answers "where does the model disagree with sealed ground truth", and crucially lets you open a cell
// and look at the actual crops. A confusion count tells you pedestrians are being called poles; only the
// crops tell you whether that is a real failure or a gold-label problem.

const OUTCOME_STYLE: Record<string, string> = {
  tp: "text-pass",
  fp: "text-block",
  fn: "text-ink-2",
};

const OUTCOME_WORDS: Record<string, string> = { tp: "Correct", fp: "False alarm", fn: "Missed" };

function cellKey(c: ConfusionCell) {
  return `${c.gt_class_id ?? "none"}:${c.pred_class_id ?? "none"}:${c.outcome}`;
}

export default function EvalDrilldown() {
  // Both lists are optional here: without them the pickers are empty, which is what a failure has
  // always looked like in this panel, so neither error is rendered.
  const goldsQ = useQuery(["goldSets"], () => api.goldSets());
  const golds: GoldSetRow[] = goldsQ.data ?? [];
  const evalsQ = useQuery(["exploreEvals"], () => api.exploreEvals());
  const evals: EvalRun[] = evalsQ.data?.evals ?? [];
  const [evalId, setEvalId] = useState<string>("");
  // The previous run's matrix stays up while the next loads; a failure reads as an empty matrix.
  const cellsQ = useQuery(evalId ? ["exploreEvalCells", evalId] : null, () => api.exploreEvalCells(evalId),
    { keepPrevious: true });
  const cells: ConfusionCell[] = evalId && !cellsQ.error ? cellsQ.data?.cells ?? [] : [];
  const [active, setActive] = useState<ConfusionCell | null>(null);
  const [patches, setPatches] = useState<EvalPatchRow[]>([]);
  const [goldId, setGoldId] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 4000); };

  // Open on the first gold set and the newest run once they arrive, unless one is already chosen.
  const firstGoldId = golds[0]?.gold_id;
  useEffect(() => { if (firstGoldId) setGoldId((cur) => cur || firstGoldId); }, [firstGoldId]);
  const firstEvalId = evals[0]?.eval_id;
  useEffect(() => { if (firstEvalId) setEvalId((cur) => cur || firstEvalId); }, [firstEvalId]);

  // An open cell and its patches belong to the run they were opened in.
  useEffect(() => {
    if (!evalId) return;
    setActive(null);
    setPatches([]);
  }, [evalId]);

  const openCell = async (c: ConfusionCell) => {
    setActive(c);
    setPatches([]);
    try {
      const r = await api.exploreEvalPatches(evalId, {
        gt_class_id: c.gt_class_id, pred_class_id: c.pred_class_id, outcome: c.outcome,
      });
      setPatches(r.patches);
    } catch (e) { flash(humanizeError(e)); }
  };

  const runEval = async () => {
    if (!goldId) { flash("Seal a gold set first"); return; }
    setBusy(true);
    flash("Scoring machine labels against the gold set...");
    try {
      const r = await api.exploreRunEval(goldId);
      if (r.error) { flash(r.error); return; }
      await evalsQ.reload();
      if (r.eval_id) setEvalId(r.eval_id);
      flash(`${r.tp} correct, ${r.fp} false alarms, ${r.fn} missed`);
    } catch (e) { flash(humanizeError(e)); } finally { setBusy(false); }
  };

  const run = evals.find((e) => e.eval_id === evalId);

  return (
    <div>
      <div className="flex items-center gap-2 mb-3 text-[12.5px] flex-wrap">
        <select value={goldId} onChange={(e) => setGoldId(e.target.value)} aria-label="Gold set"
          className="field text-[12.5px] max-w-[240px]">
          <option value="">Choose a gold set...</option>
          {golds.map((g) => <option key={g.gold_id} value={g.gold_id}>{g.name} ({g.n_objects})</option>)}
        </select>
        <button onClick={runEval} disabled={busy || !goldId}
          className="btn">
          {busy ? "Scoring..." : "Score against gold"}
        </button>

        {evals.length > 0 && (
          <>
            <span className="text-ink-3 ml-2">Run</span>
            <select value={evalId} onChange={(e) => setEvalId(e.target.value)} aria-label="Evaluation run"
              className="field text-[12.5px] max-w-[280px]">
              {evals.map((e) => (
                <option key={e.eval_id} value={e.eval_id}>
                  {e.created_at ? ago(e.created_at) : e.eval_id.slice(0, 8)} · {e.tp} correct, {e.fp} false, {e.fn} missed
                </option>
              ))}
            </select>
          </>
        )}
        {run && (
          <span className="text-ink-3">
            Precision {run.tp + run.fp ? (run.tp / (run.tp + run.fp)).toFixed(3) : "-"}
            {" · "}Recall {run.tp + run.fn ? (run.tp / (run.tp + run.fn)).toFixed(3) : "-"}
          </span>
        )}
        {msg && <span className="text-accent-2 ml-auto">{msg}</span>}
      </div>

      {!cells.length ? (
        <div className="text-[12.5px] text-ink-3 py-4 text-center">
          No evaluation yet. Choose a sealed gold set and score against it.
        </div>
      ) : (
        <div className="flex gap-3 items-start">
          {/* cells */}
          <div className="shrink-0 max-h-[440px] overflow-y-auto pr-1">
          <table className="text-[12.5px]">
            <thead>
              <tr className="text-ink-3 text-left border-b hairline">
                <th className="py-1.5 pr-3 font-medium">Gold</th><th className="pr-3 font-medium">Predicted</th>
                <th className="pr-3 font-medium">Outcome</th><th className="text-right font-medium">Count</th>
              </tr>
            </thead>
            <tbody>
              {cells.slice(0, 40).map((c) => {
                const on = active && cellKey(active) === cellKey(c);
                return (
                  <tr key={cellKey(c)} onClick={() => openCell(c)}
                    title="Open the crops behind this cell"
                    className={`border-b hairline cursor-pointer ${on ? "bg-bg-2" : "hover:bg-bg-2/50"}`}>
                    <td className="py-1.5 pr-3 text-ink-2">{c.gt_class ? humanize(c.gt_class) : "-"}</td>
                    <td className="pr-3 text-ink-2">{c.pred_class ? humanize(c.pred_class) : "-"}</td>
                    <td className={`pr-3 ${OUTCOME_STYLE[c.outcome] ?? "text-ink-3"}`}>{OUTCOME_WORDS[c.outcome] ?? humanize(c.outcome)}</td>
                    <td className="text-right text-ink tabular-nums">{num(c.count)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>

          {/* patch grid for the opened cell */}
          <div className="flex-1 min-w-0">
            {active ? (
              <>
                <div className="text-[12px] text-ink-3 mb-1.5">
                  <span className="text-ink-2">{active.gt_class ? humanize(active.gt_class) : "Nothing"}</span> seen as{" "}
                  <span className="text-ink-2">{active.pred_class ? humanize(active.pred_class) : "nothing"}</span>{" "}
                  <span className={OUTCOME_STYLE[active.outcome]}>({(OUTCOME_WORDS[active.outcome] ?? active.outcome).toLowerCase()})</span>
                  {" · "}{plural(patches.length, "crop")} shown
                </div>
                <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-1.5 max-h-[400px] overflow-auto no-scrollbar">
                  {patches.map((p) => (
                    <a key={p.patch_id} href={p.frame_id ? `/frame/${p.frame_id}` : undefined}
                      title={`IoU ${p.iou ?? "-"}, confidence ${p.conf ?? "-"}. Open the frame`}
                      className="block border border-line rounded-sm overflow-hidden hover:border-accent bg-bg-2">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      {p.crop_url ? <img src={p.crop_url} alt="" loading="lazy" className="w-full h-16 object-cover skeleton"
                            onError={(e) => { e.currentTarget.style.visibility = "hidden"; }} />
                        : <span className="flex w-full h-16 items-center justify-center text-[11px] text-ink-3">No crop</span>}
                    </a>
                  ))}
                </div>
                {!patches.length && (
                  <div className="text-[12px] text-ink-3 py-3">No crops for this cell.</div>
                )}
              </>
            ) : (
              <div className="text-[12px] text-ink-3 py-3">
                Click a row to see the objects behind it.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
