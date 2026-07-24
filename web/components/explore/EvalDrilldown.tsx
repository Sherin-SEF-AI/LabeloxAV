"use client";

import { useEffect, useState } from "react";
import { api, type GoldSetRow } from "@/lib/api";
import type { ConfusionCell, EvalPatchRow, EvalRun } from "@/lib/types";

// Model-versus-gold drill-down. The corrections table elsewhere on this page answers "what did humans fix";
// this answers "where does the model disagree with sealed ground truth", and crucially lets you open a cell
// and look at the actual crops. A confusion count tells you pedestrians are being called poles; only the
// crops tell you whether that is a real failure or a gold-label problem.

const OUTCOME_STYLE: Record<string, string> = {
  tp: "text-pass",
  fp: "text-block",
  fn: "text-ink-2",
};

function cellKey(c: ConfusionCell) {
  return `${c.gt_class_id ?? "none"}:${c.pred_class_id ?? "none"}:${c.outcome}`;
}

export default function EvalDrilldown() {
  const [golds, setGolds] = useState<GoldSetRow[]>([]);
  const [evals, setEvals] = useState<EvalRun[]>([]);
  const [evalId, setEvalId] = useState<string>("");
  const [cells, setCells] = useState<ConfusionCell[]>([]);
  const [active, setActive] = useState<ConfusionCell | null>(null);
  const [patches, setPatches] = useState<EvalPatchRow[]>([]);
  const [goldId, setGoldId] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(null), 4000); };

  useEffect(() => {
    api.goldSets().then((g) => {
      setGolds(g);
      if (g.length && !goldId) setGoldId(g[0].gold_id);
    }).catch(() => {});
    api.exploreEvals().then((r) => {
      setEvals(r.evals);
      if (r.evals.length && !evalId) setEvalId(r.evals[0].eval_id);
    }).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!evalId) { setCells([]); return; }
    setActive(null);
    setPatches([]);
    api.exploreEvalCells(evalId).then((r) => setCells(r.cells)).catch(() => setCells([]));
  }, [evalId]);

  const openCell = async (c: ConfusionCell) => {
    setActive(c);
    setPatches([]);
    try {
      const r = await api.exploreEvalPatches(evalId, {
        gt_class_id: c.gt_class_id, pred_class_id: c.pred_class_id, outcome: c.outcome,
      });
      setPatches(r.patches);
    } catch (e) { flash(String(e)); }
  };

  const runEval = async () => {
    if (!goldId) { flash("seal a gold set first"); return; }
    setBusy(true);
    flash("scoring machine labels against the gold set...");
    try {
      const r = await api.exploreRunEval(goldId);
      if (r.error) { flash(r.error); return; }
      const list = await api.exploreEvals();
      setEvals(list.evals);
      if (r.eval_id) setEvalId(r.eval_id);
      flash(`tp ${r.tp} / fp ${r.fp} / fn ${r.fn}`);
    } catch (e) { flash(String(e)); } finally { setBusy(false); }
  };

  const run = evals.find((e) => e.eval_id === evalId);

  return (
    <div>
      <div className="flex items-center gap-2 mb-2 font-mono text-[11px] flex-wrap">
        <select value={goldId} onChange={(e) => setGoldId(e.target.value)}
          className="bg-bg border border-line px-1.5 py-0.5 text-ink max-w-[220px]">
          <option value="">pick a gold set</option>
          {golds.map((g) => <option key={g.gold_id} value={g.gold_id}>{g.name} ({g.n_objects})</option>)}
        </select>
        <button onClick={runEval} disabled={busy || !goldId}
          className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
          {busy ? "scoring..." : "score against gold"}
        </button>

        {evals.length > 0 && (
          <>
            <span className="text-ink-3 ml-2">run:</span>
            <select value={evalId} onChange={(e) => setEvalId(e.target.value)}
              className="bg-bg border border-line px-1.5 py-0.5 text-ink max-w-[260px]">
              {evals.map((e) => (
                <option key={e.eval_id} value={e.eval_id}>
                  {(e.created_at ?? e.eval_id).slice(0, 19)} - tp {e.tp} fp {e.fp} fn {e.fn}
                </option>
              ))}
            </select>
          </>
        )}
        {run && (
          <span className="text-ink-3">
            precision {run.tp + run.fp ? (run.tp / (run.tp + run.fp)).toFixed(3) : "-"}
            {"  "}recall {run.tp + run.fn ? (run.tp / (run.tp + run.fn)).toFixed(3) : "-"}
          </span>
        )}
        {msg && <span className="text-accent ml-auto">{msg}</span>}
      </div>

      {!cells.length ? (
        <div className="font-mono text-xs text-ink-3 py-4 text-center">
          no evaluation yet. pick a sealed gold set and score against it.
        </div>
      ) : (
        <div className="flex gap-3 items-start">
          {/* cells */}
          <table className="font-mono text-[11px] shrink-0">
            <thead>
              <tr className="text-ink-3 text-left border-b hairline">
                <th className="py-1 pr-3">gold</th><th className="pr-3">predicted</th>
                <th className="pr-3">outcome</th><th className="text-right">count</th>
              </tr>
            </thead>
            <tbody>
              {cells.slice(0, 40).map((c) => {
                const on = active && cellKey(active) === cellKey(c);
                return (
                  <tr key={cellKey(c)} onClick={() => openCell(c)}
                    title="open the crops behind this cell"
                    className={`border-b hairline cursor-pointer ${on ? "bg-bg-2" : "hover:bg-bg-2/50"}`}>
                    <td className="py-1 pr-3 text-ink-2">{c.gt_class ?? "-"}</td>
                    <td className="pr-3 text-ink-2">{c.pred_class ?? "-"}</td>
                    <td className={`pr-3 ${OUTCOME_STYLE[c.outcome] ?? "text-ink-3"}`}>{c.outcome}</td>
                    <td className="text-right text-ink">{c.count}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {/* patch grid for the opened cell */}
          <div className="flex-1 min-w-0">
            {active ? (
              <>
                <div className="font-mono text-[11px] text-ink-3 mb-1.5">
                  {active.gt_class ?? "nothing"} <span className="text-ink-3">seen as</span>{" "}
                  {active.pred_class ?? "nothing"}{" "}
                  <span className={OUTCOME_STYLE[active.outcome]}>({active.outcome})</span>
                  {" - "}{patches.length} shown
                </div>
                <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-1.5 max-h-72 overflow-auto no-scrollbar">
                  {patches.map((p) => (
                    <a key={p.patch_id} href={p.frame_id ? `/frame/${p.frame_id}` : undefined}
                      title={`iou ${p.iou ?? "-"} conf ${p.conf ?? "-"} - open the frame`}
                      className="block border border-line hover:border-accent">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      {p.crop_url && <img src={p.crop_url} alt="" className="w-full h-16 object-cover bg-bg-2" />}
                    </a>
                  ))}
                </div>
                {!patches.length && (
                  <div className="font-mono text-[11px] text-ink-3 py-3">no crops for this cell</div>
                )}
              </>
            ) : (
              <div className="font-mono text-[11px] text-ink-3 py-3">
                click a row to see the objects behind it.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
