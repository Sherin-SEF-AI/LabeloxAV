"use client";

// M-F.0 explainable auto-labeling: the decision story for one object, fetched from /objects/{id}/explain and
// rendered as a plain-language rationale plus the structured factors (paths + scores, agreement, VLM verdict,
// calibrated confidence vs the auto-accept floor, quality flags, conflicts). It adds no inference; it shows
// what the fusion and gate already recorded, so a reviewer or a buyer can see why a label was accepted.

import { useEffect, useState } from "react";
import { api , humanizeError } from "@/lib/api";
import { Busy } from "@/components/Spinner";
import { toast } from "@/lib/toast";
import type { LabelAgentTrace, ObjectExplanation } from "@/lib/types";

const DECISION_COLOR: Record<string, string> = {
  auto_accept: "text-pass border-pass",
  review: "text-warn border-warn",
  annotate: "text-block border-block",
};

const AGENT_ACTION: Record<string, { label: string; cls: string }> = {
  fix: { label: "class fixed", cls: "border-accent/60 text-accent-2" },
  flag: { label: "maybe not an object", cls: "border-block/50 text-block" },
  ask: { label: "question", cls: "border-warn/50 text-warn" },
};

// The label agent's second opinion on the class: what it did, the question it asks the reviewer, the votes
// behind it, and a way to take a fix back. Votes are shown with their margins because the agent acts on the
// margin, not on a probability, and a reviewer deciding whether to trust it should see the same number.
function LabelAgentBlock({ la, objectId, onUndone }: { la: LabelAgentTrace; objectId: string; onUndone: () => void }) {
  const [busy, setBusy] = useState(false);
  const a = AGENT_ACTION[la.action] ?? { label: la.action, cls: "border-line text-ink-3" };
  const undo = async () => {
    setBusy(true);
    try {
      await api.labelagentUndo(objectId);
      toast("Undone. Reload the frame to see the original label.");
      onUndone();
    } catch (e) {
      toast("Could not undo: " + humanizeError(e), "warn");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="border-t hairline pt-1.5 space-y-1">
      <div className="flex items-center justify-between gap-2">
        <span className="label-caps">Label agent</span>
        <span className={`border px-1 rounded ${a.cls}`}>{a.label}</span>
      </div>
      {la.action === "fix" && <div>{la.from} &rarr; <span className="text-ink">{la.to}</span></div>}
      {la.question && <div className="text-ink">{la.question}</div>}
      {Object.entries(la.votes).map(([who, v]) => (
        <div key={who} className="flex items-center justify-between gap-2">
          <span className="text-ink-3">{who}</span>
          <span className="truncate">{v.label} <span className="text-ink-3">margin {v.margin.toFixed(2)}</span></span>
        </div>
      ))}
      {la.undoable && la.action !== "ask" && (
        <button onClick={undo} disabled={busy} aria-label="undo the label agent's change to this object"
          className="text-[11px] border border-line rounded px-1.5 py-0.5 text-ink-2 hover:border-block hover:text-block disabled:opacity-40">
          {busy ? "undoing..." : la.action === "fix" ? `undo: back to ${la.from}` : "undo"}
        </button>
      )}
    </div>
  );
}

export default function ExplainPanel({ objectId }: { objectId: string }) {
  const [ex, setEx] = useState<ObjectExplanation | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let live = true;
    setEx(null);
    setErr(null);
    api.explainObject(objectId).then((e) => live && setEx(e)).catch((e) => live && setErr(humanizeError(e)));
    return () => { live = false; };
  }, [objectId, nonce]);

  if (err) return <div className="text-[11px] text-block px-1">{err}</div>;
  if (!ex) return <div className="text-[11px] text-ink-3 px-1 flex items-center gap-1.5"><Busy />explaining...</div>;

  const cal = ex.calibration;
  const dc = DECISION_COLOR[ex.machine_decision] ?? "text-ink-3 border-line";

  return (
    <div className="text-[11.5px] text-ink-2 space-y-2 reveal">
      <div className="flex items-center gap-2">
        <span className={`border px-1.5 py-0.5 rounded uppercase ${dc}`}>{ex.machine_decision}</span>
        {ex.rare && <span className="border border-info/50 text-info px-1.5 py-0.5 rounded uppercase">rare</span>}
        <span className="text-ink-3">{ex.class_name}</span>
      </div>

      {/* plain-language rationale */}
      <ul className="space-y-1">
        {ex.summary.map((s, i) => (
          <li key={i} className="flex gap-1.5 leading-snug">
            <span className="text-ink-3 shrink-0">·</span><span>{s}</span>
          </li>
        ))}
      </ul>

      {ex.label_agent && <LabelAgentBlock la={ex.label_agent} objectId={objectId} onUndone={() => setNonce((n) => n + 1)} />}

      {/* structured factors */}
      <div className="border-t hairline pt-1.5 space-y-1">
        <div className="label-caps">Paths</div>
        {ex.paths.length === 0 && <div className="text-ink-3">no model proposals recorded</div>}
        {ex.paths.map((p, i) => (
          <div key={i} className="flex items-center justify-between gap-2">
            <span className="truncate">{p.label} <span className="text-ink-3">{p.class_name}</span></span>
            <span className="flex items-center gap-1.5 shrink-0">
              <span className={p.verdict === "overruled" ? "text-block" : "text-ink-3"}>{p.verdict}</span>
              <span className="text-ink">{p.conf.toFixed(2)}</span>
            </span>
          </div>
        ))}
      </div>

      {cal.calibrated != null && (
        <div className="border-t hairline pt-1.5 flex items-center justify-between">
          <span className="label-caps">Calibrated confidence</span>
          <span>
            <span className={cal.auto_accept_floor != null && cal.calibrated >= cal.auto_accept_floor ? "text-pass" : "text-warn"}>{cal.calibrated}</span>
            <span className="text-ink-3"> / floor {cal.auto_accept_floor} · raw {cal.raw}</span>
          </span>
        </div>
      )}

      <div className="flex flex-wrap gap-1.5 pt-0.5">
        <span className={`border px-1 rounded ${ex.agreement ? "border-pass/50 text-pass" : "border-line text-ink-3"}`}>
          {ex.agreement ? "paths agree" : "no agreement"}
        </span>
        {ex.vlm && <span className={`border px-1 rounded ${ex.vlm.confirmed ? "border-pass/50 text-pass" : "border-block/50 text-block"}`}>
          VLM {ex.vlm.confirmed ? "confirmed" : "unconfirmed"}
        </span>}
        {ex.mask_box_disagree && <span className="border border-block/50 text-block px-1 rounded">mask/box disagree</span>}
        {ex.quality_flags.map((q) => <span key={q} className="border border-block/50 text-block px-1 rounded">{q}</span>)}
      </div>
    </div>
  );
}
