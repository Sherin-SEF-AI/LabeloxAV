"use client";

// A vision model's opinion on one label, with its one-sentence reason (tool #13). Reusable: mount it anywhere an
// object is reviewed.
//
//   <VlmReason objectId={id} />                         loads the stored check, offers "Ask the vision model"
//   <VlmReason objectId={id} check={c} />               shows a check you already fetched (no request)
//   <VlmReason objectId={id} compact />                 one line, for a list row or a tile tooltip
//
// The verdict is a machine's (it lives in machine_verdict, never in review) and says so: the reason is there so a
// person can overrule it in a glance, and a check made before the label last changed is marked as about an old
// label rather than shown as if it applied.

import { useEffect, useState } from "react";

import { humanizeError } from "@/lib/api";
import { ago, humanize } from "@/lib/format";
import { verdictLabel, verdictTone } from "@/lib/tools/quality";
import { checkLabels, getVlmChecks, type VlmCheck } from "@/lib/tools/qualityApi";

const TONE_TEXT = { pass: "text-pass", warn: "text-warn", block: "text-block", muted: "text-ink-3", info: "text-info" };

export default function VlmReason({ objectId, check, compact = false, canAsk = true, onChecked }: {
  objectId: string;
  /** A check already in hand. Undefined: this component loads the stored one. Null: known to be unchecked. */
  check?: VlmCheck | null;
  compact?: boolean;
  /** Show the button that asks the model now (a few seconds of local GPU). */
  canAsk?: boolean;
  onChecked?: (c: VlmCheck) => void;
}) {
  const [c, setC] = useState<VlmCheck | null>(check ?? null);
  const [loading, setLoading] = useState(check === undefined);
  const [asking, setAsking] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (check !== undefined) { setC(check); setLoading(false); return; }
    let stop = false;
    setLoading(true); setErr(null);
    getVlmChecks([objectId])
      .then((r) => { if (!stop) setC(r.checks[objectId] ?? null); })
      .catch((e) => { if (!stop) setErr(humanizeError(e)); })
      .finally(() => { if (!stop) setLoading(false); });
    return () => { stop = true; };
  }, [objectId, check]);

  const ask = async () => {
    setAsking(true); setErr(null);
    try {
      const r = await checkLabels([objectId]);
      const got = r.results[objectId];
      if (got) { setC(got); onChecked?.(got); }
      else setErr(r.errors[0] ?? "The model gave no answer for this label.");
    } catch (e) {
      setErr(humanizeError(e));
    } finally {
      setAsking(false);
    }
  };

  const tone = verdictTone(c?.verdict);
  const askBtn = canAsk && (
    <button type="button" onClick={ask} disabled={asking} className="btn text-[11.5px]"
      data-tip="Ask the local vision model whether this label is right. Takes a few seconds.">
      {asking ? "Asking..." : c ? "Ask again" : "Ask the vision model"}
    </button>
  );

  if (compact) {
    if (loading) return <span className="text-[11px] text-ink-3">VLM...</span>;
    if (!c) return canAsk ? askBtn || null : <span className="text-[11px] text-ink-3">not checked</span>;
    return (
      <span className={`text-[11.5px] ${TONE_TEXT[tone]}`} title={`${c.reason}${c.stale ? " (about an earlier label)" : ""}`}>
        VLM: {verdictLabel(c)}{c.stale ? " (old label)" : ""}
      </span>
    );
  }

  return (
    <div className="rounded border hairline bg-bg-2/60 px-2.5 py-2 space-y-1.5" aria-live="polite">
      <div className="flex items-center gap-2">
        <span className="label-caps">Vision model</span>
        {c && <span className={`text-[12px] font-medium ${TONE_TEXT[tone]}`}>{verdictLabel(c)}</span>}
        <span className="ml-auto">{askBtn}</span>
      </div>
      {loading && <div className="text-[12px] text-ink-3" role="status">Loading the last check...</div>}
      {!loading && !c && !err && (
        <div className="text-[12px] text-ink-3">Not checked yet.</div>
      )}
      {c && (
        <>
          <p className="text-[12.5px] text-ink leading-snug">&ldquo;{c.reason || "No reason given."}&rdquo;</p>
          <div className="text-[11px] text-ink-3 flex flex-wrap gap-x-2">
            {c.given_class && <span>about the label {humanize(c.given_class)}</span>}
            {c.stale && <span className="text-warn">the label has changed since</span>}
            <span>{c.model}</span>
            {c.latency_ms != null && <span>{(c.latency_ms / 1000).toFixed(1)} s</span>}
            <span>{c.cost_usd === 0 ? "local, no cost" : "cost in the LLM ledger"}</span>
            {c.checked_at && <span>{ago(c.checked_at)}</span>}
          </div>
        </>
      )}
      {err && <div className="text-[12px] text-block" role="alert">{err}</div>}
    </div>
  );
}
