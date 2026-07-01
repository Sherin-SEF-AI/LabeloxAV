"use client";

import { useState } from "react";
import { api, type AgentPlan } from "@/lib/api";

// The frame agent, surfaced in the editor. It runs a dry-run plan first (writes nothing), shows what it
// would auto-accept vs route to a human plus any consistency-critic flags, and only then lets a reviewer
// commit. Every commit is one reversible run, so the Revert button undoes it exactly. This is the
// "human supervises exceptions" surface: you see the 80% the system is sure about before it touches them.

export default function AgentPanel({ frameId, onApplied }: { frameId: string; onApplied?: () => void }) {
  const [plan, setPlan] = useState<AgentPlan | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);

  const doPlan = async () => {
    setBusy("plan"); setMsg(null); setRunId(null);
    try { setPlan(await api.agentPlan(frameId)); }
    catch (e) { setMsg("plan failed: " + String(e)); }
    finally { setBusy(null); }
  };
  const doCommit = async () => {
    setBusy("commit"); setMsg(null);
    try {
      const r = await api.agentRun(frameId);
      setRunId(r.run_id);
      setMsg(`committed: ${r.applied} changed`);
      setPlan(null);
      onApplied?.();
    } catch (e) { setMsg("commit failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };
  const doRevert = async () => {
    if (!runId) return;
    setBusy("revert"); setMsg(null);
    try {
      const r = await api.agentRevert(runId);
      setMsg(`reverted ${r.reverted}${r.skipped ? `, skipped ${r.skipped}` : ""}`);
      setRunId(null);
      onApplied?.();
    } catch (e) { setMsg("revert failed: " + String(e)); }
    finally { setBusy(null); }
  };

  const c = plan?.counts;
  return (
    <div className="border-t hairline pt-2 mt-2">
      <div className="flex items-center gap-2 px-1 pb-1.5">
        <span className="font-display text-[10px] font-semibold uppercase tracking-wider text-ink-3">Agent</span>
        <span className="font-mono text-[9px] text-ink-3/70">auto-accept the sure ones</span>
        <button onClick={doPlan} disabled={!!busy}
          className="ml-auto font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-50">
          {busy === "plan" ? "planning..." : "dry-run"}
        </button>
      </div>

      {c && (
        <div className="px-1 space-y-1.5">
          <div className="grid grid-cols-3 gap-1 font-mono text-[10px]">
            <span className="text-pass">accept {c.auto_accept}</span>
            <span className="text-warn">review {c.review}</span>
            <span className="text-ink-3">annotate {c.annotate}</span>
          </div>
          {(c.demoted_by_critic > 0 || Object.keys(plan!.critic_flags).length > 0) && (
            <div className="font-mono text-[9px] text-ink-3">
              critic vetoed {c.demoted_by_critic}
              {Object.keys(plan!.critic_flags).length > 0 && (
                <span className="text-ink-3/70"> ({Object.entries(plan!.critic_flags).map(([k, n]) => `${k}:${n}`).join(", ")})</span>
              )}
            </div>
          )}
          <div className="max-h-32 overflow-auto no-scrollbar space-y-0.5">
            {plan!.items.filter((i) => i.changes_state).slice(0, 40).map((i) => (
              <div key={i.object_id} title={i.reason} className="flex items-center gap-1.5 font-mono text-[9.5px]">
                <span className={i.action === "auto_accept" ? "text-pass" : i.action === "review" ? "text-warn" : "text-ink-3"}>
                  {i.action === "auto_accept" ? "✓" : i.action === "review" ? "○" : "✎"}
                </span>
                <span className="text-ink-2 truncate flex-1">{i.class_name}</span>
                <span className="text-ink-3/70">{i.conf.toFixed(2)}</span>
                {!i.critic_ok && <span className="text-block" title={i.critic_reasons.join("; ")}>!</span>}
              </div>
            ))}
            {plan!.items.filter((i) => i.changes_state).length === 0 && (
              <div className="font-mono text-[9.5px] text-ink-3">no changes — already settled</div>
            )}
          </div>
          <button onClick={doCommit} disabled={!!busy || c.auto_accept + c.review + c.annotate === c.unchanged}
            className="w-full font-mono text-[10px] border border-accent/40 bg-accent/10 text-accent px-2 py-1 rounded hover:bg-accent/20 disabled:opacity-40">
            {busy === "commit" ? "committing..." : "commit (reversible)"}
          </button>
        </div>
      )}

      {runId && (
        <div className="px-1 pt-1.5">
          <button onClick={doRevert} disabled={!!busy}
            className="w-full font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-block hover:text-block disabled:opacity-50">
            {busy === "revert" ? "reverting..." : "revert last run"}
          </button>
        </div>
      )}
      {msg && <div className="px-1 pt-1 font-mono text-[9.5px] text-ink-3">{msg}</div>}
    </div>
  );
}
