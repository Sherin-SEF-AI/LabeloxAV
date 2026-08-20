"use client";

import { useEffect, useState } from "react";
import { api, type AgentPlan, type ReanalyzeResult, humanizeError } from "@/lib/api";
import { trackOp, trackRun } from "@/lib/canvasOps";
import { Busy } from "@/components/Spinner";
import { OpPrecisionChip, fetchOpState } from "@/components/agent/OpPrecision";

// The frame agent, surfaced in the editor. It runs a dry-run plan first (writes nothing), shows what it
// would auto-accept vs route to a human plus any consistency-critic flags, and only then lets a reviewer
// commit. Every commit is one reversible run, so the Revert button undoes it exactly. This is the
// "human supervises exceptions" surface: you see the 80% the system is sure about before it touches them.

export default function AgentPanel({ frameId, sessionId, selectedId, onApplied, embedded = false }: { frameId: string; sessionId?: string | null; selectedId?: string | null; onApplied?: () => void; embedded?: boolean }) {
  const [plan, setPlan] = useState<AgentPlan | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [previewed, setPreviewed] = useState<Set<string>>(new Set());
  const [msg, setMsg] = useState<string | null>(null);
  const [intent, setIntent] = useState<{ action: string; classes: string[] | string;
                                         conf_min: number | null } | null>(null);
  const [cmd, setCmd] = useState("");
  const [suggestions, setSuggestions] = useState<{ action: string; label: string }[]>([]);
  const [reanalysis, setReanalysis] = useState<ReanalyzeResult | null>(null);
  const [copilot, setCopilot] = useState<{ pattern: { from_name: string; to_name: string; to_class: number; count: number } | null; candidates: string[] } | null>(null);

  useEffect(() => {
    let live = true;
    api.agentSuggest(frameId).then((r) => { if (live) setSuggestions(r.suggestions.slice(0, 3)); }).catch(() => {});
    api.agentCopilotPattern().then((r) => { if (live) setCopilot(r); }).catch(() => {});
    return () => { live = false; };
  }, [frameId]);

  const doBatchFix = async () => {
    if (!(await guardUnmeasured("copilot_batch", "batch-fix"))) return;
    if (!copilot?.pattern || !copilot.candidates.length) return;
    setBusy("batch"); setMsg(null);
    try {
      const r = await api.agentCopilotBatchFix(copilot.candidates, copilot.pattern.to_class);
      setMsg(`batch-fixed ${r.relabeled} similar ${copilot.pattern.from_name} -> ${copilot.pattern.to_name}, routed to review (reversible)`);
      setCopilot(null); onApplied?.();
    } catch (e) { setMsg("batch fix failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doAttributes = async () => {
    if (!(await guardUnmeasured("attribute", "fill attributes"))) return;
    setBusy("attrs"); setMsg(null);
    try {
      const p = await api.agentAttributesPlan(frameId);
      if (!p.counts.attrs_filled) { setMsg("no derivable attributes to fill on this frame"); return; }
      const r = await api.agentAttributes(frameId);
      const by = Object.entries(r.counts.by_attr).map(([k, n]) => `${k}:${n}`).join(", ");
      setMsg(`filled ${r.counts.attrs_filled} attrs on ${r.objects_updated} objects (${by})`);
      onApplied?.();
    } catch (e) { setMsg("attribute fill failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doRelabel = async () => {
    if (!(await guardUnmeasured("relabel", "relabel this frame"))) return;
    setBusy("relabel"); setMsg(null);
    try {
      const p = await api.agentRelabelPlan(frameId);
      if (!p.counts.relabel_keep && !p.counts.relabel_review) { setMsg("labels look right (no confident corrections)"); return; }
      const preview = p.items.slice(0, 3).map((i) => `${i.from_name}→${i.to_name}`).join(", ");
      const r = await api.agentRelabel(frameId);
      setMsg(`relabeled ${r.relabeled} (${r.counts.relabel_keep} fixed, ${r.counts.relabel_review} to review): ${preview}`);
      onApplied?.();
    } catch (e) { setMsg("relabel failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doCuboids = async () => {
    if (!(await guardUnmeasured("cuboid", "fit 3D boxes"))) return;
    setBusy("cuboids"); setMsg(null);
    try {
      const p = await api.agentCuboidsPlan(frameId);
      if (!p.counts.total) { setMsg("no liftable vehicles/VRUs on this frame"); return; }
      const r = await api.agentCuboids(frameId);
      setMsg(`fit ${r.attached} 3D boxes (${r.counts.auto_accept} auto, ${r.counts.review} review, ${r.counts.skip} skipped)`);
      onApplied?.();
    } catch (e) { setMsg("fit 3D failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doCrossCam = async () => {
    if (!selectedId) return;
    setBusy("xcam"); setMsg(null);
    try {
      const p = await api.agentCrossCamPlan(selectedId);
      if (!p.counts.targets) { setMsg(p.reason ? p.reason : "no other synchronized cameras for this object"); return; }
      const r = await api.agentCrossCam(selectedId);
      setMsg(r.created ? `propagated to ${r.created} camera view${r.created > 1 ? "s" : ""}` : "not visible in any other camera");
      onApplied?.();
    } catch (e) { setMsg("cross-camera failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doPropagate = async () => {
    if (!selectedId) return;
    setBusy("track"); setMsg(null);
    try {
      const p = await api.agentPropagatePlan(selectedId);
      if (p.counts.total_steps === 0) { setMsg("nothing to propagate (no adjacent frames / drifts immediately)"); return; }
      const r = await api.agentPropagate(selectedId);
      setMsg(`tracked across ${r.created} frames (${r.counts.auto_accept} auto, ${r.counts.review} review${p.counts.appearance_used ? "" : ", no drift model"})`);
      onApplied?.();
    } catch (e) { setMsg("auto-track failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doCommand = async () => {
    const text = cmd.trim();
    if (!text) return;
    setBusy("cmd"); setMsg(null);
    try {
      const r = await api.agentCommand(frameId, text);
      setMsg(r.summary);
      // The resolved intent, shown rather than discarded. services/agent/nl.py returns it precisely so a
      // person sees what the agent understood before and after it acts, and this panel was reading it only
      // to decide whether to refresh. An agent whose interpretation is invisible is one you have to trust
      // instead of check.
      setIntent(r.intent);
      if (["accept", "revert"].includes(r.intent.action) && !r.blocked) onApplied?.();
    // Clear the intent on failure: a stale "understood as accept" beside an error reads as though
    // the agent acted, which is the opposite of what happened.
    } catch (e) { setIntent(null); setMsg("command failed: " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doPlan = async () => {
    setBusy("plan"); setMsg(null); setRunId(null);
    try { setPlan(await api.agentPlan(frameId)); }
    catch (e) { setMsg("plan failed: " + humanizeError(e)); }
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
    } catch (e) { setMsg("commit failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };
  // An operation with no measurement behind it cannot apply straight away. It has to be previewed first,
  // and what it produces goes to review rather than being auto-accepted. This is the product behaviour the
  // 404 from the harness is for, not a degraded mode: the alternative is a button that writes to the corpus
  // on the strength of a number nobody computed.
  const guardUnmeasured = async (opType: string, label: string): Promise<boolean> => {
    const st = await fetchOpState(opType);
    if (st.measured) return true;
    if (previewed.has(opType)) return true;
    setPreviewed((v) => new Set(v).add(opType));
    setMsg(`${label} has no measured precision. Previewed instead of applied; press again to apply, and `
      + "results will route to review.");
    return false;
  };

  // Reanalyse is the one action here that previews before it applies no matter how well measured it is.
  // Its redaction half writes blurred pixels over the stored image, and the unredacted original is
  // deliberately never kept, so there is no revert to fall back on: the preview IS the safety mechanism.
  // Everything else on this panel is reversible, which is why nothing else works this way.
  const doReanalyzePlan = async () => {
    setBusy("reanalyze"); setMsg(null); setReanalysis(null);
    try {
      const r = await trackOp("reanalyze", "reanalyse this frame",
        () => api.agentReanalyzePlan(frameId),
        (res) => `${res.redaction.faces_added} faces, ${res.redaction.plates_added} plates, `
          + `${res.findings.length} label findings`);
      setReanalysis(r);
      if (!r.redaction.faces_added && !r.redaction.plates_added && !r.findings.length) {
        setMsg(`nothing to fix: checked ${r.redaction.windows} annotated people/vehicles and every region `
          + "is already redacted");
      }
    } catch (e) { setMsg("reanalyse failed: " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doReanalyzeApply = async () => {
    setBusy("reanalyze"); setMsg(null);
    try {
      const r = await trackOp("reanalyze", "applying reanalyse",
        () => api.agentReanalyze(frameId),
        (res) => `blurred ${res.redaction.faces_added + res.redaction.plates_added}, `
          + `queued ${res.persisted}`);
      const blurred = r.redaction.faces_added + r.redaction.plates_added;
      setMsg(`blurred ${blurred} missed region${blurred === 1 ? "" : "s"}`
        + ` and queued ${r.persisted} label finding${r.persisted === 1 ? "" : "s"} for review`
        + (r.findings_dropped ? ` (${r.findings_dropped} lower-ranked findings not queued)` : ""));
      setReanalysis(null);
      onApplied?.();
    } catch (e) { setMsg("reanalyse failed (needs reviewer role): " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const doReanalyzeSession = async () => {
    if (!sessionId) return;
    setBusy("reanalyze-all"); setMsg(null);
    try {
      const { run_id } = await api.agentReanalyzeAll({ session_id: sessionId });
      setMsg("re-checking every frame in this session; follow it in the console");
      // Deliberately not awaited: the run outlives this press by minutes, and blocking the panel on it would
      // make the editor unusable for exactly as long as the job takes. The op it registers is what carries
      // the progress, in the canvas console and the modal alike.
      void trackRun("reanalyze", "reanalysing this session", run_id,
        async (id) => await api.agentRunStatus(id));
    } catch (e) { setMsg("could not start the session re-check: " + humanizeError(e)); }
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
    } catch (e) { setMsg("revert failed: " + humanizeError(e)); }
    finally { setBusy(null); }
  };

  const c = plan?.counts;
  return (
    <div className={embedded ? "" : "border-t hairline pt-2 mt-2"}>
      <div className="flex items-center gap-2 px-1 pb-1.5">
        {!embedded && <span className="font-display text-[10px] font-semibold uppercase tracking-wider text-ink-3">Agent</span>}
        <span className="font-mono text-[9px] text-ink-3/70">auto-accept the sure ones</span>
        <button onClick={doPlan} disabled={!!busy}
          className={`ml-auto flex items-center gap-1.5 font-mono text-[10px] border px-2 py-1 rounded hover:border-accent disabled:opacity-50 ${busy === "plan" ? "running border-accent/40" : "border-line"}`}>
          {busy === "plan" && <Busy />}{busy === "plan" ? "planning..." : "dry-run"}
        </button>
      </div>

      <div className="px-1 pb-1.5 grid grid-cols-2 gap-1">
        <button onClick={doCuboids} disabled={!!busy}
          title="lift every 2D vehicle/VRU box on this frame to a 3D cuboid (monocular, reprojection-validated)"
          className="font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">
          {busy === "cuboids" ? "fitting..." : "fit 3D boxes"} <OpPrecisionChip opType="cuboid" />
        </button>
        <button onClick={doAttributes} disabled={!!busy}
          title="auto-fill the derivable attributes (occlusion, truncation, static, direction)"
          className="font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">
          {busy === "attrs" ? "filling..." : "fill attributes"} <OpPrecisionChip opType="attribute" />
        </button>
        <button onClick={doRelabel} disabled={!!busy}
          title="an independent model re-reads every box and corrects the class where it decisively disagrees (reversible)"
          className={`col-span-2 flex items-center justify-center gap-1.5 font-mono text-[10px] border px-2 py-1 rounded hover:border-accent disabled:opacity-40 ${busy === "relabel" ? "running border-accent/40" : "border-line"}`}>
          {busy === "relabel" && <Busy />}{busy === "relabel" ? "re-reading labels..." : "relabel this frame (AI reasoning)"} <OpPrecisionChip opType="relabel" />
        </button>
        <button onClick={doReanalyzePlan} disabled={!!busy}
          title="look again at this frame: re-check the face/plate redaction inside every annotated person and vehicle, and re-run every label check at once"
          className={`${sessionId ? "" : "col-span-2 "}flex items-center justify-center gap-1.5 font-mono text-[10px] border px-2 py-1 rounded hover:border-accent disabled:opacity-40 ${busy === "reanalyze" ? "running border-accent/40" : "border-line"}`}>
          {busy === "reanalyze" && <Busy />}{busy === "reanalyze" ? "reanalysing..." : "reanalyse this frame"}
        </button>
        {sessionId && (
          <button onClick={doReanalyzeSession} disabled={!!busy}
            title="reanalyse every frame in this session in the background; progress shows in the console"
            className="font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">
            {busy === "reanalyze-all" ? "starting..." : "reanalyse session"}
          </button>
        )}
      </div>

      {reanalysis && (
        <div className="px-1 pb-1.5">
          {/* Shown before anything is written, because the redaction half cannot be undone: the unredacted
              original is deliberately never stored. Every other action on this panel is reversible. */}
          <div className="border border-warn/40 rounded p-2 space-y-1.5">
            <div className="font-mono text-[10px] text-ink-2">
              checked {reanalysis.redaction.windows} annotated {reanalysis.redaction.windows === 1 ? "person/vehicle" : "people and vehicles"}
              {reanalysis.redaction.already_covered > 0 && `, ${reanalysis.redaction.already_covered} region(s) already redacted`}
            </div>
            <div className="font-mono text-[10px] text-ink">
              would blur <span className="text-warn">{reanalysis.redaction.faces_added}</span> missed face
              {reanalysis.redaction.faces_added === 1 ? "" : "s"} and{" "}
              <span className="text-warn">{reanalysis.redaction.plates_added}</span> missed plate
              {reanalysis.redaction.plates_added === 1 ? "" : "s"} (permanent)
            </div>
            <div className="font-mono text-[10px] text-ink">
              and queue <span className="text-accent">{reanalysis.findings.length}</span> label finding
              {reanalysis.findings.length === 1 ? "" : "s"} for review (nothing is relabelled)
              {reanalysis.findings_dropped > 0 && (
                <span className="text-ink-3"> · {reanalysis.findings_dropped} lower-ranked not queued</span>
              )}
            </div>
            {Object.keys(reanalysis.systemic).length > 0 && (
              <div className="font-mono text-[9px] text-ink-3 border-t hairline pt-1.5">
                fires on the whole frame, so counted rather than queued:{" "}
                {Object.entries(reanalysis.systemic).map(([rule, n]) => `${rule} (${n})`).join(", ")}
                <div className="text-ink-3/70">a rule that objects to every object is describing the pipeline, not the labels</div>
              </div>
            )}
            {reanalysis.findings.length > 0 && (
              <ul className="space-y-0.5 max-h-24 overflow-auto">
                {reanalysis.findings.slice(0, 6).map((f, i) => (
                  <li key={`${f.object_id}-${i}`} className="font-mono text-[9px] text-ink-3 truncate">
                    <span className="text-ink-2">{f.detail.rule}</span> · {f.detail.reason}
                  </li>
                ))}
              </ul>
            )}
            <div className="flex gap-1.5">
              <button onClick={doReanalyzeApply} disabled={!!busy}
                className="flex-1 font-mono text-[10px] border border-warn/60 text-warn px-2 py-1 rounded hover:bg-warn/10 disabled:opacity-40">
                apply
              </button>
              <button onClick={() => setReanalysis(null)} disabled={!!busy}
                className="font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">
                discard
              </button>
            </div>
          </div>
        </div>
      )}

      {copilot?.pattern && copilot.candidates.length > 0 && (
        <div className="px-1 pb-1.5">
          <div className="border border-accent/40 bg-accent/5 rounded p-1.5">
            <div className="font-mono text-[9.5px] text-ink-2">
              you relabeled <span className="text-accent">{copilot.pattern.from_name} → {copilot.pattern.to_name}</span> {copilot.pattern.count}x
            </div>
            <button onClick={doBatchFix} disabled={!!busy}
              className="w-full mt-1 font-mono text-[10px] border border-accent/40 bg-accent/10 text-accent px-2 py-1 rounded hover:bg-accent/20 disabled:opacity-40">
              {busy === "batch" ? "fixing..." : `batch-fix ${copilot.candidates.length} similar → review`}
            </button>
          </div>
        </div>
      )}

      {suggestions.length > 0 && (
        <div className="px-1 pb-1.5">
          <div className="font-mono text-[9px] uppercase tracking-wider text-ink-3/70 mb-0.5">suggested</div>
          {suggestions.map((s, i) => (
            <div key={i} className="flex items-center gap-1.5 font-mono text-[9.5px] text-ink-3">
              <span className="text-accent">›</span><span className="truncate">{s.label}</span>
            </div>
          ))}
        </div>
      )}

      {selectedId && (
        <div className="px-1 pb-1.5">
          <button onClick={doPropagate} disabled={!!busy}
            title="label once, carry this object across the clip both ways; stops where it drifts"
            className="w-full font-mono text-[10px] border border-accent/40 bg-accent/5 text-accent px-2 py-1 rounded hover:bg-accent/15 disabled:opacity-40">
            {busy === "track" ? "auto-tracking..." : "auto-track selected object →"}
          </button>
          <button onClick={doCrossCam} disabled={!!busy}
            title="project this object's 3D box into the other synchronized cameras and label it there"
            className="w-full mt-1 font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">
            {busy === "xcam" ? "projecting..." : "propagate to other cameras →"}
          </button>
        </div>
      )}

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
              <div className="font-mono text-[9.5px] text-ink-3">no changes, already settled</div>
            )}
          </div>
          <button onClick={doCommit} disabled={!!busy || c.auto_accept + c.review + c.annotate === c.unchanged}
            className="w-full font-mono text-[10px] border border-accent/40 bg-accent/10 text-accent px-2 py-1 rounded hover:bg-accent/20 disabled:opacity-40">
            {busy === "commit" ? "committing..." : "commit (reversible)"}
          </button>
        </div>
      )}

      <div className="px-1 pt-1.5 flex items-center gap-1">
        <input value={cmd} onChange={(e) => setCmd(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter") doCommand(); }}
          placeholder="tell the agent, e.g. accept two-wheelers above 0.9"
          className="flex-1 min-w-0 bg-bg-2 border border-line rounded px-1.5 py-1 font-mono text-[10px] text-ink-2 placeholder:text-ink-3/60 focus:border-accent outline-none" />
        <button onClick={doCommand} disabled={!!busy || !cmd.trim()}
          className="font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">
          {busy === "cmd" ? "…" : "ask"}
        </button>
      </div>

      {runId && (
        <div className="px-1 pt-1.5">
          <button onClick={doRevert} disabled={!!busy}
            className="w-full font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-block hover:text-block disabled:opacity-50">
            {busy === "revert" ? "reverting..." : "revert last run"}
          </button>
        </div>
      )}
      {msg && <div className="px-1 pt-1 font-mono text-[9.5px] text-ink-3">{msg}</div>}
      {intent && (
        <div className="px-1 pt-0.5 font-mono text-[9.5px] text-ink-3">
          understood as{" "}
          <span className="text-accent">{intent.action}</span>
          {Array.isArray(intent.classes) && intent.classes.length > 0 && (
            <> on <span className="text-ink">{intent.classes.join(", ")}</span></>
          )}
          {!Array.isArray(intent.classes) && intent.classes && (
            <> on <span className="text-ink">{String(intent.classes)}</span></>
          )}
          {intent.conf_min != null && <> above <span className="text-ink">{intent.conf_min}</span></>}
          {(!intent.classes || (Array.isArray(intent.classes) && !intent.classes.length)) &&
            intent.conf_min == null && <span className="text-warn"> nothing in particular</span>}
        </div>
      )}
    </div>
  );
}
