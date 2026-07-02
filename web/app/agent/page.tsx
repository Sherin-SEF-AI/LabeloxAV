"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type AuditReport, type PromotionProposalRow } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { Spinner } from "@/components/Spinner";

// The Agent Console: the corpus-level home for the autonomous agent. Self-healing QA (the error daemon
// sweep + temporal auto-repair) and the ranked fix queue it produces. Programs 3-5 (data intelligence,
// flywheel, copilot) add their sections here.

type Cand = { candidate_id: string; object_id: string; kind: string; score: number; detail: Record<string, unknown>; proposed_label?: { class_name?: string } | null };

const KIND_LABEL: Record<string, string> = {
  critic_flag: "consistency", near_dup_inconsistent: "near-dup", policy_violation: "policy",
  track_inconsistent: "track flip", cross_cam_inconsistent: "cross-cam", embedding_outlier: "outlier",
  confident_learning: "confident-wrong",
};

export default function AgentConsole() {
  const router = useRouter();
  const [queue, setQueue] = useState<Cand[]>([]);
  const [summary, setSummary] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const [audit, setAudit] = useState<{ status?: string; created_at?: string; report: AuditReport | null } | null>(null);
  const loadAudit = useCallback(async () => { try { setAudit(await api.agentAuditLatest()); } catch { /* none yet */ } }, []);
  const load = useCallback(async () => {
    setLoading(true);
    try { const q = await api.agentErrorQueue(); setQueue(q.candidates); setSummary(q.summary); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); loadAudit(); }, [load, loadAudit]);

  const runAudit = async () => {
    setBusy("audit"); setMsg(null);
    try {
      const r = await api.agentAuditRun({ sample_size: 200, vlm_calls: 60 });
      setMsg("overnight audit running in the background - sampling auto-accepts, VLM + critic spot-checks");
      const poll = async (n: number) => {
        const a = await api.agentAuditLatest(); setAudit(a);
        if (a.status === "committed" || a.status === "error") { load(); return; }
        if (n > 0) setTimeout(() => poll(n - 1), 5000);
      };
      void r; poll(60);
    } catch (e) { setMsg("audit failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };

  const sweep = async () => {
    setBusy("sweep"); setMsg(null);
    try {
      const r = await api.agentErrorSweep(8);
      setMsg("error sweep running in the background — the fix queue refreshes as sessions complete");
      // poll a couple of times for results
      setTimeout(load, 6000); setTimeout(load, 15000);
      void r;
    } catch (e) { setMsg("sweep failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };

  const [relabel, setRelabel] = useState<{ frames: number; relabel_keep: number; relabel_review: number } | null>(null);
  const [relabelDone, setRelabelDone] = useState(false);
  const relabelAll = async () => {
    setBusy("relabel"); setMsg(null); setRelabel(null); setRelabelDone(false);
    try {
      const r = await api.agentRelabelAll({ max_frames: 300 });
      setMsg("relabel running across the corpus — an independent model is re-reading every box");
      // poll progress until the background run reports committed
      const poll = async (n: number) => {
        try {
          const s = await api.agentRunStatus(r.run_id);
          setRelabel({ frames: s.counts.frames ?? 0, relabel_keep: s.counts.relabel_keep ?? 0, relabel_review: s.counts.relabel_review ?? 0 });
          if (s.status === "committed" || s.status === "error") { setRelabelDone(true); load(); return; }
        } catch { /* keep polling */ }
        if (n > 0) setTimeout(() => poll(n - 1), 4000);
      };
      poll(60);
    } catch (e) { setMsg("relabel failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };

  const estimateEgo = async () => {
    setBusy("ego"); setMsg(null);
    try {
      const r = await api.estimateEgoMasks();
      setMsg(`ego-hood masks: ${r.with_hood}/${r.cameras} cameras have a detected hood${r.no_hood.length ? ` (no hood: ${r.no_hood.slice(0, 4).join(", ")})` : ""}`);
    } catch (e) { setMsg("ego-mask estimation failed: " + String(e)); }
    finally { setBusy(null); }
  };
  const backfillPii = async () => {
    setBusy("pii"); setMsg(null);
    try {
      await api.piiBackfill(2000);
      setMsg("PII backfill running in the background: blurring faces/plates on pre-gate frames, overwriting the stored image in place");
    } catch (e) { setMsg("PII backfill failed (needs plate/face weights): " + String(e)); }
    finally { setBusy(null); }
  };
  const redetectAll = async () => {
    setBusy("redetect"); setMsg(null);
    try {
      const r = await api.redetectAll(true);
      setMsg(`full re-detection started (run ${r.run_id.slice(0, 8)}): PII backfill, then re-run every session with thing/stuff + ego-hood + de-dup + oversize gates, one at a time on the GPU`);
    } catch (e) { setMsg("re-detection failed (GPU may be reserved for training): " + String(e)); }
    finally { setBusy(null); }
  };

  const repair = async () => {
    setBusy("repair"); setMsg(null);
    try {
      const p = await api.agentTemporalRepairPlan();
      if (!p.counts.relabels) { setMsg(`no safe track-flip relabels (scanned ${p.counts.tracks} tracks, ${p.counts.flipped_tracks} flipped, ${p.counts.skipped_static ?? 0} corrupt)`); return; }
      const r = await api.agentTemporalRepair();
      setMsg(`temporal auto-repair: relabeled ${r.relabeled} outliers to their track majority (reversible, run ${r.run_id.slice(0, 8)})`);
    } catch (e) { setMsg("repair failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };

  const [gaps, setGaps] = useState<string[] | null>(null);
  const [ask, setAsk] = useState("");
  const [askResult, setAskResult] = useState<{ understood: string; count: number } | null>(null);
  const [report, setReport] = useState<{ size: { sessions: number; objects: number; human_labeled: number }; coverage_gaps: string[]; fix_queue_total: number; scenarios: Record<string, number> } | null>(null);

  const doAsk = async () => {
    const t = ask.trim(); if (!t) return;
    setBusy("ask"); setMsg(null);
    try { const r = await api.agentAsk(t); setAskResult({ understood: r.understood, count: r.count }); }
    catch (e) { setMsg("query failed: " + String(e)); }
    finally { setBusy(null); }
  };
  const doReport = async () => {
    setBusy("report"); setMsg(null);
    try { setReport(await api.agentReport()); }
    catch (e) { setMsg("report failed: " + String(e)); }
    finally { setBusy(null); }
  };
  const mine = async (what: "scenarios" | "disagreements") => {
    setBusy(what); setMsg(null);
    try {
      if (what === "scenarios") { const r = await api.agentMineScenarios(); setMsg(`mined ${r.persisted} safety scenarios (${Object.entries(r.by_kind).map(([k, n]) => `${k}:${n}`).join(", ") || "none"}) — see Scenarios`); }
      else { const r = await api.agentMineDisagreements(); setMsg(`mined ${r.persisted} model-disagreement frames${r.top[0] ? ` (top: ${r.top[0].tag})` : ""} — see Scenarios`); }
    } catch (e) { setMsg("mine failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };
  const coverage = async () => {
    setBusy("coverage"); setMsg(null);
    try { const r = await api.agentCoverage(); setGaps(r.gaps); }
    catch (e) { setMsg("coverage failed: " + String(e)); }
    finally { setBusy(null); }
  };

  const cycle = async () => {
    setBusy("cycle"); setMsg(null);
    try { const r = await api.agentTrainingCycle(true); setMsg(`flywheel cycle (dry-run): would auto-accept ${r.tick.auto_accept}, review ${r.tick.review}, annotate ${r.tick.annotate} across ${r.tick.frames} top-value frames`); }
    catch (e) { setMsg("cycle failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };
  const drift = async () => {
    setBusy("drift"); setMsg(null);
    try {
      const r = await api.agentGoldDrift();
      setMsg(r.status === "rolled_back" ? `GOLD DRIFT: champion regressed ${r.baseline_map}→${r.current_map} — rolled back + paused loop`
        : r.status === "healthy" ? `champion healthy on gold (${r.current_map} vs baseline ${r.baseline_map})`
        : r.status === "cannot_evaluate" ? `champion ${r.champion} (baseline mAP ${r.baseline_map}) — gold set not materialized here`
        : "no champion registered");
    } catch (e) { setMsg("gold-drift check failed: " + String(e)); }
    finally { setBusy(null); }
  };

  const act = async (c: Cand, kind: "confirm" | "dismiss") => {
    try { await (kind === "confirm" ? api.errorConfirm(c.candidate_id) : api.errorDismiss(c.candidate_id)); load(); }
    catch (e) { setMsg(String(e)); }
  };

  const [driftDiag, setDriftDiag] = useState<{ report: { hypothesis: string; proposed_action: { kind: string } } | null } | null>(null);
  useEffect(() => { api.agentDriftLatest().then(setDriftDiag).catch(() => {}); }, []);
  const investigateDrift = async () => {
    setBusy("driftinv"); setMsg(null);
    try {
      const r = await api.agentDriftInvestigate();
      if (!r.breached.length) { setMsg("no drift breach right now - governance is holding within tolerance"); return; }
      setMsg(`drift breach (${r.breached.join(", ")}) - investigating root cause in the background`);
      setTimeout(() => api.agentDriftLatest().then(setDriftDiag).catch(() => {}), 4000);
    } catch (e) { setMsg("drift investigation failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };

  const [doc, setDoc] = useState<string | null>(null);
  const genDoc = async (kind: "datasheet" | "weekly") => {
    setBusy("doc"); setMsg(null); setDoc(null);
    try {
      const r = kind === "datasheet" ? await api.agentDocDatasheet() : await api.agentDocWeekly();
      setDoc(r.markdown); setMsg(`${kind} drafted and stored (${r.uri.split("/").slice(-2).join("/")})`);
    } catch (e) { setMsg("doc generation failed: " + String(e)); }
    finally { setBusy(null); }
  };

  const [props, setProps] = useState<PromotionProposalRow[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const loadProps = useCallback(async () => { try { setProps(await api.agentOntologyProposals()); } catch { /* none */ } }, []);
  useEffect(() => { loadProps(); }, [loadProps]);
  const scanOntology = async () => {
    setBusy("ontscan"); setMsg(null);
    try { const r = await api.agentOntologyScan(40); setMsg(`scanned ${r.scanned} fallbacks -> ${r.proposals} promotion proposals`); await loadProps(); }
    catch (e) { setMsg("ontology scan failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };
  const decide = async (id: string, action: "approve" | "reject") => {
    setBusy(id); setMsg(null);
    try {
      if (action === "approve") {
        const nm = (names[id] || props.find((p) => p.proposal_id === id)?.suggested_name || "").trim();
        if (!nm) { setMsg("give the new class a name first"); return; }
        const r = await api.agentOntologyApprove(id, nm); setMsg(`minted ${r.name} (#${r.class_id}), relabeled ${r.relabeled} - reversible run ${r.run_id.slice(0, 8)}`);
      } else { await api.agentOntologyReject(id); setMsg("proposal rejected"); }
      await loadProps();
    } catch (e) { setMsg("decision failed (needs reviewer role): " + String(e)); }
    finally { setBusy(null); }
  };

  const reason = (c: Cand) => (c.detail?.reason as string) || (c.detail?.reasons ? (c.detail.reasons as string[]).join("; ") : (c.detail?.note as string) || JSON.stringify(c.detail).slice(0, 80));

  return (
    <PageShell active="AGENT" subtitle="CONSOLE" right={loading ? <Spinner label="loading" /> : <span className="font-mono text-xs text-ink-3">{queue.length} in fix queue</span>}>
      <div className="h-full overflow-auto">
        <div className="max-w-5xl mx-auto p-6 space-y-5">
          <div>
            <h1 className="text-xl text-ink font-semibold">Agent Console</h1>
            <p className="text-ink-3 text-sm mt-1 max-w-2xl">Autonomous QA over the whole corpus: the agent finds likely-wrong labels and fixes the obvious ones itself. Everything it does is reversible and provenance-stamped.</p>
          </div>

          {/* Overnight Auditor: standing watchdog + morning report */}
          <div className="panel p-4 border border-accent/30">
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <div className="text-ink font-medium">Overnight Auditor <span className="font-mono text-[10px] text-accent">morning report</span></div>
                <div className="text-ink-3 text-xs mt-1">Patrols the day&apos;s auto-accepted labels within a token budget: VLM spot-checks + cross-frame consistency + control-sample precision. Suspects are queued to review (reversible). Runs nightly off-hours; run it now below.</div>
                {audit?.report ? (
                  <div className="mt-2 font-mono text-[11px] text-ink-2 space-y-0.5">
                    {audit.report.notes.map((n, i) => <div key={i}>· {n}</div>)}
                    <div className="text-ink-3 mt-1">
                      sampled {audit.report.sampled} · vlm-checked {audit.report.vlm_checked} · <span className="text-warn">{audit.report.vlm_disagreements} vlm-disagree</span> · budget {audit.report.budget.used}/{audit.report.budget.max_calls}
                      {audit.created_at ? ` · ${new Date(audit.created_at).toLocaleString()}` : ""}
                    </div>
                    {audit.report.confusion_movers.length > 0 && (
                      <div className="text-ink-3">movers: {audit.report.confusion_movers.map((m) => `${m.from}->${m.to} (${m.n})${m.concentrated_in ? ` in ${m.concentrated_in}` : ""}`).join(", ")}</div>
                    )}
                  </div>
                ) : <div className="mt-2 font-mono text-[11px] text-ink-3">no audit yet</div>}
              </div>
              <button onClick={runAudit} disabled={!!busy} className="shrink-0 font-mono text-[11px] border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded hover:bg-accent/20 disabled:opacity-40">{busy === "audit" ? "auditing..." : "run audit now"}</button>
            </div>
          </div>

          {/* Ask the dataset (conversational corpus query) */}
          <div className="panel p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-ink font-medium text-sm">Ask the dataset</span>
              <span className="font-mono text-[10px] text-ink-3">plain-language corpus query</span>
              <button onClick={doReport} disabled={!!busy} className="ml-auto font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">{busy === "report" ? "…" : "dataset report"}</button>
            </div>
            <div className="flex items-center gap-2">
              <input value={ask} onChange={(e) => setAsk(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") doAsk(); }}
                placeholder="e.g. two-wheelers going against traffic at night on the highway"
                className="flex-1 bg-bg-2 border border-line rounded px-2.5 py-1.5 font-mono text-[11px] text-ink-2 placeholder:text-ink-3/60 focus:border-accent outline-none" />
              <button onClick={doAsk} disabled={!!busy || !ask.trim()} className="font-mono text-[11px] border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded hover:bg-accent/20 disabled:opacity-40">{busy === "ask" ? "…" : "ask"}</button>
            </div>
            {askResult ? <div className="mt-2 font-mono text-[11px] text-ink-2"><span className="text-pass">{askResult.count} frames</span> · understood as <span className="text-ink-3">{askResult.understood}</span></div> : null}
            {report ? (
              <div className="mt-3 border-t hairline pt-3 grid grid-cols-2 md:grid-cols-4 gap-3 font-mono text-[11px]">
                <div><div className="text-ink-3 text-[10px] uppercase">objects</div><div className="text-ink text-base tabular-nums">{report.size.objects.toLocaleString()}</div><div className="text-ink-3">{report.size.human_labeled.toLocaleString()} human</div></div>
                <div><div className="text-ink-3 text-[10px] uppercase">sessions</div><div className="text-ink text-base tabular-nums">{report.size.sessions.toLocaleString()}</div></div>
                <div><div className="text-ink-3 text-[10px] uppercase">coverage gaps</div><div className="text-warn text-base tabular-nums">{report.coverage_gaps.length}</div></div>
                <div><div className="text-ink-3 text-[10px] uppercase">fix queue</div><div className="text-block text-base tabular-nums">{report.fix_queue_total}</div><div className="text-ink-3">{Object.values(report.scenarios).reduce((a, b) => a + b, 0)} scenarios</div></div>
              </div>
            ) : null}
          </div>

          {/* Relabel: the reasoning layer improves accuracy across the corpus */}
          <div className="panel p-4 border border-accent/30">
            <div className="flex items-start gap-3">
              <div className="flex-1">
                <div className="text-ink font-medium">Relabel all frames <span className="font-mono text-[10px] text-accent">AI reasoning</span></div>
                <div className="text-ink-3 text-xs mt-1">Re-read every machine-labelled box with an independent model and correct the class wherever it decisively disagrees with the current label. Decisive corrections are applied and kept; moderate ones are applied but routed to review. One reversible run per frame, so it can be undone wholesale. To relabel a single frame, use the Agent panel in the editor.</div>
              </div>
              <button onClick={relabelAll} disabled={!!busy} className="shrink-0 font-mono text-[11px] border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded hover:bg-accent/20 disabled:opacity-40">{busy === "relabel" ? "starting..." : "relabel all frames"}</button>
            </div>
            {relabel ? (
              <div className="mt-3 font-mono text-[11px] text-ink-2">
                {relabelDone ? "done" : "running"} · scanned {relabel.frames} frames · <span className="text-pass">{relabel.relabel_keep} fixed</span> · <span className="text-warn">{relabel.relabel_review} routed to review</span>
              </div>
            ) : null}
          </div>

          {/* Corpus re-detection: fix existing frames with the new detection gates */}
          <div className="panel p-4 border border-line">
            <div className="text-ink font-medium">Corpus re-detection <span className="font-mono text-[10px] text-ink-3">label quality</span></div>
            <div className="text-ink-3 text-xs mt-1">The detection pipeline now enforces thing/stuff (no boxed trees, barriers, sky), an auto-estimated ego-hood mask (no self-labeling), fusion de-duplication (one object, one box), and an oversize reviewer rule (no frame-spanning boxes). Those shape new labels; run this to fix the frames already in the corpus. Sequential on one GPU, yields to training.</div>
            <div className="mt-3 flex flex-wrap gap-2">
              <button onClick={estimateEgo} disabled={!!busy} className="font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "ego" ? "estimating..." : "1. estimate ego-hood masks"}</button>
              <button onClick={backfillPii} disabled={!!busy} className="font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "pii" ? "starting..." : "2. PII backfill (pre-gate frames)"}</button>
              <button onClick={redetectAll} disabled={!!busy} className="font-mono text-[11px] border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded hover:bg-accent/20 disabled:opacity-40">{busy === "redetect" ? "starting..." : "3. re-detect all frames"}</button>
            </div>
          </div>

          {/* Self-healing actions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="panel p-4">
              <div className="text-ink font-medium">Error sweep</div>
              <div className="text-ink-3 text-xs mt-1">Run every detector across the corpus (consistency critic, embedding outliers, near-duplicate divergence, policy violations, confident-wrong) and refresh the ranked fix queue.</div>
              <button onClick={sweep} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded hover:bg-accent/20 disabled:opacity-40">{busy === "sweep" ? "sweeping..." : "run error sweep"}</button>
            </div>
            <div className="panel p-4">
              <div className="text-ink font-medium">Temporal auto-repair</div>
              <div className="text-ink-3 text-xs mt-1">Relabel track class-flip outliers to their strong track majority automatically. Corrupt tracks (static-class majority) are left for a human. Reversible.</div>
              <button onClick={repair} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "repair" ? "repairing..." : "run temporal repair"}</button>
            </div>
          </div>

          {/* Data intelligence: the system finds what is worth labeling */}
          <div>
            <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3 mb-2">Data intelligence — find what matters</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Safety scenarios</div>
                <div className="text-ink-3 text-xs mt-1">Mine near-misses (low TTC), high-risk interactions, and hard-brake events into the scenario queue.</div>
                <button onClick={() => mine("scenarios")} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "scenarios" ? "mining..." : "mine safety scenarios"}</button>
              </div>
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Model disagreement</div>
                <div className="text-ink-3 text-xs mt-1">Surface frames where the champion and challenger detectors voted different classes — the highest-value labels + a regression signal.</div>
                <button onClick={() => mine("disagreements")} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "disagreements" ? "mining..." : "mine disagreements"}</button>
              </div>
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Coverage gaps</div>
                <div className="text-ink-3 text-xs mt-1">Profile the corpus and name the thin cells (rare classes, missing weather/time/road coverage).</div>
                <button onClick={coverage} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "coverage" ? "analyzing..." : "coverage report"}</button>
              </div>
            </div>
            {gaps ? (
              <div className="panel mt-3 p-4">
                <div className="font-mono text-[10px] uppercase text-ink-3 mb-1.5">coverage gaps ({gaps.length})</div>
                <ul className="space-y-0.5">{gaps.slice(0, 10).map((g, i) => <li key={i} className="font-mono text-[11px] text-ink-2">· {g}</li>)}</ul>
              </div>
            ) : null}
          </div>

          {/* Self-improving loop */}
          <div>
            <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3 mb-2">Self-improving loop</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Flywheel cycle</div>
                <div className="text-ink-3 text-xs mt-1">One turn of the loop: mine the highest-value frames, auto-accept the sure ones, escalate the rest; retrains when enough corrections accumulate. Dry-run preview.</div>
                <button onClick={cycle} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "cycle" ? "running..." : "run flywheel cycle"}</button>
              </div>
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Gold-drift monitor</div>
                <div className="text-ink-3 text-xs mt-1">Re-evaluate the serving champion on the gold set; if it has regressed beyond tolerance, roll back to the prior champion and pause the loop.</div>
                <button onClick={drift} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "drift" ? "checking..." : "check gold drift"}</button>
              </div>
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Drift Investigator</div>
                <div className="text-ink-3 text-xs mt-1">On a drift breach, root-cause it: the affected slice, worst classes/scenes/sessions, a common factor, and a proposed action. Proposes only.</div>
                {driftDiag?.report ? (
                  <div className="mt-2 font-mono text-[10.5px] text-ink-2">
                    <div>{driftDiag.report.hypothesis}</div>
                    <div className="text-ink-3 mt-0.5">proposed: {driftDiag.report.proposed_action.kind}</div>
                  </div>
                ) : null}
                <button onClick={investigateDrift} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "driftinv" ? "investigating..." : "investigate drift"}</button>
              </div>
            </div>
          </div>

          {/* Documentation agent */}
          <div>
            <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3 mb-2">Documentation</h2>
            <div className="panel p-4">
              <div className="flex items-center gap-2">
                <div className="flex-1 text-ink-3 text-xs">Auto-draft the buyer-diligence artifacts from the platform&apos;s own metrics: dataset datasheet (composition, coverage, known gaps) and the weekly quality report (precision, drift, promotions). Model cards are drafted per model via the API.</div>
                <button onClick={() => genDoc("datasheet")} disabled={!!busy} className="shrink-0 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "doc" ? "..." : "datasheet"}</button>
                <button onClick={() => genDoc("weekly")} disabled={!!busy} className="shrink-0 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">weekly report</button>
              </div>
              {doc ? <pre className="mt-3 max-h-64 overflow-auto no-scrollbar bg-bg-2 rounded p-3 font-mono text-[10.5px] text-ink-2 whitespace-pre-wrap">{doc.slice(0, 4000)}</pre> : null}
            </div>
          </div>

          {/* Ontology Steward */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3">Ontology Steward — grow the ontology</h2>
              <button onClick={scanOntology} disabled={!!busy} className="ml-auto font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">{busy === "ontscan" ? "scanning..." : "scan fallback clusters"}</button>
            </div>
            {props.length ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {props.map((p) => (
                  <div key={p.proposal_id} className="panel p-3">
                    <div className="flex gap-1 mb-2">
                      {p.sample_object_ids.slice(0, 6).map((oid) => (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img key={oid} src={`/api/objects/${oid}/crop`} alt="" className="w-12 h-12 object-cover rounded bg-bg-2" />
                      ))}
                    </div>
                    <div className="font-mono text-[10.5px] text-ink-2">{p.member_count} in {p.from_class} · looks like {p.confusion_classes.map((c) => `${c.class} ${Math.round(c.share * 100)}%`).join(", ") || "nothing known"}</div>
                    <div className="flex items-center gap-1.5 mt-2">
                      <input value={names[p.proposal_id] ?? p.suggested_name ?? ""} onChange={(e) => setNames((s) => ({ ...s, [p.proposal_id]: e.target.value }))}
                        placeholder="new class name" className="flex-1 min-w-0 bg-bg-2 border border-line rounded px-1.5 py-1 font-mono text-[10.5px] text-ink-2 focus:border-accent outline-none" />
                      <button onClick={() => decide(p.proposal_id, "approve")} disabled={!!busy} className="font-mono text-[10px] border border-pass text-pass px-2 py-1 rounded disabled:opacity-40">approve</button>
                      <button onClick={() => decide(p.proposal_id, "reject")} disabled={!!busy} className="font-mono text-[10px] border border-line text-ink-3 px-2 py-1 rounded hover:text-ink disabled:opacity-40">reject</button>
                    </div>
                  </div>
                ))}
              </div>
            ) : <div className="panel p-4 text-ink-3 text-sm">No promotion proposals. Scan the fallback clusters to find classes that have earned their way in.</div>}
          </div>

          {msg ? <div className="font-mono text-[11px] text-warn">{msg}</div> : null}

          {/* Fix queue */}
          <div className="panel">
            <div className="flex items-center gap-3 px-4 py-3 border-b hairline">
              <div className="text-ink font-medium">Fix queue</div>
              <div className="text-ink-3 text-xs">likely-wrong labels, worst first</div>
              <div className="ml-auto font-mono text-[10px] text-ink-3">{Object.entries(summary).map(([k, n]) => `${k}:${n}`).join("  ")}</div>
            </div>
            {loading && !queue.length ? <div className="p-6"><Spinner label="loading" /></div> : queue.length ? (
              <table className="w-full text-sm">
                <thead className="text-ink-3 font-mono text-[11px] uppercase border-b hairline">
                  <tr><th className="text-left font-normal px-3 py-2 w-28">kind</th><th className="text-left font-normal px-3 py-2 w-16">score</th><th className="text-left font-normal px-3 py-2">why</th><th className="px-3 py-2 w-40"></th></tr>
                </thead>
                <tbody>
                  {queue.map((c) => (
                    <tr key={c.candidate_id} className="border-b hairline hover:bg-bg-2">
                      <td className="px-3 py-2"><span className="font-mono text-[10px] border border-line px-1.5 py-0.5 rounded text-ink-2">{KIND_LABEL[c.kind] || c.kind}</span></td>
                      <td className="px-3 py-2 font-mono text-ink-3">{c.score.toFixed(2)}</td>
                      <td className="px-3 py-2 text-ink-2 text-xs">{reason(c)}{c.proposed_label?.class_name ? <span className="text-pass"> → {c.proposed_label.class_name}</span> : null}</td>
                      <td className="px-3 py-2 text-right font-mono text-[10px]">
                        <button onClick={() => act(c, "confirm")} className="border border-pass text-pass px-1.5 py-0.5 rounded mr-1">confirm</button>
                        <button onClick={() => act(c, "dismiss")} className="border border-line text-ink-3 px-1.5 py-0.5 rounded hover:text-ink">dismiss</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="px-4 py-10 text-center text-ink-3 text-sm">Fix queue is empty. Run an error sweep to scan the corpus.</div>}
          </div>
        </div>
      </div>
    </PageShell>
  );
}
