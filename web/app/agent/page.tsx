"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api, type AuditReport, type PromotionProposalRow } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import ActivityLog from "@/components/agent/ActivityLog";
import InterruptedRuns from "@/components/agent/InterruptedRuns";
import JobWatcher from "@/components/agent/JobWatcher";
import ContaminationPanel from "@/components/agent/ContaminationPanel";
import Inspector from "@/components/shell/Inspector";
import EvidencePanel, { type EvidenceSubject } from "@/components/inspect/EvidencePanel";
import LineageObjects from "@/components/inspect/LineageObjects";
import ActionResultPanel, { type ActionResult } from "@/components/inspect/ActionResultPanel";
import { describeFailure } from "@/lib/actionError";
import { type ActivityLog as Log, emptyLog, record } from "@/lib/activityLog";
import { Spinner } from "@/components/Spinner";

// The Agent Console: the corpus-level home for the autonomous agent. Self-healing QA (the error daemon
// sweep + temporal auto-repair) and the ranked fix queue it produces. Programs 3-5 (data intelligence,
// flywheel, copilot) add their sections here.

type Cand = { candidate_id: string; object_id: string; kind: string; score: number; detail: Record<string, unknown>; proposed_label?: { class_name?: string } | null };

const KIND_LABEL: Record<string, string> = {
  critic_flag: "consistency", near_dup_inconsistent: "near-dup", policy_violation: "policy",
  track_inconsistent: "track flip", cross_cam_inconsistent: "cross-cam", embedding_outlier: "outlier",
  confident_learning: "confident-wrong",
  // Was missing, so every one of these rendered as the raw slug. It is also the kind that dominates the
  // queue, because the error sweep never clears it (services/errordetect/queue.py scopes its delete by
  // the kinds it was asked to run, and this is not one of them).
  vlm_confusion: "vlm disagrees",
  // The bulk of this queue and it rendered as a raw slug. Its claim ("another box covers the same object
  // with higher confidence") is the reason the panel draws the frame's other boxes: one box cannot show
  // a duplicate.
  reanalyze: "re-check",
};

export default function AgentConsole() {
  const router = useRouter();
  const [queue, setQueue] = useState<Cand[]>([]);
  const [summary, setSummary] = useState<Record<string, number>>({});
  // Which row the evidence panel is showing, and which row the keyboard is on. One index rather than an
  // id: the queue is ordered worst-first and acting on a row removes it, so the cursor has to be able to
  // stay put and pick up whatever moved into that position.
  const [cursor, setCursor] = useState(-1);
  const [subject, setSubject] = useState<EvidenceSubject | null>(null);
  // The rail serves two tables. A lineage is a set rather than one object, so it shows the set first and
  // the evidence for whichever member is picked; a fix-queue row goes straight to the evidence.
  const [lineage, setLineage] = useState<{ from_name: string; to_name: string } | null>(null);
  // The last action's full payload. Every handler used to report through setMsg alone, which is a summary
  // written into a transcript at the bottom of a long page: out of sight from the button, and lossy at the
  // point of rendering. The transcript stays, because a running record of what was done is worth having;
  // this is the result itself, beside the button that asked for it.
  const [result, setResult] = useState<ActionResult | null>(null);
  const showResult = useCallback((kind: string, label: string, data: unknown,
                                  destination?: { href: string; label: string }) => {
    setResult({ kind, label, data, at: Date.now(), destination });
    setSubject(null);
    setLineage(null);
  }, []);
  const [busy, setBusy] = useState<string | null>(null);
  // The transcript replaces a single shared message string. `setMsg` is kept as the name every call site
  // already uses and now appends an entry instead of overwriting one, so a background result landing minutes
  // later sits beside the earlier ones rather than erasing them. Passing null clears nothing and records
  // nothing, which is what the old `setMsg(null)` at the top of each handler meant.
  const [log, setLog] = useState<Log>(emptyLog);
  const setMsg = useCallback((m: string | null, status: "ok" | "failed" = "ok", hint?: string) => {
    if (m === null) return;
    setLog((l) => record(l, "", status, m, Date.now(), hint));
  }, []);
  // Report a failure from what the status actually says. Thirteen call sites here asserted "(needs reviewer
  // role)" on every failure of an action, so a busy GPU (503) read as a permissions problem.
  const failed = useCallback((action: string, e: unknown) => {
    const f = describeFailure(action, e);
    setLog((l) => record(l, action, "failed", f.message, Date.now(), f.hint));
  }, []);
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
      showResult("audit", "overnight audit", r);
      setMsg("overnight audit running in the background - sampling auto-accepts, VLM + critic spot-checks");
      const poll = async (n: number) => {
        const a = await api.agentAuditLatest(); setAudit(a);
        if (a.status === "committed" || a.status === "error") { load(); return; }
        if (n > 0) setTimeout(() => poll(n - 1), 5000);
      };
      void r; poll(60);
    } catch (e) { failed("audit", e); }
    finally { setBusy(null); }
  };

  const sweep = async () => {
    setBusy("sweep"); setMsg(null);
    try {
      const r = await api.agentErrorSweep(8);
      showResult("sweep", "error sweep", r);
      setMsg("error sweep running in the background: the fix queue refreshes as sessions complete");
      // poll a couple of times for results
      setTimeout(load, 6000); setTimeout(load, 15000);
      void r;
    } catch (e) { failed("sweep", e); }
    finally { setBusy(null); }
  };

  const [relabel, setRelabel] = useState<{ frames: number; relabel_keep: number; relabel_review: number } | null>(null);
  const [relabelDone, setRelabelDone] = useState(false);
  const relabelAll = async () => {
    setBusy("relabel"); setMsg(null); setRelabel(null); setRelabelDone(false);
    try {
      const r = await api.agentRelabelAll({ max_frames: 300 });
      showResult("relabel", "relabel the corpus", r);
      setMsg("relabel running across the corpus: an independent model is re-reading every box");
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
    } catch (e) { failed("relabel", e); }
    finally { setBusy(null); }
  };

  const estimateEgo = async () => {
    setBusy("ego"); setMsg(null);
    try {
      const r = await api.estimateEgoMasks();
      setMsg(`ego-hood masks: ${r.with_hood}/${r.cameras} cameras have a detected hood${r.no_hood.length ? ` (no hood: ${r.no_hood.slice(0, 4).join(", ")})` : ""}`);
      // The line truncates no_hood at four cameras. Which cameras have no detected hood is the actionable
      // part, so the panel gets all of them.
      showResult("ego", "estimate ego-hood masks", r);
    } catch (e) { failed("ego-mask estimation", e); }
    finally { setBusy(null); }
  };
  const backfillPii = async () => {
    setBusy("pii"); setMsg(null);
    try {
      const r = await api.piiBackfill(2000);
      showResult("pii", "PII backfill", r);
      setMsg("PII backfill running in the background: blurring faces/plates on pre-gate frames, overwriting the stored image in place");
    } catch (e) { failed("PII backfill", e); }
    finally { setBusy(null); }
  };
  const redetectAll = async () => {
    setBusy("redetect"); setMsg(null);
    try {
      const r = await api.redetectAll(true);
      showResult("redetect", "full re-detection", r);
      setMsg(`full re-detection started (run ${r.run_id.slice(0, 8)}): PII backfill, then re-run every session with thing/stuff + ego-hood + de-dup + oversize gates, one at a time on the GPU`);
    } catch (e) { failed("re-detection", e); }
    finally { setBusy(null); }
  };

  const repair = async () => {
    setBusy("repair"); setMsg(null);
    try {
      const p = await api.agentTemporalRepairPlan();
      showResult("repair", "temporal repair plan", p);
      if (!p.counts.relabels) { setMsg(`no safe track-flip relabels (scanned ${p.counts.tracks} tracks, ${p.counts.flipped_tracks} flipped, ${p.counts.skipped_static ?? 0} corrupt)`); return; }
      const r = await api.agentTemporalRepair();
      setMsg(`temporal auto-repair: relabeled ${r.relabeled} outliers to their track majority (reversible, run ${r.run_id.slice(0, 8)})`);
    } catch (e) { failed("repair", e); }
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
    catch (e) { failed("query", e); }
    finally { setBusy(null); }
  };
  const doReport = async () => {
    setBusy("report"); setMsg(null);
    try { setReport(await api.agentReport()); }
    catch (e) { failed("report", e); }
    finally { setBusy(null); }
  };
  const mine = async (what: "scenarios" | "disagreements") => {
    setBusy(what); setMsg(null);
    try {
      if (what === "scenarios") {
        const r = await api.agentMineScenarios();
        setMsg(`mined ${r.persisted} safety scenarios (${Object.entries(r.by_kind).map(([k, n]) => `${k}:${n}`).join(", ") || "none"})`);
        showResult("scenarios", "mine safety scenarios", r,
          { href: "/scenarios", label: `open the ${r.persisted} scenarios` });
      } else {
        const r = await api.agentMineDisagreements();
        setMsg(`mined ${r.persisted} model-disagreement frames${r.top[0] ? ` (top: ${r.top[0].tag})` : ""}`);
        showResult("disagreements", "mine disagreements", r,
          { href: "/scenarios", label: `open the ${r.persisted} disagreement frames` });
      }
    } catch (e) { failed("mine", e); }
    finally { setBusy(null); }
  };
  const coverage = async () => {
    setBusy("coverage"); setMsg(null);
    try {
      const r = await api.agentCoverage();
      setGaps(r.gaps);
      // The card keeps its ten-gap preview; the panel gets class balance, per-axis scene coverage and the
      // geo histogram, which the page returned and never drew.
      showResult("coverage", "coverage report", r);
    }
    catch (e) { failed("coverage", e); }
    finally { setBusy(null); }
  };

  const cycle = async () => {
    setBusy("cycle"); setMsg(null);
    try {
      const r = await api.agentTrainingCycle(true);
      setMsg(`flywheel cycle (dry-run): would auto-accept ${r.tick.auto_accept}, review ${r.tick.review}, annotate ${r.tick.annotate} across ${r.tick.frames} top-value frames`);
      showResult("cycle", "flywheel cycle (dry-run)", r);
    }
    catch (e) { failed("cycle", e); }
    finally { setBusy(null); }
  };
  const drift = async () => {
    setBusy("drift"); setMsg(null);
    try {
      const r = await api.agentGoldDrift();
      showResult("drift", "gold-drift check", r);
      setMsg(r.status === "rolled_back" ? `GOLD DRIFT: champion regressed ${r.baseline_map}→${r.current_map}, rolled back + paused loop`
        : r.status === "healthy" ? `champion healthy on gold (${r.current_map} vs baseline ${r.baseline_map})`
        : r.status === "cannot_evaluate" ? `champion ${r.champion} (baseline mAP ${r.baseline_map}), gold set not materialized here`
        : "no champion registered");
    } catch (e) { failed("gold-drift check", e); }
    finally { setBusy(null); }
  };

  // Memoised because the keyboard effect depends on it: recreated every render, the listener would be
  // torn down and rebound on every keystroke that changed any state on this page.
  const act = useCallback(async (c: Cand, kind: "confirm" | "dismiss") => {
    try { await (kind === "confirm" ? api.errorConfirm(c.candidate_id) : api.errorDismiss(c.candidate_id)); load(); }
    catch (e) { failed(kind === "confirm" ? "confirm" : "dismiss", e); }
  }, [load, failed]);

  // Declared before `open`, which reads it: a const arrow is in the temporal dead zone until its line
  // runs, and a useCallback dependency array is evaluated during render.
  const reason = useCallback((c: Cand) => (c.detail?.reason as string) || (c.detail?.reasons ? (c.detail.reasons as string[]).join("; ") : (c.detail?.note as string) || JSON.stringify(c.detail).slice(0, 80)), []);

  // A row and the panel are the same selection; opening one sets the other. The reason text is the row's
  // own words, so the panel shows exactly what the table was showing plus the thing it was not: the object.
  const open = useCallback((c: Cand, i: number) => {
    setCursor(i);
    setLineage(null);
    setSubject({
      objectId: c.object_id,
      suggestion: c.proposed_label?.class_name
        ? { class_id: 0, class_name: c.proposed_label.class_name } : null,
      text: reason(c), kind: KIND_LABEL[c.kind] || c.kind, score: c.score, detail: c.detail,
    });
  }, [reason]);

  // j/k rather than arrows, matching the triage page: arrows scroll the panel, and this list is long
  // enough that a reviewer wants both. e.repeat is refused because a held key would otherwise stack
  // verdicts on rows nobody looked at, which is the failure the review grid already guards against.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      if (!queue.length) return;
      if (e.key === "j" || e.key === "k") {
        e.preventDefault();
        const next = e.key === "j"
          ? Math.min(cursor + 1, queue.length - 1)
          : Math.max(cursor - 1, 0);
        open(queue[next], next);
      } else if (e.key === "Escape") {
        setSubject(null); setCursor(-1);
      } else if ((e.key === "c" || e.key === "d") && cursor >= 0 && queue[cursor]) {
        e.preventDefault();
        void act(queue[cursor], e.key === "c" ? "confirm" : "dismiss");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [queue, cursor, open, act]);

  // After a row leaves the queue, hold the position rather than the row: the next-worst candidate slides
  // into it and that is the one a reviewer wants next. Clamped, so acting on the last row lands on the
  // new last row instead of nothing.
  useEffect(() => {
    if (cursor < 0) return;
    if (!queue.length) { setSubject(null); setCursor(-1); return; }
    const at = Math.min(cursor, queue.length - 1);
    if (!subject || queue[at]?.object_id !== subject.objectId) open(queue[at], at);
  }, [queue]);   // eslint-disable-line react-hooks/exhaustive-deps

  const [driftDiag, setDriftDiag] = useState<{ report: { hypothesis: string; proposed_action: { kind: string } } | null } | null>(null);
  useEffect(() => { api.agentDriftLatest().then(setDriftDiag).catch(() => {}); }, []);
  const investigateDrift = async () => {
    setBusy("driftinv"); setMsg(null);
    try {
      const r = await api.agentDriftInvestigate();
      showResult("driftinv", "drift investigation", r);
      if (!r.breached.length) { setMsg("no drift breach right now - governance is holding within tolerance"); return; }
      setMsg(`drift breach (${r.breached.join(", ")}) - investigating root cause in the background`);
      // The diagnosis arrives after the investigation runs, so replace the panel with it rather than
      // leaving the panel showing the request that started it.
      setTimeout(() => api.agentDriftLatest().then((d) => {
        setDriftDiag(d);
        showResult("driftinv", "drift diagnosis", d);
      }).catch(() => {}), 4000);
    } catch (e) { failed("drift investigation", e); }
    finally { setBusy(null); }
  };

  const [doc, setDoc] = useState<string | null>(null);
  const genDoc = async (kind: "datasheet" | "weekly") => {
    setBusy("doc"); setMsg(null); setDoc(null);
    try {
      const r = kind === "datasheet" ? await api.agentDocDatasheet() : await api.agentDocWeekly();
      setDoc(r.markdown); setMsg(`${kind} drafted and stored (${r.uri.split("/").slice(-2).join("/")})`);
    } catch (e) { failed("doc generation", e); }
    finally { setBusy(null); }
  };

  const [props, setProps] = useState<PromotionProposalRow[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  const loadProps = useCallback(async () => { try { setProps(await api.agentOntologyProposals()); } catch { /* none */ } }, []);
  useEffect(() => { loadProps(); }, [loadProps]);
  const scanOntology = async () => {
    setBusy("ontscan"); setMsg(null);
    try {
      const r = await api.agentOntologyScan(40);
      showResult("ontscan", "scan for ontology gaps", r);
      setMsg(`scanned ${r.scanned} fallbacks -> ${r.proposals} promotion proposals`);
      await loadProps();
    }
    catch (e) { failed("ontology scan", e); }
    finally { setBusy(null); }
  };
  const decide = async (id: string, action: "approve" | "reject") => {
    setBusy(id); setMsg(null);
    try {
      if (action === "approve") {
        const nm = (names[id] || props.find((p) => p.proposal_id === id)?.suggested_name || "").trim();
        if (!nm) { setMsg("give the new class a name first"); return; }
        const r = await api.agentOntologyApprove(id, nm);
        showResult("ontology", `mint ${nm}`, r);
        setMsg(`minted ${r.name} (#${r.class_id}), relabeled ${r.relabeled} - reversible run ${r.run_id.slice(0, 8)}`);
      } else { await api.agentOntologyReject(id); setMsg("proposal rejected"); }
      await loadProps();
    } catch (e) { failed("decision", e); }
    finally { setBusy(null); }
  };

  const [orders, setOrders] = useState<{ order_id: string; priority: number; summary: string; status: string }[]>([]);
  const loadOrders = useCallback(async () => { try { setOrders(await api.agentFleetOrders("proposed")); } catch { /* none */ } }, []);
  useEffect(() => { loadOrders(); }, [loadOrders]);
  const planFleet = async () => {
    setBusy("fleet"); setMsg(null);
    try {
      const r = await api.agentFleetPlan();
      showResult("fleet", "plan fleet collection", r);
      setMsg(`fused ${r.gaps} gaps + ${r.vehicles} vehicles -> ${r.orders} collection orders`);
      await loadOrders();
    }
    catch (e) { failed("fleet plan", e); }
    finally { setBusy(null); }
  };
  const dispatchOrder = async (id: string) => {
    setBusy(id); setMsg(null);
    try { await api.agentFleetDispatch(id, "dispatched"); await loadOrders(); }
    catch (e) { failed("dispatch", e); }
    finally { setBusy(null); }
  };

  const [buyer, setBuyer] = useState("");
  const [buyerRes, setBuyerRes] = useState<{ status: string; understood?: string; fulfillment?: { requested: number | null; available: number; fulfillable: number; shortfall: number }; guidance?: string | null; datasheet_uri?: string } | null>(null);
  const askBuyer = async (confirm = false) => {
    if (!buyer.trim()) return;
    setBusy("buyer"); setMsg(null);
    try { setBuyerRes(await api.agentBuyerSpec(buyer.trim(), confirm, confirm ? "buyer-" + Date.now() : undefined)); if (confirm) setMsg("slice composed, sealed export launched, datasheet drafted"); }
    catch (e) { failed("buyer spec", e); }
    finally { setBusy(null); }
  };

  const [ops, setOps] = useState("");
  const [opsRes, setOpsRes] = useState<{ plan: { source: string; steps: { tool: string; mutating: boolean }[] }; status: string; results?: { tool: string; result: Record<string, unknown> }[]; pending?: { tool: string } | null } | null>(null);
  const askOps = async (confirm = false) => {
    if (!ops.trim()) return;
    setBusy("ops"); setMsg(null);
    try { setOpsRes(await api.agentOpsAsk(ops.trim(), confirm)); }
    catch (e) { failed("ops", e); }
    finally { setBusy(null); }
  };

  return (
    <PageShell active="AGENT" subtitle="CONSOLE" right={loading ? <Spinner label="loading" /> : <span className="font-mono text-xs text-ink-3">{queue.length} in fix queue</span>}>
      <div className="flex h-full min-h-0">
        <div className="flex-1 min-w-0 overflow-auto">
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
                {audit?.report?.notes ? (
                  <div className="mt-2 font-mono text-[11px] text-ink-2 space-y-0.5">
                    {audit.report.notes.map((n, i) => <div key={i}>· {n}</div>)}
                    <div className="text-ink-3 mt-1">
                      sampled {audit.report.sampled ?? 0} · vlm-checked {audit.report.vlm_checked ?? 0} · <span className="text-warn">{audit.report.vlm_disagreements ?? 0} vlm-disagree</span> · budget {audit.report.budget?.used ?? 0}/{audit.report.budget?.max_calls ?? 0}
                      {audit.created_at ? ` · ${new Date(audit.created_at).toLocaleString()}` : ""}
                    </div>
                    {(audit.report.confusion_movers?.length ?? 0) > 0 && (
                      <div className="text-ink-3">movers: {audit.report.confusion_movers.map((m) => `${m.from}->${m.to} (${m.n})${m.concentrated_in ? ` in ${m.concentrated_in}` : ""}`).join(", ")}</div>
                    )}
                  </div>
                ) : <div className="mt-2 font-mono text-[11px] text-ink-3">{audit?.status === "running" ? "audit running..." : "no audit yet"}</div>}
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
            <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3 mb-2">Data intelligence: find what matters</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Safety scenarios</div>
                <div className="text-ink-3 text-xs mt-1">Mine near-misses (low TTC), high-risk interactions, and hard-brake events into the scenario queue.</div>
                <button onClick={() => mine("scenarios")} disabled={!!busy} className="mt-3 font-mono text-[11px] border border-line px-3 py-1.5 rounded hover:border-accent disabled:opacity-40">{busy === "scenarios" ? "mining..." : "mine safety scenarios"}</button>
              </div>
              <div className="panel p-4">
                <div className="text-ink font-medium text-sm">Model disagreement</div>
                <div className="text-ink-3 text-xs mt-1">Surface frames where the champion and challenger detectors voted different classes ,  the highest-value labels + a regression signal.</div>
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

          {/* Fleet Dispatch */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3">Fleet dispatch: collect what the corpus lacks</h2>
              <button onClick={planFleet} disabled={!!busy} className="ml-auto font-mono text-[10px] border border-line px-2 py-1 rounded hover:border-accent disabled:opacity-40">{busy === "fleet" ? "planning..." : "plan collection orders"}</button>
            </div>
            {orders.length ? (
              <div className="panel divide-y divide-line/50">
                {orders.map((o) => (
                  <div key={o.order_id} className="flex items-center gap-2 px-3 py-2">
                    <span className="font-mono text-[10px] text-accent w-8">{o.priority.toFixed(1)}</span>
                    <span className="font-mono text-[10.5px] text-ink-2 flex-1">{o.summary}</span>
                    <button onClick={() => dispatchOrder(o.order_id)} disabled={!!busy} className="font-mono text-[10px] border border-accent/40 text-accent px-2 py-0.5 rounded disabled:opacity-40">dispatch</button>
                  </div>
                ))}
              </div>
            ) : <div className="panel p-4 text-ink-3 text-sm">No collection orders. Plan orders to turn the coverage gaps into fleet acquisition tasks.</div>}
          </div>

          {/* Buyer Curation Agent */}
          <div>
            <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3 mb-2">Buyer curation: spec to sealed dataset</h2>
            <div className="panel p-4">
              <div className="flex items-center gap-1.5">
                <input value={buyer} onChange={(e) => setBuyer(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") askBuyer(false); }}
                  placeholder="e.g. 10k night-rain frames with an autorickshaw and a VRU, balanced across cities"
                  className="flex-1 min-w-0 bg-bg-2 border border-line rounded px-2 py-1.5 font-mono text-[11px] text-ink-2 placeholder:text-ink-3/60 focus:border-accent outline-none" />
                <button onClick={() => askBuyer(false)} disabled={!!busy || !buyer.trim()} className="font-mono text-[11px] border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded hover:bg-accent/20 disabled:opacity-40">{busy === "buyer" ? "checking..." : "check"}</button>
              </div>
              {buyerRes?.fulfillment ? (
                <div className="mt-3 font-mono text-[11px] text-ink-2">
                  <div className="text-ink-3">{buyerRes.understood}</div>
                  <div className="mt-1">requested {buyerRes.fulfillment.requested ?? "any"} · <span className="text-pass">{buyerRes.fulfillment.available} available</span>{buyerRes.fulfillment.shortfall ? <span className="text-warn"> · short {buyerRes.fulfillment.shortfall}</span> : <span className="text-pass"> · fully fulfillable</span>}</div>
                  {buyerRes.guidance ? <div className="text-warn mt-1">{buyerRes.guidance}</div> : null}
                  {buyerRes.fulfillment.available > 0 && buyerRes.status === "analyzed" ? (
                    <button onClick={() => askBuyer(true)} disabled={!!busy} className="mt-2 font-mono text-[10px] border border-accent/40 bg-accent/10 text-accent px-2 py-1 rounded disabled:opacity-40">compose slice &amp; export</button>
                  ) : null}
                  {buyerRes.datasheet_uri ? <div className="text-ink-3 mt-1">datasheet: {buyerRes.datasheet_uri.split("/").slice(-2).join("/")}</div> : null}
                </div>
              ) : null}
            </div>
          </div>

          {/* Operations Agent */}
          <div>
            <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3 mb-2">Ask LabeloxAV: operate in sentences</h2>
            <div className="panel p-4">
              <div className="flex items-center gap-1.5">
                <input value={ops} onChange={(e) => setOps(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") askOps(false); }}
                  placeholder="e.g. show coverage gaps · list sessions in BLR · export accepted frames to coco"
                  className="flex-1 min-w-0 bg-bg-2 border border-line rounded px-2 py-1.5 font-mono text-[11px] text-ink-2 placeholder:text-ink-3/60 focus:border-accent outline-none" />
                <button onClick={() => askOps(false)} disabled={!!busy || !ops.trim()} className="font-mono text-[11px] border border-accent/50 bg-accent/10 text-accent px-3 py-1.5 rounded hover:bg-accent/20 disabled:opacity-40">{busy === "ops" ? "planning..." : "ask"}</button>
              </div>
              {opsRes ? (
                <div className="mt-3 font-mono text-[10.5px] text-ink-2">
                  <div className="text-ink-3">plan ({opsRes.plan.source}): {opsRes.plan.steps.map((s) => s.tool).join(" -> ") || "no plan"}</div>
                  {opsRes.results?.map((r, i) => (
                    <pre key={i} className="mt-1 bg-bg-2 rounded p-2 whitespace-pre-wrap max-h-40 overflow-auto no-scrollbar">{r.tool}: {JSON.stringify(r.result).slice(0, 600)}</pre>
                  ))}
                  {opsRes.pending ? (
                    <div className="mt-2 flex items-center gap-2">
                      <span className="text-warn">step &quot;{opsRes.pending.tool}&quot; mutates data.</span>
                      <button onClick={() => askOps(true)} disabled={!!busy} className="font-mono text-[10px] border border-warn text-warn px-2 py-1 rounded disabled:opacity-40">confirm &amp; run</button>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>
          </div>

          {/* Ontology Steward */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <h2 className="font-mono text-[11px] uppercase tracking-wide text-ink-3">Ontology Steward: grow the ontology</h2>
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

          <InterruptedRuns onResumed={load} />

          <JobWatcher />

          <ActivityLog log={log} onClear={() => setLog((l) => ({ entries: [], seq: l.seq }))} />

          {/* What a past relabel run did, grouped by the decision rather than by the object. Fifty
              thousand individual rewrites are not reviewable; the hundred decisions behind them are. */}
          <ContaminationPanel
            onInspect={(l) => { setLineage(l); setSubject(null); setCursor(-1); }}
            inspecting={lineage ? `${lineage.from_name}->${lineage.to_name}` : null} />

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
                  {queue.map((c, i) => (
                    <tr key={c.candidate_id} onClick={() => open(c, i)}
                      title="show the object this is about"
                      className={`border-b hairline cursor-pointer ${i === cursor ? "bg-accent/10" : "hover:bg-bg-2"}`}>
                      <td className="px-3 py-2"><span className="font-mono text-[10px] border border-line px-1.5 py-0.5 rounded text-ink-2">{KIND_LABEL[c.kind] || c.kind}</span></td>
                      <td className="px-3 py-2 font-mono text-ink-3">{c.score.toFixed(2)}</td>
                      <td className="px-3 py-2 text-ink-2 text-xs">{reason(c)}{c.proposed_label?.class_name ? <span className="text-pass"> → {c.proposed_label.class_name}</span> : null}</td>
                      <td className="px-3 py-2 text-right font-mono text-[10px]">
                        {/* stopPropagation: acting on a row must not also re-open it, which would refetch
                            the object that just left the queue. */}
                        <button onClick={(e) => { e.stopPropagation(); act(c, "confirm"); }} className="border border-pass text-pass px-1.5 py-0.5 rounded mr-1">confirm</button>
                        <button onClick={(e) => { e.stopPropagation(); act(c, "dismiss"); }} className="border border-line text-ink-3 px-1.5 py-0.5 rounded hover:text-ink">dismiss</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="px-4 py-10 text-center text-ink-3 text-sm">Fix queue is empty. Run an error sweep to scan the corpus.</div>}
            {queue.length > 0 && (
              <div className="border-t hairline px-4 py-1.5 font-mono text-[10px] text-ink-3">
                <span className="text-ink-2">j</span>/<span className="text-ink-2">k</span> move
                <span className="mx-2">·</span>
                <span className="text-pass">c</span> confirm
                <span className="mx-2">·</span>
                <span className="text-block">d</span> dismiss
                <span className="mx-2">·</span>
                <span className="text-ink-2">esc</span> close
              </div>
            )}
          </div>
        </div>
        </div>
        {/* One rail, three things it can be showing: the result of the last action, the objects behind a
            lineage, or the evidence for one object. They are exclusive because they answer the same
            question - "what is this about" - and stacking them would bury whichever you asked for last. */}
        <Inspector title={result ? "result" : lineage ? "lineage" : "evidence"} side="right" width="w-[26rem]"
          meta={result ? undefined : lineage ? undefined
            : (cursor >= 0 && queue.length ? `${cursor + 1}/${queue.length}` : undefined)}>
          {result ? (
            <ActionResultPanel result={result}
              onOpenObject={(id) => { setResult(null); setSubject({ objectId: id }); }}
              onOpenFrame={(id) => router.push(`/frame/${id}`)} />
          ) : (<>
          {lineage && (
            <div className="border-b hairline">
              <LineageObjects fromName={lineage.from_name} toName={lineage.to_name}
                selectedId={subject?.objectId ?? null}
                onPick={(objectId) => setSubject({
                  objectId,
                  text: `Moved by a past relabel run: ${lineage.from_name} -> ${lineage.to_name}.`,
                  kind: "relabel lineage",
                  // No suggestion button here. Reverting the whole lineage is the row's own action and
                  // is a bulk decision; offering a per-object relabel beside it would be two ways to
                  // undo the same thing that do not agree on scope.
                  suggestion: null,
                })} />
            </div>
          )}
          <EvidencePanel
            subject={subject}
            onResolved={() => load()}
            actions={cursor >= 0 && queue[cursor] ? [
              { key: "confirm", label: "confirm", tone: "accept" as const,
                hint: "the label is wrong; apply the proposed class and mark it reviewed",
                run: async () => { await api.errorConfirm(queue[cursor].candidate_id); load(); } },
              { key: "dismiss", label: "dismiss", tone: "reject" as const,
                hint: "the label is fine; this candidate was a false alarm",
                run: async () => { await api.errorDismiss(queue[cursor].candidate_id); load(); } },
            ] : []}
          />
          </>)}
        </Inspector>
      </div>
    </PageShell>
  );
}
