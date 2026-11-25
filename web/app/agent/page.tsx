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
import { invalidate, useQuery } from "@/lib/useQuery";
import { REVIEW_QUEUE_KEY } from "@/lib/queryKeys";
import { T } from "@/lib/i18n";
import EmptyState from "@/components/ui/EmptyState";
import Thumb from "@/components/engine/Thumb";
import { ago, humanize, num } from "@/lib/format";

// The Agent Console: the corpus-level home for the autonomous agent. Self-healing QA (the error daemon
// sweep + temporal auto-repair) and the ranked fix queue it produces. Programs 3-5 (data intelligence,
// flywheel, copilot) add their sections here.

type Cand = { candidate_id: string; object_id: string; kind: string; score: number; detail: Record<string, unknown>; proposed_label?: { class_name?: string } | null };

// Stable empties: the queue effects below depend on the array's identity, and a fresh [] on every render
// while nothing has loaded would re-run them each time.
const NO_CANDS: Cand[] = [];
const NO_SUMMARY: Record<string, number> = {};
const NO_PROPS: PromotionProposalRow[] = [];
const NO_ORDERS: { order_id: string; priority: number; summary: string; status: string }[] = [];

const KIND_LABEL: Record<string, string> = {
  critic_flag: "Inconsistent", near_dup_inconsistent: "Near-copy differs", policy_violation: "Breaks a rule",
  track_inconsistent: "Class flips in track", cross_cam_inconsistent: "Cameras disagree", embedding_outlier: "Looks unusual",
  confident_learning: "Confidently wrong",
  // Was missing, so every one of these rendered as the raw slug. It is also the kind that dominates the
  // queue, because the error sweep never clears it (services/errordetect/queue.py scopes its delete by
  // the kinds it was asked to run, and this is not one of them).
  vlm_confusion: "Vision model disagrees",
  // The bulk of this queue and it rendered as a raw slug. Its claim ("another box covers the same object
  // with higher confidence") is the reason the panel draws the frame's other boxes: one box cannot show
  // a duplicate.
  reanalyze: "Re-check",
};

// Server-written notes name classes by their stored codes ("traffic_sign"); shown as words.
function wordsOf(text: string): string {
  return text.replace(/\b([a-z0-9]+(?:_[a-z0-9]+)+)\b/g, (m) => m.replace(/_/g, " "));
}

// One automated action: what it does in a sentence, the rest behind "How this works", and its button(s).
function ActionCard({ title, desc, more, action, children, className = "" }: {
  title: string; desc: string; more?: React.ReactNode; action?: React.ReactNode; children?: React.ReactNode; className?: string;
}) {
  return (
    <section className={`panel p-4 flex flex-col gap-2.5 ${className}`}>
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <h3 className="section-title">{title}</h3>
          <p className="section-desc mt-1">{desc}</p>
        </div>
        {action && <div className="shrink-0 flex flex-wrap gap-2 justify-end">{action}</div>}
      </div>
      {more && (
        <details className="text-[12px] text-ink-3">
          <summary className="cursor-pointer select-none text-ink-2 hover:text-ink w-fit">How this works</summary>
          <div className="mt-1.5 leading-relaxed">{more}</div>
        </details>
      )}
      {children}
    </section>
  );
}

// A titled group of cards, so the page reads as a handful of sections rather than one long column.
function Group({ title, desc, right, children }: { title: string; desc?: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="space-y-2.5">
      <div className="flex items-end gap-3">
        <div className="min-w-0">
          <h2 className="section-title">{title}</h2>
          {desc && <p className="section-desc mt-0.5">{desc}</p>}
        </div>
        {right && <div className="ml-auto shrink-0">{right}</div>}
      </div>
      {children}
    </div>
  );
}

export default function AgentConsole() {
  const router = useRouter();
  // The fix queue. Cached, so coming back to this page shows the last queue while it refreshes. A failed
  // load was never rendered here and still is not: the queue simply stays as it was.
  const queueQ = useQuery(["agentErrorQueue"], () => api.agentErrorQueue());
  const queue: Cand[] = queueQ.data?.candidates ?? NO_CANDS;
  const summary: Record<string, number> = queueQ.data?.summary ?? NO_SUMMARY;
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
  // True during any fetch of the queue, the first or a refresh, as the flag it replaces was.
  const loading = queueQ.validating;
  const load = queueQ.reload;

  // The latest audit, or nothing: there may not have been one yet, so a failure is not shown.
  const auditQ = useQuery<{ status?: string; created_at?: string; report: AuditReport | null }>(
    ["agentAuditLatest"], () => api.agentAuditLatest());
  const audit = auditQ.data ?? null;
  const setAudit = auditQ.setData;

  const runAudit = async () => {
    setBusy("audit"); setMsg(null);
    try {
      const r = await api.agentAuditRun({ sample_size: 200, vlm_calls: 60 });
      showResult("audit", "overnight audit", r);
      setMsg("Audit running in the background: spot-checking today's auto-accepted labels");
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
      setMsg("Error sweep running in the background. The fix queue refreshes as each session finishes.");
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
      setMsg("Relabel running: a second model is re-reading every machine-drawn box");
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
      setMsg(`Bonnet masks: ${r.with_hood} of ${r.cameras} cameras show the vehicle's own bonnet${r.no_hood.length ? ` (no hood: ${r.no_hood.slice(0, 4).join(", ")})` : ""}`);
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
      setMsg("Blurring faces and number plates on older frames in the background");
    } catch (e) { failed("PII backfill", e); }
    finally { setBusy(null); }
  };
  const redetectAll = async () => {
    setBusy("redetect"); setMsg(null);
    try {
      const r = await api.redetectAll(true);
      showResult("redetect", "full re-detection", r);
      setMsg(`Re-detection started: blurring first, then every session re-detected one at a time on the GPU (run ${r.run_id.slice(0, 8)})`);
    } catch (e) { failed("re-detection", e); }
    finally { setBusy(null); }
  };

  const repair = async () => {
    setBusy("repair"); setMsg(null);
    try {
      const p = await api.agentTemporalRepairPlan();
      showResult("repair", "temporal repair plan", p);
      if (!p.counts.relabels) { setMsg(`Nothing safe to repair: ${p.counts.tracks} tracks checked, ${p.counts.flipped_tracks} flip class, ${p.counts.skipped_static ?? 0} left for a person`); return; }
      const r = await api.agentTemporalRepair();
      setMsg(`Track repair: ${r.relabeled} labels changed to match the rest of their track (can be undone, run ${r.run_id.slice(0, 8)})`);
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
        setMsg(`Found ${r.persisted} safety scenarios${Object.keys(r.by_kind).length ? ` (${Object.entries(r.by_kind).map(([k, n]) => `${n} ${humanize(k).toLowerCase()}`).join(", ")})` : ""}`);
        showResult("scenarios", "mine safety scenarios", r,
          { href: "/scenarios", label: `open the ${r.persisted} scenarios` });
      } else {
        const r = await api.agentMineDisagreements();
        setMsg(`Found ${r.persisted} frames where the models disagree${r.top[0] ? ` (most common: ${humanize(r.top[0].tag).toLowerCase()})` : ""}`);
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
      setMsg(`Preview of one loop turn: across ${r.tick.frames} high-value frames it would accept ${r.tick.auto_accept}, send ${r.tick.review} to review and ${r.tick.annotate} to labelling`);
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
      setMsg(r.status === "rolled_back" ? `Live model got worse on the gold set (${r.baseline_map} to ${r.current_map}): rolled back and paused the loop`
        : r.status === "healthy" ? `Live model is healthy on the gold set (${r.current_map}, was ${r.baseline_map})`
        : r.status === "cannot_evaluate" ? `Cannot check live model ${r.champion}: the gold set is not available here (last score ${r.baseline_map})`
        : "No live model registered yet");
    } catch (e) { failed("gold-drift check", e); }
    finally { setBusy(null); }
  };

  // Memoised because the keyboard effect depends on it: recreated every render, the listener would be
  // torn down and rebound on every keystroke that changed any state on this page.
  const act = useCallback(async (c: Cand, kind: "confirm" | "dismiss") => {
    try {
      await (kind === "confirm" ? api.errorConfirm(c.candidate_id) : api.errorDismiss(c.candidate_id));
      // This page's queue, and the error-candidate lists the quality pages cache.
      void invalidate("agentErrorQueue", REVIEW_QUEUE_KEY);
    }
    catch (e) { failed(kind === "confirm" ? "confirm" : "dismiss", e); }
  }, [failed]);

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

  // Best-effort: the panel is simply absent when there is no diagnosis or it could not be read.
  const driftQ = useQuery<{ report: { hypothesis: string; proposed_action: { kind: string } } | null }>(
    ["agentDriftLatest"], () => api.agentDriftLatest());
  const driftDiag = driftQ.data ?? null;
  const setDriftDiag = driftQ.setData;
  const investigateDrift = async () => {
    setBusy("driftinv"); setMsg(null);
    try {
      const r = await api.agentDriftInvestigate();
      showResult("driftinv", "drift investigation", r);
      if (!r.breached.length) { setMsg("No drift right now: everything is within tolerance"); return; }
      setMsg(`Drift found (${r.breached.map(humanize).join(", ")}): looking for the cause in the background`);
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
      setDoc(r.markdown); setMsg(`${kind === "datasheet" ? "Datasheet" : "Weekly report"} drafted and saved (${r.uri.split("/").slice(-2).join("/")})`);
    } catch (e) { failed("doc generation", e); }
    finally { setBusy(null); }
  };

  const propsQ = useQuery(["agentOntologyProposals"], () => api.agentOntologyProposals());
  const props: PromotionProposalRow[] = propsQ.data ?? NO_PROPS;
  const [names, setNames] = useState<Record<string, string>>({});
  const loadProps = propsQ.reload;
  const scanOntology = async () => {
    setBusy("ontscan"); setMsg(null);
    try {
      const r = await api.agentOntologyScan(40);
      showResult("ontscan", "scan for ontology gaps", r);
      setMsg(`Looked at ${r.scanned} catch-all labels: ${r.proposals} could become new classes`);
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
        if (!nm) { setMsg("Give the new class a name first"); return; }
        const r = await api.agentOntologyApprove(id, nm);
        showResult("ontology", `mint ${nm}`, r);
        setMsg(`Added class ${r.name} and moved ${r.relabeled} objects into it (can be undone, run ${r.run_id.slice(0, 8)})`);
        // A new class exists and objects moved into it: the cached ontology and run lists are stale.
        void invalidate("ontology", "agentRuns");
      } else { await api.agentOntologyReject(id); setMsg("Proposal rejected"); }
      await loadProps();
    } catch (e) { failed("decision", e); }
    finally { setBusy(null); }
  };

  const ordersQ = useQuery(["agentFleetOrders", "proposed"], () => api.agentFleetOrders("proposed"));
  const orders: { order_id: string; priority: number; summary: string; status: string }[] = ordersQ.data ?? NO_ORDERS;
  const loadOrders = ordersQ.reload;
  const planFleet = async () => {
    setBusy("fleet"); setMsg(null);
    try {
      const r = await api.agentFleetPlan();
      showResult("fleet", "plan fleet collection", r);
      setMsg(`Planned ${r.orders} collection orders from ${r.gaps} gaps and ${r.vehicles} vehicles`);
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
    try { setBuyerRes(await api.agentBuyerSpec(buyer.trim(), confirm, confirm ? "buyer-" + Date.now() : undefined)); if (confirm) setMsg("Dataset assembled, sealed export started, datasheet drafted"); }
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

  const btn = (key: string, idle: string, working: string, onClick: () => void, primary = false) => (
    <button type="button" onClick={onClick} disabled={!!busy} className={primary ? "btn btn-primary" : "btn"}>
      {busy === key ? working : idle}
    </button>
  );

  return (
    <PageShell active="AGENT" title="Agent console"
      subtitle="Automated checks that find likely-wrong labels and fix the obvious ones. Everything it does can be undone."
      right={loading ? <Spinner label="loading" /> : <span className="text-[12.5px] text-ink-3">{num(queue.length)} in fix queue</span>}>
      <div className="flex h-full min-h-0">
        <div className="flex-1 min-w-0 overflow-auto">
        <div className="max-w-5xl mx-auto px-6 py-5 space-y-7">

          {/* Overnight Auditor: standing watchdog + morning report */}
          <section className="panel p-4 border-accent/40">
            <div className="flex items-start gap-3">
              <div className="flex-1 min-w-0">
                <h2 className="section-title">Overnight audit</h2>
                <p className="section-desc mt-1">Each night a sample of the day&apos;s auto-accepted labels is double-checked; anything suspect goes to review.</p>
              </div>
              <button type="button" onClick={runAudit} disabled={!!busy} className="btn btn-primary shrink-0">
                {busy === "audit" ? "Auditing..." : "Run audit now"}
              </button>
            </div>
            {audit?.report?.notes ? (
              <div className="mt-3 border-t hairline pt-3 space-y-2">
                <div className="flex items-baseline gap-2">
                  <span className="label-caps">Last report</span>
                  {audit.created_at && <span className="text-[12px] text-ink-3">{ago(audit.created_at)}</span>}
                </div>
                <ul className="text-[12.5px] text-ink-2 space-y-1 list-disc pl-5 marker:text-ink-3">
                  {audit.report.notes.map((n, i) => <li key={i}>{wordsOf(n)}</li>)}
                </ul>
                <div className="text-[12px] text-ink-3">
                  {num(audit.report.sampled ?? 0)} labels sampled · {num(audit.report.vlm_checked ?? 0)} checked by the vision model ·{" "}
                  <span className="text-warn">{num(audit.report.vlm_disagreements ?? 0)} disagreed</span> ·{" "}
                  {num(audit.report.budget?.used ?? 0)} of {num(audit.report.budget?.max_calls ?? 0)} checks used
                </div>
                {(audit.report.confusion_movers?.length ?? 0) > 0 && (
                  <div className="text-[12px] text-ink-3">
                    Most confused: {audit.report.confusion_movers.map((m) => `${humanize(m.from)} → ${humanize(m.to).toLowerCase()} (${m.n})`).join(", ")}
                  </div>
                )}
              </div>
            ) : (
              <div className="mt-3 text-[12.5px] text-ink-3">{audit?.status === "running" ? "Audit running..." : "No audit has run yet."}</div>
            )}
          </section>

          {/* Fix queue: the page's working list, first after the audit that feeds it */}
          <section className="panel">
            <div className="panel-head flex-wrap">
              <h2 className="section-title">Fix queue</h2>
              <span className="text-[12px] text-ink-3">Labels most likely to be wrong, worst first</span>
              <div className="ml-auto flex flex-wrap gap-1.5">
                {Object.entries(summary).map(([k, n]) => (
                  <span key={k} className="text-[11px] border border-line rounded px-1.5 py-px text-ink-3">
                    {KIND_LABEL[k] || humanize(k)} <span className="text-ink-2 tabular-nums">{num(n)}</span>
                  </span>
                ))}
              </div>
            </div>
            {loading && !queue.length ? <div className="p-6"><Spinner label="loading" /></div> : queue.length ? (
              <div className="max-h-[520px] overflow-y-auto">
              <table className="data-table">
                <thead>
                  <tr><th className="w-40">Problem</th><th className="w-16 num">Score</th><th>Why</th><th className="w-44"></th></tr>
                </thead>
                <tbody>
                  {queue.map((c, i) => (
                    <tr key={c.candidate_id} onClick={() => open(c, i)}
                      title="show the object this is about"
                      className={`cursor-pointer ${i === cursor ? "bg-accent/10" : ""}`}>
                      <td><span className="text-[11.5px] border border-line px-1.5 py-0.5 rounded text-ink-2 whitespace-nowrap">{KIND_LABEL[c.kind] || humanize(c.kind)}</span></td>
                      <td className="num font-mono text-ink-3">{c.score.toFixed(2)}</td>
                      <td>{wordsOf(reason(c))}{c.proposed_label?.class_name ? <span className="text-pass"> → {humanize(c.proposed_label.class_name)}</span> : null}</td>
                      <td className="text-right whitespace-nowrap space-x-1.5">
                        {/* stopPropagation: acting on a row must not also re-open it, which would refetch
                            the object that just left the queue. */}
                        <button type="button" onClick={(e) => { e.stopPropagation(); act(c, "confirm"); }} className="btn text-pass"
                          data-tip="The label is wrong: apply the fix"><T k="action.confirm_short">Confirm</T></button>
                        <button type="button" onClick={(e) => { e.stopPropagation(); act(c, "dismiss"); }} className="btn"
                          data-tip="The label is fine: drop this from the queue"><T k="action.dismiss">Dismiss</T></button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              </div>
            ) : (
              <EmptyState compact icon="check" title="Fix queue is empty"
                action={btn("sweep", "Run error sweep", "Sweeping...", sweep, true)}>
                No label is currently suspected of being wrong. An error sweep checks the whole corpus again.
              </EmptyState>
            )}
            {queue.length > 0 && (
              <div className="border-t hairline px-4 py-2 text-[12px] text-ink-3 flex flex-wrap gap-x-4 gap-y-1">
                <span><kbd className="font-mono text-ink-2">J</kbd> / <kbd className="font-mono text-ink-2">K</kbd> next / previous</span>
                <span><kbd className="font-mono text-pass">C</kbd> confirm</span>
                <span><kbd className="font-mono text-block">D</kbd> dismiss</span>
                <span><kbd className="font-mono text-ink-2">Esc</kbd> close</span>
              </div>
            )}
          </section>

          <InterruptedRuns onResumed={load} />

          <JobWatcher />

          <ActivityLog log={log} onClear={() => setLog((l) => ({ entries: [], seq: l.seq }))} />

          {/* Self-healing: corpus-wide fixes */}
          <Group title="Fix labels across the corpus" desc="Every run is recorded and can be undone from Agent jobs.">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <ActionCard title="Error sweep" desc="Re-run every mistake detector across the corpus and refresh the fix queue."
                more="Looks for labels that disagree with their near-copies, break a class rule, flip class within a track, look unlike the rest of their class, or that the model is confident are wrong."
                action={btn("sweep", "Run error sweep", "Sweeping...", sweep)} />
              <ActionCard title="Track repair" desc="Where one frame of a track has a different class from the rest, change it to match."
                more="Only changes a label when most of the track clearly agrees. Tracks that are inconsistent throughout are left for a person."
                action={btn("repair", "Repair tracks", "Repairing...", repair)} />
              <ActionCard title="Relabel all frames" className="md:col-span-2"
                desc="A second model re-reads every machine-drawn box and corrects the class where it clearly disagrees."
                more={<>Clear corrections are applied and kept; less certain ones are applied but also sent to review. Each frame is one run, so the whole pass can be undone. To relabel a single frame, use the Agent panel in the editor.</>}
                action={btn("relabel", "Relabel all frames", "Starting...", relabelAll)}>
                {relabel ? (
                  <div className="text-[12.5px] text-ink-2">
                    {relabelDone ? "Done" : "Running"} · {num(relabel.frames)} frames checked · <span className="text-pass">{num(relabel.relabel_keep)} fixed</span> · <span className="text-warn">{num(relabel.relabel_review)} sent to review</span>
                  </div>
                ) : null}
              </ActionCard>
              <ActionCard title="Re-detect with the current rules" className="md:col-span-2"
                desc="Bring frames detected under older rules up to date. Run the three steps in order; it uses the GPU one session at a time."
                more="Newer detection skips scenery such as trees and sky, ignores the vehicle's own bonnet, keeps one box per object and refuses boxes that cover the whole frame. Older frames were labelled before those rules existed. The run steps aside when training needs the GPU.">
                <div className="flex flex-wrap gap-2">
                  {btn("ego", "1. Find the bonnet in each camera", "Finding...", estimateEgo)}
                  {btn("pii", "2. Blur faces and plates on older frames", "Starting...", backfillPii)}
                  {btn("redetect", "3. Re-detect all frames", "Starting...", redetectAll)}
                </div>
              </ActionCard>
            </div>
          </Group>

          {/* Data intelligence: the system finds what is worth labeling */}
          <Group title="Find what is worth labelling">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <ActionCard title="Safety scenarios" desc="Find near-misses, risky interactions and hard braking, and add them to Scenarios."
                action={btn("scenarios", "Find", "Searching...", () => mine("scenarios"))} />
              <ActionCard title="Model disagreement" desc="Find frames where the live and candidate models chose different classes: the most useful labels to check."
                action={btn("disagreements", "Find", "Searching...", () => mine("disagreements"))} />
              <ActionCard title="Coverage gaps" desc="List what the corpus is short of: rare classes, weather, time of day, road types."
                action={btn("coverage", "Report", "Analysing...", coverage)} />
            </div>
            {gaps ? (
              <div className="panel p-4">
                <div className="label-caps mb-1.5">Coverage gaps ({num(gaps.length)})</div>
                <ul className="space-y-0.5 list-disc pl-5 marker:text-ink-3">{gaps.slice(0, 10).map((g, i) => <li key={i} className="text-[12.5px] text-ink-2">{wordsOf(g)}</li>)}</ul>
              </div>
            ) : null}
          </Group>

          {/* Self-improving loop */}
          <Group title="Keep the model improving">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <ActionCard title="Preview one loop turn" desc="See what one turn of the improvement loop would accept, review and send to labelling. Changes nothing."
                action={btn("cycle", "Preview", "Running...", cycle)} />
              <ActionCard title="Gold-set check" desc="Re-score the live model on the gold set. If it has got worse, roll back and pause the loop."
                action={btn("drift", "Check now", "Checking...", drift)} />
              <ActionCard title="Investigate drift" desc="When the data drifts, find the likely cause and propose a fix. Proposes only."
                action={btn("driftinv", "Investigate", "Investigating...", investigateDrift)}>
                {driftDiag?.report ? (
                  <div className="text-[12.5px] text-ink-2 border-t hairline pt-2">
                    <div>{driftDiag.report.hypothesis}</div>
                    <div className="text-ink-3 mt-0.5">Proposed: {humanize(driftDiag.report.proposed_action.kind).toLowerCase()}</div>
                  </div>
                ) : null}
              </ActionCard>
            </div>
          </Group>

          {/* Questions in plain language */}
          <Group title="Ask in plain language">
            <div className="space-y-3">
              <section className="panel p-4 space-y-2.5">
                <div className="flex items-center gap-2">
                  <h3 className="section-title">Ask the dataset</h3>
                  <span className="section-desc">Count frames matching a description</span>
                  <button type="button" onClick={doReport} disabled={!!busy} className="btn ml-auto">{busy === "report" ? "Loading..." : "Dataset summary"}</button>
                </div>
                <div className="flex items-center gap-2">
                  <input value={ask} onChange={(e) => setAsk(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") doAsk(); }}
                    aria-label="Describe the frames to find"
                    placeholder="e.g. two-wheelers going against traffic at night on the highway"
                    className="field flex-1" />
                  <button type="button" onClick={doAsk} disabled={!!busy || !ask.trim()} className="btn btn-primary">{busy === "ask" ? "Asking..." : "Ask"}</button>
                </div>
                {askResult ? <div className="text-[12.5px] text-ink-2"><span className="text-pass">{num(askResult.count)} frames</span> · understood as <span className="text-ink-3">{askResult.understood}</span></div> : null}
                {report ? (
                  <div className="border-t hairline pt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-[12px]">
                    <div><div className="label-caps">Objects</div><div className="text-ink text-[18px] font-semibold tabular-nums">{num(report.size.objects)}</div><div className="text-ink-3">{num(report.size.human_labeled)} by people</div></div>
                    <div><div className="label-caps">Sessions</div><div className="text-ink text-[18px] font-semibold tabular-nums">{num(report.size.sessions)}</div></div>
                    <div><div className="label-caps">Coverage gaps</div><div className="text-warn text-[18px] font-semibold tabular-nums">{num(report.coverage_gaps.length)}</div></div>
                    <div><div className="label-caps">Fix queue</div><div className="text-block text-[18px] font-semibold tabular-nums">{num(report.fix_queue_total)}</div><div className="text-ink-3">{num(Object.values(report.scenarios).reduce((a, b) => a + b, 0))} scenarios</div></div>
                  </div>
                ) : null}
              </section>

              <section className="panel p-4 space-y-2.5">
                <div className="flex items-center gap-2">
                  <h3 className="section-title">Ask LabeloxAV to do something</h3>
                  <span className="section-desc">Anything that changes data asks you first</span>
                </div>
                <div className="flex items-center gap-2">
                  <input value={ops} onChange={(e) => setOps(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") askOps(false); }}
                    aria-label="What should LabeloxAV do"
                    placeholder="e.g. show coverage gaps · list sessions in BLR · export accepted frames to COCO"
                    className="field flex-1 min-w-0" />
                  <button type="button" onClick={() => askOps(false)} disabled={!!busy || !ops.trim()} className="btn btn-primary">{busy === "ops" ? "Planning..." : "Ask"}</button>
                </div>
                {opsRes ? (
                  <div className="text-[12.5px] text-ink-2">
                    <div className="text-ink-3">Plan: {opsRes.plan.steps.map((s) => humanize(s.tool)).join(" → ") || "nothing to do"}</div>
                    {opsRes.results?.map((r, i) => (
                      <pre key={i} className="mt-1.5 bg-bg-2 rounded p-2 text-[12px] whitespace-pre-wrap max-h-40 overflow-auto no-scrollbar">{humanize(r.tool)}: {JSON.stringify(r.result).slice(0, 600)}</pre>
                    ))}
                    {opsRes.pending ? (
                      <div className="mt-2 flex items-center gap-2">
                        <span className="text-warn">&ldquo;{humanize(opsRes.pending.tool)}&rdquo; will change data.</span>
                        <button type="button" onClick={() => askOps(true)} disabled={!!busy} className="btn btn-danger">Confirm and run</button>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </section>

              <section className="panel p-4 space-y-2.5">
                <div className="flex items-center gap-2">
                  <h3 className="section-title">Build a dataset from a request</h3>
                  <span className="section-desc">Check what is available, then assemble and export it</span>
                </div>
                <div className="flex items-center gap-2">
                  <input value={buyer} onChange={(e) => setBuyer(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") askBuyer(false); }}
                    aria-label="Describe the dataset"
                    placeholder="e.g. 10k night-rain frames with an autorickshaw and a pedestrian, balanced across cities"
                    className="field flex-1 min-w-0" />
                  <button type="button" onClick={() => askBuyer(false)} disabled={!!busy || !buyer.trim()} className="btn btn-primary">{busy === "buyer" ? "Checking..." : "Check"}</button>
                </div>
                {buyerRes?.fulfillment ? (
                  <div className="text-[12.5px] text-ink-2">
                    <div className="text-ink-3">{buyerRes.understood}</div>
                    <div className="mt-1">Asked for {buyerRes.fulfillment.requested != null ? num(buyerRes.fulfillment.requested) : "any number"} · <span className="text-pass">{num(buyerRes.fulfillment.available)} available</span>{buyerRes.fulfillment.shortfall ? <span className="text-warn"> · {num(buyerRes.fulfillment.shortfall)} short</span> : <span className="text-pass"> · can be met in full</span>}</div>
                    {buyerRes.guidance ? <div className="text-warn mt-1">{buyerRes.guidance}</div> : null}
                    {buyerRes.fulfillment.available > 0 && buyerRes.status === "analyzed" ? (
                      <button type="button" onClick={() => askBuyer(true)} disabled={!!busy} className="btn btn-primary mt-2">Assemble and export</button>
                    ) : null}
                    {buyerRes.datasheet_uri ? <div className="text-ink-3 mt-1">Datasheet: {buyerRes.datasheet_uri.split("/").slice(-2).join("/")}</div> : null}
                  </div>
                ) : null}
              </section>
            </div>
          </Group>

          {/* Ontology Steward */}
          <Group title="New classes" desc="Catch-all labels that keep collecting the same kind of object may deserve a class of their own."
            right={btn("ontscan", "Look for new classes", "Looking...", scanOntology)}>
            {props.length ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {props.map((p) => (
                  <div key={p.proposal_id} className="panel p-3.5">
                    <div className="flex gap-1.5 mb-2.5">
                      {p.sample_object_ids.slice(0, 6).map((oid) => (
                        <Thumb key={oid} src={`/api/objects/${oid}/crop`} className="w-12 h-12 rounded" />
                      ))}
                    </div>
                    <div className="text-[12.5px] text-ink-2">
                      {num(p.member_count)} objects labelled {humanize(p.from_class).toLowerCase()} · look like {p.confusion_classes.map((c) => `${humanize(c.class).toLowerCase()} ${Math.round(c.share * 100)}%`).join(", ") || "nothing known"}
                    </div>
                    <div className="flex items-center gap-1.5 mt-2.5">
                      <input value={names[p.proposal_id] ?? p.suggested_name ?? ""} onChange={(e) => setNames((s) => ({ ...s, [p.proposal_id]: e.target.value }))}
                        aria-label="Name for the new class"
                        placeholder="New class name" className="field flex-1 min-w-0" />
                      <button type="button" onClick={() => decide(p.proposal_id, "approve")} disabled={!!busy} className="btn btn-primary">Add class</button>
                      <button type="button" onClick={() => decide(p.proposal_id, "reject")} disabled={!!busy} className="btn"><T k="action.reject">Reject</T></button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="panel">
                <EmptyState compact icon="objects" title="No new classes proposed">
                  Look for new classes to check whether any catch-all label has grown a group worth naming.
                </EmptyState>
              </div>
            )}
          </Group>

          {/* Fleet Dispatch */}
          <Group title="Collection orders" desc="Turn coverage gaps into requests for vehicles to record what is missing."
            right={btn("fleet", "Plan orders", "Planning...", planFleet)}>
            {orders.length ? (
              <div className="panel divide-y divide-line/50">
                {orders.map((o) => (
                  <div key={o.order_id} className="flex items-center gap-3 px-4 py-2.5">
                    <span className="text-[12px] text-accent-2 w-20 shrink-0 tabular-nums" data-tip="Higher is more urgent">Priority {o.priority.toFixed(1)}</span>
                    <span className="text-[12.5px] text-ink-2 flex-1">{o.summary}</span>
                    <button type="button" onClick={() => dispatchOrder(o.order_id)} disabled={!!busy} className="btn">Send to fleet</button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="panel">
                <EmptyState compact icon="route" title="No collection orders">
                  Plan orders to turn today&apos;s coverage gaps into recording requests for the fleet.
                </EmptyState>
              </div>
            )}
          </Group>

          {/* Documentation agent */}
          <Group title="Documents">
            <ActionCard title="Draft a document" desc="Write a dataset datasheet or this week's quality report from the platform's own numbers."
              more="The datasheet covers composition, coverage and known gaps. The weekly report covers precision, drift and model promotions. Model cards are drafted per model through the API."
              action={<>
                <button type="button" onClick={() => genDoc("datasheet")} disabled={!!busy} className="btn">{busy === "doc" ? "Drafting..." : "Datasheet"}</button>
                <button type="button" onClick={() => genDoc("weekly")} disabled={!!busy} className="btn">Weekly report</button>
              </>}>
              {doc ? <pre className="max-h-64 overflow-auto no-scrollbar bg-bg-2 rounded p-3 text-[12px] text-ink-2 whitespace-pre-wrap">{doc.slice(0, 4000)}</pre> : null}
            </ActionCard>
          </Group>

          {/* What a past relabel run did, grouped by the decision rather than by the object. Fifty
              thousand individual rewrites are not reviewable; the hundred decisions behind them are. */}
          <ContaminationPanel
            onInspect={(l) => { setLineage(l); setSubject(null); setCursor(-1); }}
            inspecting={lineage ? `${lineage.from_name}->${lineage.to_name}` : null} />
        </div>
        </div>
        {/* One rail, three things it can be showing: the result of the last action, the objects behind a
            lineage, or the evidence for one object. They are exclusive because they answer the same
            question - "what is this about" - and stacking them would bury whichever you asked for last. */}
        <Inspector title={result ? "Result" : lineage ? "Objects changed" : "Evidence"} side="right" width="w-[26rem]"
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
                  text: `Moved by a past relabel run: ${humanize(lineage.from_name)} → ${humanize(lineage.to_name).toLowerCase()}.`,
                  kind: "Past relabel",
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
              { key: "confirm", label: "Confirm", tone: "accept" as const,
                hint: "the label is wrong; apply the proposed class and mark it reviewed",
                run: async () => { await api.errorConfirm(queue[cursor].candidate_id); void invalidate("agentErrorQueue", REVIEW_QUEUE_KEY); } },
              { key: "dismiss", label: "Dismiss", tone: "reject" as const,
                hint: "the label is fine; this candidate was a false alarm",
                run: async () => { await api.errorDismiss(queue[cursor].candidate_id); void invalidate("agentErrorQueue", REVIEW_QUEUE_KEY); } },
            ] : []}
          />
          </>)}
        </Inspector>
      </div>
    </PageShell>
  );
}
