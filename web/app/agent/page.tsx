"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
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

  const load = useCallback(async () => {
    setLoading(true);
    try { const q = await api.agentErrorQueue(); setQueue(q.candidates); setSummary(q.summary); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { load(); }, [load]);

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

  const reason = (c: Cand) => (c.detail?.reason as string) || (c.detail?.reasons ? (c.detail.reasons as string[]).join("; ") : (c.detail?.note as string) || JSON.stringify(c.detail).slice(0, 80));

  return (
    <PageShell active="AGENT" subtitle="CONSOLE" right={loading ? <Spinner label="loading" /> : <span className="font-mono text-xs text-ink-3">{queue.length} in fix queue</span>}>
      <div className="h-full overflow-auto">
        <div className="max-w-5xl mx-auto p-6 space-y-5">
          <div>
            <h1 className="text-xl text-ink font-semibold">Agent Console</h1>
            <p className="text-ink-3 text-sm mt-1 max-w-2xl">Autonomous QA over the whole corpus: the agent finds likely-wrong labels and fixes the obvious ones itself. Everything it does is reversible and provenance-stamped.</p>
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
            </div>
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
