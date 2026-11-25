"use client";

import { useEffect, useState } from "react";
import { T } from "@/lib/i18n";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";
import { invalidate, useQuery } from "@/lib/useQuery";
import type { Campaign, CampaignDetail } from "@/lib/types";
import EmptyState from "@/components/ui/EmptyState";
import LoadState from "@/components/shell/LoadState";
import { humanize, num } from "@/lib/format";

// Campaigns: the improvement loop, run by the system instead of by somebody remembering to.
//
// Every stage of improving a stuck class already existed and a person stood between each pair of them:
// read the gate's deficit, mine, judge, wait for humans, retrain, promote, decide whether to go again.
// A class stalled the moment nobody was watching.
//
// The board is built around the two things an operator actually needs to see: which campaign is waiting
// on them, and which one has stopped paying. Everything else is detail behind a click.

const STAGES = ["mine", "judge", "label", "train", "evaluate", "promote"] as const;
// What each stage does, for the stage rail's tooltips.
const STAGE_TIP: Record<string, string> = {
  mine: "Find frames likely to contain the class", judge: "Have a model check the found frames",
  label: "People label the batch", train: "Retrain the detector", evaluate: "Measure the new model",
  promote: "Make it the live model if it passes the gate",
};

const STATUS_TONE: Record<string, string> = {
  running: "text-accent",
  pending: "text-ink-3",
  blocked: "text-block",
  succeeded: "text-pass",
  exhausted: "text-warn",
  stopped: "text-ink-3",
};

function Bar({ value, target }: { value: number | null; target: number }) {
  const pct = Math.min(100, Math.round(((value ?? 0) / Math.max(target, 1e-6)) * 100));
  return (
    <div className="w-24 h-2 bg-bg-2 rounded-sm overflow-hidden relative">
      <div className="h-full bg-accent rounded-sm" style={{ width: `${pct}%` }} />
      {/* The target as a line rather than a number: the question is "how close", not "what is it". */}
      <div className="absolute top-0 right-0 h-full w-px bg-pass" />
    </div>
  );
}

const NO_ROWS: Campaign[] = [];

export default function CampaignsPage() {
  const [open, setOpen] = useState<string | null>(null);
  const rowsQ = useQuery(["campaigns"], () => api.campaigns());
  const rows: Campaign[] = rowsQ.data?.campaigns ?? NO_ROWS;
  // keepPrevious: switching to another campaign leaves the last one's detail up until the new one
  // arrives. Closing clears it, which is why the null case is spelled out.
  const detailQ = useQuery(open ? ["campaign", open] : null, () => api.campaign(open as string),
    { keepPrevious: true });
  const detail: CampaignDetail | null = open ? detailQ.data ?? null : null;
  const [busy, setBusy] = useState(false);
  const [creating, setCreating] = useState(false);
  const confirm = useConfirm();

  const [name, setName] = useState("");
  const [className, setClassName] = useState("");
  const [target, setTarget] = useState(0.6);
  const [budget, setBudget] = useState(2000);

  // A failed read is announced, as it was; the list and the detail keep whatever they last held.
  const rowsErr = rowsQ.error;
  const detailErr = detailQ.error;
  useEffect(() => { if (rowsErr) toast(humanizeError(rowsErr), "error"); }, [rowsErr]);
  useEffect(() => { if (detailErr) toast(humanizeError(detailErr), "error"); }, [detailErr]);

  const run = (fn: () => Promise<void>) => async () => {
    setBusy(true);
    try { await fn(); } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const create = run(async () => {
    if (!name.trim() || !className.trim()) { toast("Give the campaign a name and a class", "error"); return; }
    await api.createCampaign({ name: name.trim(), class_name: className.trim(),
                               target_value: target, label_budget: budget });
    setName(""); setClassName(""); setCreating(false);
    await invalidate("campaigns");
    toast("Campaign created. It waits for your approval before each stage.", "success");
  });

  const tick = (id: string) => run(async () => {
    const out = await api.tickCampaign(id);
    toast(out.detail || `${out.action}${out.stage ? `: ${out.stage}` : ""}`,
          out.action === "failed" ? "error" : "success");
    // The list, and this campaign's detail if it is the one open.
    await invalidate("campaigns", ["campaign", id]);
  });

  const approve = (id: string, stage: string) => run(async () => {
    await api.approveCampaignStage(id, stage);
    toast(`Ran ${stage}`, "success");
    await invalidate("campaigns", ["campaign", id]);
  });

  const stop = (id: string) => run(async () => {
    if (!(await confirm({ title: "Stop this campaign?",
                          body: "It keeps its history and stops advancing.",
                          confirmLabel: "Stop" }))) return;
    await api.stopCampaign(id);
    await invalidate("campaigns", ["campaign", id]);
  });

  const createButton = (
    <button type="button" onClick={() => setCreating(true)} className="btn btn-primary">New campaign</button>
  );

  return (
    <PageShell active="CAMPAIGNS" title="Campaigns"
      subtitle="Improve one weak class at a time: find examples, label them, retrain, and check the result"
      primaryAction={
        <button type="button" onClick={() => setCreating((c) => !c)}
          className={creating ? "btn" : "btn btn-primary"}>
          {creating ? "Cancel" : "New campaign"}
        </button>
      }>
      <div className="page-wrap space-y-4">
        {creating && (
          <section className="panel">
            <div className="panel-head"><h2 className="section-title">New campaign</h2></div>
            <form className="p-3.5 space-y-3" onSubmit={(e) => { e.preventDefault(); void create(); }}>
              <div className="flex items-end gap-3 text-[12.5px] flex-wrap">
                <label className="flex flex-col gap-1">
                  <span className="text-ink-2">Name</span>
                  <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Cattle recall push"
                    className="field w-56" />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-ink-2">Class to improve</span>
                  <input value={className} onChange={(e) => setClassName(e.target.value)}
                    placeholder="e.g. cattle" className="field w-44" />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-ink-2">Target recall (0 to 1)</span>
                  <input type="number" step="0.05" min="0.05" max="1" value={target}
                    onChange={(e) => setTarget(Number(e.target.value))} className="field w-28" />
                </label>
                <label className="flex flex-col gap-1">
                  <span className="text-ink-2">Label budget</span>
                  <input type="number" step="100" min="1" value={budget}
                    onChange={(e) => setBudget(Number(e.target.value))} className="field w-28" />
                </label>
                <button type="submit" disabled={busy} className="btn btn-primary">
                  <T k="action.create">Create</T>
                </button>
              </div>
              <p className="section-desc">
                The budget counts labels people will draw, not hours. Each stage waits for your approval
                until you put the campaign on autopilot.
              </p>
            </form>
          </section>
        )}

        {rowsQ.loading && !rows.length ? <LoadState loading rows={3} /> : rows.length === 0 ? (
          !creating && (
            <div className="panel">
              <EmptyState icon="flag" title="No campaigns yet" action={createButton}>
                A campaign takes a class the model keeps getting wrong and runs the improvement loop for it:
                find more examples, have people label them, retrain, and test. It stops when the class reaches
                its target, the label budget runs out, or it stops getting better.
              </EmptyState>
            </div>
          )
        ) : (
          <section className="panel overflow-x-auto">
            <div className="panel-head">
              <h2 className="section-title">Campaigns</h2>
              <span className="text-[12px] text-ink-3">{num(rows.length)}</span>
            </div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Name</th><th>Class</th><th>Status</th><th className="num">Round</th>
                  <th>Progress to target</th><th className="num">Labels used</th><th></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((c) => (
                  <tr key={c.campaign_id} className={open === c.campaign_id ? "bg-accent/10" : ""}>
                    <td>
                      <button type="button" onClick={() => setOpen(open === c.campaign_id ? null : c.campaign_id)}
                        aria-expanded={open === c.campaign_id}
                        className="text-ink hover:text-accent-2 text-left">{c.name}</button>
                    </td>
                    <td>{humanize(c.class_name)}</td>
                    <td className={STATUS_TONE[c.status] ?? "text-ink-3"}>
                      {humanize(c.status)}
                      {c.stalled_iterations > 0 && c.status === "running" && (
                        <span className="text-warn"> · no gain for {c.stalled_iterations} {c.stalled_iterations === 1 ? "round" : "rounds"}</span>
                      )}
                    </td>
                    <td className="num">{c.iteration} of {c.max_iterations}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <Bar value={c.best_value} target={c.target_value} />
                        <span className="text-ink-3 tabular-nums">
                          {c.best_value != null ? c.best_value.toFixed(2) : "-"} / {c.target_value}
                        </span>
                      </div>
                    </td>
                    <td className="num">{num(c.labels_spent)} / {num(c.label_budget)}</td>
                    <td className="text-right whitespace-nowrap space-x-3">
                      {!["succeeded", "exhausted", "stopped"].includes(c.status) && (
                        <>
                          <button type="button" onClick={tick(c.campaign_id)} disabled={busy}
                            data-tip="Move the campaign on by one step now"
                            className="link disabled:opacity-40">Advance</button>
                          <button type="button" onClick={stop(c.campaign_id)} disabled={busy}
                            className="text-block hover:underline disabled:opacity-40"><T k="action.stop">Stop</T></button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        {detail && (
          <section className="panel">
            <div className="panel-head">
              <h2 className="section-title">{detail.name}</h2>
              <span className="text-[12px] text-ink-3">
                Next: {humanize(detail.next_stage)}
                {detail.halt_reason ? ` · ${detail.halt_reason}` : ""}
              </span>
            </div>
            <div className="p-3.5 space-y-4">
              {/* The stage rail: where this iteration has got to, and what is waiting on a person. */}
              <div className="flex gap-1.5 flex-wrap">
                {STAGES.map((stage) => {
                  const steps = detail.steps.filter(
                    (s) => s.iteration === Math.max(1, detail.iteration) && s.stage === stage);
                  const last = steps[steps.length - 1];
                  const tone = !last ? "border-line text-ink-3"
                    : last.status === "done" ? "border-pass/60 text-pass"
                    : last.status === "waiting" ? "border-warn/60 text-warn"
                    : last.status === "failed" ? "border-block/60 text-block"
                    : "border-accent/60 text-accent-2";
                  return (
                    <div key={stage} className={`rounded border px-2.5 py-1.5 ${tone}`} data-tip={STAGE_TIP[stage]}>
                      <div className="text-[12.5px]">{humanize(stage)}</div>
                      {last?.awaiting && (
                        <div className="text-[11px] text-ink-3 max-w-[14rem]">
                          {last.awaiting}
                        </div>
                      )}
                      {last?.status === "waiting" && last.awaiting?.startsWith("approval") && (
                        <button type="button" onClick={approve(detail.campaign_id, stage)} disabled={busy}
                          className="btn btn-primary mt-1.5">
                          Approve and run
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>

              <table className="data-table">
                <thead>
                  <tr>
                    <th className="num">Round</th><th>Stage</th><th>Status</th><th>Detail</th>
                  </tr>
                </thead>
                <tbody>
                  {[...detail.steps].reverse().slice(0, 20).map((s) => (
                    <tr key={s.step_id}>
                      <td className="num">{s.iteration}</td>
                      <td>{humanize(s.stage)}</td>
                      <td className={s.status === "done" ? "text-pass"
                        : s.status === "failed" ? "text-block"
                        : s.status === "waiting" ? "text-warn" : "text-ink-3"}>{humanize(s.status)}</td>
                      <td className="text-ink-3 truncate max-w-[26rem]">
                        {s.awaiting || Object.entries(s.detail)
                          .filter(([k]) => !k.endsWith("_ids"))
                          .map(([k, v]) => `${humanize(k)}: ${String(v).slice(0, 40)}`).join(" · ")}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </div>
    </PageShell>
  );
}
