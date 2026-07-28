"use client";

import { useCallback, useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import PageShell from "@/components/shell/PageShell";
import { useConfirm } from "@/components/ConfirmProvider";
import { toast } from "@/lib/toast";
import type { Campaign, CampaignDetail } from "@/lib/types";

// Campaigns: the improvement loop, run by the system instead of by somebody remembering to.
//
// Every stage of improving a stuck class already existed and a person stood between each pair of them:
// read the gate's deficit, mine, judge, wait for humans, retrain, promote, decide whether to go again.
// A class stalled the moment nobody was watching.
//
// The board is built around the two things an operator actually needs to see: which campaign is waiting
// on them, and which one has stopped paying. Everything else is detail behind a click.

const STAGES = ["mine", "judge", "label", "train", "evaluate", "promote"] as const;

const STATUS_TONE: Record<string, string> = {
  running: "text-accent",
  pending: "text-ink-3",
  blocked: "text-block",
  succeeded: "text-pass",
  exhausted: "text-warn",
  stopped: "text-ink-4",
};

function Bar({ value, target }: { value: number | null; target: number }) {
  const pct = Math.min(100, Math.round(((value ?? 0) / Math.max(target, 1e-6)) * 100));
  return (
    <div className="w-24 h-1.5 bg-line/50 relative">
      <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
      {/* The target as a line rather than a number: the question is "how close", not "what is it". */}
      <div className="absolute top-0 right-0 h-full w-px bg-pass" />
    </div>
  );
}

export default function CampaignsPage() {
  const [rows, setRows] = useState<Campaign[]>([]);
  const [open, setOpen] = useState<string | null>(null);
  const [detail, setDetail] = useState<CampaignDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [creating, setCreating] = useState(false);
  const confirm = useConfirm();

  const [name, setName] = useState("");
  const [className, setClassName] = useState("");
  const [target, setTarget] = useState(0.6);
  const [budget, setBudget] = useState(2000);

  const load = useCallback(async () => {
    try { setRows((await api.campaigns()).campaigns); }
    catch (e) { toast(humanizeError(e), "error"); }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (!open) { setDetail(null); return; }
    api.campaign(open).then(setDetail).catch((e) => toast(humanizeError(e), "error"));
  }, [open]);

  const run = (fn: () => Promise<void>) => async () => {
    setBusy(true);
    try { await fn(); } catch (e) { toast(humanizeError(e), "error"); } finally { setBusy(false); }
  };

  const create = run(async () => {
    if (!name.trim() || !className.trim()) { toast("name and class are required", "error"); return; }
    await api.createCampaign({ name: name.trim(), class_name: className.trim(),
                               target_value: target, label_budget: budget });
    setName(""); setClassName(""); setCreating(false);
    await load();
    toast("campaign created; it waits for approval before doing anything", "success");
  });

  const tick = (id: string) => run(async () => {
    const out = await api.tickCampaign(id);
    toast(out.detail || `${out.action}${out.stage ? `: ${out.stage}` : ""}`,
          out.action === "failed" ? "error" : "success");
    await load();
    if (open === id) setDetail(await api.campaign(id));
  });

  const approve = (id: string, stage: string) => run(async () => {
    await api.approveCampaignStage(id, stage);
    toast(`ran ${stage}`, "success");
    await load();
    setDetail(await api.campaign(id));
  });

  const stop = (id: string) => run(async () => {
    if (!(await confirm({ title: "Stop this campaign?",
                          body: "It keeps its history and stops advancing.",
                          confirmLabel: "Stop" }))) return;
    await api.stopCampaign(id);
    await load();
  });

  return (
    <PageShell active="CAMPAIGNS" title="Campaigns"
      subtitle="the improvement loop, run by the system"
      primaryAction={
        <button onClick={() => setCreating((c) => !c)}
          className="border border-accent px-2 py-1 font-mono text-[11px] text-accent hover:bg-accent/10">
          {creating ? "cancel" : "new campaign"}
        </button>
      }>
      <div className="p-4 space-y-4 max-w-6xl">
        {creating && (
          <section className="panel p-3 space-y-2">
            <div className="flex items-center gap-2 font-mono text-[11px] flex-wrap">
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="campaign name"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-44" />
              <input value={className} onChange={(e) => setClassName(e.target.value)}
                placeholder="class (e.g. cattle)"
                className="bg-bg border border-line px-1.5 py-0.5 text-ink w-40" />
              <label className="flex items-center gap-1 text-ink-3">
                target recall
                <input type="number" step="0.05" min="0.05" max="1" value={target}
                  onChange={(e) => setTarget(Number(e.target.value))}
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink w-20" />
              </label>
              <label className="flex items-center gap-1 text-ink-3">
                label budget
                <input type="number" step="100" min="1" value={budget}
                  onChange={(e) => setBudget(Number(e.target.value))}
                  className="bg-bg border border-line px-1.5 py-0.5 text-ink w-24" />
              </label>
              <button onClick={create} disabled={busy}
                className="border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
                create
              </button>
            </div>
            <div className="font-mono text-[10px] text-ink-3">
              The budget is in labels, not hours: the batches a campaign builds are human time, and no
              wall-clock limit constrains that. Every stage waits for approval until you put it on autopilot.
            </div>
          </section>
        )}

        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            campaigns ({rows.length})
          </div>
          {rows.length === 0 ? (
            <div className="p-4 font-mono text-[11px] text-ink-3">
              No campaigns. One picks a class the gate keeps blocking on, then chains mining, judging,
              review, retraining and a promotion attempt until it hits the target, spends its budget, or
              stops improving.
            </div>
          ) : (
            <table className="w-full font-mono text-[11px]">
              <thead>
                <tr className="text-ink-3 text-left border-b hairline">
                  <th className="py-1">name</th><th>class</th><th>status</th><th>iter</th>
                  <th>progress</th><th>budget</th><th></th>
                </tr>
              </thead>
              <tbody>
                {rows.map((c) => (
                  <tr key={c.campaign_id} className="border-b hairline">
                    <td className="py-1">
                      <button onClick={() => setOpen(open === c.campaign_id ? null : c.campaign_id)}
                        className="text-ink hover:text-accent">{c.name}</button>
                    </td>
                    <td className="text-ink-2">{c.class_name}</td>
                    <td className={STATUS_TONE[c.status] ?? "text-ink-3"}>
                      {c.status}
                      {c.stalled_iterations > 0 && c.status === "running" && (
                        <span className="text-warn"> · {c.stalled_iterations} flat</span>
                      )}
                    </td>
                    <td className="text-ink-3">{c.iteration}/{c.max_iterations}</td>
                    <td>
                      <div className="flex items-center gap-2">
                        <Bar value={c.best_value} target={c.target_value} />
                        <span className="text-ink-3 tabular-nums">
                          {c.best_value != null ? c.best_value.toFixed(3) : "-"} / {c.target_value}
                        </span>
                      </div>
                    </td>
                    <td className="text-ink-3 tabular-nums">{c.labels_spent}/{c.label_budget}</td>
                    <td className="text-right space-x-2">
                      {!["succeeded", "exhausted", "stopped"].includes(c.status) && (
                        <>
                          <button onClick={tick(c.campaign_id)} disabled={busy}
                            className="text-ink-3 hover:text-accent disabled:opacity-40">tick</button>
                          <button onClick={stop(c.campaign_id)} disabled={busy}
                            className="text-ink-3 hover:text-block disabled:opacity-40">stop</button>
                        </>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>

        {detail && (
          <section className="panel">
            <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3
                            border-b hairline px-3 py-2">
              <span>{detail.name}</span>
              <span className="normal-case text-ink-4">
                next: {detail.next_stage}
                {detail.halt_reason ? ` · ${detail.halt_reason}` : ""}
              </span>
            </div>
            <div className="p-3 space-y-3">
              {/* The stage rail: where this iteration has got to, and what is waiting on a person. */}
              <div className="flex gap-1 flex-wrap">
                {STAGES.map((stage) => {
                  const steps = detail.steps.filter(
                    (s) => s.iteration === Math.max(1, detail.iteration) && s.stage === stage);
                  const last = steps[steps.length - 1];
                  const tone = !last ? "border-line text-ink-4"
                    : last.status === "done" ? "border-pass text-pass"
                    : last.status === "waiting" ? "border-warn text-warn"
                    : last.status === "failed" ? "border-block text-block"
                    : "border-accent text-accent";
                  return (
                    <div key={stage} className={`border px-2 py-1 ${tone}`}>
                      <div className="font-mono text-[11px]">{stage}</div>
                      {last?.awaiting && (
                        <div className="font-mono text-[9.5px] text-ink-3 max-w-[14rem]">
                          {last.awaiting}
                        </div>
                      )}
                      {last?.status === "waiting" && last.awaiting?.startsWith("approval") && (
                        <button onClick={approve(detail.campaign_id, stage)} disabled={busy}
                          className="mt-1 font-mono text-[10px] text-accent hover:underline">
                          approve and run
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>

              <table className="w-full font-mono text-[11px]">
                <thead>
                  <tr className="text-ink-3 text-left border-b hairline">
                    <th className="py-1">iter</th><th>stage</th><th>status</th><th>detail</th>
                  </tr>
                </thead>
                <tbody>
                  {[...detail.steps].reverse().slice(0, 20).map((s) => (
                    <tr key={s.step_id} className="border-b hairline">
                      <td className="py-1 text-ink-3">{s.iteration}</td>
                      <td className="text-ink-2">{s.stage}</td>
                      <td className={s.status === "done" ? "text-pass"
                        : s.status === "failed" ? "text-block"
                        : s.status === "waiting" ? "text-warn" : "text-ink-3"}>{s.status}</td>
                      <td className="text-ink-3 truncate max-w-[26rem]">
                        {s.awaiting || Object.entries(s.detail)
                          .filter(([k]) => !k.endsWith("_ids"))
                          .map(([k, v]) => `${k}=${String(v).slice(0, 40)}`).join(" ")}
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
