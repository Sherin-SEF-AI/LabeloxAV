"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { AlItem, ErrorCandidateRow } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import ScoreBar from "@/components/shell/ScoreBar";
import { ConfBar } from "@/components/StateBadge";

// M4.0 + M4.1 unified review queue: the highest-value active-learning items to label, and the
// error candidates flagged on already-accepted data. The human governor spends touches here, on the
// hardest and most valuable cases, and confirms the errors auto-accept got wrong.

export default function ReviewQueuePage() {
  const router = useRouter();
  const [tab, setTab] = useState<"value" | "errors">("value");
  const [items, setItems] = useState<AlItem[]>([]);
  const [errs, setErrs] = useState<ErrorCandidateRow[]>([]);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [al, ec] = await Promise.all([api.alScore(undefined, 60), api.errorCandidates("pending", 80)]);
    setItems(al.items);
    setErrs(ec);
  }, []);

  useEffect(() => { load(); }, [load]);

  const confirm = async (id: string) => { await api.errorConfirm(id); setMsg("confirmed as error (fed to retrain)"); await load(); };
  const dismiss = async (id: string) => { await api.errorDismiss(id); await load(); };
  const runDetect = async () => { const r = await api.errorRun(); setMsg(`detected ${r.persisted}`); await load(); };

  return (
    <PageShell active="REVIEW" title="Review Queue"
      right={msg ? <span className="text-warn">{msg}</span> : undefined}
      primaryAction={
        <>
          <button onClick={() => setTab("value")} className={`px-2 py-1 border ${tab === "value" ? "border-accent text-accent" : "border-line text-ink-3"}`}>value queue ({items.length})</button>
          <button onClick={() => setTab("errors")} className={`px-2 py-1 border ${tab === "errors" ? "border-accent text-accent" : "border-line text-ink-3"}`}>error candidates ({errs.length})</button>
          {tab === "errors" && <button onClick={runDetect} className="border border-line px-2 py-1 hover:border-accent">re-run detection</button>}
        </>
      }>
      <div className="p-4 space-y-3 font-mono text-[11px]">

        {tab === "value" ? (
          <table className="w-full">
            <thead><tr className="text-ink-3 text-left border-b hairline"><th className="px-2 py-1">class</th><th>conf</th><th>value</th><th>uncertain</th><th>diverse</th><th>rare</th><th>err</th><th></th></tr></thead>
            <tbody>
              {items.map((it) => (
                <tr key={it.object_id} className="border-b hairline hover:bg-line">
                  <td className="px-2 py-1 text-ink-2">{it.class_name}</td>
                  <td><ConfBar conf={it.conf} /></td>
                  <td className="text-accent">{it.value.toFixed(3)}</td>
                  <td><ScoreBar value={it.scores.uncertainty} showValue={false} /></td>
                  <td><ScoreBar value={it.scores.diversity} showValue={false} /></td>
                  <td><ScoreBar value={it.scores.rarity} showValue={false} /></td>
                  <td><ScoreBar value={it.scores.error_prone} showValue={false} tone="warn" /></td>
                  <td className="text-right pr-2"><button onClick={() => router.push(`/frame/${it.frame_id}`)} className="text-info hover:text-accent">label →</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <table className="w-full">
            <thead><tr className="text-ink-3 text-left border-b hairline"><th className="px-2 py-1">kind</th><th>score</th><th>proposed fix</th><th>detail</th><th></th></tr></thead>
            <tbody>
              {errs.map((e) => (
                <tr key={e.candidate_id} className="border-b hairline">
                  <td className="px-2 py-1 text-ink-2">{e.kind}</td>
                  <td><ScoreBar value={e.score} tone="warn" /></td>
                  <td className="text-info">{e.proposed_label?.class_name || "(review)"}</td>
                  <td className="text-ink-3 truncate max-w-[280px]">{JSON.stringify(e.detail)}</td>
                  <td className="text-right pr-2 space-x-2">
                    <button onClick={() => confirm(e.candidate_id)} className="text-block hover:text-accent">confirm</button>
                    <button onClick={() => dismiss(e.candidate_id)} className="text-ink-3 hover:text-ink">dismiss</button>
                  </td>
                </tr>
              ))}
              {!errs.length && <tr><td colSpan={5} className="text-ink-3 text-center py-4">no error candidates (run detection)</td></tr>}
            </tbody>
          </table>
        )}
      </div>
    </PageShell>
  );
}
