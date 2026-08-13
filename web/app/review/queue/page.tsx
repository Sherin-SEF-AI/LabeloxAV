"use client";

import { Fragment, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { AlItem, ErrorCandidateRow } from "@/lib/types";
import dynamic from "next/dynamic";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import ScoreBar from "@/components/shell/ScoreBar";
import { ConfBar } from "@/components/StateBadge";

const ExplainPanel = dynamic(() => import("@/components/ExplainPanel"), { ssr: false });

// M4.0 + M4.1 unified review queue: the highest-value active-learning items to label, and the
// error candidates flagged on already-accepted data. The human governor spends touches here, on the
// hardest and most valuable cases, and confirms the errors auto-accept got wrong.

export default function ReviewQueuePage() {
  const router = useRouter();
  const [tab, setTab] = useState<"value" | "errors" | "recall" | "control">("value");
  const [items, setItems] = useState<AlItem[]>([]);
  const [errs, setErrs] = useState<ErrorCandidateRow[]>([]);
  // Objects the detector missed entirely, found by a track gap, an open-vocabulary hit or a region
  // proposal. These have always been mined and never been rulable, which is why the per-channel reliability
  // fit read an empty set and the priors stayed guesses.
  const [recall, setRecall] = useState<Awaited<ReturnType<typeof api.recallCandidates>>["candidates"]>([]);
  const [recallProg, setRecallProg] =
    useState<Awaited<ReturnType<typeof api.recallProgress>> | null>(null);
  // The gate's own auto-accepts, mirrored into an always-reviewed stream. The fraction judged incorrect is
  // the measured precision the drift detector watches; 601 were seeded and none had ever been judged,
  // because nothing listed them.
  const [control, setControl] = useState<Awaited<ReturnType<typeof api.controlPending>>["samples"]>([]);
  const [precision, setPrecision] =
    useState<Awaited<ReturnType<typeof api.governPrecision>> | null>(null);
  const [msg, setMsg] = useState<string | null>(null);
  const [whyOpen, setWhyOpen] = useState<string | null>(null);  // M-F.0 expanded rationale row
  const [err, setErr] = useState<unknown>(null);
  const [sortQ, setSortQ] = useState(false);  // M-F.1 sort the value queue by lowest label quality first
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const [al, ec, rc, rp, cs, pr] = await Promise.all([
        api.alScore(undefined, 60), api.errorCandidates("pending", 80),
        api.recallCandidates("pending", 100), api.recallProgress(),
        api.controlPending(100), api.governPrecision(),
      ]);
      setItems(al.items);
      setErrs(ec);
      setRecall(rc.candidates);
      setRecallProg(rp);
      setControl(cs.samples);
      setPrecision(pr);
    } catch (e) {
      // There was a finally and no catch, so a failed fetch left the table empty and silent. An empty
      // review queue is a meaningful state here (the shift is done) and must not be what a dropped request
      // looks like.
      setErr(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const confirm = async (id: string) => { await api.errorConfirm(id); setMsg("confirmed as error (fed to retrain)"); await load(); };
  const dismiss = async (id: string) => { await api.errorDismiss(id); await load(); };
  const runDetect = async () => { const r = await api.errorRun(); setMsg(`detected ${r.persisted}`); await load(); };
  const ruleControl = async (id: string, verdict: "correct" | "incorrect") => {
    await api.controlVerdict(id, verdict);
    setMsg(verdict === "correct" ? "the gate was right here" : "recorded as a gate error");
    await load();
  };
  const ruleRecall = async (id: string, verdict: "confirmed" | "rejected") => {
    const r = await api.recallRule(id, verdict);
    setMsg(verdict === "confirmed"
      ? (r.object_routed_to_review ? "confirmed; the object is in the review queue" : "confirmed")
      : "rejected; the object was left alone");
    await load();
  };

  return (
    <PageShell active="REVIEW" title="Review Queue"
      right={msg ? <span className="text-warn">{msg}</span> : undefined}
      primaryAction={
        <>
          <button onClick={() => setTab("value")} className={`px-2 py-1 border ${tab === "value" ? "border-accent text-accent" : "border-line text-ink-3"}`}>value queue ({items.length})</button>
          <button onClick={() => setTab("errors")} className={`px-2 py-1 border ${tab === "errors" ? "border-accent text-accent" : "border-line text-ink-3"}`}>error candidates ({errs.length})</button>
          <button onClick={() => setTab("recall")} className={`px-2 py-1 border ${tab === "recall" ? "border-accent text-accent" : "border-line text-ink-3"}`}>missed objects ({recall.length})</button>
          <button onClick={() => setTab("control")} className={`px-2 py-1 border ${tab === "control" ? "border-accent text-accent" : "border-line text-ink-3"}`}>gate audit ({control.length})</button>
          {tab === "errors" && <button onClick={runDetect} className="border border-line px-2 py-1 hover:border-accent">re-run detection</button>}
        </>
      }>
      <div className="p-4 space-y-3 font-mono text-[11px]">
        {/* A failure says so and offers a retry, instead of an empty table that reads as a finished shift. */}
        {err != null && <LoadState error={err} onRetry={() => void load()} />}

        {tab === "value" ? (
          <table className="w-full">
            <thead><tr className="text-ink-3 text-left border-b hairline"><th className="px-2 py-1 w-14"></th><th>class</th><th>conf</th>
              <th><button onClick={() => setSortQ((s) => !s)} title="sort by lowest label quality first"
                className={sortQ ? "text-accent" : "hover:text-ink"}>quality{sortQ ? " ↓" : ""}</button></th>
              <th>value</th><th>uncertain</th><th>diverse</th><th>rare</th><th>err</th><th></th></tr></thead>
            <tbody>
              {loading && items.length === 0 && Array.from({ length: 10 }).map((_, r) => (
                <tr key={`sk${r}`} className="border-b hairline">
                  {Array.from({ length: 9 }).map((__, c) => (
                    <td key={c} className="px-2 py-1.5"><span className="block h-3 skeleton" style={{ width: c === 0 ? "80%" : "60%", animationDelay: `${(r * 8 + c) * 40}ms` }} /></td>
                  ))}
                </tr>
              ))}
              {(sortQ ? [...items].sort((a, b) => (a.quality_score ?? 1) - (b.quality_score ?? 1)) : items).map((it) => (
                <Fragment key={it.object_id}>
                  <tr className="border-b hairline hover:bg-line">
                    {/* The crop, because this is visual triage. Without it the queue asks a reviewer to rank
                        boxes they cannot see, and every judgement costs a round trip into the editor. The
                        endpoint has existed all along; only the img tag was missing. Lazy so a 200-row queue
                        does not fire 200 requests before the first screenful is readable. */}
                    <td className="px-2 py-1">
                      <img src={`/api/objects/${it.object_id}/crop`} alt="" loading="lazy"
                        className="w-12 h-12 object-cover rounded bg-bg-2 border border-line"
                        onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = "hidden"; }} />
                    </td>
                    <td className="px-2 py-1 text-ink-2">{it.class_name}</td>
                    <td><ConfBar conf={it.conf} /></td>
                    <td>{it.quality_score != null
                      ? <span className={it.quality_score >= 0.4 ? "text-pass" : it.quality_score >= 0.25 ? "text-warn" : "text-block"}>{it.quality_score.toFixed(2)}</span>
                      : <span className="text-ink-3">-</span>}</td>
                    <td className="text-accent">{it.value.toFixed(3)}</td>
                    <td><ScoreBar value={it.scores.uncertainty} /></td>
                    <td><ScoreBar value={it.scores.diversity} /></td>
                    <td><ScoreBar value={it.scores.rarity} /></td>
                    <td><ScoreBar value={it.scores.error_prone} tone="warn" /></td>
                    <td className="text-right pr-2 space-x-2 whitespace-nowrap">
                      <button onClick={() => setWhyOpen((w) => (w === it.object_id ? null : it.object_id))}
                        className={whyOpen === it.object_id ? "text-accent" : "text-ink-3 hover:text-ink"}>why</button>
                      <button onClick={() => router.push(`/frame/${it.frame_id}`)} className="text-info hover:text-accent">label →</button>
                    </td>
                  </tr>
                  {whyOpen === it.object_id && (
                    <tr className="border-b hairline"><td colSpan={10} className="px-3 py-2 bg-bg-2">
                      <ExplainPanel objectId={it.object_id} />
                    </td></tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        ) : tab === "errors" ? (
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
        ) : tab === "recall" ? (
          <>
            {/* Per channel, not only a total: the reliability fit needs a minimum number of verdicts per
                channel before it will apply a measurement, so a healthy-looking total can hide a channel
                nobody has ruled on and the priors for it stay guesses. */}
            {recallProg && (
              <div className="text-ink-3">
                {recallProg.judged} of {recallProg.total} judged
                {Object.entries(recallProg.per_channel).map(([ch, t]) => (
                  <span key={ch} className="ml-3">
                    {ch}: <span className="text-ink-2">{t.confirmed + t.rejected}</span>/{t.confirmed + t.rejected + t.pending}
                  </span>
                ))}
              </div>
            )}
            <table className="w-full">
              <thead><tr className="text-ink-3 text-left border-b hairline">
                <th className="px-2 py-1">found by</th><th>value</th><th>frame</th><th></th>
              </tr></thead>
              <tbody>
                {recall.map((c) => (
                  <tr key={c.candidate_id} className="border-b hairline">
                    <td className="px-2 py-1 text-ink-2">{c.channels.join(", ")}</td>
                    <td><ScoreBar value={c.fn_value} tone="warn" /></td>
                    <td>
                      <button onClick={() => router.push(`/frame/${c.frame_id}?focus=${c.object_id}`)}
                        className="text-info hover:text-accent">{c.frame_id.slice(0, 8)}</button>
                    </td>
                    <td className="text-right pr-2 space-x-2">
                      <button onClick={() => ruleRecall(c.candidate_id, "confirmed")}
                        title="a real object the detector missed; route it to the review queue"
                        className="text-pass hover:text-accent">confirm</button>
                      <button onClick={() => ruleRecall(c.candidate_id, "rejected")}
                        title="a bad suggestion; the object is left alone"
                        className="text-ink-3 hover:text-ink">reject</button>
                    </td>
                  </tr>
                ))}
                {!recall.length && (
                  <tr><td colSpan={4} className="text-ink-3 text-center py-4">
                    nothing mined yet. Recall mining runs per session and finds objects the detector missed.
                  </td></tr>
                )}
              </tbody>
            </table>
          </>
        ) : (
          <>
            {/* The number this queue exists to produce. Null is not zero: it means nobody has judged a
                single sample, which is what the drift detector has been watching all along. */}
            <div className="text-ink-3">
              measured gate precision:{" "}
              {precision?.precision == null
                ? <span className="text-warn">not measured yet ({precision?.pending ?? 0} awaiting a verdict)</span>
                : <span className="text-ink-2">{(precision.precision * 100).toFixed(1)}% over {precision.reviewed} judged</span>}
            </div>
            <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
              {control.map((c) => (
                <div key={c.sample_id} className="border border-line">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={c.crop_url} alt="" className="w-full h-20 object-cover bg-bg-2" />
                  <div className="px-1 py-0.5 text-[10px] text-ink-2 truncate" title={c.class_name}>
                    {c.class_name} {c.conf != null ? c.conf.toFixed(2) : ""}
                  </div>
                  <div className="flex border-t hairline">
                    <button onClick={() => ruleControl(c.sample_id, "correct")}
                      title="the gate labelled this correctly"
                      className="flex-1 text-pass text-[10px] py-0.5 hover:bg-pass/10">right</button>
                    <button onClick={() => ruleControl(c.sample_id, "incorrect")}
                      title="the gate got this wrong; it counts against measured precision"
                      className="flex-1 text-block text-[10px] py-0.5 hover:bg-block/10 border-l hairline">wrong</button>
                  </div>
                </div>
              ))}
            </div>
            {!control.length && (
              <div className="text-ink-3 text-center py-4">
                nothing awaiting a verdict. The gate mirrors a small random share of its auto-accepts here.
              </div>
            )}
          </>
        )}
      </div>
    </PageShell>
  );
}
