"use client";

import { Fragment, useCallback, useState } from "react";
import { T } from "@/lib/i18n";
import { useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import { toastError } from "@/lib/toast";
import type { AlItem, ErrorCandidateRow } from "@/lib/types";
import dynamic from "next/dynamic";
import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import ScoreBar from "@/components/shell/ScoreBar";
import { ConfBar } from "@/components/StateBadge";
import { invalidate, useQuery } from "@/lib/useQuery";
import { REVIEW_QUEUE_KEY } from "@/lib/queryKeys";
import EmptyState from "@/components/ui/EmptyState";
import { humanize, shortId } from "@/lib/format";

const ExplainPanel = dynamic(() => import("@/components/ExplainPanel"), { ssr: false });

// M4.0 + M4.1 unified review queue: the highest-value active-learning items to label, and the
// error candidates flagged on already-accepted data. The human governor spends touches here, on the
// hardest and most valuable cases, and confirms the errors auto-accept got wrong.

export default function ReviewQueuePage() {
  const router = useRouter();
  const [tab, setTab] = useState<"value" | "errors" | "recall" | "control">("value");
  // The six reads behind the four tabs are fetched and replaced together, all or nothing: the tab counts
  // are read against each other, and a half-refreshed set would have one tab describing a newer queue
  // than the next. Cached, so returning here from a frame shows the queue while it refreshes.
  const q = useQuery([REVIEW_QUEUE_KEY],
    () => Promise.all([
      api.alScore(undefined, 60), api.errorCandidates("pending", 80),
      api.recallCandidates("pending", 100), api.recallProgress(),
      api.controlPending(100), api.governPrecision(),
    ]));
  const [al, ec, rc, rp, cs, pr] = q.data ?? [];
  const items: AlItem[] = al?.items ?? [];
  const errs: ErrorCandidateRow[] = ec ?? [];
  // Objects the detector missed entirely, found by a track gap, an open-vocabulary hit or a region
  // proposal. These have always been mined and never been rulable, which is why the per-channel reliability
  // fit read an empty set and the priors stayed guesses.
  const recall = rc?.candidates ?? [];
  const recallProg = rp ?? null;
  // The gate's own auto-accepts, mirrored into an always-reviewed stream. The fraction judged incorrect is
  // the measured precision the drift detector watches; 601 were seeded and none had ever been judged,
  // because nothing listed them.
  const control = cs?.samples ?? [];
  const precision = pr ?? null;
  const [msg, setMsg] = useState<string | null>(null);
  const [whyOpen, setWhyOpen] = useState<string | null>(null);  // M-F.0 expanded rationale row
  const [sortQ, setSortQ] = useState(false);  // M-F.1 sort the value queue by lowest label quality first
  const loading = q.validating;
  // There was a finally and no catch, so a failed fetch left the table empty and silent. An empty
  // review queue is a meaningful state here (the shift is done) and must not be what a dropped request
  // looks like. A request in flight supersedes the last failure.
  const err = q.validating ? null : q.error ?? null;
  const load = q.reload;

  // Every verdict below goes through this. The read path already surfaces its failures; these five did
  // not, so a rejected verdict was an unhandled promise: the row stayed exactly where it was, nothing
  // said why, and the reviewer clicked it again. A verdict that did not take has to say so.
  const act = useCallback(async (what: string, run: () => Promise<string | null>) => {
    try {
      const note = await run();
      if (note) setMsg(note);
    } catch (e) {
      setMsg(null);
      toastError(`${humanize(what)} failed: ${humanizeError(e)}`);
      return;
    }
    await load();
    // A verdict here moves rows in the error-candidate grid, the review queues and home's waiting count.
    void invalidate("errorPrecision", "triage", "/api/home");
  }, [load]);

  const confirm = (id: string) =>
    act("confirm", async () => { await api.errorConfirm(id); return "Confirmed as an error; it goes into the next training run"; });
  const dismiss = (id: string) =>
    act("dismiss", async () => { await api.errorDismiss(id); return "Dismissed"; });
  const runDetect = () =>
    act("detection run", async () => { const r = await api.errorRun(); return `Found ${r.persisted} possible errors`; });
  const ruleControl = (id: string, verdict: "correct" | "incorrect") =>
    act("verdict", async () => {
      await api.controlVerdict(id, verdict);
      return verdict === "correct" ? "Recorded: the label was right" : "Recorded: the label was wrong";
    });
  const ruleRecall = (id: string, verdict: "confirmed" | "rejected") =>
    act("verdict", async () => {
      const r = await api.recallRule(id, verdict);
      return verdict === "confirmed"
        ? (r.object_routed_to_review ? "Confirmed; the object is now in the review queue" : "Confirmed")
        : "Rejected; nothing was changed";
    });

  const TABS: { key: typeof tab; label: string; count: number; tip: string; warn?: boolean }[] = [
    { key: "value", label: "Most useful to label", count: items.length,
      tip: "Objects where a decision teaches the model the most" },
    { key: "errors", label: "Possible errors", count: errs.length,
      tip: "Accepted labels that checks think are wrong" },
    { key: "recall", label: "Missed objects", count: recall.length,
      tip: "Objects the detector may have missed entirely" },
    // The whole backlog, not the page of it that was fetched. This read `control.length`, which is capped at
    // 100 by the request, so a queue of 601 unjudged samples announced itself as 100: the one number that says
    // how far the gate is from being measured, understated six-fold on the tab somebody decides whether to open.
    { key: "control", label: "Spot-check auto-accepts", count: precision?.pending ?? control.length,
      tip: "A random sample of labels accepted without a person", warn: Boolean(precision?.pending) },
  ];
  const hasRows = tab === "value" ? items.length > 0 : tab === "errors" ? errs.length > 0
    : tab === "recall" ? recall.length > 0 : control.length > 0;

  return (
    <PageShell active="REVIEW" title="Review Queue"
      subtitle="Decide on the objects where a person's call matters most"
      right={msg ? <span role="status" className="text-[12px] text-ink-2">{msg}</span> : undefined}
      filters={
        <>
          <div role="tablist" aria-label="Review queues" className="flex items-center gap-1.5">
            {TABS.map((tb) => (
              <button key={tb.key} role="tab" aria-selected={tab === tb.key} onClick={() => setTab(tb.key)} data-tip={tb.tip}
                className={`chip ${tab === tb.key ? "chip-on" : tb.warn ? "text-warn border-warn/45" : ""}`}>
                {tb.label}<span className="tabular-nums opacity-80">{tb.count.toLocaleString()}</span>
              </button>
            ))}
          </div>
          {tab === "errors" && <button onClick={runDetect} className="btn ml-auto">Check for errors again</button>}
        </>
      }>
      <div className="page-wrap space-y-3 text-[12.5px]">
        {/* A failure says so and offers a retry, instead of an empty table that reads as a finished shift. */}
        {err != null && <LoadState error={err} onRetry={() => void load()} />}

        {!err && !loading && !hasRows && (
          <div className="panel">
            {tab === "value" ? (
              <EmptyState icon="check" title="Nothing waiting to be labelled">
                Objects appear here once auto-label has run on a session and left some for a person.
              </EmptyState>
            ) : tab === "errors" ? (
              <EmptyState icon="flag" title="No possible errors found"
                action={<button onClick={runDetect} className="btn">Check for errors</button>}>
                The checks look for accepted labels that disagree with the model or with their own track.
              </EmptyState>
            ) : tab === "recall" ? (
              <EmptyState icon="search" title="No missed objects to check">
                Missed objects are found per session, from gaps in tracks and from a second detector.
              </EmptyState>
            ) : (
              <EmptyState icon="check" title="No auto-accepts waiting for a check">
                A small random share of what is accepted automatically is set aside here for a person to check.
              </EmptyState>
            )}
          </div>
        )}

        {tab === "value" ? (
          (loading || items.length > 0) && (
          <div className="panel overflow-x-auto">
          <table className="data-table">
            <thead><tr>
              <th className="w-16"><span className="sr-only">Crop</span></th>
              <th>Class</th><th>Confidence</th>
              <th><button onClick={() => setSortQ((s) => !s)} title="Sort by lowest label quality first" aria-pressed={sortQ}
                className={`uppercase ${sortQ ? "text-accent-2" : "hover:text-ink"}`}>Label quality{sortQ ? " ↓" : ""}</button></th>
              <th data-tip="How much a decision here helps the model, all signals combined">Value</th>
              <th>Uncertain</th><th>Unusual</th><th>Rare class</th><th>Error-prone</th><th></th></tr></thead>
            <tbody>
              {loading && items.length === 0 && Array.from({ length: 10 }).map((_, r) => (
                <tr key={`sk${r}`}>
                  {Array.from({ length: 9 }).map((__, c) => (
                    <td key={c}><span className="block h-3 skeleton" style={{ width: c === 0 ? "80%" : "60%", animationDelay: `${(r * 8 + c) * 40}ms` }} /></td>
                  ))}
                </tr>
              ))}
              {(sortQ ? [...items].sort((a, b) => (a.quality_score ?? 1) - (b.quality_score ?? 1)) : items).map((it) => (
                <Fragment key={it.object_id}>
                  <tr>
                    {/* The crop, because this is visual triage. Without it the queue asks a reviewer to rank
                        boxes they cannot see, and every judgement costs a round trip into the editor. The
                        endpoint has existed all along; only the img tag was missing. Lazy so a 200-row queue
                        does not fire 200 requests before the first screenful is readable. */}
                    <td className="!py-1.5">
                      <img src={`/api/objects/${it.object_id}/crop`} alt="" loading="lazy"
                        className="w-12 h-12 object-cover rounded border border-line skeleton"
                        onLoad={(e) => { e.currentTarget.classList.remove("skeleton"); }}
                        onError={(e) => { (e.currentTarget as HTMLImageElement).style.visibility = "hidden"; }} />
                    </td>
                    <td title={it.class_name}><span className="text-ink">{humanize(it.class_name)}</span></td>
                    <td><ConfBar conf={it.conf} /></td>
                    <td className="tabular-nums">{it.quality_score != null
                      ? <span className={it.quality_score >= 0.4 ? "text-pass" : it.quality_score >= 0.25 ? "text-warn" : "text-block"}>{it.quality_score.toFixed(2)}</span>
                      : <span className="text-ink-3">-</span>}</td>
                    <td className="tabular-nums"><span className="text-accent-2">{it.value.toFixed(2)}</span></td>
                    <td><ScoreBar value={it.scores.uncertainty} /></td>
                    <td><ScoreBar value={it.scores.diversity} /></td>
                    <td><ScoreBar value={it.scores.rarity} /></td>
                    <td><ScoreBar value={it.scores.error_prone} tone="warn" /></td>
                    <td className="text-right whitespace-nowrap">
                      <button onClick={() => setWhyOpen((w) => (w === it.object_id ? null : it.object_id))} aria-expanded={whyOpen === it.object_id}
                        className={`mr-3 ${whyOpen === it.object_id ? "text-accent-2" : "text-ink-3 hover:text-ink"}`}>Why</button>
                      <button onClick={() => router.push(`/frame/${it.frame_id}`)} className="btn">Label</button>
                    </td>
                  </tr>
                  {whyOpen === it.object_id && (
                    <tr><td colSpan={10} className="bg-bg-2">
                      <ExplainPanel objectId={it.object_id} />
                    </td></tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
          </div>
          )
        ) : tab === "errors" ? (
          errs.length > 0 && (
          <div className="panel overflow-x-auto">
          <table className="data-table">
            <thead><tr><th>Check</th><th>Score</th><th>Suggested class</th><th>Detail</th><th></th></tr></thead>
            <tbody>
              {errs.map((e) => (
                <tr key={e.candidate_id}>
                  <td title={e.kind}><span className="text-ink">{humanize(e.kind)}</span></td>
                  <td><ScoreBar value={e.score} tone="warn" /></td>
                  <td className="text-ink-2">{e.proposed_label?.class_name ? humanize(e.proposed_label.class_name) : <span className="text-ink-3">Needs a look</span>}</td>
                  <td className="text-ink-3 truncate max-w-[320px]" title={JSON.stringify(e.detail)}>{detailText(e.detail)}</td>
                  <td className="text-right whitespace-nowrap space-x-2">
                    <button onClick={() => confirm(e.candidate_id)} className="btn btn-danger"
                      title="Yes, this label is wrong; it is sent back for a fix"><T k="action.confirm_short">Confirm</T></button>
                    <button onClick={() => dismiss(e.candidate_id)} className="btn"><T k="action.dismiss">Dismiss</T></button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
          )
        ) : tab === "recall" ? (
          <>
            {/* Per channel, not only a total: the reliability fit needs a minimum number of verdicts per
                channel before it will apply a measurement, so a healthy-looking total can hide a channel
                nobody has ruled on and the priors for it stay guesses. */}
            {recallProg && recallProg.total > 0 && (
              <div className="text-ink-3 flex flex-wrap gap-x-4 gap-y-1">
                <span className="text-ink-2">{recallProg.judged.toLocaleString()} of {recallProg.total.toLocaleString()} checked</span>
                {Object.entries(recallProg.per_channel).map(([ch, t]) => (
                  <span key={ch}>
                    {humanize(ch)}: <span className="text-ink-2 tabular-nums">{t.confirmed + t.rejected}</span> of {t.confirmed + t.rejected + t.pending}
                  </span>
                ))}
              </div>
            )}
            {recall.length > 0 && (
            <div className="panel overflow-x-auto">
            <table className="data-table">
              <thead><tr><th>Found by</th><th>Value</th><th>Frame</th><th></th></tr></thead>
              <tbody>
                {recall.map((c) => (
                  <tr key={c.candidate_id}>
                    <td className="text-ink-2">{c.channels.map(humanize).join(", ")}</td>
                    <td><ScoreBar value={c.fn_value} tone="warn" /></td>
                    <td>
                      <button onClick={() => router.push(`/frame/${c.frame_id}?focus=${c.object_id}`)} className="link">
                        Open frame <span className="font-mono text-[11px] text-ink-3">{shortId(c.frame_id)}</span>
                      </button>
                    </td>
                    <td className="text-right whitespace-nowrap space-x-2">
                      <button onClick={() => ruleRecall(c.candidate_id, "confirmed")}
                        title="A real object the detector missed; it goes to the review queue"
                        className="btn text-pass border-pass/45"><T k="action.confirm_short">Confirm</T></button>
                      <button onClick={() => ruleRecall(c.candidate_id, "rejected")}
                        title="A bad suggestion; nothing is changed"
                        className="btn"><T k="action.reject">Reject</T></button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
            )}
          </>
        ) : (
          <>
            {/* The number this queue exists to produce. Null is not zero: it means nobody has judged a
                single sample, which is what the drift detector has been watching all along. */}
            <div className="text-ink-3">
              How often auto-accept is right:{" "}
              {precision?.precision == null
                ? <span className="text-warn">not measured yet ({(precision?.pending ?? 0).toLocaleString()} waiting for a check)</span>
                : <span className="text-ink-2">{(precision.precision * 100).toFixed(1)}% of {precision.reviewed.toLocaleString()} checked</span>}
            </div>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-2.5">
              {control.map((c) => (
                <div key={c.sample_id} className="panel overflow-hidden">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={c.crop_url} alt={humanize(c.class_name)} className="w-full h-24 object-cover skeleton"
                    onLoad={(e) => { e.currentTarget.classList.remove("skeleton"); }} />
                  <div className="px-2 py-1 text-[12px] text-ink-2 truncate flex items-center gap-1" title={c.class_name}>
                    <span className="truncate">{humanize(c.class_name)}</span>
                    {c.conf != null && <span className="ml-auto text-[11px] text-ink-3 tabular-nums">{Math.round(c.conf * 100)}%</span>}
                  </div>
                  <div className="flex border-t hairline">
                    <button onClick={() => ruleControl(c.sample_id, "correct")}
                      title="The label is right"
                      className="flex-1 text-pass text-[12px] py-1 hover:bg-pass/10">Right</button>
                    <button onClick={() => ruleControl(c.sample_id, "incorrect")}
                      title="The label is wrong; it counts against how often auto-accept is right"
                      className="flex-1 text-block text-[12px] py-1 hover:bg-block/10 border-l hairline">Wrong</button>
                  </div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </PageShell>
  );
}

// An error candidate's evidence, as "key: value" pairs rather than a JSON blob.
function detailText(detail: Record<string, unknown> | null | undefined): string {
  if (!detail) return "";
  return Object.entries(detail)
    .filter(([, v]) => v !== null && v !== undefined && typeof v !== "object")
    .map(([k, v]) => `${humanize(k)}: ${typeof v === "number" && !Number.isInteger(v) ? v.toFixed(2) : String(v)}`)
    .join(" · ");
}
