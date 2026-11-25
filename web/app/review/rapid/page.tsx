"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useRouter } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import { api, humanizeError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { toast } from "@/lib/toast";
import { undoPayload, verdictPayload } from "@/lib/reviewVerdict";
import { describeScope, triageQuery } from "@/lib/triageScope";
import { acceptState, useCurrentUser } from "@/lib/user";
import type { TriageRow } from "@/lib/types";
import { invalidate, useQuery } from "@/lib/useQuery";
import EmptyState from "@/components/ui/EmptyState";
import { humanize } from "@/lib/format";

// Rapid review: one crop, one keystroke.
//
// The review queue ranks objects well and then makes each verdict cost a page. Open the frame, find the
// object among forty, judge it, click, wait, go back. Reviewing is the loop's throughput bottleneck, and
// almost all of that cost is navigation rather than judgement.
//
// This is the same queue with the navigation removed. The crop fills the screen, A accepts, R rejects, C
// reclassifies, and the next one is already decoded because it was prefetched while you were looking at
// this one.
//
// Two decisions that make it usable rather than merely fast:
//
// - **Optimistic, with a real undo.** The verdict is applied to the local queue immediately and sent in the
//   background, because waiting on a round trip per object puts the network latency back into the loop this
//   page exists to remove. U reverses the last one, which is what makes acting fast safe.
// - **A verdict is never inferred from a keypress you did not mean.** Held keys do not repeat into
//   multiple verdicts, and the panel ignores input while the reclassify box is open.

type Verdict = "accept" | "reject" | "reclass" | "skip";
type Decided = { row: TriageRow; verdict: Verdict; className?: string };

const PREFETCH = 4;

// How far a crop may be enlarged. Filling the pane turned a 30-pixel rider into a wall of blocks that no
// longer looked like anything; past about 5x an upscale adds size and no detail. The surrounding view
// (the same object with more of the frame around it) is what answers "what is this" for small objects.
const MAX_UPSCALE = 5;
const CONTEXT_PAD = 2;
const cropUrl = (id: string) => `/api/objects/${id}/crop`;
const contextUrl = (id: string) => `/api/objects/${id}/crop?pad=${CONTEXT_PAD}`;

function RapidBody() {
  // Bound to the locale this render used: the server and hydration see English, then the stored language.
  const t = useT();
  const params = useSearchParams();
  const router = useRouter();
  const me = useCurrentUser();

  const [queue, setQueue] = useState<TriageRow[]>([]);
  const [index, setIndex] = useState(0);
  const [done, setDone] = useState<Decided[]>([]);
  const [loading, setLoading] = useState(true);
  const [reclassOpen, setReclassOpen] = useState(false);
  const [classQuery, setClassQuery] = useState("");
  // The crop's own pixel size, read when it decodes, so it can be drawn at a capped multiple of it.
  const [natural, setNatural] = useState<{ id: string; w: number; h: number } | null>(null);
  // The class list for the reclassify box. Shared with every other page that reads the ontology; without
  // it the box simply has no suggestions, which is what a failure has always looked like here.
  const ontologyQ = useQuery(["ontology"], () => api.ontology());
  const classes: string[] = useMemo(() => (ontologyQ.data?.classes ?? []).map((c) => c.name), [ontologyQ.data]);
  const shownAt = useRef<number>(Date.now());
  const inFlight = useRef(0);

  // Every scope the triage API accepts, not just the session. A batch mined for one class had no URL that
  // would open it while this page forwarded one parameter out of five.
  const scope = params.toString();
  const scopeLabel = describeScope(params);
  const current = queue[index] ?? null;

  // Deliberately not through the query cache. This queue is consumed by position, one keystroke per row:
  // painting the last visit's rows and swapping in the fresh ones a moment later would put a different
  // object under a verdict already on its way. It is fetched once per scope and then only walked.
  useEffect(() => {
    (async () => {
      try {
        setQueue(await api.triage(triageQuery(new URLSearchParams(scope))));
      } catch (e) { toast(humanizeError(e), "error"); } finally { setLoading(false); }
    })();
  }, [scope]);

  useEffect(() => { shownAt.current = Date.now(); }, [index]);

  // Decode the next few crops while this one is on screen. Without it every verdict is followed by a blank
  // pane for as long as the next image takes, which is exactly the pause this page exists to remove.
  useEffect(() => {
    for (let i = index + 1; i <= index + PREFETCH && i < queue.length; i++) {
      const img = new Image();
      img.src = cropUrl(queue[i].object_id);
      const ctx = new Image();
      ctx.src = contextUrl(queue[i].object_id);
    }
  }, [index, queue]);

  const send = useCallback(async (row: TriageRow, verdict: Verdict, className?: string) => {
    if (verdict === "skip") return;
    inFlight.current += 1;
    try {
      await api.review(row.object_id, verdictPayload(verdict, {
        reviewer: me?.name ?? "anon",
        acceptState: acceptState(me?.role),
        className,
        timeSpentMs: Date.now() - shownAt.current,
      }));
      // The object has left the queues other pages cache, and home's waiting count with it.
      void invalidate("triage", "/api/home");
    } catch (e) {
      toast(`${row.class_name}: ${humanizeError(e)}`, "error");
    } finally { inFlight.current -= 1; }
  }, [me]);

  const decide = useCallback((verdict: Verdict, className?: string) => {
    const row = queue[index];
    if (!row) return;
    setDone((d) => [...d, { row, verdict, className }]);
    setIndex((i) => i + 1);
    void send(row, verdict, className);
  }, [queue, index, send]);

  const undo = useCallback(() => {
    // Read the stack rather than mutating inside a state updater. An updater must be pure, and this one
    // fired the network call, so React re-invoking it (StrictMode does, in development) sent the reversal
    // twice.
    const last = done[done.length - 1];
    if (!last) return;
    setDone((d) => d.slice(0, -1));
    setIndex((i) => Math.max(0, i - 1));
    if (last.verdict === "skip") return;
    // The reversal is a real one: the object goes back to needing review on the server, not just in this
    // tab. An undo that only moved the cursor would leave the mistake committed.
    //
    // Success is claimed only once the server confirms it. This toast used to fire synchronously beside
    // the request, so when every reversal was being rejected the annotator was still told it had worked.
    void api.review(last.row.object_id, undoPayload(last.row, { reviewer: me?.name ?? "anon" }))
      .then(() => {
        toast(`Undid ${last.verdict} on ${humanize(last.row.class_name).toLowerCase()}`, "success");
        void invalidate("triage", "/api/home");
      })
      .catch((e) => toast(`Could not undo ${last.verdict} on ${humanize(last.row.class_name).toLowerCase()}: `
                          + `${humanizeError(e)}. The verdict still stands.`, "error"));
  }, [done, me]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Held keys must not repeat into a stack of verdicts on objects nobody looked at.
      if (e.repeat) return;
      if (reclassOpen) {
        if (e.key === "Escape") { setReclassOpen(false); setClassQuery(""); }
        return;
      }
      const k = e.key.toLowerCase();
      if (k === "a") { e.preventDefault(); decide("accept"); }
      else if (k === "r") { e.preventDefault(); decide("reject"); }
      else if (k === "c") { e.preventDefault(); setReclassOpen(true); }
      else if (k === "s" || e.key === " ") { e.preventDefault(); decide("skip"); }
      else if (k === "u") { e.preventDefault(); undo(); }
      else if (k === "o" && current) { router.push(`/frame/${current.frame_id}`); }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [decide, undo, reclassOpen, current, router]);

  const matches = useMemo(() => {
    const q = classQuery.trim().toLowerCase();
    return (q ? classes.filter((c) => c.includes(q)) : classes).slice(0, 8);
  }, [classQuery, classes]);

  const counts = useMemo(() => {
    const c = { accept: 0, reject: 0, reclass: 0, skip: 0 };
    for (const d of done) c[d.verdict] += 1;
    return c;
  }, [done]);

  if (loading) {
    return (
      <PageShell active="RAPID REVIEW" title="Rapid review">
        <div className="h-full flex items-center justify-center text-[13px] text-ink-3" role="status">Loading the queue...</div>
      </PageShell>
    );
  }

  return (
    <PageShell
      active="RAPID REVIEW"
      title="Rapid review"
      // A scoped queue ends early by design. Saying what it was scoped to is the difference between "the
      // batch is finished" and "the corpus looks empty and something is broken".
      subtitle={scopeLabel ? `${scopeLabel} · one object, one key` : "One object at a time, one key per decision"}
      right={
        <span className="text-[12px] text-ink-3 tabular-nums flex items-center gap-2.5">
          <span className="text-ink-2">{Math.min(index + 1, queue.length)} of {queue.length}</span>
          <span className="text-pass" title="Accepted">{counts.accept} accepted</span>
          <span className="text-block" title="Rejected">{counts.reject} rejected</span>
          <span className="text-warn" title="Reclassified">{counts.reclass} reclassified</span>
          <span title="Skipped">{counts.skip} skipped</span>
        </span>
      }
    >
      <div className="h-full flex flex-col">
        {!current ? (
          <div className="flex-1 flex items-center justify-center p-8">
            <EmptyState icon={queue.length ? "check" : "list"}
              title={queue.length ? "Queue cleared" : "Nothing to review here"}
              action={<button onClick={() => router.push("/review/queue")} className="btn">Back to the review queue</button>}>
              {queue.length
                ? `${done.length} decided: ${counts.accept} accepted, ${counts.reject} rejected, ${counts.reclass} reclassified, ${counts.skip} skipped.`
                : scopeLabel ? `No objects match ${scopeLabel}.` : "No objects are waiting for a decision right now."}
            </EmptyState>
          </div>
        ) : (
          <>
            <div className="flex-1 min-h-0 flex bg-bg-2 relative">
              <div className="flex-1 min-w-0 min-h-0 flex flex-col items-center justify-center p-6 gap-2">
                {/* Keyed on the object id so React swaps the element rather than reusing it, which would
                    briefly show the previous crop under the new label. Drawn at up to MAX_UPSCALE times its own
                    size and never past the pane; object-contain keeps the aspect inside that box. */}
                <img key={current.object_id} src={cropUrl(current.object_id)}
                  alt={humanize(current.class_name)}
                  onLoad={(e) => setNatural({ id: current.object_id, w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
                  style={natural?.id === current.object_id
                    ? { width: natural.w * MAX_UPSCALE, height: natural.h * MAX_UPSCALE }
                    : undefined}
                  className="max-w-full max-h-full min-h-0 object-contain rounded border border-line bg-bg" />
                {natural?.id === current.object_id && (
                  <div className="text-[11.5px] text-ink-3 tabular-nums">
                    Crop {natural.w} × {natural.h} px{natural.w < 120 && natural.h < 120 ? ", a small object: check the frame view" : ""}
                  </div>
                )}
              </div>

              {/* What the object is and why it is here, and the same object with the frame around it. */}
              <aside className="w-[340px] shrink-0 border-l hairline bg-panel overflow-y-auto p-4 space-y-4">
                <div>
                  <div className="text-[18px] font-semibold text-ink leading-tight" title={current.class_name}>{humanize(current.class_name)}</div>
                  <div className="mt-1 text-[12.5px] text-ink-3">
                    {current.conf > 0 ? `${Math.round(current.conf * 100)}% confident` : "No detector score"}
                    {current.source ? ` · ${humanize(current.source)}` : ""}
                    {current.import_format ? ` · ${current.import_format}` : ""}
                  </div>
                </div>
                {current.why && (
                  <div>
                    <div className="label-caps mb-1.5">Why it needs you</div>
                    <div className="flex flex-wrap gap-1.5">
                      {(current.flags?.length ? current.flags.map((f) => f.label) : current.why.split(/,\s*/)).map((w, i) => (
                        <span key={`${w}-${i}`} className="text-[11.5px] px-1.5 py-0.5 rounded border border-warn/40 text-warn bg-warn/10">
                          {humanize(w)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <div>
                  <div className="label-caps mb-1.5">In the frame</div>
                  <img key={`ctx-${current.object_id}`} src={contextUrl(current.object_id)} alt={`${humanize(current.class_name)} with its surroundings`}
                    className="w-full max-h-[300px] object-contain rounded border border-line skeleton"
                    onLoad={(e) => { e.currentTarget.classList.remove("skeleton"); }}
                    onError={(e) => { e.currentTarget.style.display = "none"; }} />
                  <button onClick={() => router.push(`/frame/${current.frame_id}`)} className="link text-[12.5px] mt-2">
                    Open the whole frame (O)
                  </button>
                </div>
              </aside>

              {reclassOpen && (
                <div className="absolute inset-0 bg-bg/80 flex items-center justify-center">
                  <div className="panel p-3 w-[min(22rem,90vw)] space-y-2">
                    <div className="section-title">Reclassify as</div>
                    {/* eslint-disable-next-line jsx-a11y/no-autofocus -- opened by the C key in a one-keystroke loop; a box that did not take the next keystroke would break it */}
                    <input autoFocus aria-label="reclassify as" value={classQuery} onChange={(e) => setClassQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && matches[0]) {
                          setReclassOpen(false); setClassQuery("");
                          decide("reclass", matches[0]);
                        }
                      }}
                      placeholder="Type a class name"
                      className="field w-full text-[13px]" />
                    <div className="space-y-0.5">
                      {matches.map((c, i) => (
                        <button key={c}
                          onClick={() => { setReclassOpen(false); setClassQuery(""); decide("reclass", c); }}
                          className={`flex w-full items-center text-left px-2 py-1 rounded text-[12.5px]
                                      ${i === 0 ? "text-ink bg-head" : "text-ink-2 hover:text-ink hover:bg-head/60"}`}>
                          <span title={c}>{humanize(c)}</span>
                          {i === 0 && <span className="ml-auto text-[11px] text-ink-3">Enter</span>}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="shrink-0 border-t hairline px-4 py-2.5 flex flex-wrap items-center gap-x-5 gap-y-2 text-[12.5px]">
              {/* The verdict row, in the reviewer's own language. i18n shipped with four locales and one
                  consumer (the onboarding tour), so the surface an annotator spends a whole shift on -
                  this one - was English-only in a product built for Indian roads. The hotkeys stay Latin
                  because they are physical keys, and the class name below is never translated: renaming a
                  class in the interface would make it impossible to talk about with the ontology. */}
              <Key k="A" label={t("action.accept", "Accept")} tone="text-pass" onClick={() => decide("accept")} />
              <Key k="R" label={t("action.reject", "Reject")} tone="text-block" onClick={() => decide("reject")} />
              <Key k="C" label={t("action.reclassify", "Reclassify")} tone="text-warn" onClick={() => setReclassOpen(true)} />
              <Key k="S" label={t("action.skip", "Skip")} tone="text-ink-2" onClick={() => decide("skip")} />
              <Key k="U" label={t("action.undo", "Undo")} tone="text-ink-2" onClick={undo} />
              <Key k="O" label="Open the frame" tone="text-ink-2"
                onClick={() => router.push(`/frame/${current.frame_id}`)} />
              <span className="ml-auto text-ink-3">
                {inFlight.current > 0 ? t("editor.unsaved", "saving...") : t("editor.saved", "saved")}
              </span>
            </div>
          </>
        )}
      </div>
    </PageShell>
  );
}

function Key({ k, label, tone, onClick }: {
  k: string; label: string; tone: string; onClick: () => void;
}) {
  return (
    <button onClick={onClick} className={`flex items-center gap-1.5 ${tone} hover:opacity-80`}>
      <kbd className="min-w-[20px] text-center rounded border border-line bg-head px-1.5 py-0.5 text-[11px] font-mono text-ink-2">{k}</kbd>
      {label}
    </button>
  );
}

export default function RapidReviewPage() {
  return <Suspense fallback={null}><RapidBody /></Suspense>;
}
