"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { useRouter } from "next/navigation";

import PageShell from "@/components/shell/PageShell";
import { api, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { describeScope, triageQuery } from "@/lib/triageScope";
import { acceptState, useCurrentUser } from "@/lib/user";
import type { TriageRow } from "@/lib/types";

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

function RapidBody() {
  const params = useSearchParams();
  const router = useRouter();
  const me = useCurrentUser();

  const [queue, setQueue] = useState<TriageRow[]>([]);
  const [index, setIndex] = useState(0);
  const [done, setDone] = useState<Decided[]>([]);
  const [loading, setLoading] = useState(true);
  const [reclassOpen, setReclassOpen] = useState(false);
  const [classQuery, setClassQuery] = useState("");
  const [classes, setClasses] = useState<string[]>([]);
  const shownAt = useRef<number>(Date.now());
  const inFlight = useRef(0);

  // Every scope the triage API accepts, not just the session. A batch mined for one class had no URL that
  // would open it while this page forwarded one parameter out of five.
  const scope = params.toString();
  const scopeLabel = describeScope(params);
  const current = queue[index] ?? null;

  useEffect(() => {
    (async () => {
      try {
        setQueue(await api.triage(triageQuery(new URLSearchParams(scope))));
      } catch (e) { toast(humanizeError(e), "error"); } finally { setLoading(false); }
    })();
    api.ontology().then((o) => setClasses(o.classes.map((c) => c.name))).catch(() => {});
  }, [scope]);

  useEffect(() => { shownAt.current = Date.now(); }, [index]);

  // Decode the next few crops while this one is on screen. Without it every verdict is followed by a blank
  // pane for as long as the next image takes, which is exactly the pause this page exists to remove.
  useEffect(() => {
    for (let i = index + 1; i <= index + PREFETCH && i < queue.length; i++) {
      const img = new Image();
      img.src = `/api/objects/${queue[i].object_id}/crop`;
    }
  }, [index, queue]);

  const send = useCallback(async (row: TriageRow, verdict: Verdict, className?: string) => {
    if (verdict === "skip") return;
    inFlight.current += 1;
    try {
      await api.review(row.object_id, {
        reviewer: me?.name ?? "anon",
        action: verdict === "accept" ? "confirm" : verdict === "reject" ? "reject" : "reclassify",
        class_name: className,
        state: verdict === "reject" ? "rejected" : acceptState(me?.role),
        // Real elapsed time, so the productivity numbers this feeds are measurements rather than a
        // constant. A hardcoded value here would quietly corrupt every throughput report downstream.
        time_spent_ms: Math.max(0, Date.now() - shownAt.current),
      });
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
    setDone((d) => {
      if (!d.length) return d;
      const last = d[d.length - 1];
      setIndex((i) => Math.max(0, i - 1));
      // The reversal is a real one: the object goes back to needing review on the server, not just in this
      // tab. An undo that only moved the cursor would leave the mistake committed.
      if (last.verdict !== "skip") {
        void api.review(last.row.object_id, {
          reviewer: me?.name ?? "anon", action: "confirm", state: "needs_review", time_spent_ms: 0,
        }).catch((e) => toast(humanizeError(e), "error"));
      }
      toast(`undid ${last.verdict} on ${last.row.class_name}`, "success");
      return d.slice(0, -1);
    });
  }, [me]);

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
    return <PageShell active="RAPID REVIEW" title="Rapid review"><div className="p-4 font-mono text-[11px] text-ink-3">loading the queue...</div></PageShell>;
  }

  return (
    <PageShell
      active="RAPID REVIEW"
      title="Rapid review"
      // A scoped queue ends early by design. Saying what it was scoped to is the difference between "the
      // batch is finished" and "the corpus looks empty and something is broken".
      subtitle={scopeLabel ? `${scopeLabel} · one crop, one keystroke` : "one crop, one keystroke"}
      right={
        <span className="font-mono text-[11px] text-ink-3">
          {index}/{queue.length} · <span className="text-pass">{counts.accept}a</span>
          {" "}<span className="text-block">{counts.reject}r</span>
          {" "}<span className="text-warn">{counts.reclass}c</span>
          {" "}{counts.skip}s
        </span>
      }
    >
      <div className="h-full flex flex-col">
        {!current ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8">
            <div className="font-mono text-[13px] text-ink">
              {queue.length ? "queue cleared" : "nothing in the queue"}
            </div>
            <div className="font-mono text-[11px] text-ink-3">
              {done.length} decided · {counts.accept} accepted, {counts.reject} rejected,
              {" "}{counts.reclass} reclassified, {counts.skip} skipped
            </div>
            <button onClick={() => router.push("/review/queue")}
              className="border border-line px-3 py-1 font-mono text-[11px] text-ink-2 hover:border-accent">
              back to the review queue
            </button>
          </div>
        ) : (
          <>
            <div className="flex-1 min-h-0 flex items-center justify-center bg-bg-2 relative">
              {/* Keyed on the object id so React swaps the element rather than reusing it, which would
                  briefly show the previous crop under the new label. */}
              {/* h-full w-full, not max-h/max-w: a crop of a distant rider is a few dozen pixels, and
                  max-* never scales up, so the object under review rendered as a thumbnail in the middle of
                  an empty screen. object-contain keeps the aspect ratio while filling the pane, and
                  pixelated rendering is honest about the upscale rather than smearing it. */}
              <img key={current.object_id} src={`/api/objects/${current.object_id}/crop`}
                alt={current.class_name}
                style={{ imageRendering: "pixelated" }}
                className="h-full w-full object-contain p-6" />

              <div className="absolute top-3 left-3 panel px-3 py-2 space-y-0.5">
                <div className="font-mono text-[14px] text-ink">{current.class_name}</div>
                <div className="font-mono text-[10.5px] text-ink-3">
                  conf {current.conf.toFixed(2)} · {current.source}
                  {current.import_format ? ` · ${current.import_format}` : ""}
                </div>
                {current.why && (
                  <div className="font-mono text-[10.5px] text-warn max-w-[22rem]">{current.why}</div>
                )}
              </div>

              {reclassOpen && (
                <div className="absolute inset-0 bg-bg/80 flex items-center justify-center">
                  <div className="panel p-3 w-[min(22rem,90vw)] space-y-2">
                    <div className="font-mono text-[11px] uppercase text-ink-3">reclassify as</div>
                    <input autoFocus value={classQuery} onChange={(e) => setClassQuery(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && matches[0]) {
                          setReclassOpen(false); setClassQuery("");
                          decide("reclass", matches[0]);
                        }
                      }}
                      placeholder="type a class name"
                      className="w-full bg-bg border border-line px-2 py-1 font-mono text-[12px] text-ink" />
                    <div className="space-y-0.5">
                      {matches.map((c, i) => (
                        <button key={c}
                          onClick={() => { setReclassOpen(false); setClassQuery(""); decide("reclass", c); }}
                          className={`block w-full text-left px-2 py-0.5 font-mono text-[11.5px]
                                      ${i === 0 ? "text-ink bg-panel-2" : "text-ink-2 hover:text-ink"}`}>
                          {c}{i === 0 ? "  (enter)" : ""}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="shrink-0 border-t hairline px-4 py-2 flex items-center gap-4 font-mono text-[11px]">
              <Key k="A" label="accept" tone="text-pass" onClick={() => decide("accept")} />
              <Key k="R" label="reject" tone="text-block" onClick={() => decide("reject")} />
              <Key k="C" label="reclassify" tone="text-warn" onClick={() => setReclassOpen(true)} />
              <Key k="S" label="skip" tone="text-ink-3" onClick={() => decide("skip")} />
              <Key k="U" label="undo" tone="text-ink-3" onClick={undo} />
              <Key k="O" label="open the frame" tone="text-ink-3"
                onClick={() => router.push(`/frame/${current.frame_id}`)} />
              <span className="ml-auto text-ink-4">
                {inFlight.current > 0 ? "saving..." : "saved"}
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
      <kbd className="border border-line px-1.5 py-0.5 text-[10px] text-ink-2">{k}</kbd>
      {label}
    </button>
  );
}

export default function RapidReviewPage() {
  return <Suspense fallback={null}><RapidBody /></Suspense>;
}
