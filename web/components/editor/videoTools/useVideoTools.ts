"use client";

// State and actions for agent A's editor tools: Track mask (SAM 2 video), Describe (phrase to labels) and
// Sweep (one key per decision, with a correction fanned out in one undoable batch).
//
// One hook rather than three so the frame page gains one call and one layer, not a dozen pieces of state:
// that page is 2,500 lines and heavily tested, and every line added there is a line its tests read. The
// rules each tool follows are pure functions in lib/tools/videoLogic.ts.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { EdObject } from "@/components/editor/useEditor";
import { useStableCallback } from "@/components/editor/useStableCallback";
import { humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import { useToolJob, type ToolJob } from "@/lib/tools/toolJobs";
import {
  acceptDescribed, describeFrame, describeSession, fanoutApply, fanoutPreview, fanoutUndo, startTrack,
  type DescribeResponse, type FanoutPreview, type TrackRequest, type TrackResult,
} from "@/lib/tools/videoApi";
import {
  defaultTicks, fanoutCount, fanoutMessage, nextSweepTarget, sweepQueue, toggleTick, trackRequestForClick,
  type FanoutGroup, type FanoutTicks, type LastTrack,
} from "@/lib/tools/videoLogic";

const LAST_TRACK_KEY = "lbx.video.lastTrack";
const SWEEP_KEY = "lbx.video.sweep";

// sessionStorage, guarded: the frame page remounts on every frame change, and a track made on one frame
// is corrected on another, so the last track and the sweep have to outlive the page. Private mode and a
// full quota throw, and the tools still work then, only without crossing frames.
function readStore<T>(key: string): T | null {
  try {
    const raw = globalThis.sessionStorage?.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}
function writeStore(key: string, value: unknown): void {
  try {
    if (value == null) globalThis.sessionStorage?.removeItem(key);
    else globalThis.sessionStorage?.setItem(key, JSON.stringify(value));
  } catch {
    /* not persisted; this page still has it */
  }
}

export type StoredTrack = NonNullable<LastTrack> & { empty_frame_ids: string[]; class_name: string | null };

export type FanoutOffer = { objectId: string; from: string; to: string };

export type VideoToolsOptions = {
  frameId: string;
  sessionId: string | null;
  /** The class a new label gets (the editor's current class). */
  className: string | null;
  selected: EdObject | null;
  objects: EdObject[];
  /** Re-read this frame's objects. */
  refresh: () => unknown;
  select: (id: string | null) => void;
  /** True (and says so) when the frame is view-only. */
  viewOnly: () => boolean;
  /** Sweep runs in Review mode, where A, X and 1 to 9 decide; called when a sweep starts. */
  onSweepStart: () => void;
};

export function useVideoTools(o: VideoToolsOptions) {
  // The effects below read the page's callbacks through a ref: the page hands a fresh options object every
  // render, and an effect keyed on it would run on every render.
  const oRef = useRef(o);
  oRef.current = o;
  // ---- Track mask ----
  const [span, setSpan] = useState({ before: 10, after: 10 });
  const [trackJobId, setTrackJobId] = useState<string | null>(null);
  const [lastTrack, setLastTrackState] = useState<StoredTrack | null>(null);
  useEffect(() => { setLastTrackState(readStore<StoredTrack>(LAST_TRACK_KEY)); }, []);
  const setLastTrack = useCallback((t: StoredTrack | null) => { setLastTrackState(t); writeStore(LAST_TRACK_KEY, t); }, []);

  const onTrackDone = useStableCallback((job: ToolJob) => {
    setTrackJobId(null);
    if (job.status !== "done") {
      if (job.status === "error") toast(`mask tracking failed: ${job.error ?? "unknown error"}`, "error");
      return;
    }
    const r = (job.result ?? {}) as unknown as TrackResult;
    if (r.track_id && r.window_frame_ids) {
      setLastTrack({ track_id: r.track_id, window_frame_ids: r.window_frame_ids, empty_frame_ids: r.empty_frame_ids ?? [],
                     class_name: (job.params?.class_name as string | undefined) ?? lastTrack?.class_name ?? null });
    }
    const made = r.created?.length ?? 0;
    const fixed = r.updated?.length ?? 0;
    const msg = r.method === "sam_propagate_fallback"
      ? `SAM 2 video was unavailable, so the box was carried by optical flow instead: ${made} frames (${r.fallback_reason ?? ""})`
      : `tracked over ${r.frames ?? "?"} frames: ${made} new, ${fixed} corrected${r.removed?.length ? `, ${r.removed.length} dropped` : ""}`
        + `${r.empty_frame_ids?.length ? `; not visible on ${r.empty_frame_ids.length}` : ""}. Click on any of them to correct it.`;
    toast(msg, r.method === "sam_propagate_fallback" ? "warn" : "success", 9000);
    void o.refresh();
  });
  const { job: trackJob } = useToolJob(trackJobId, { everyMs: 800, onDone: onTrackDone });

  const runTrack = useStableCallback(async (req: TrackRequest) => {
    if (o.viewOnly()) return;
    if (trackJobId) { toast("a track is still running; wait for it to finish", "warn"); return; }
    try {
      const j = await startTrack(req);
      setTrackJobId(j.job_id);
    } catch (e) {
      toast(`could not start tracking: ${humanizeError(e)}`, "error");
    }
  });

  const onTrackPoint = useStableCallback((pt: number[], label: number) => {
    const req = trackRequestForClick({ frameId: o.frameId, pt, label, className: o.className, last: lastTrack,
                                       before: span.before, after: span.after });
    if (!req) {
      toast(label ? "pick a class first: the tracked object is labelled with the class in hand"
                  : "shift-click marks what is not the object; click on the object first", "info");
      return;
    }
    void runTrack(req);
  });

  const trackSelected = useStableCallback(() => {
    const s = o.selected;
    if (!s || s.isNew || s.id.startsWith("tmp-")) { toast("select a saved object to track it", "info"); return; }
    void runTrack({ frame_id: o.frameId, object_id: s.id, prompts: [], before: span.before, after: span.after });
  });

  const inLastTrack = !!lastTrack && lastTrack.window_frame_ids.includes(o.frameId);
  const forgetTrack = useCallback(() => setLastTrack(null), [setLastTrack]);

  // ---- Describe ----
  const [describeOpen, setDescribeOpen] = useState(false);
  const [phrase, setPhrase] = useState("");
  const [describing, setDescribing] = useState(false);
  const [described, setDescribed] = useState<DescribeResponse | null>(null);
  const [ticks, setTicks] = useState<boolean[]>([]);
  const [describeClass, setDescribeClass] = useState<string | null>(null);
  const [sessionJobId, setSessionJobId] = useState<string | null>(null);

  const openDescribe = useStableCallback(() => {
    if (o.viewOnly()) return;
    setDescribeOpen(true);
    if (!describeClass && o.className) setDescribeClass(o.className);
  });
  const closeDescribe = useStableCallback(() => { setDescribeOpen(false); setDescribed(null); setTicks([]); });

  const runDescribe = useStableCallback(async () => {
    const text = phrase.trim();
    if (!text) return;
    setDescribing(true);
    try {
      const r = await describeFrame(o.frameId, text);
      setDescribed(r);
      setTicks(r.proposals.map(() => true));
      if (!r.proposals.length) toast(`nothing on this frame matched "${r.phrase}"`, "info");
    } catch (e) {
      toast(`describe failed: ${humanizeError(e)}`, "error");
    } finally {
      setDescribing(false);
    }
  });

  const acceptDescribe = useStableCallback(async () => {
    if (o.viewOnly() || !described) return;
    const chosen = described.proposals.filter((_, i) => ticks[i]);
    const cls = describeClass ?? o.className;
    if (!chosen.length || !cls) { toast(cls ? "tick at least one proposal" : "pick a class to file them under", "info"); return; }
    try {
      const r = await acceptDescribed(o.frameId, { phrase: described.phrase, class_name: cls,
        proposals: chosen.map((p) => ({ bbox: p.bbox, polygons: p.polygons, conf: p.conf })) });
      toast(`${r.created.length} labelled ${r.class_name} from "${r.phrase}"`, "success");
      closeDescribe();
      void o.refresh();
    } catch (e) {
      toast(`could not save the proposals: ${humanizeError(e)}`, "error");
    }
  });

  const describeWholeSession = useStableCallback(async () => {
    const cls = describeClass ?? o.className;
    if (!o.sessionId || !cls || !phrase.trim()) return;
    try {
      const j = await describeSession({ session_id: o.sessionId, phrase: phrase.trim(), class_name: cls });
      setSessionJobId(j.job_id);
      toast("queued over the whole session; finds go to review, not straight into the labels", "info", 8000);
    } catch (e) {
      toast(`could not queue the session run: ${humanizeError(e)}`, "error");
    }
  });
  const { job: sessionJob } = useToolJob(sessionJobId, {
    everyMs: 3000,
    onDone: (j) => {
      if (j.status === "done") {
        const r = j.result as { written?: number; frames?: number; frames_with_finds?: number } | null;
        toast(`described the session: ${r?.written ?? 0} proposals on ${r?.frames_with_finds ?? 0} of ${r?.frames ?? 0} frames, waiting in review`,
              "success", 10000);
      } else if (j.status === "error") {
        toast(`the session run failed: ${j.error ?? "unknown error"}`, "error");
      }
    },
  });

  const canvasProposals = useMemo(() => {
    if (!described?.proposals.length) return null;
    return described.proposals.map((p, i) => ({
      bbox: p.bbox, polygons: p.polygons, on: !!ticks[i], label: `${described.phrase} ${p.conf.toFixed(2)}`,
    }));
  }, [described, ticks]);

  // ---- Sweep ----
  const [sweepOn, setSweepOnState] = useState(false);
  const [decided, setDecided] = useState<Set<string>>(() => new Set());
  const queue = useMemo(() => sweepQueue(o.objects, decided), [o.objects, decided]);
  const total = useMemo(() => sweepQueue(o.objects).length + decided.size, [o.objects, decided]);
  const setSweepOn = useCallback((v: boolean) => { setSweepOnState(v); writeStore(SWEEP_KEY, v ? 1 : null); }, []);

  const startedRef = useRef(false);
  useEffect(() => {
    // A sweep carries across frames. Restored once per page; the page puts itself in Review mode.
    if (startedRef.current) return;
    startedRef.current = true;
    if (readStore<number>(SWEEP_KEY)) { setSweepOnState(true); oRef.current.onSweepStart(); }
  }, []);
  useEffect(() => { setDecided(new Set()); }, [o.frameId]);

  // With a sweep on, something is always in front of the person: when the selection is not in the queue
  // (the frame just opened, or the last decision emptied it) the head of the queue is selected.
  const selectedId = o.selected?.id ?? null;
  useEffect(() => {
    if (!sweepOn) return;
    if (selectedId && queue.includes(selectedId)) return;
    const next = nextSweepTarget(queue, null);
    if (next && next !== selectedId) oRef.current.select(next);
  }, [sweepOn, queue, selectedId]);

  const toggleSweep = useStableCallback(() => {
    if (!sweepOn && o.viewOnly()) return;
    const on = !sweepOn;
    setSweepOn(on);
    if (on) {
      o.onSweepStart();
      setDecided(new Set());
      const first = nextSweepTarget(sweepQueue(o.objects), null);
      o.select(first);
      toast(first ? "sweep: A accept, X reject, 1 to 9 change the class, N pass, Y stop"
                  : "sweep: nothing left to decide on this frame; ] goes to the next", "info", 6000);
    }
  });

  /** Called after a decision on `id` (accept, reject or relabel): it leaves the queue and the next comes up. */
  const sweepAfter = useStableCallback((id: string) => {
    if (!sweepOn) return;
    const nextDecided = new Set(decided).add(id);
    setDecided(nextDecided);
    const rest = sweepQueue(o.objects, nextDecided);
    const next = nextSweepTarget(rest, id);
    o.select(next);
    if (!next) toast("this frame is swept; ] for the next frame", "success");
  });

  const sweepSkip = useStableCallback(() => {
    if (!sweepOn) return;
    o.select(nextSweepTarget(queue, selectedId));
  });

  // ---- Fan-out ----
  const [offer, setOffer] = useState<FanoutOffer | null>(null);
  const [preview, setPreview] = useState<FanoutPreview | null>(null);
  const [fanTicks, setFanTicks] = useState<FanoutTicks>({ track: [], lookalikes: [], confusion: [] });
  const [fanBusy, setFanBusy] = useState(false);

  const offerFanout = useStableCallback(async (f: FanoutOffer) => {
    setOffer(f);
    setPreview(null);
    try {
      const p = await fanoutPreview({ object_id: f.objectId, from_class: f.from, to_class: f.to });
      setPreview(p);
      const t = defaultTicks(p);
      setFanTicks(t);
      if (!fanoutCount(t) && !p.track.objects.length && !p.lookalikes.objects.length && !p.confusion.objects.length) {
        setOffer(null);   // nothing else carries this mistake; saying so would be noise on every correction
      }
    } catch (e) {
      setOffer(null);
      toast(`could not look for the same mistake elsewhere: ${humanizeError(e)}`, "error");
    }
  });
  const dismissFanout = useCallback(() => { setOffer(null); setPreview(null); }, []);
  const tickFanout = useCallback((g: FanoutGroup, id: string) => setFanTicks((t) => toggleTick(t, g, id)), []);

  const applyFanout = useStableCallback(async () => {
    if (!offer || o.viewOnly()) return;
    setFanBusy(true);
    try {
      const r = await fanoutApply({ object_id: offer.objectId, to_class: offer.to, track_object_ids: fanTicks.track,
                                    lookalike_object_ids: fanTicks.lookalikes, confusion_object_ids: fanTicks.confusion });
      const runId = r.run_id;
      toast(fanoutMessage(r), "success", 12000, runId ? {
        label: "undo",
        run: async () => {
          try {
            const u = await fanoutUndo(runId);
            toast(`undone: ${u.reverted} restored${u.skipped ? `, ${u.skipped} left alone (edited since)` : ""}`, "info");
            void o.refresh();
          } catch (e) {
            toast(`could not undo: ${humanizeError(e)}`, "error");
          }
        },
      } : undefined);
      dismissFanout();
      void o.refresh();
    } catch (e) {
      toast(`could not apply the correction: ${humanizeError(e)}`, "error");
    } finally {
      setFanBusy(false);
    }
  });

  return {
    // track
    span, setSpan, trackJob, trackRunning: !!trackJobId, lastTrack, inLastTrack, forgetTrack, onTrackPoint,
    trackSelected,
    // describe
    describeOpen, openDescribe, closeDescribe, phrase, setPhrase, describing, described, ticks, setTicks,
    describeClass, setDescribeClass, runDescribe, acceptDescribe, describeWholeSession, sessionJob, canvasProposals,
    // sweep
    sweepOn, toggleSweep, sweepAfter, sweepSkip, sweepLeft: queue.length, sweepTotal: total,
    // fan-out
    offer, preview, fanTicks, fanBusy, offerFanout, dismissFanout, tickFanout, applyFanout,
  };
}

export type VideoTools = ReturnType<typeof useVideoTools>;
