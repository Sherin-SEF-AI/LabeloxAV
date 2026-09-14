"use client";

// The event lane: typed spans drawn across the same frames the crop strip shows.
//
// A track event has a time extent, which is the whole reason it exists rather than being another entry in
// `Track.intents`. So the lane is laid out on the track's own frames, one column per crop, and a span is a
// bar covering the columns it contains. Drag across columns to create; the type picker opens on release.
//
// Proposed spans are hatched. A heuristic proposal and a human's span must not look the same: the two
// proposers behind them read IPM monocular estimates whose frame-to-frame noise is larger than the effect
// they look for, so a reviewer needs to see at a glance which bars are claims and which are suggestions.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api, humanizeError } from "@/lib/api";
import { toast } from "@/lib/toast";
import type { TrackEvent, TrackEventType, TrackItem } from "@/lib/types";

// How far a dragged span edge may be pulled to reach a changepoint.
//
// Small on purpose. An edge dropped in the middle of a steady stretch means what it says, and hauling it
// several frames to the nearest event would move a span the annotator placed deliberately. Three frames
// at 3fps is one second, which is about the width of a hand-drag's imprecision.
const SNAP_WINDOW = 3;

const STATE_TONE: Record<string, string> = {
  accepted: "bg-pass/60 border-pass",
  proposed: "bg-warn/20 border-warn",
  rejected: "bg-block/20 border-block",
};

// Column geometry, shared with the crop strip above: w-28 (7rem) cells with gap-2 (0.5rem) between them.
const CELL_REM = 7;
const GAP_REM = 0.5;
const widthFor = (n: number) => `${n * CELL_REM + (n - 1) * GAP_REM}rem`;

const HATCH = {
  backgroundImage:
    "repeating-linear-gradient(45deg, transparent 0 4px, rgba(255,255,255,0.14) 4px 8px)",
};

export default function EventLane({ trackId, items, onChanged }: {
  trackId: string;
  items: TrackItem[];
  onChanged?: () => void;
}) {
  const [events, setEvents] = useState<TrackEvent[]>([]);
  const [types, setTypes] = useState<TrackEventType[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [drag, setDrag] = useState<[number, number] | null>(null);
  const [pending, setPending] = useState<[number, number] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const dragging = useRef(false);

  const load = useCallback(async () => {
    try {
      const r = await api.trackEvents(trackId);
      setEvents(r.events);
      setTypes(r.event_types);
    } catch (e) {
      toast(humanizeError(e), "error");
    } finally {
      setLoaded(true);
    }
  }, [trackId]);

  useEffect(() => { void load(); }, [load]);

  // Column index for a timestamp: the last frame at or before it. Spans are stored against frames, so an
  // event always lands on a real column rather than on a fractional position between two.
  const indexOf = useCallback((ts: number) => {
    let lo = 0;
    for (let i = 0; i < items.length; i++) if (items[i].ts_ns <= ts) lo = i;
    return lo;
  }, [items]);

  const bars = useMemo(() => events.map((e) => {
    const a = indexOf(e.start_ts_ns);
    const b = Math.max(a, indexOf(e.end_ts_ns));
    return { e, a, span: b - a + 1 };
  }), [events, indexOf]);

  const applicable = useMemo(() => types.filter((t) => t.applicable), [types]);

  // Where the track's motion actually changes, as snap targets for a span edge.
  //
  // A dragged edge lands where the mouse came up, which is within a frame or two of the moment that
  // matters and almost never on it. Snapping is only honest if there is a measured moment to snap to,
  // which is why this is a request rather than a heuristic: the server refuses when the signal it was
  // asked for does not exist, and 86% of real tracks legitimately have no changepoint at all.
  const [cps, setCps] = useState<{ index: number; sigmas: number; before: number; after: number }[]>([]);
  const [snapOn, setSnapOn] = useState(true);

  useEffect(() => {
    let live = true;
    api.trackChangepoints(trackId)
      .then((r) => { if (live) setCps(r.changepoints.map((c) => ({ index: c.index, sigmas: c.sigmas, before: c.before, after: c.after }))); })
      // Quietly: a track with no dynamics computed is the common case and not an error the annotator can
      // act on. The absence shows as no snap marks, which is what it means.
      .catch(() => { if (live) setCps([]); });
    return () => { live = false; };
  }, [trackId]);

  /**
   * Pull an index onto the nearest changepoint, when one is close enough.
   *
   * The window is deliberately small. An edge dragged into the middle of a steady stretch means what it
   * says, and hauling it several frames to the nearest event would move a span placed deliberately.
   */
  const snap = useCallback((i: number) => {
    if (!snapOn || !cps.length) return i;
    const near = cps.filter((c) => Math.abs(c.index - i) <= SNAP_WINDOW);
    if (!near.length) return i;
    return near.reduce((a, b) => (Math.abs(a.index - i) <= Math.abs(b.index - i) ? a : b)).index;
  }, [cps, snapOn]);

  const endDrag = useCallback(() => {
    if (!dragging.current) return;
    dragging.current = false;
    setDrag((d) => {
      // Both edges snap independently: a span usually starts at one event and ends at another.
      if (d) setPending([snap(Math.min(d[0], d[1])), snap(Math.max(d[0], d[1]))]);
      return null;
    });
  }, [snap]);

  useEffect(() => {
    // On window, not on the lane: releasing outside the lane must still finish the drag, or the next click
    // anywhere extends a span the annotator thought they had already finished.
    window.addEventListener("mouseup", endDrag);
    return () => window.removeEventListener("mouseup", endDrag);
  }, [endDrag]);

  async function create(eventType: string) {
    if (!pending) return;
    const [a, b] = pending;
    setPending(null);
    try {
      await api.createTrackEvent(trackId, {
        event_type: eventType,
        start_frame_id: items[a].frame_id,
        end_frame_id: items[b].frame_id,
      });
      toast(`${eventType} over ${b - a + 1} frames`, "success");
      await load();
      onChanged?.();
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }

  async function setState(eventId: string, state: string) {
    try {
      await api.updateTrackEvent(eventId, { state });
      await load();
      onChanged?.();
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }

  if (!loaded) return null;

  return (
    <section className="panel">
      <div className="flex items-center gap-2 font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
        <span>events ({events.length})</span>
        <span className="text-ink-3/70 normal-case">drag across the lane to mark a span</span>
        {!applicable.length && (
          <span className="ml-auto text-warn normal-case">
            no event types apply to a track of this class
          </span>
        )}
      </div>

      <div className="p-3 space-y-1.5">
        {/* Off is a real choice: 86% of tracks have no changepoint at all, and on the rest an annotator
            may be marking something the speed does not show. */}
        {cps.length > 0 && (
          <label className="flex items-center gap-1.5 font-mono text-[10px] text-ink-3 cursor-pointer">
            <input type="checkbox" checked={snapOn} onChange={(e) => setSnapOn(e.target.checked)} />
            snap span edges to the {cps.length} point{cps.length === 1 ? "" : "s"} where this
            track&apos;s speed changes
          </label>
        )}
        <div className="flex gap-2 overflow-x-auto select-none">
          {items.map((it, i) => {
            const inDrag = drag && i >= Math.min(drag[0], drag[1]) && i <= Math.max(drag[0], drag[1]);
            const cp = cps.find((c) => c.index === i);
            return (
              <button key={it.object_id} aria-label={`mark from frame ${i + 1}`}
                onMouseDown={() => { dragging.current = true; setDrag([i, i]); }}
                onMouseEnter={() => { if (dragging.current) setDrag((d) => (d ? [d[0], i] : null)); }}
                title={cp
                  ? `the speed changes here: ${cp.before.toFixed(0)} to ${cp.after.toFixed(0)} km/h`
                  : undefined}
                className={`relative shrink-0 w-28 h-6 border ${inDrag ? "bg-accent/30 border-accent" : "border-line hover:border-ink-3"}`}>
                {/* Where the motion actually changes. Drawn so an annotator can see what an edge will
                    snap to before dragging, rather than being surprised by it afterwards. */}
                {cp && (
                  <span className={`absolute inset-y-0 left-0 w-0.5 ${snapOn ? "bg-accent" : "bg-ink-3"}`} />
                )}
              </button>
            );
          })}
        </div>

        {/* One row per event. Two events on one track routinely overlap in time, and stacking them into a
            single strip hides whichever is shorter. */}
        {bars.map(({ e, a, span }) => (
          <div key={e.event_id} className="flex gap-2 overflow-x-auto items-center">
            {a > 0 && <div className="shrink-0" style={{ width: widthFor(a) }} />}
            <button
              onClick={() => setSelected(selected === e.event_id ? null : e.event_id)}
              aria-label={`${e.event_type}, ${e.state}`}
              title={`${e.event_type} - ${e.state}${e.source === "heuristic" ? " (heuristic proposal)" : ""}`}
              style={{ width: widthFor(span), ...(e.state === "proposed" ? HATCH : {}) }}
              className={`shrink-0 h-6 border px-1.5 text-left font-mono text-[10px] text-ink truncate ${STATE_TONE[e.state] ?? "bg-line border-line"}`}>
              {e.event_type}
              {e.confidence != null && <span className="ml-1 text-ink-3">{e.confidence.toFixed(2)}</span>}
            </button>
            {selected === e.event_id && (
              <div className="shrink-0 flex items-center gap-1.5 pl-2 font-mono text-[10px]">
                {e.state !== "accepted" && (
                  <button onClick={() => setState(e.event_id, "accepted")}
                    className="text-pass hover:underline">accept</button>
                )}
                {e.state !== "rejected" && (
                  <button onClick={() => setState(e.event_id, "rejected")}
                    className="text-block hover:underline">reject</button>
                )}
                <span className="text-ink-3">{e.source}</span>
                {e.evidence && Object.keys(e.evidence).length > 0 && (
                  // Why it fired, not merely that it did. A proposal a reviewer cannot interrogate is one
                  // they learn to accept without looking.
                  <span className="text-ink-3 truncate max-w-[24rem]">
                    {Object.entries(e.evidence).filter(([k]) => k !== "method")
                      .map(([k, v]) => `${k}=${v}`).join("  ")}
                  </span>
                )}
              </div>
            )}
          </div>
        ))}

        {pending && (
          <div className="border border-line bg-bg-2 p-2 space-y-1">
            <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3">
              frames {pending[0] + 1} to {pending[1] + 1} - what happened?
            </div>
            <div className="flex flex-wrap gap-1 items-center">
              {applicable.map((t) => (
                <button key={t.name} onClick={() => create(t.name)} title={t.definition}
                  className="border border-line px-1.5 py-0.5 font-mono text-[10px] text-ink-2 hover:border-accent hover:text-ink">
                  {t.name}
                </button>
              ))}
              <button onClick={() => setPending(null)}
                className="ml-auto font-mono text-[10px] text-ink-3 hover:text-ink">cancel</button>
            </div>
            {/* The definition is the interface, so it is on hover of every choice rather than something an
                annotator has to go and look up while deciding. */}
            <div className="font-mono text-[10px] text-ink-3">hover a type for its labeling definition</div>
          </div>
        )}
      </div>
    </section>
  );
}
