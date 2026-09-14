"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import type { Track, TrackItem, Tracklet } from "@/lib/types";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import Inspector from "@/components/shell/Inspector";
import { StateBadge } from "@/components/StateBadge";

// M2.5 keyframe + interpolation video workspace: a track scrubber with keyframe markers, interpolated
// frames shown distinctly from human ones, and one-action edit-propagation across a segment.
//
// Keyframe economy is the thing this page is for and the numbers that measure it were computed and shown
// nowhere. `services/temporal/tracklet.py` has produced `frames_per_keyframe` and `suggest_keyframes` for
// a long time, both exposed as endpoints with client bindings, and a grep across the whole web tree found
// no consumer for either. So an annotator could not see how many frames one correction was covering, and
// had no way to know which frame to correct next.
//
// The second half is the drift lane. The scrubber coloured by `source`, which says where a box came from
// and nothing about whether it is still on its object. That question had an answer inside
// propagate_agent's walk and only at creation time. Measured once it could be asked of the corpus: 64.6%
// of machine fill has drifted, against 32.4% of detector output judged the same way, so it discriminates
// and it is worth a lane of its own.

function cellColor(it: TrackItem): string {
  if (it.is_keyframe || it.source === "human") return "#56D364"; // human keyframe
  if (it.source === "interpolated") return "#E3B341";            // machine-filled (interpolated)
  if (it.source === "propagated") return "#58A6FF";              // propagated
  return "#6C727A";                                              // detected
}

// Drift is its own lane rather than a recolouring of the source lane. They answer different questions and
// an annotator needs both: where the box came from, and whether it is still on the thing.
const DRIFT_COLOR: Record<string, string> = {
  drifted: "#F85149",   // fell through the appearance floor, or left the anchor's size band
  ok: "#56D364",
  unknown: "#3D444D",   // could not be measured, which is not the same as fine
};

type DriftRow = {
  object_id: string; verdict: "drifted" | "ok" | "unknown"; similarity: number | null; why: string;
};

export default function TimelineWorkspace() {
  const router = useRouter();
  const trackId = String(useParams().trackId);
  const [track, setTrack] = useState<Track | null>(null);
  const [stats, setStats] = useState<Tracklet | null>(null);
  const [suggested, setSuggested] = useState<Set<string>>(new Set());
  const [drift, setDrift] = useState<Record<string, DriftRow> | null>(null);
  const [driftCounts, setDriftCounts] = useState<{ drifted: number; ok: number; unknown: number } | null>(null);
  const [driftBusy, setDriftBusy] = useState(false);
  const [sel, setSel] = useState<number>(0);
  const [method, setMethod] = useState<"linear" | "cubic">("linear");
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    const t = await api.track(trackId);
    setTrack(t);
    setSel((s) => Math.min(s, t.items.length - 1));
    // The two keyframe-economy calls, which had no consumer anywhere in the app. Failures are said out
    // loud: a silently absent budget looks identical to a track where no frame is worth correcting.
    api.tracklet(trackId).then(setStats)
      .catch((e) => setMsg(`keyframe stats unavailable: ${humanizeError(e)}`));
    api.suggestKeyframes(trackId)
      .then((r) => setSuggested(new Set(r.suggestions.map((s) => s.object_id))))
      .catch((e) => setMsg(`keyframe suggestions unavailable: ${humanizeError(e)}`));
  }, [trackId]);

  useEffect(() => { load(); }, [load]);

  const it = track?.items[sel];

  const interpolate = async () => { const r = await api.interpolateKeyframed(trackId, method); setMsg(`interpolated ${r.created} frames (${r.method}, ${r.keyframes} keyframes)`); await load(); };
  const markKeyframe = async () => { if (!it) return; await api.setKeyframe(it.object_id, !(it.is_keyframe || it.source === "human")); setMsg("keyframe set - open the frame to adjust its box, then re-interpolate"); await load(); };
  const reinterpolate = async () => { if (!it) return; const r = await api.reinterpolate(it.object_id, method); setMsg(`re-interpolated ${r.created} frames in the adjacent segments`); await load(); };

  /**
   * Ask which machine-filled boxes have slid off their anchor.
   *
   * On demand rather than on load: it encodes one crop per box on a GPU that a training job may hold, so
   * running it every time somebody opened a track would put the page in a queue behind training.
   */
  const checkDrift = async () => {
    setDriftBusy(true);
    setMsg("checking appearance against each box's nearest anchor...");
    try {
      const r = await api.trackDrift(trackId);
      const by: Record<string, DriftRow> = {};
      for (const row of r.rows) by[row.object_id] = row;
      setDrift(by);
      setDriftCounts(r.counts ?? null);
      setMsg(r.reason
        ? r.reason
        : `${r.counts?.drifted ?? 0} of ${r.checked} machine-filled boxes have drifted`
          + `${r.counts?.unknown ? `, ${r.counts.unknown} could not be measured` : ""}`);
    } catch (e) {
      setMsg(`drift check failed: ${humanizeError(e)}`);
    } finally {
      setDriftBusy(false);
    }
  };

  const kfCount = track?.items.filter((x) => x.is_keyframe || x.source === "human").length ?? 0;
  const interpCount = track?.items.filter((x) => x.source === "interpolated").length ?? 0;

  // What one correction is currently buying. The number an annotator is optimising, and the reason the
  // suggestions below are worth following.
  const perKeyframe = useMemo(() => {
    if (stats?.frames_per_keyframe != null) return stats.frames_per_keyframe;
    if (!track || !kfCount) return null;
    return track.n_frames / kfCount;
  }, [stats, track, kfCount]);

  const selDrift = it ? drift?.[it.object_id] : undefined;

  return (
    <div className="min-h-screen flex flex-col">
      <PageHeaderBar
        title="Timeline"
        subtitle={trackId.slice(0, 8)}
        meta={
          <>
            {/* Same: with no history, the track this timeline is of is the right destination, not "/". */}
            <BackButton fallback={`/track/${trackId}`} label="track" />
            {track && <span>{track.n_frames} frames · {kfCount} keyframes · {interpCount} interpolated</span>}
            {/* The economy figure. Median across the corpus is one touched frame per 93-frame track, so
                this number starts very high and coming down is the work. */}
            {perKeyframe != null && (
              <span className={perKeyframe > 40 ? "text-warn" : "text-accent"}
                title="frames covered by each keyframe. One correction propagates across its segment, so this is what a correction is worth here.">
                {perKeyframe.toFixed(1)} frames per keyframe
              </span>
            )}
            <span className="ml-1">method:</span>
            {(["linear", "cubic"] as const).map((m) => (
              <button key={m} onClick={() => setMethod(m)} className={`border px-2 py-1 ${method === m ? "border-accent text-accent" : "border-line text-ink-3"}`}>{m}</button>
            ))}
          </>
        }
        right={msg && <span className="text-warn">{msg}</span>}
        primaryAction={
          <span className="flex items-center gap-2">
            <button onClick={checkDrift} disabled={driftBusy}
              title="compare each machine-filled box to its nearest anchor crop (uses the GPU, yields to training)"
              className="border border-line px-2 py-1 hover:border-accent disabled:opacity-40">
              {driftBusy ? "checking..." : "check drift"}
            </button>
            <button onClick={interpolate} className="border border-line px-2 py-1 hover:border-accent">interpolate between keyframes</button>
          </span>
        }
      />

      <div className="flex-1 flex min-h-0">
        <main className="flex-1 overflow-auto p-4 space-y-4 min-w-0">
          {/* scrubber */}
          <div className="panel p-3">
          <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">source</div>
          <div className="flex items-end gap-px h-12 overflow-x-auto">
            {track?.items.map((x, i) => (
              <button key={x.object_id} onClick={() => setSel(i)}
                title={`${x.source}${x.is_keyframe ? " (keyframe)" : ""}`
                       + `${suggested.has(x.object_id) ? " - correcting this frame buys the most" : ""}`}
                style={{ background: cellColor(x), height: x.is_keyframe || x.source === "human" ? "100%" : "60%", outline: i === sel ? "2px solid #E7E9EB" : "none" }}
                className="w-2 shrink-0 relative">
                {/* Where the next correction is worth the most, from the curvature of the trajectory.
                    Computed by suggest_keyframes since it was written and never rendered anywhere. */}
                {suggested.has(x.object_id) && (
                  <span className="absolute -top-1.5 left-0 w-full text-center text-[8px] leading-none text-accent">▾</span>
                )}
              </button>
            ))}
          </div>
          <div className="flex gap-4 mt-2 font-mono text-[10px] text-ink-3">
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#56D364" }} />human keyframe</span>
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#E3B341" }} />interpolated</span>
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#58A6FF" }} />propagated</span>
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#6C727A" }} />detected</span>
            {suggested.size > 0 && <span className="text-accent">▾ correcting here buys the most frames</span>}
          </div>

          {/* drift lane: a second view over the same columns, answering a different question */}
          {drift && (
            <div className="mt-3 pt-3 border-t hairline">
              <div className="font-mono text-[10px] uppercase text-ink-3 mb-1">
                still on the object?
              </div>
              <div className="flex items-end gap-px h-4 overflow-x-auto">
                {track?.items.map((x, i) => {
                  const d = drift[x.object_id];
                  return (
                    <button key={x.object_id} onClick={() => setSel(i)}
                      title={d ? `${d.verdict}: ${d.why}` : "an anchor, not machine fill: nothing to check"}
                      style={{ background: d ? DRIFT_COLOR[d.verdict] : "transparent",
                               outline: i === sel ? "2px solid #E7E9EB" : "none" }}
                      className="w-2 h-full shrink-0" />
                  );
                })}
              </div>
              <div className="flex gap-4 mt-2 font-mono text-[10px] text-ink-3">
                <span><span className="inline-block w-2 h-2 mr-1" style={{ background: DRIFT_COLOR.drifted }} />drifted{driftCounts ? ` (${driftCounts.drifted})` : ""}</span>
                <span><span className="inline-block w-2 h-2 mr-1" style={{ background: DRIFT_COLOR.ok }} />still on it{driftCounts ? ` (${driftCounts.ok})` : ""}</span>
                {/* Never folded into "drifted". Unmeasurable is a different answer from wrong, and
                    painting it red would tell an annotator that good boxes are bad because a GPU is busy. */}
                <span><span className="inline-block w-2 h-2 mr-1" style={{ background: DRIFT_COLOR.unknown }} />not measurable{driftCounts ? ` (${driftCounts.unknown})` : ""}</span>
                <span className="text-ink-3">anchors are blank: there is nothing to check on a box somebody drew</span>
              </div>
            </div>
          )}
        </div>
        </main>

        {/* selected frame */}
        {it && (
          <Inspector title="frame" side="right">
            <div className="p-3 flex flex-col gap-4">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={it.crop_url} alt="" className="w-full h-32 object-cover bg-bg-2 border border-line cursor-pointer" onClick={() => router.push(`/frame/${it.frame_id}?focus=${it.object_id}`)} />
              <div className="font-mono text-[11px] space-y-1">
                <div className="text-ink-2">{it.class_name} <span className="text-ink-3">frame {sel + 1}/{track?.n_frames}</span></div>
                <div className="text-ink-3">source: <span style={{ color: cellColor(it) }}>{it.source}{it.interp_source ? ` (${it.interp_source})` : ""}</span></div>
                <div className="text-ink-3">state: <StateBadge state={it.state} /></div>
                {selDrift && (
                  <div className="text-ink-3">
                    appearance: <span style={{ color: DRIFT_COLOR[selDrift.verdict] }}>{selDrift.why}</span>
                  </div>
                )}
                {suggested.has(it.object_id) && (
                  <div className="text-accent">
                    correcting this frame buys the most: the box turns hardest here, so the segments either
                    side are where interpolation is furthest from the truth
                  </div>
                )}
                <div className="flex flex-wrap gap-2 pt-2">
                  <button onClick={markKeyframe} className={`border px-2 py-1 ${it.is_keyframe || it.source === "human" ? "border-pass text-pass" : "border-line hover:border-accent"}`}>{it.is_keyframe || it.source === "human" ? "keyframe ✓" : "mark keyframe"}</button>
                  <button onClick={reinterpolate} className="border border-line px-2 py-1 hover:border-accent">re-interpolate segment</button>
                  <button onClick={() => router.push(`/frame/${it.frame_id}?focus=${it.object_id}`)} className="border border-line px-2 py-1 hover:border-accent">edit box</button>
                </div>
              </div>
            </div>
          </Inspector>
        )}
      </div>
    </div>
  );
}
