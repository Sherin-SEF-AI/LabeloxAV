"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api, humanizeError } from "@/lib/api";
import type { Track, TrackItem, Tracklet } from "@/lib/types";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import Inspector from "@/components/shell/Inspector";
import { StateBadge } from "@/components/StateBadge";
import LoadState from "@/components/shell/LoadState";
import { Stat } from "@/components/ui/kit";
import { humanize, num, shortId } from "@/lib/format";

// Ontology class names read as words ("two wheeler") but keep their case: "suv" is not "Suv".
const classLabel = (n: string | null | undefined) => (n ?? "").replace(/_/g, " ");

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
  const [loadErr, setLoadErr] = useState<unknown>(null);

  const load = useCallback(async () => {
    let t: Track;
    try {
      t = await api.track(trackId);
    } catch (e) {
      setLoadErr(e);
      return;
    }
    setLoadErr(null);
    setTrack(t);
    setSel((s) => Math.min(s, t.items.length - 1));
    // The two keyframe-economy calls, which had no consumer anywhere in the app. Failures are said out
    // loud: a silently absent budget looks identical to a track where no frame is worth correcting.
    api.tracklet(trackId).then(setStats)
      .catch((e) => setMsg(`Keyframe stats unavailable: ${humanizeError(e)}`));
    api.suggestKeyframes(trackId)
      .then((r) => setSuggested(new Set(r.suggestions.map((s) => s.object_id))))
      .catch((e) => setMsg(`Keyframe suggestions unavailable: ${humanizeError(e)}`));
  }, [trackId]);

  useEffect(() => { load(); }, [load]);

  const it = track?.items[sel];

  const interpolate = async () => { const r = await api.interpolateKeyframed(trackId, method); setMsg(`Interpolated ${r.created} frames (${r.method}, ${r.keyframes} keyframes)`); await load(); };
  const markKeyframe = async () => { if (!it) return; await api.setKeyframe(it.object_id, !(it.is_keyframe || it.source === "human")); setMsg("Keyframe set. Open the frame to adjust its box, then re-interpolate."); await load(); };
  const reinterpolate = async () => { if (!it) return; const r = await api.reinterpolate(it.object_id, method); setMsg(`Re-interpolated ${r.created} frames in the adjacent segments`); await load(); };

  /**
   * Ask which machine-filled boxes have slid off their anchor.
   *
   * On demand rather than on load: it encodes one crop per box on a GPU that a training job may hold, so
   * running it every time somebody opened a track would put the page in a queue behind training.
   */
  const checkDrift = async () => {
    setDriftBusy(true);
    setMsg("Checking appearance against each box's nearest anchor...");
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
      setMsg(`Drift check failed: ${humanizeError(e)}`);
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
  const isKey = (x: TrackItem) => !!(x.is_keyframe || x.source === "human");
  // Messages that report a failure read as a warning; progress and results read as plain status.
  const msgIsBad = !!msg && /unavailable|failed/i.test(msg);

  return (
    <div className="min-h-screen flex flex-col">
      <PageHeaderBar
        title="Timeline"
        subtitle={track ? `${classLabel(track.dominant)} track · ${shortId(trackId)}` : shortId(trackId)}
        meta={
          <>
            {/* Same: with no history, the track this timeline is of is the right destination, not "/". */}
            <BackButton fallback={`/track/${trackId}`} label="Track" />
            <span className="ml-2">Interpolation</span>
            <span className="inline-flex" role="group" aria-label="interpolation method">
              {(["linear", "cubic"] as const).map((m) => (
                <button key={m} type="button" onClick={() => setMethod(m)} aria-pressed={method === m}
                  className={`btn ${method === m ? "btn-on" : ""} ${m === "linear" ? "rounded-r-none" : "rounded-l-none -ml-px"}`}>
                  {m === "linear" ? "Linear" : "Cubic"}
                </button>
              ))}
            </span>
          </>
        }
        right={msg && <span className={`max-w-[520px] truncate ${msgIsBad ? "text-warn" : "text-ink-2"}`} title={msg}>{msg}</span>}
        primaryAction={
          <span className="flex items-center gap-2">
            <button type="button" onClick={checkDrift} disabled={driftBusy}
              title="compare each machine-filled box to its nearest anchor crop (uses the GPU, yields to training)"
              className="btn">
              {driftBusy ? "Checking..." : "Check drift"}
            </button>
            <button type="button" onClick={interpolate} className="btn btn-primary">Interpolate between keyframes</button>
          </span>
        }
      />

      <div className="flex-1 flex min-h-0">
        <main className="flex-1 overflow-auto p-4 space-y-4 min-w-0">
          {(loadErr || !track) && (
            <LoadState loading={!track && !loadErr} error={loadErr} rows={4} onRetry={() => { setLoadErr(null); load(); }} />
          )}

          {track && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <Stat label="Frames" value={num(track.n_frames)} sub={classLabel(track.dominant)} />
              <Stat label="Keyframes" value={num(kfCount)} sub="drawn or confirmed by a person" tone={kfCount ? "pass" : undefined} />
              <Stat label="Interpolated" value={num(interpCount)} sub="filled between keyframes" />
              {/* The economy figure. Median across the corpus is one touched frame per 93-frame track, so
                  this number starts very high and coming down is the work. */}
              <Stat label="Frames per keyframe" value={perKeyframe != null ? perKeyframe.toFixed(1) : "-"}
                tone={perKeyframe == null ? undefined : perKeyframe > 40 ? "warn" : "accent"}
                sub="what one correction is worth here; lower is better" />
            </div>
          )}

          {/* scrubber */}
          {track && (
          <section className="panel">
          <div className="panel-head">
            <h2 className="section-title">Where each box came from</h2>
            <span className="ml-auto text-[12px] text-ink-3">Click a frame to inspect it</span>
          </div>
          <div className="p-3.5">
          <div className="flex items-end gap-px h-24 overflow-x-auto pt-2">
            {track.items.map((x, i) => (
              <button key={x.object_id} type="button" onClick={() => setSel(i)}
                aria-label={`frame ${i + 1}: ${x.source ?? "detected"}${isKey(x) ? ", keyframe" : ""}`}
                aria-pressed={i === sel}
                title={`frame ${i + 1}: ${x.source}${x.is_keyframe ? " (keyframe)" : ""}`
                       + `${suggested.has(x.object_id) ? " - correcting this frame buys the most" : ""}`}
                style={{ background: cellColor(x), height: isKey(x) ? "100%" : "60%", outline: i === sel ? "2px solid rgb(var(--c-ink))" : "none" }}
                className="flex-1 min-w-[6px] max-w-[24px] shrink-0 relative rounded-[1px]">
                {/* Where the next correction is worth the most, from the curvature of the trajectory.
                    Computed by suggest_keyframes since it was written and never rendered anywhere. */}
                {suggested.has(x.object_id) && (
                  <span className="absolute -top-3 left-0 w-full text-center text-[11px] leading-none text-accent" aria-hidden>▾</span>
                )}
              </button>
            ))}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-[12px] text-ink-3">
            <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: "#56D364" }} />Human keyframe</span>
            <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: "#E3B341" }} />Interpolated</span>
            <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: "#58A6FF" }} />Propagated</span>
            <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: "#6C727A" }} />Detected</span>
            {suggested.size > 0 && <span className="text-accent">▾ Correcting here buys the most frames</span>}
          </div>

          {/* drift lane: a second view over the same columns, answering a different question */}
          {drift ? (
            <div className="mt-4 pt-3 border-t hairline">
              <div className="section-title mb-2">Still on the object?</div>
              <div className="flex items-end gap-px h-5 overflow-x-auto">
                {track.items.map((x, i) => {
                  const d = drift[x.object_id];
                  return (
                    <button key={x.object_id} type="button" onClick={() => setSel(i)} aria-pressed={i === sel}
                      aria-label={`frame ${i + 1}: ${d ? `${d.verdict}, ${d.why}` : "an anchor, not machine fill"}`}
                      title={d ? `${d.verdict}: ${d.why}` : "an anchor, not machine fill: nothing to check"}
                      style={{ background: d ? DRIFT_COLOR[d.verdict] : "transparent",
                               outline: i === sel ? "2px solid rgb(var(--c-ink))" : "none" }}
                      className="flex-1 min-w-[6px] max-w-[24px] h-full shrink-0 rounded-[1px]" />
                  );
                })}
              </div>
              <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[12px] text-ink-3">
                <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: DRIFT_COLOR.drifted }} />Drifted{driftCounts ? ` (${driftCounts.drifted})` : ""}</span>
                <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: DRIFT_COLOR.ok }} />Still on it{driftCounts ? ` (${driftCounts.ok})` : ""}</span>
                {/* Never folded into "drifted". Unmeasurable is a different answer from wrong, and
                    painting it red would tell an annotator that good boxes are bad because a GPU is busy. */}
                <span><span className="inline-block w-2.5 h-2.5 mr-1.5 rounded-sm align-[-1px]" style={{ background: DRIFT_COLOR.unknown }} />Not measurable{driftCounts ? ` (${driftCounts.unknown})` : ""}</span>
                <span className="text-ink-3">Anchors are blank: there is nothing to check on a box somebody drew.</span>
              </div>
            </div>
          ) : (
            <p className="section-desc mt-4 pt-3 border-t hairline">
              Check drift compares every machine-filled box with its nearest keyframe and adds a lane showing
              which boxes have slid off the object.
            </p>
          )}
          </div>
        </section>
          )}

          {track && (
            <details className="panel px-3.5 py-2.5 text-[12.5px] text-ink-2">
              <summary className="cursor-pointer text-ink-2 hover:text-ink">How this works</summary>
              <div className="mt-2 space-y-1.5 section-desc">
                <p>A keyframe is a frame where a person drew or confirmed the box. Every other frame is filled by
                  machine: propagated from a neighbour, or interpolated between two keyframes.</p>
                <p>Correcting a frame and marking it as a keyframe re-fills the segments on either side, so one
                  correction fixes many frames. The ▾ markers show where the trajectory turns hardest, which is where
                  interpolation is furthest from the truth.</p>
              </div>
            </details>
          )}
        </main>

        {/* selected frame */}
        {it && (
          <Inspector title="Selected frame" side="right">
            <div className="p-3 flex flex-col gap-4">
              <button type="button" className="block w-full" aria-label={`open this ${it.class_name} in the frame editor`}
                onClick={() => router.push(`/frame/${it.frame_id}?focus=${it.object_id}`)}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={it.crop_url} alt="" className="w-full h-32 object-cover bg-bg-2 border border-line cursor-pointer" />
              </button>
              <div className="text-[12.5px] space-y-1.5">
                <div className="text-ink">{classLabel(it.class_name)} <span className="text-ink-3">· frame {sel + 1} of {track?.n_frames}</span></div>
                <div className="text-ink-3">Source: <span style={{ color: cellColor(it) }}>{humanize(it.source ?? "detected")}{it.interp_source ? ` (${humanize(it.interp_source)})` : ""}</span></div>
                <div className="text-ink-3 flex items-center gap-1.5">State: <StateBadge state={it.state} /></div>
                {selDrift && (
                  <div className="text-ink-3">
                    Appearance: <span style={{ color: DRIFT_COLOR[selDrift.verdict] }}>{selDrift.why}</span>
                  </div>
                )}
                {suggested.has(it.object_id) && (
                  <div className="text-accent">
                    Correcting this frame buys the most: the box turns hardest here, so the segments either
                    side are where interpolation is furthest from the truth.
                  </div>
                )}
                <div className="flex flex-wrap gap-2 pt-2">
                  <button type="button" onClick={() => router.push(`/frame/${it.frame_id}?focus=${it.object_id}`)} className="btn btn-primary">Edit box</button>
                  <button type="button" onClick={markKeyframe} className={`btn ${isKey(it) ? "btn-on" : ""}`}>{isKey(it) ? "Keyframe ✓" : "Mark keyframe"}</button>
                  <button type="button" onClick={reinterpolate} className="btn">Re-interpolate segment</button>
                </div>
              </div>
            </div>
          </Inspector>
        )}
      </div>
    </div>
  );
}
