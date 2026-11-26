"use client";

// The floating cards for agent A's editor tools, drawn over the canvas (top left; the layers panel holds
// the top right). Presentational: the state and the calls are in ./useVideoTools.

import { useEffect, useRef } from "react";

import { getUser } from "@/lib/user";
import type { FanoutGroup } from "@/lib/tools/videoLogic";

import type { VideoTools } from "./useVideoTools";

const REVIEWER_ROLES = new Set(["reviewer", "admin"]);

export default function VideoToolsLayer({
  vt, mode, tool, frameId, classes, onGoto, onTool, keyOf,
}: {
  vt: VideoTools;
  mode: string;
  tool: string;
  frameId: string;
  classes: { id: number; name: string }[];
  onGoto: (frameId: string) => void;
  onTool: (tool: string) => void;
  /** The key a binding answers to now (rebindable), for the hints. */
  keyOf: (bindingId: string) => string;
}) {
  const showTrack = mode === "objects" && (tool === "track-mask" || vt.trackRunning || vt.inLastTrack);
  const showLauncher = mode === "objects" || mode === "review";
  return (
    <>
      {vt.sweepOn && <SweepBar vt={vt} keyOf={keyOf} />}
      <div className="absolute top-14 left-3 z-20 w-[300px] max-w-[calc(100%-230px)] space-y-2 pointer-events-none">
        {showLauncher && !vt.describeOpen && (
          <div className="flex gap-1.5 pointer-events-auto">
            {mode === "objects" && (
              <button type="button" onClick={vt.openDescribe} className="btn text-[11.5px]"
                title="type what you see; YOLO-World and SAM propose it on this frame">
                Describe <span className="kbd">{keyOf("action:describe")}</span>
              </button>
            )}
            <button type="button" onClick={vt.toggleSweep} className={`btn text-[11.5px] ${vt.sweepOn ? "btn-primary" : ""}`}
              title="walk this frame's machine labels one at a time: one key per decision">
              {vt.sweepOn ? "Stop sweep" : "Sweep"} <span className="kbd">{keyOf("action:sweep")}</span>
            </button>
            {mode === "objects" && tool !== "track-mask" && (
              <button type="button" onClick={() => onTool("track-mask")} className="btn text-[11.5px]"
                title="click an object once and SAM 2 follows its mask through the clip">
                Track mask <span className="kbd">{keyOf("tool:track-mask")}</span>
              </button>
            )}
          </div>
        )}
        {showTrack && <TrackCard vt={vt} frameId={frameId} onGoto={onGoto} />}
        {vt.describeOpen && <DescribeCard vt={vt} classes={classes} />}
        {vt.offer && <FanoutCard vt={vt} />}
      </div>
    </>
  );
}

function SweepBar({ vt, keyOf }: { vt: VideoTools; keyOf: (id: string) => string }) {
  const done = vt.sweepTotal - vt.sweepLeft;
  return (
    <div role="status" className="absolute top-3 left-1/2 -translate-x-1/2 z-20 panel px-3 py-1.5 flex items-center gap-3 text-[12px]">
      <span className="label-caps text-accent">Sweep</span>
      <span className="text-ink-2 tabular-nums">{done} of {vt.sweepTotal} decided</span>
      <span className="text-ink-3">
        <span className="kbd">{keyOf("action:reviewAccept")}</span> accept{" "}
        <span className="kbd">{keyOf("action:reviewReject")}</span> reject{" "}
        <span className="kbd">1-9</span> class{" "}
        <span className="kbd">{keyOf("action:sweepSkip")}</span> pass{" "}
        <span className="kbd">]</span> next frame
      </span>
      <button type="button" onClick={vt.toggleSweep} className="text-ink-3 hover:text-ink">stop ({keyOf("action:sweep")})</button>
    </div>
  );
}

function TrackCard({ vt, frameId, onGoto }: { vt: VideoTools; frameId: string; onGoto: (f: string) => void }) {
  const job = vt.trackJob;
  const last = vt.lastTrack;
  return (
    <div className="panel p-2.5 space-y-2 text-[12px] pointer-events-auto" aria-label="Track mask">
      <div className="flex items-center justify-between">
        <span className="section-title">Track mask</span>
        <span className="text-[11px] text-ink-3">SAM 2 video</span>
      </div>
      {vt.trackRunning ? (
        <div className="space-y-1">
          <div className="text-ink-2">{job?.status === "pending" ? "waiting for the GPU worker" : (job?.stage ?? "starting")}</div>
          <div className="h-1.5 bg-line overflow-hidden" role="progressbar" aria-label="tracking progress" aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={Math.round((job?.progress ?? 0) * 100)}>
            <div className="h-full bg-accent transition-[width]" style={{ width: `${Math.round((job?.progress ?? 0) * 100)}%` }} />
          </div>
        </div>
      ) : vt.inLastTrack ? (
        <p className="text-ink-2">
          Click where the mask should be to correct this frame; <span className="text-block">shift-click</span> what is
          not the object. The whole track is re-run with every correction.
        </p>
      ) : (
        <p className="text-ink-2">
          Click the object once. Its mask is followed {vt.span.before} frames back and {vt.span.after} forward,
          as one track sent to review.
        </p>
      )}
      <div className="flex items-center gap-2 text-[11.5px]">
        <label className="flex items-center gap-1 text-ink-3">back
          <input type="number" min={0} max={120} value={vt.span.before} aria-label="frames before"
            onChange={(e) => vt.setSpan((s) => ({ ...s, before: Math.max(0, Math.min(120, Number(e.target.value) || 0)) }))}
            className="field w-14 px-1 py-0.5" />
        </label>
        <label className="flex items-center gap-1 text-ink-3">forward
          <input type="number" min={0} max={120} value={vt.span.after} aria-label="frames after"
            onChange={(e) => vt.setSpan((s) => ({ ...s, after: Math.max(0, Math.min(120, Number(e.target.value) || 0)) }))}
            className="field w-14 px-1 py-0.5" />
        </label>
        <button type="button" onClick={vt.trackSelected} disabled={vt.trackRunning} className="btn text-[11.5px] ml-auto"
          title="track the selected label from its box">track selected</button>
      </div>
      {last && (
        <div className="space-y-1">
          <div className="flex items-center justify-between text-[11px] text-ink-3">
            <span>last track, {last.window_frame_ids.length} frames</span>
            <button type="button" onClick={vt.forgetTrack} className="hover:text-ink">start a new one</button>
          </div>
          {/* The progress strip: one cell per frame of the window, so a gap or a drift can be found and
              clicked to. Grey where SAM 2 did not see the object. */}
          <div className="flex gap-px" aria-label="tracked frames">
            {last.window_frame_ids.map((fid, i) => {
              const empty = last.empty_frame_ids.includes(fid);
              const here = fid === frameId;
              return (
                <button key={fid} type="button" onClick={() => onGoto(fid)}
                  aria-label={`go to tracked frame ${i + 1}`}
                  title={`frame ${i + 1} of ${last.window_frame_ids.length}${empty ? ": not seen" : ""}`}
                  className={`h-3 flex-1 min-w-[3px] ${empty ? "bg-line" : "bg-pass/70"} ${here ? "outline outline-1 outline-accent" : ""}`} />
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function DescribeCard({ vt, classes }: { vt: VideoTools; classes: { id: number; name: string }[] }) {
  const user = getUser();
  const canSession = !user || REVIEWER_ROLES.has(user.role);
  const props = vt.described?.proposals ?? [];
  const nTicked = vt.ticks.filter(Boolean).length;
  const sj = vt.sessionJob;
  // Focused after the keystroke that opened the card has finished: focusing during it (autoFocus) put the
  // `/` itself into the box.
  const inputRef = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const t = setTimeout(() => inputRef.current?.focus(), 0);
    return () => clearTimeout(t);
  }, []);
  // When the phrase is itself a class name ("autorickshaw"), that is the class to file the finds under.
  const asked = vt.described?.phrase;
  const setCls = vt.setDescribeClass;
  useEffect(() => {
    if (!asked) return;
    const norm = asked.trim().toLowerCase().replace(/[\s-]+/g, "_");
    if (classes.some((c) => c.name === norm)) setCls(norm);
  }, [asked, classes, setCls]);
  return (
    <div className="panel p-2.5 space-y-2 text-[12px] pointer-events-auto" aria-label="Describe">
      <div className="flex items-center justify-between">
        <span className="section-title">Describe</span>
        <button type="button" onClick={vt.closeDescribe} className="text-ink-3 hover:text-ink" aria-label="close describe">close</button>
      </div>
      <form onSubmit={(e) => { e.preventDefault(); void vt.runDescribe(); }} className="flex gap-1.5">
        <input ref={inputRef} value={vt.phrase} onChange={(e) => vt.setPhrase(e.target.value)} maxLength={80}
          placeholder="e.g. hand cart, water tanker" aria-label="what to find"
          onKeyDown={(e) => { if (e.key === "Escape") vt.closeDescribe(); }}
          className="field flex-1 px-2 py-1" />
        <button type="submit" disabled={vt.describing || !vt.phrase.trim()} className="btn btn-primary text-[11.5px]">
          {vt.describing ? "finding" : "find"}
        </button>
      </form>
      <label className="flex items-center gap-1.5 text-ink-3">file under
        <select value={vt.describeClass ?? ""} onChange={(e) => vt.setDescribeClass(e.target.value || null)}
          aria-label="class for the described objects" className="field flex-1 px-1 py-0.5">
          <option value="">choose a class</option>
          {classes.map((c) => <option key={c.id} value={c.name}>{c.name}</option>)}
        </select>
      </label>
      {vt.described && (
        props.length ? (
          <div className="space-y-1">
            <ul className="max-h-[180px] overflow-y-auto space-y-0.5">
              {props.map((p, i) => (
                <li key={i}>
                  <label className="flex items-center gap-2">
                    <input type="checkbox" checked={!!vt.ticks[i]}
                      onChange={() => vt.setTicks((t) => t.map((v, j) => (j === i ? !v : v)))} />
                    <span className="tabular-nums text-ink-2">{p.conf.toFixed(2)}</span>
                    <span className="text-ink-3 truncate">{Math.round(p.bbox[2] - p.bbox[0])} x {Math.round(p.bbox[3] - p.bbox[1])} px{p.polygons.length ? "" : ", box only"}</span>
                  </label>
                </li>
              ))}
            </ul>
            <p className="text-[11px] text-ink-3">{vt.described.note}</p>
            <button type="button" onClick={() => void vt.acceptDescribe()} disabled={!nTicked || !vt.describeClass}
              className="btn btn-primary w-full text-[11.5px]">accept {nTicked} as {vt.describeClass ?? "..."}</button>
          </div>
        ) : <p className="text-ink-3">nothing on this frame matched &ldquo;{vt.described.phrase}&rdquo;.</p>
      )}
      {canSession && (
        <div className="border-t hairline pt-2 space-y-1">
          <button type="button" onClick={() => void vt.describeWholeSession()}
            disabled={!vt.phrase.trim() || !vt.describeClass || (!!sj && !["done", "error", "canceled"].includes(sj.status))}
            className="btn w-full text-[11.5px]" title="run the phrase over every frame of this drive; finds go to review">
            find on every frame of this drive
          </button>
          {sj && (
            <div className="text-[11px] text-ink-3">
              {sj.status === "pending"
                ? (sj.quiet_hours ? "queued: auto-labelling waits until 18:00 IST (office hours)" : "queued for the GPU worker")
                : sj.status === "running" ? `${sj.stage ?? "running"} (${Math.round(sj.progress * 100)}%)`
                : sj.status === "error" ? `failed: ${sj.error ?? ""}` : sj.status}
            </div>
          )}
          <p className="text-[11px] text-ink-3">YOLO-World and SAM only: the Florence-2 second opinion is not installed here.</p>
        </div>
      )}
    </div>
  );
}

const GROUPS: { key: FanoutGroup; label: string; hint: string }[] = [
  { key: "track", label: "rest of the track", hint: "the same object on other frames" },
  { key: "lookalikes", label: "look-alikes", hint: "visually similar objects anywhere (DINOv3)" },
  { key: "confusion", label: "same confusion", hint: "machine labels in this drive whose own model votes said this class" },
];

function FanoutCard({ vt }: { vt: VideoTools }) {
  const p = vt.preview;
  const offer = vt.offer!;
  const n = vt.fanTicks.track.length + vt.fanTicks.lookalikes.length + vt.fanTicks.confusion.length;
  return (
    <div className="panel p-2.5 space-y-2 text-[12px] pointer-events-auto" aria-label="Apply the correction elsewhere">
      <div className="flex items-center justify-between">
        <span className="section-title">Same mistake elsewhere?</span>
        <button type="button" onClick={vt.dismissFanout} className="text-ink-3 hover:text-ink">dismiss</button>
      </div>
      <div className="text-ink-2">{offer.from} &rarr; <span className="text-ink">{offer.to}</span></div>
      {!p ? <div className="text-ink-3" role="status">looking...</div> : (
        <>
          {GROUPS.map((g) => {
            const rows = g.key === "track" ? p.track.objects : g.key === "lookalikes" ? p.lookalikes.objects : p.confusion.objects;
            const ticked = vt.fanTicks[g.key];
            return (
              <div key={g.key} className="space-y-1">
                <div className="flex items-center justify-between">
                  <span className="label-caps" title={g.hint}>{g.label}</span>
                  <span className="text-ink-3 tabular-nums">{ticked.length} of {rows.length}</span>
                </div>
                {g.key === "lookalikes" && !rows.length && p.lookalikes.reason && (
                  <div className="text-[11px] text-ink-3">{p.lookalikes.reason}</div>
                )}
                {rows.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {rows.slice(0, 24).map((r) => {
                      const on = ticked.includes(r.object_id);
                      return (
                        // eslint-disable-next-line @next/next/no-img-element -- an authenticated crop, not a static asset
                        <button key={r.object_id} type="button" disabled={!r.selectable}
                          onClick={() => vt.tickFanout(g.key, r.object_id)}
                          title={`${r.class_name}, ${r.state}${r.human ? ", a person's label: left alone" : ""}${r.already ? ", already this class" : ""}`}
                          className={`w-10 h-10 border overflow-hidden ${on ? "border-accent" : "border-line opacity-50"} disabled:opacity-30`}>
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={r.crop_url} alt={r.class_name} className="w-full h-full object-cover" loading="lazy" />
                        </button>
                      );
                    })}
                    {rows.length > 24 && <span className="text-[11px] text-ink-3 self-end">+{rows.length - 24}</span>}
                  </div>
                )}
              </div>
            );
          })}
          <button type="button" onClick={() => void vt.applyFanout()} disabled={!n || vt.fanBusy}
            className="btn btn-primary w-full text-[11.5px]">
            {vt.fanBusy ? "applying" : `make ${n} more ${offer.to}`}
          </button>
          <p className="text-[11px] text-ink-3">One batch: a single undo takes all of it back. A person&apos;s label is never changed.</p>
        </>
      )}
    </div>
  );
}
