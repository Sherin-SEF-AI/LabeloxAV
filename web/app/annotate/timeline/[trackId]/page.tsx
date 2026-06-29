"use client";

import { useCallback, useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { Track, TrackItem } from "@/lib/types";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import Inspector from "@/components/shell/Inspector";
import { StateBadge } from "@/components/StateBadge";

// M2.5 keyframe + interpolation video workspace: a track scrubber with keyframe markers, interpolated
// frames shown distinctly from human ones, and one-action edit-propagation across a segment.

function cellColor(it: TrackItem): string {
  if (it.is_keyframe || it.source === "human") return "#56D364"; // human keyframe
  if (it.source === "interpolated") return "#E3B341";            // machine-filled (interpolated)
  if (it.source === "propagated") return "#58A6FF";              // propagated
  return "#6C727A";                                              // detected
}

export default function TimelineWorkspace() {
  const router = useRouter();
  const trackId = String(useParams().trackId);
  const [track, setTrack] = useState<Track | null>(null);
  const [sel, setSel] = useState<number>(0);
  const [method, setMethod] = useState<"linear" | "cubic">("linear");
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    const t = await api.track(trackId);
    setTrack(t);
    setSel((s) => Math.min(s, t.items.length - 1));
  }, [trackId]);

  useEffect(() => { load(); }, [load]);

  const it = track?.items[sel];

  const interpolate = async () => { const r = await api.interpolateKeyframed(trackId, method); setMsg(`interpolated ${r.created} frames (${r.method}, ${r.keyframes} keyframes)`); await load(); };
  const markKeyframe = async () => { if (!it) return; await api.setKeyframe(it.object_id, !(it.is_keyframe || it.source === "human")); setMsg("keyframe set - open the frame to adjust its box, then re-interpolate"); await load(); };
  const reinterpolate = async () => { if (!it) return; const r = await api.reinterpolate(it.object_id, method); setMsg(`re-interpolated ${r.created} frames in the adjacent segments`); await load(); };

  const kfCount = track?.items.filter((x) => x.is_keyframe || x.source === "human").length ?? 0;
  const interpCount = track?.items.filter((x) => x.source === "interpolated").length ?? 0;

  return (
    <div className="min-h-screen flex flex-col">
      <PageHeaderBar
        title="Timeline"
        subtitle={trackId.slice(0, 8)}
        meta={
          <>
            <BackButton />
            {track && <span>{track.n_frames} frames · {kfCount} keyframes · {interpCount} interpolated</span>}
            <span className="ml-1">method:</span>
            {(["linear", "cubic"] as const).map((m) => (
              <button key={m} onClick={() => setMethod(m)} className={`border px-2 py-1 ${method === m ? "border-accent text-accent" : "border-line text-ink-3"}`}>{m}</button>
            ))}
          </>
        }
        right={msg && <span className="text-warn">{msg}</span>}
        primaryAction={
          <button onClick={interpolate} className="border border-line px-2 py-1 hover:border-accent">interpolate between keyframes</button>
        }
      />

      <div className="flex-1 flex min-h-0">
        <main className="flex-1 overflow-auto p-4 space-y-4 min-w-0">
          {/* scrubber */}
          <div className="panel p-3">
          <div className="flex items-end gap-px h-12 overflow-x-auto">
            {track?.items.map((x, i) => (
              <button key={x.object_id} onClick={() => setSel(i)}
                title={`${x.source}${x.is_keyframe ? " (keyframe)" : ""}`}
                style={{ background: cellColor(x), height: x.is_keyframe || x.source === "human" ? "100%" : "60%", outline: i === sel ? "2px solid #E7E9EB" : "none" }}
                className="w-2 shrink-0" />
            ))}
          </div>
          <div className="flex gap-4 mt-2 font-mono text-[10px] text-ink-3">
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#56D364" }} />human keyframe</span>
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#E3B341" }} />interpolated</span>
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#58A6FF" }} />propagated</span>
            <span><span className="inline-block w-2 h-2 mr-1" style={{ background: "#6C727A" }} />detected</span>
          </div>
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
