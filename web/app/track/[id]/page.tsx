"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { Ontology, Track } from "@/lib/types";
import { classColor } from "@/lib/colors";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";

// Tracklet editor: scan a track across frames as a strip, spot class flips (the cells that disagree
// with the dominant class glow red), and fix the whole track in one action. One relabel corrects every
// frame. Click a crop to jump into that frame in the editor.

export default function TrackEditor() {
  const router = useRouter();
  const { id } = useParams<{ id: string }>();
  const [track, setTrack] = useState<Track | null>(null);
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const load = useCallback(async () => {
    const [t, o] = await Promise.all([api.track(id), api.ontology()]);
    setTrack(t);
    setOnto(o);
  }, [id]);

  useEffect(() => {
    load().catch((e) => setMsg(String(e)));
  }, [load]);

  const relabel = useCallback(
    async (className: string) => {
      setBusy(true);
      try {
        const r = await api.relabelTrack(id, className);
        setMsg(`relabeled ${r.relabeled} frames to ${className}`);
        await load();
      } catch (e) {
        setMsg(String(e));
      } finally {
        setBusy(false);
      }
    },
    [id, load],
  );

  const onDelete = useCallback(async () => {
    if (!confirm("Delete this entire track (all its objects)?")) return;
    setBusy(true);
    try {
      await api.deleteTrack(id);
      router.push("/");
    } catch (e) {
      setMsg(String(e));
      setBusy(false);
    }
  }, [id, router]);

  const onInterpolate = useCallback(async () => {
    setBusy(true);
    try {
      const r = await api.interpolateTrack(id);
      setMsg(r.created ? `interpolated ${r.created} gap frames (confirm them in the editor)` : "no gaps to fill");
      await load();
    } catch (e) {
      setMsg(String(e));
    } finally {
      setBusy(false);
    }
  }, [id, load]);

  const filtered = useMemo(
    () => (onto ? onto.classes.filter((c) => c.name.includes(search.toLowerCase().replace(/\s/g, "_"))) : []),
    [onto, search],
  );

  if (!track) return <div className="min-h-screen flex items-center justify-center font-mono text-ink-3">{msg ?? "loading track..."}</div>;

  return (
    <div className="min-h-screen flex flex-col">
      <div className="flex items-center gap-3 px-4 h-11 border-b hairline shrink-0">
        <BackButton />
        <button onClick={() => router.push("/")} className="font-display font-bold" title="home (triage)">
          Labelox<span className="text-accent">AV</span>
        </button>
        <span className="font-mono text-xs text-ink-3">/ TRACK <span className="text-ink-2">{id.slice(0, 8)}</span></span>
      </div>

      <PageHeaderBar
        title="Track"
        subtitle={id.slice(0, 8)}
        meta={
          <>
            <span className="text-ink-3">{track.n_frames} frames</span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 inline-block" style={{ background: classColor(track.items[0]?.class_id ?? 0) }} />
              {track.dominant}
            </span>
            {track.flips ? (
              <span className="text-block">class flips: {Object.keys(track.classes).length} classes</span>
            ) : (
              <span className="text-pass">consistent</span>
            )}
          </>
        }
      />

      <main className="flex-1 overflow-auto p-4 space-y-4">
        {msg && <div className="panel px-3 py-1.5 font-mono text-[11px] text-warn">{msg}</div>}

        {/* timeline strip */}
        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            timeline ({track.n_frames}) — click a frame to open it in the editor
          </div>
          <div className="p-3 flex gap-2 overflow-x-auto">
            {track.items.map((it, i) => {
              const flip = it.class_name !== track.dominant;
              return (
                <button key={it.object_id} onClick={() => router.push(`/frame/${it.frame_id}?focus=${it.object_id}`)}
                  className={`shrink-0 w-28 border ${flip ? "border-block" : "border-line"} hover:border-accent`}
                  title={`frame ${i + 1}: ${it.class_name}`}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={it.crop_url} alt={it.class_name} className="w-full h-20 object-cover bg-bg-2" />
                  <div className="flex items-center gap-1 px-1 py-0.5 font-mono text-[10px]">
                    <span className="w-2 h-2 inline-block shrink-0" style={{ background: classColor(it.class_id) }} />
                    <span className={`truncate ${flip ? "text-block" : "text-ink-2"}`}>{it.class_name}</span>
                    <span className="ml-auto text-ink-3">{i + 1}</span>
                  </div>
                </button>
              );
            })}
          </div>
        </section>

        {/* relabel whole track */}
        <section className="panel max-w-md">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">
            relabel entire track (one fix, all {track.n_frames} frames)
          </div>
          <div className="p-3">
            <input value={search} onChange={(e) => setSearch(e.target.value)} placeholder="search class..."
              className="w-full bg-panel border border-line px-2 py-1 font-mono text-[11px] text-ink mb-2" />
            <div className="max-h-48 overflow-auto grid grid-cols-2 gap-1">
              {filtered.slice(0, 40).map((c) => (
                <button key={c.id} disabled={busy} onClick={() => relabel(c.name)}
                  className="flex items-center gap-1.5 px-1.5 py-1 border border-line font-mono text-[11px] text-ink-2 text-left hover:border-accent hover:text-ink disabled:opacity-50">
                  <span className="w-2.5 h-2.5 inline-block shrink-0" style={{ background: classColor(c.id) }} />
                  <span className="truncate">{c.name}</span>
                  {c.india && <span className="ml-auto text-accent">*</span>}
                </button>
              ))}
            </div>
            <div className="mt-3 flex items-center gap-2">
              <button onClick={onInterpolate} disabled={busy}
                title="fill the gaps between this track's keyframes with interpolated boxes (no drift)"
                className="font-mono text-[11px] border border-line text-ink-2 px-2 py-1 hover:border-accent disabled:opacity-50">
                interpolate gaps
              </button>
              <button onClick={() => router.push(`/annotate/timeline/${id}`)}
                title="open the keyframe + interpolation video timeline workspace"
                className="font-mono text-[11px] border border-accent text-accent px-2 py-1 hover:bg-accent/10">
                timeline workspace →
              </button>
              <button onClick={onDelete} disabled={busy}
                className="font-mono text-[11px] border border-line text-ink-3 px-2 py-1 hover:border-block hover:text-block disabled:opacity-50">
                delete track
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
