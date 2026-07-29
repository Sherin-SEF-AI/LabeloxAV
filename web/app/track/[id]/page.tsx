"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useSmartBack } from "@/lib/nav";
import { useConfirm } from "@/components/ConfirmProvider";
import { api , humanizeError } from "@/lib/api";
import type { IntentVocab, Ontology, Track } from "@/lib/types";
import { classColor } from "@/lib/colors";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";

// Tracklet editor: scan a track across frames as a strip, spot class flips (the cells that disagree
// with the dominant class glow red), and fix the whole track in one action. One relabel corrects every
// frame. Click a crop to jump into that frame in the editor.

export default function TrackEditor() {
  const confirm = useConfirm();
  const router = useRouter();
  const goBack = useSmartBack();
  const { id } = useParams<{ id: string }>();
  const [track, setTrack] = useState<Track | null>(null);
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [vocab, setVocab] = useState<IntentVocab | null>(null);

  const load = useCallback(async () => {
    const [t, o, v] = await Promise.all([api.track(id), api.ontology(), api.intentVocab().catch(() => null)]);
    setTrack(t);
    setOnto(o);
    setVocab(v);
  }, [id]);

  // M-F.2 intent kind (vehicle | vru | null) from the dominant class's l1 subclass
  const intentKind = useMemo(() => {
    if (!onto || !track) return null;
    const l1 = onto.classes.find((c) => c.name === track.dominant)?.l1;
    if (["two_wheeler", "three_wheeler", "four_wheeler", "heavy"].includes(l1 || "")) return "vehicle";
    if (l1 === "vru") return "vru";
    return null;
  }, [onto, track]);

  const proposeIntent = useCallback(async () => {
    setBusy(true);
    try { const r = await api.intentPropose(id); setMsg(r.proposed.length ? `proposed: ${r.proposed.join(", ")}` : "no geometric intent (left unknown)"); await load(); }
    catch (e) { setMsg(humanizeError(e)); } finally { setBusy(false); }
  }, [id, load]);
  const vlmIntent = useCallback(async () => {
    setBusy(true); setMsg("asking the VLM...");
    try { const r = await api.intentVlm(id); setMsg(r.proposed ? `VLM proposed: ${r.proposed}` : `VLM: ${r.reason ?? "unclear (unknown)"}`); await load(); }
    catch (e) { setMsg(humanizeError(e)); } finally { setBusy(false); }
  }, [id, load]);
  const setIntent = useCallback(async (intent: string) => {
    if (!intentKind) return;
    setBusy(true);
    try { await api.intentSet(id, intent, intentKind); setMsg(intent === "unknown" ? "cleared intent" : `confirmed intent: ${intent}`); await load(); }
    catch (e) { setMsg(humanizeError(e)); } finally { setBusy(false); }
  }, [id, intentKind, load]);

  useEffect(() => {
    load().catch((e) => setMsg(humanizeError(e)));
  }, [load]);

  const relabel = useCallback(
    async (className: string) => {
      setBusy(true);
      try {
        const r = await api.relabelTrack(id, className);
        setMsg(`relabeled ${r.relabeled} frames to ${className}`);
        await load();
      } catch (e) {
        setMsg(humanizeError(e));
      } finally {
        setBusy(false);
      }
    },
    [id, load],
  );

  const onDelete = useCallback(async () => {
    // The app's dialog, not the browser's. A native confirm is styled differently from every other
    // destructive prompt here and a browser is free to suppress it, which would delete a whole track on a
    // single click with no prompt at all.
    if (!(await confirm({ title: "Delete this entire track?",
                          body: "Every object on it goes too, and this cannot be undone.",
                          danger: true, confirmLabel: "Delete track" }))) return;
    setBusy(true);
    try {
      await api.deleteTrack(id);
      goBack();
    } catch (e) {
      setMsg(humanizeError(e));
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
      setMsg(humanizeError(e));
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
            timeline ({track.n_frames}) - click a frame to open it in the editor
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

        {/* M-F.2 behavior / intent (track-level, from a closed vocabulary; proposals need human confirmation) */}
        {intentKind && (
          <section className="panel max-w-2xl">
            <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2 flex items-center justify-between">
              <span>behavior / intent ({intentKind})</span>
              <span className="flex items-center gap-2">
                <button onClick={proposeIntent} disabled={busy} className="border border-line px-1.5 py-0.5 rounded text-ink-3 hover:border-accent disabled:opacity-40">propose from trajectory</button>
                {intentKind === "vru" && <button onClick={vlmIntent} disabled={busy} className="border border-info/50 text-info px-1.5 py-0.5 rounded hover:bg-info/10 disabled:opacity-40">propose from VLM</button>}
              </span>
            </div>
            <div className="p-3 space-y-3 font-mono text-[11px]">
              {/* current intents */}
              <div className="flex flex-wrap gap-1.5">
                {(track.intents || []).length === 0 && <span className="text-ink-3">no intent yet. propose from trajectory or VLM, or set one below.</span>}
                {(track.intents || []).map((it, i) => (
                  <span key={i} title={`source ${it.source} · confidence ${it.confidence}`}
                    className={`inline-flex items-center gap-1 border px-1.5 py-0.5 rounded ${it.status === "confirmed" ? "border-pass/60 text-pass" : "border-warn/60 text-warn"}`}>
                    {it.intent}
                    <span className="text-ink-3 text-[9px] uppercase">{it.source}</span>
                    {it.status === "proposed" && it.source !== "human" && (
                      <button onClick={() => setIntent(it.intent)} disabled={busy} title="confirm this intent" className="text-pass hover:text-accent ml-0.5">✓</button>
                    )}
                  </span>
                ))}
              </div>
              {/* set from the closed vocabulary */}
              <div className="flex flex-wrap items-center gap-1.5 border-t hairline pt-2">
                <span className="text-ink-3 uppercase text-[9px] w-full">set intent (closed vocabulary)</span>
                {(intentKind === "vehicle" ? vocab?.vehicle : vocab?.vru)?.map((v) => (
                  <button key={v} onClick={() => setIntent(v)} disabled={busy}
                    className="border border-line px-1.5 py-0.5 rounded text-ink-2 hover:border-accent disabled:opacity-40">{v}</button>
                ))}
                <button onClick={() => setIntent("unknown")} disabled={busy}
                  className="border border-line px-1.5 py-0.5 rounded text-ink-3 hover:border-block disabled:opacity-40">unknown</button>
              </div>
            </div>
          </section>
        )}

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
