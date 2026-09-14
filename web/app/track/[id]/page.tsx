"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { useSmartBack } from "@/lib/nav";
import { useConfirm } from "@/components/ConfirmProvider";
import { api , humanizeError } from "@/lib/api";
import type { IntentVocab, Ontology, OntologyClass, Track } from "@/lib/types";
import { classColor } from "@/lib/colors";
import { cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle } from "@/lib/gridSelection";
import { toast } from "@/lib/toast";
import { acceptState, useCurrentUser } from "@/lib/user";
import BackButton from "@/components/BackButton";
import PageHeaderBar from "@/components/shell/PageHeaderBar";
import EventLane from "@/components/track/EventLane";
import ClassPopover from "@/components/editor/properties/ClassPopover";

// Tube review: scan a track across frames as a strip, confirm the whole thing with one key, and fix the
// odd wrong crop in place instead of leaving for the frame editor and coming back.
//
// The strip is the lever this page exists for. 7,512 tracks carry ten or more objects and between them
// hold 549,038 objects, 95% of the corpus, so a reviewer who can look at a whole track at once and press
// one key is standing in for fifty per-frame verdicts. The measured shape of the work says the same
// thing from the other side: the median track is 93 frames and the median number of frames a person has
// ever touched on one is 1.
//
// Three things make that keystroke honest rather than fast. Accepting a track moves state and nothing
// else, so an interpolated box stays interpolated and the Review row names who approved it. The cells
// that disagree with the dominant class still glow red, so a flip is visible before the key is pressed.
// And an id-switch count is reported after, because a re-identification event is where a whole-track
// verdict may have covered two different physical objects.

const CELL = 96;          // px per crop in the sheet the server builds, and per tile on screen
const SHEET_CHUNK = 200;  // crops per sheet request; the server caps at 400, and the longest track is 697

type Placement = { sheet: string; row: number; col: number; ok: boolean };

export default function TrackEditor() {
  const confirm = useConfirm();
  const router = useRouter();
  const goBack = useSmartBack();
  const { id } = useParams<{ id: string }>();
  const me = useCurrentUser();
  const [track, setTrack] = useState<Track | null>(null);
  const [onto, setOnto] = useState<Ontology | null>(null);
  const [search, setSearch] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [vocab, setVocab] = useState<IntentVocab | null>(null);

  // Strip review state
  const [placements, setPlacements] = useState<Record<string, Placement>>({});
  const [cursor, setCursor] = useState(0);
  const [selection, setSelection] = useState<ReadonlySet<number>>(new Set());
  const [anchor, setAnchor] = useState<number | null>(null);
  const [decided, setDecided] = useState(0);
  const [pickOpen, setPickOpen] = useState(false);
  const startedAt = useRef<number>(Date.now());
  const batchStart = useRef<number>(Date.now());
  const cursorTile = useRef<HTMLDivElement | null>(null);

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

  const items = useMemo(() => track?.items ?? [], [track]);

  // One sprite sheet per chunk instead of one request per crop. The strip used to issue N <img src=/crop>
  // requests, so a 200-frame track opened 200 connections and a 697-frame one opened 697; the server
  // already had POST /api/objects/crops, which groups by frame so each JPEG is decoded once.
  useEffect(() => {
    if (!items.length) { setPlacements({}); return; }
    let cancelled = false;
    (async () => {
      const ids = items.map((it) => it.object_id);
      for (let i = 0; i < ids.length; i += SHEET_CHUNK) {
        const chunk = ids.slice(i, i + SHEET_CHUNK);
        try {
          const s = await api.cropSheet(chunk, CELL);
          if (cancelled) return;
          if (!s.sheet) continue;
          const sheet = s.sheet;
          // Merged rather than replaced, so the first chunk stays on screen while the rest arrive.
          setPlacements((prev) => {
            const next = { ...prev };
            for (const p of s.placements) next[p.object_id] = { sheet, row: p.row, col: p.col, ok: p.ok };
            return next;
          });
        } catch (e) {
          if (!cancelled) toast(humanizeError(e), "error");
          return;
        }
      }
    })();
    return () => { cancelled = true; };
  }, [items.map((it) => it.object_id).join(",")]);

  // Keep the cursor on screen. A 697-tile strip scrolls far past the viewport, and a keyboard mode that
  // moves a cursor you cannot see is worse than no keyboard mode.
  useEffect(() => {
    cursorTile.current?.scrollIntoView({ block: "nearest", inline: "nearest" });
  }, [cursor]);

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

  /** Confirm every frame on the track. The headline keystroke, and the reason this page was rebuilt. */
  const acceptWholeTrack = useCallback(async () => {
    if (!track) return;
    setBusy(true);
    const elapsed = Math.max(0, Date.now() - batchStart.current);
    batchStart.current = Date.now();
    try {
      const r = await api.acceptTrack(id, { time_spent_ms: elapsed });
      setDecided((d) => d + r.accepted);
      const parts = [`accepted ${r.accepted} frames as ${r.state}`];
      if (r.clamped) parts.push("clamped to your role's ceiling");
      if (r.skipped_human.length) parts.push(`${r.skipped_human.length} left alone (already human)`);
      // Not a blocker, but the one thing that can make a whole-track verdict wrong, so it is said out loud.
      if (r.id_switch_events) parts.push(`${r.id_switch_events} id-switch event(s) on this track: check the strip`);
      setMsg(parts.join(" - "));
      if (r.run_id) toast(`accepted ${r.accepted} frames`, "success");
      await load();
    } catch (e) {
      setMsg(humanizeError(e));
    } finally {
      setBusy(false);
    }
  }, [id, track, load]);

  /**
   * Apply a verdict to the selected tiles, or to the tile under the cursor when nothing is selected.
   *
   * Deliberately not `targetIndices`' whole-page fallback: on this page the no-selection fallback for
   * accept is the whole track (see the keymap), and routing that through bulk review would restate 93
   * object ids the server can resolve from one track id.
   */
  const decideSome = useCallback(async (action: "confirm" | "reject" | "reclassify", className?: string) => {
    const idx = targetIndices(selection, cursor);
    const rows = idx.map((i) => items[i]).filter(Boolean);
    if (!rows.length) return;
    const ids = rows.map((r) => r.object_id);
    const elapsed = Math.max(0, Date.now() - batchStart.current);
    batchStart.current = Date.now();
    setSelection(new Set());
    setAnchor(null);
    setDecided((d) => d + ids.length);
    try {
      const res = await api.bulkReview(
        ids, action, className,
        action === "reject" ? "rejected" : acceptState(me?.role),
        undefined, { time_spent_ms: elapsed },
      );
      const lost = (res.skipped_stale?.length ?? 0) + (res.skipped_missing?.length ?? 0);
      if (lost) {
        toast(`${res.updated} applied, ${lost} skipped (changed or gone since this page loaded)`, "error");
        setDecided((d) => d - lost);
      } else {
        toast(`${res.updated} ${action === "reject" ? "rejected" : action === "reclassify" ? `set to ${className}` : "accepted"}`, "success");
      }
      await load();
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, [selection, cursor, items, me, load]);

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
  }, [id, confirm, goBack]);

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

  const addClass = useCallback(async (raw: string) => {
    try {
      const c = await api.addClass(raw);
      setOnto(await api.ontology());
      await decideSome("reclassify", c.name);
    } catch (e) {
      toast(humanizeError(e), "error");
    }
  }, [decideSome]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return;
      // The class search box and the popover's own input live on this page; typing in either must not
      // move the cursor or accept a track.
      const el = e.target as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      const n = items.length;
      if (!n) return;
      const k = e.key.toLowerCase();

      if (k === "arrowleft" || k === "h") { e.preventDefault(); step("left", e.shiftKey); }
      else if (k === "arrowright" || k === "l") { e.preventDefault(); step("right", e.shiftKey); }
      else if (k === " ") { e.preventDefault(); setSelection((s) => toggle(s, cursor)); setAnchor(cursor); }
      else if (k === "escape") { setSelection(new Set()); setAnchor(null); }
      else if (k === "c") { e.preventDefault(); setPickOpen(true); }
      else if (k === "o") { const it = items[cursor]; if (it) router.push(`/frame/${it.frame_id}?focus=${it.object_id}`); }
      else if (k === "r") { e.preventDefault(); void decideSome("reject"); }
      else if (k === "a") {
        e.preventDefault();
        // One rule, and it is printed above the strip: with nothing selected, A means the whole track,
        // because on this page the track is the unit. With a selection it means those frames, so a
        // reviewer who has picked out the three bad crops is not forced to accept all ninety.
        if (selection.size > 0) void decideSome("confirm");
        else void acceptWholeTrack();
      }

      function step(dir: "left" | "right", extend: boolean) {
        const next = moveCursor(cursor, dir, n, 1);
        setCursor(next);
        if (!extend) { setSelection(new Set()); setAnchor(null); return; }
        const a = anchor ?? cursor;
        setAnchor(a);
        setSelection(new Set(rangeBetween(a, next)));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cursor, anchor, items, selection, decideSome, acceptWholeTrack, router]);

  // Clamp the cursor when the track shrinks under it (a reject removes rows on reload).
  useEffect(() => {
    setCursor((c) => cursorAfterRemoval(c, [], items.length));
  }, [items.length]);

  const perMin = useMemo(() => {
    const mins = (Date.now() - startedAt.current) / 60000;
    return mins > 0.05 && decided > 0 ? Math.round(decided / mins) : null;
  }, [decided]);

  const filtered = useMemo(
    () => (onto ? onto.classes.filter((c) => c.name.includes(search.toLowerCase().replace(/\s/g, "_"))) : []),
    [onto, search],
  );

  if (!track) return <div className="min-h-screen flex items-center justify-center font-mono text-ink-3">{msg ?? "loading track..."}</div>;

  const current = items[cursor];

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
            {decided > 0 && <span className="text-ink-3">{decided} decided</span>}
            {perMin != null && <span className="text-accent">{perMin}/min</span>}
          </>
        }
      />

      <main className="flex-1 overflow-auto p-4 space-y-4">
        {msg && <div className="panel px-3 py-1.5 font-mono text-[11px] text-warn">{msg}</div>}

        {/* The event lane sits directly under the strip and shares its column geometry, so a span reads
            against the crops it covers rather than against a separate time axis. */}
        {/* tube review strip */}
        <section className="panel">
          <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2 flex items-center gap-3 flex-wrap">
            <span>timeline ({track.n_frames})</span>
            <span className="text-ink-2 normal-case">
              <b className="text-accent">A</b> accept whole track (or the selection) ·
              {" "}<b className="text-accent">C</b> change class ·
              {" "}<b className="text-accent">R</b> reject ·
              {" "}<b className="text-accent">space</b> select ·
              {" "}<b className="text-accent">shift+←→</b> range ·
              {" "}<b className="text-accent">O</b> open frame
            </span>
            <span className="ml-auto flex items-center gap-2">
              {selection.size > 0 && <span className="text-accent normal-case">{selection.size} selected</span>}
              <button onClick={acceptWholeTrack} disabled={busy}
                title="confirm every frame on this track (state only: no box moves and no class is asserted)"
                className="border border-pass/60 text-pass px-2 py-0.5 rounded hover:bg-pass/10 disabled:opacity-40 normal-case">
                accept all {track.n_frames}
              </button>
            </span>
          </div>
          <div className="p-3 flex gap-2 overflow-x-auto">
            {items.map((it, i) => {
              const flip = it.class_name !== track.dominant;
              const p = placements[it.object_id];
              const isCursor = i === cursor;
              const isSel = selection.has(i);
              return (
                <div
                  key={it.object_id}
                  ref={isCursor ? cursorTile : undefined}
                  role="button"
                  tabIndex={-1}
                  onClick={(e) => {
                    if (e.shiftKey) {
                      const a = anchor ?? cursor;
                      setAnchor(a);
                      setSelection(new Set(rangeBetween(a, i)));
                    } else if (e.altKey) {
                      setSelection((s) => toggle(s, i));
                      setAnchor(i);
                    } else {
                      setSelection(new Set());
                      setAnchor(null);
                    }
                    setCursor(i);
                  }}
                  onDoubleClick={() => router.push(`/frame/${it.frame_id}?focus=${it.object_id}`)}
                  className={`shrink-0 cursor-pointer border ${
                    isCursor ? "border-accent" : isSel ? "border-info" : flip ? "border-block" : "border-line"
                  } ${isSel ? "bg-info/10" : ""} hover:border-accent`}
                  style={{ width: CELL }}
                  title={`frame ${i + 1}: ${it.class_name} (${it.state}, ${it.source}) - click to select, double-click to open`}>
                  {p && p.ok ? (
                    // One image, positioned. Each tile is a window onto the same sprite sheet.
                    <div
                      aria-label={it.class_name}
                      style={{
                        width: CELL, height: CELL,
                        backgroundImage: `url(${p.sheet})`,
                        backgroundPosition: `-${p.col * CELL}px -${p.row * CELL}px`,
                      }}
                    />
                  ) : (
                    <div className="bg-bg-2 flex items-center justify-center font-mono text-[9px] text-ink-3"
                      style={{ width: CELL, height: CELL }}>
                      {p ? "no crop" : "..."}
                    </div>
                  )}
                  <div className="flex items-center gap-1 px-1 py-0.5 font-mono text-[10px]">
                    <span className="w-2 h-2 inline-block shrink-0" style={{ background: classColor(it.class_id) }} />
                    <span className={`truncate ${flip ? "text-block" : "text-ink-2"}`}>{it.class_name}</span>
                    <span className="ml-auto text-ink-3">{i + 1}</span>
                  </div>
                  {/* State and source, because "accept the track" is a claim about these and a reviewer
                      should be able to see which frames it will actually move. */}
                  <div className="px-1 pb-0.5 font-mono text-[9px] flex items-center gap-1">
                    <span className={it.state === "accepted" ? "text-pass" : it.state === "rejected" ? "text-block" : "text-ink-3"}>
                      {it.state}
                    </span>
                    <span className="ml-auto text-ink-3 truncate">{it.source}</span>
                  </div>
                </div>
              );
            })}
          </div>
          {/* Fix in place. Anchored to the cursor tile, so the picker opens over the crop it will change
              rather than sending the reviewer to the frame editor and back. */}
          {onto && (
            <ClassPopover
              anchorRef={cursorTile} open={pickOpen} onClose={() => setPickOpen(false)}
              classes={onto.classes} currentId={current?.class_id ?? null}
              onPick={(c: OntologyClass) => { void decideSome("reclassify", c.name); }}
              onAdd={(raw: string) => { void addClass(raw); }} />
          )}
        </section>

        <EventLane trackId={id} items={track.items} />

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
