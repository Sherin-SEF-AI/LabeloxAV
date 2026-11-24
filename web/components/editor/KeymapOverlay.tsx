"use client";

// The frame editor's "?" overlay: the shortcut reference, with the tool and action keys rebindable in
// place.
//
// The shared overlay (components/shell/ShortcutOverlay.tsx) is a printed list. That is right for the rest
// of the app, where the keys are not the editor's to change, and wrong here in one specific way: it would
// go on printing B for the box tool after an annotator had moved it. So the editor mounts this one
// instead. It opens on the same key and the same `lbx:shortcuts` event, shows the same fixed rows, and
// reads the tool and action rows from the live keymap (lib/editor/keymap.ts), which is also what the
// keyboard handler resolves from. The list and the behaviour cannot disagree because they are one value.
//
// Click a key, press the new one. Escape cancels the capture without closing the overlay. A key that is
// reserved or already taken in a mode the binding is active in is refused, with the reason, and nothing
// changes.

import { useEffect, useState, useSyncExternalStore } from "react";

import { GLOBAL } from "@/components/shell/ShortcutOverlay";
import {
  bindings, getKeymap, getKeymapServerSnapshot, hydrateKeymap, resetKeymap, setBinding, subscribeKeymap,
  type Binding,
} from "@/lib/editor/keymap";
import { MODES } from "@/lib/editor/registry";
import { isTypingTarget } from "@/lib/editor/keys";

// Keys this editor binds that the shared list does not carry.
const EDITOR_ROWS: { keys: string; label: string }[] = [
  { keys: "Arrows", label: "nudge the selection 1 px (unlocked objects; cuboids are skipped)" },
  { keys: "Shift Arrows", label: "nudge the selection 10 px" },
  { keys: "Del", label: "delete every selected object" },
  { keys: "Tab", label: "select the riskiest object not yet dealt with" },
  { keys: "Shift Enter", label: "accept every object you did not touch" },
];

// The shared list's single-letter rows are the rebindable actions. They are printed from the keymap
// below, with the key actually in force, so the static copies are left out rather than shown twice.
const FIXED = [...GLOBAL.filter((r) => !/^[A-Za-z]$/.test(r.keys)), ...EDITOR_ROWS];

const MODIFIER_KEYS = new Set(["Shift", "Control", "Alt", "Meta", "AltGraph", "CapsLock"]);

const show = (key: string) => key.toUpperCase();

// "Objects" for a tool one mode has; nothing for one every mode has, where a list of seven names is noise.
const scope = (b: Binding) => (b.modes.length === MODES.length ? "" : b.modes.join(", "));

export default function KeymapOverlay() {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  // The binding waiting for a key, and why the last key offered was refused.
  const [capturing, setCapturing] = useState<string | null>(null);
  const [refusal, setRefusal] = useState<string | null>(null);
  const overrides = useSyncExternalStore(subscribeKeymap, getKeymap, getKeymapServerSnapshot);

  useEffect(() => { hydrateKeymap(); }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "?" && !isTypingTarget(e.target)) { e.preventDefault(); setOpen((o) => !o); }
      else if (e.key === "Escape") setOpen(false);
    };
    const onEvt = () => setOpen(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener("lbx:shortcuts", onEvt);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("lbx:shortcuts", onEvt); };
  }, []);

  // Closing mid-capture must not leave a binding armed behind a closed overlay.
  useEffect(() => { if (!open) { setCapturing(null); setRefusal(null); } }, [open]);

  // While a binding is waiting, the next key belongs to it and to nothing else. Captured on window in the
  // capture phase with propagation stopped: the editor's keymap is a bubble-phase listener on the same
  // window, and without this the key being offered would also switch tool, and Escape would also close the
  // overlay and clear the selection.
  useEffect(() => {
    if (!capturing) return;
    const onKey = (e: KeyboardEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (e.key === "Escape") { setCapturing(null); setRefusal(null); return; }
      // A modifier on its own is somebody on the way to a key, not a key.
      if (MODIFIER_KEYS.has(e.key)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) { setRefusal("a single key, without Cmd, Ctrl or Alt"); return; }
      const r = setBinding(capturing, e.key);
      if (r.ok) { setCapturing(null); setRefusal(null); }
      else setRefusal(r.reason);
    };
    // The rebind button has focus, and a button is clicked by the keyup of Space. Without this, offering
    // Space (which is refused) would also disarm the capture on the way back up.
    const onUp = (e: KeyboardEvent) => { e.preventDefault(); e.stopPropagation(); };
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("keyup", onUp, true);
    return () => {
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("keyup", onUp, true);
    };
  }, [capturing]);

  if (!open) return null;
  const needle = q.toLowerCase();
  const match = (label: string, keys: string) => !needle || label.toLowerCase().includes(needle) || keys.toLowerCase().includes(needle);

  const all = bindings(overrides);
  const rebindable = [
    { title: "actions", rows: all.filter((b) => b.kind === "action" && match(b.label, b.key)) },
    { title: "tools", rows: all.filter((b) => b.kind === "tool" && match(b.label, b.key)) },
  ].filter((s) => s.rows.length);
  const fixed = FIXED.filter((r) => match(r.label, r.keys));
  const changed = Object.keys(overrides).length;

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[12vh] bg-bg/70"
      onClick={(e) => { if (e.target === e.currentTarget) setOpen(false); }}>
      {/* A dialog to assistive technology too: it covers the editor and takes the keyboard. */}
      <div role="dialog" aria-modal="true" aria-labelledby="keymap-overlay-title"
        className="panel w-[34rem] max-h-[70vh] flex flex-col">
        <div className="flex items-center gap-2 px-3 h-10 border-b hairline shrink-0">
          <span id="keymap-overlay-title" className="text-[10.5px] font-medium uppercase tracking-[0.06em] tracking-wide text-ink-3">keyboard shortcuts</span>
          <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="search shortcuts..."
            className="ml-auto w-48 bg-bg border border-line px-2 py-1 text-[12px] text-ink" />
          <button onClick={() => setOpen(false)} className="text-[12px] text-ink-3 hover:text-accent">esc</button>
        </div>
        <div className="overflow-y-auto p-3 space-y-3">
          {fixed.length > 0 && (
            <div>
              <div className="text-[10.5px] font-medium uppercase tracking-[0.06em] tracking-wide text-ink-3 mb-1">global</div>
              <div className="space-y-0.5">
                {fixed.map((r, i) => (
                  <div key={`${r.keys}-${i}`} className="flex items-center justify-between gap-4 text-[12px]">
                    <span className="text-ink-2 truncate">{r.label}</span>
                    <span className="text-accent shrink-0">{r.keys}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {rebindable.map((s) => (
            <div key={s.title}>
              <div className="flex items-baseline gap-2 mb-1">
                <span className="text-[10.5px] font-medium uppercase tracking-[0.06em] tracking-wide text-ink-3">{s.title}</span>
                <span className="text-[10.5px] text-ink-3/70">click a key to change it</span>
              </div>
              <div className="space-y-0.5">
                {s.rows.map((b) => {
                  const armed = capturing === b.id;
                  return (
                    <div key={b.id} className="flex items-center justify-between gap-4 text-[12px]">
                      <span className="text-ink-2 truncate">
                        {b.label}
                        {scope(b) && <span className="text-ink-3/70"> · {scope(b)}</span>}
                        {b.key !== b.default && <span className="text-ink-3/70"> · was {show(b.default)}</span>}
                      </span>
                      <button
                        onClick={() => { setRefusal(null); setCapturing(armed ? null : b.id); }}
                        aria-label={armed ? `press a key for ${b.label}` : `rebind ${b.label}, currently ${show(b.key)}`}
                        aria-pressed={armed}
                        title="click, then press the new key. Escape cancels."
                        className={`shrink-0 min-w-[26px] px-1.5 py-0.5 border rounded text-center ${
                          armed ? "border-accent bg-accent/10 text-accent" : "border-line text-accent hover:border-accent"}`}>
                        {armed ? "press a key" : show(b.key)}
                      </button>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
          {!fixed.length && !rebindable.length && <div className="text-[12px] text-ink-3 text-center py-4">no shortcuts match</div>}
        </div>
        <div className="flex items-center gap-3 px-3 h-9 border-t hairline shrink-0 text-[11.5px]">
          {/* The refusal stays until the next attempt. A message that vanished on a timer would be gone
              before somebody looking at the keyboard looked back up. */}
          {refusal
            ? <span role="alert" className="text-block truncate">{refusal}</span>
            : <span className="text-ink-3 truncate">
                {capturing ? "press the new key, or Escape to cancel" : changed ? `${changed} changed from the defaults` : "these are the default keys"}
              </span>}
          <button onClick={() => { resetKeymap(); setCapturing(null); setRefusal(null); }} disabled={!changed}
            className="ml-auto shrink-0 border border-line px-2 py-0.5 text-ink-2 hover:border-accent disabled:opacity-40">
            Reset to defaults
          </button>
        </div>
      </div>
    </div>
  );
}
