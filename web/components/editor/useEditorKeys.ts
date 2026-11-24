"use client";

// The frame editor's keyboard, as a hook.
//
// What a key means is decided in lib/editor/keys.ts, which knows nothing about React. This is the thin
// part that cannot be pure: listening on window, reading where the event landed, and handing the decision
// back to the page to carry out.
//
// The listener is attached once. It used to be torn down and re-attached whenever any of eighteen
// dependencies changed, and still read ten values the dependency list had not kept up with. Reading
// through a ref that is rewritten every render means there is no list to keep up.

import { useEffect, useRef, useSyncExternalStore } from "react";

import {
  getKeymap, getKeymapServerSnapshot, hydrateKeymap, subscribeKeymap, type Overrides,
} from "@/lib/editor/keymap";
import { isTypingTarget, ownsArrowKeys, resolveKey, type KeyContext, type KeyIntent } from "@/lib/editor/keys";
import { intentAllowed } from "@/lib/editor/readOnly";

export type EditorKeyState = Omit<KeyContext, "typing" | "arrowsOwned" | "overrides">;

export type EditorKeysOptions = {
  /** The editor's state as the resolver needs it, read at the moment a key arrives. */
  state: () => EditorKeyState;
  /** Carry out what the key meant. */
  run: (intent: Exclude<KeyIntent, { t: "pan" }>, e: KeyboardEvent) => void;
  /** Space held or released. */
  onPan: (panning: boolean) => void;
  /**
   * The frame is view-only, read at the moment a key arrives. Keys that would edit are then refused
   * here, before `run`, by the one rule in lib/editor/readOnly.ts.
   */
  readOnly?: () => boolean;
  /** A key was refused because the frame is view-only. */
  onBlocked?: (intent: KeyIntent) => void;
};

/** Binds the editor keymap on window. Returns the annotator's key overrides, for whatever prints keys. */
export function useEditorKeys(opts: EditorKeysOptions): Overrides {
  const ref = useRef(opts);
  ref.current = opts;
  const overrides = useSyncExternalStore(subscribeKeymap, getKeymap, getKeymapServerSnapshot);

  // In an effect, not at module load: the first client render has to match the server's, which has no
  // localStorage to read.
  useEffect(() => { hydrateKeymap(); }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const o = ref.current;
      const r = resolveKey(e, {
        ...o.state(),
        typing: isTypingTarget(e.target),
        // defaultPrevented covers the widgets that handle arrows without a role to recognise them by:
        // the menu bar listens on document and has already answered by the time this runs.
        arrowsOwned: e.defaultPrevented || ownsArrowKeys(e.target),
        overrides: getKeymap(),
      });
      if (!r) return;
      // Still prevented when refused: Cmd S must not fall through to the browser's "save page as".
      if (r.preventDefault) e.preventDefault();
      if (o.readOnly?.() && !intentAllowed(r.intent)) {
        // Not for a held key: one message per press, not thirty a second.
        if (!e.repeat) o.onBlocked?.(r.intent);
        return;
      }
      if (r.intent.t === "pan") o.onPan(true);
      else o.run(r.intent, e);
    };
    const onUp = (e: KeyboardEvent) => { if (e.code === "Space") ref.current.onPan(false); };
    window.addEventListener("keydown", onKey);
    window.addEventListener("keyup", onUp);
    return () => { window.removeEventListener("keydown", onKey); window.removeEventListener("keyup", onUp); };
  }, []);

  return overrides;
}
