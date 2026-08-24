"use client";

// An anchored popover for the editor rail. There was no popover primitive in the tree, and the frame
// editor is the wrong place to add the first one casually, because it owns a global keymap.
//
// THE ESCAPE PROBLEM, which is the whole reason this is its own file.
//
// The frame page registers its keymap on `window` in the bubble phase, and its Escape branch clears the
// SAM candidate and then the selection. Every other popover in this codebase (MenuBar, NotificationBell)
// closes on a `document` keydown that does not stop propagation. Copy that here and pressing Escape to
// dismiss the class list also wipes the annotator's multi-selection, silently, with no way to connect
// cause to effect. ClassPopover.test.tsx pins this: remove the stopPropagation below and that test fails.
//
// The same applies to the digits. The page relabels the selection on 1-9, and with focus on a list row
// inside the popover its `typing()` guard does not apply, so one keypress would both pick here and
// relabel there: two dispatches, two history entries. Consumers pass `onKey` and return true for any key
// they have handled.
//
// stopPropagation is what stops the key; the capture phase is belt and braces. A document-bubble listener
// would already run before a window-bubble one, so capture is not load-bearing against the page as it is
// written today. It is used anyway because that ordering is a property of where the page happens to bind,
// not something this file can enforce: move the page keymap to `document` and bubble order becomes
// registration order, which the page wins by mounting first. Capture holds either way.

import { useCallback, useEffect, useRef, type RefObject } from "react";

export default function Popover({ anchorRef, open, onClose, label, width = "w-56", onKey, children }: {
  /** Used to exclude the trigger from click-outside, and to restore focus on close. */
  anchorRef: RefObject<HTMLElement | null>;
  open: boolean;
  onClose: () => void;
  label: string;
  width?: string;
  /** Return true when the popover has consumed the key, so it is never seen by the page keymap. */
  onKey?: (e: KeyboardEvent) => boolean;
  children: React.ReactNode;
}) {
  const boxRef = useRef<HTMLDivElement | null>(null);

  const closeAndRestore = useCallback(() => {
    onClose();
    // Without this a keyboard user lands back at the top of the document, several tab stops from where
    // they were.
    anchorRef.current?.focus();
  }, [onClose, anchorRef]);

  useEffect(() => {
    if (!open) return;

    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (boxRef.current?.contains(t)) return;
      // The anchor is excluded as well: without it, clicking "change" to dismiss fires close here and
      // toggle on the button, and the popover reopens in the same tick.
      if (anchorRef.current?.contains(t)) return;
      onClose();
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        e.stopPropagation();
        closeAndRestore();
        return;
      }
      if (onKey?.(e)) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKeyDown, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKeyDown, true);
    };
  }, [open, onClose, closeAndRestore, onKey, anchorRef]);

  if (!open) return null;

  return (
    // Right-aligned and opening downward with no collision logic, which is sound here and nowhere else:
    // the anchor is pinned near the top of a fixed 340px rail, so there is no edge to collide with. It is
    // mounted in the panel header rather than in the scroll body, or the body's overflow-y-auto clips it.
    <div ref={boxRef} role="dialog" aria-label={label}
      className={`absolute right-0 top-full mt-1 z-30 ${width} bg-panel border border-line rounded shadow-xl`}>
      {children}
    </div>
  );
}
