"use client";

import { useEffect, useRef } from "react";

// What a modal owes a keyboard, in one place.
//
// The app has a dozen `fixed inset-0` overlays and each had grown its own subset of the contract: one
// closed on Escape, one moved focus in, none kept Tab inside, and none gave focus back, so closing a
// dialog dropped a keyboard user at the top of the document with the thing they had been working on
// somewhere behind them. A screen reader was told nothing at all for most of them, since a div with a
// dark background is not a dialog to anything but an eye.
//
// The hook does the behaviour; the element still declares what it is:
//
//   const ref = useDialog<HTMLDivElement>(open, close);
//   <div ref={ref} role="dialog" aria-modal="true" aria-labelledby={titleId} tabIndex={-1}> ... </div>
//
// While `open`:
//   - focus moves in: to the element marked `autoFocus` / `data-autofocus` if there is one, else the first
//     focusable element, else the container itself (hence tabIndex={-1} on it);
//   - Tab and Shift+Tab cycle inside the container;
//   - Escape calls onClose;
//   - when it closes or unmounts, focus returns to whatever had it before.
//
// Dialogs stack (a confirm over the console), and only the top one answers the keyboard, so one Escape
// closes one layer.

const FOCUSABLE = [
  "a[href]", "button:not([disabled])", "input:not([disabled]):not([type='hidden'])",
  "select:not([disabled])", "textarea:not([disabled])", "summary", "[tabindex]:not([tabindex='-1'])",
].join(",");

const stack: HTMLElement[] = [];

function visible(el: HTMLElement): boolean {
  if (el.hidden || el.getAttribute("aria-hidden") === "true") return false;
  // checkVisibility is the only cheap test that sees an ancestor's display: none. Where it does not exist
  // (older browsers, jsdom) every candidate is kept, which at worst makes Tab stop on a hidden control.
  const check = (el as HTMLElement & { checkVisibility?: () => boolean }).checkVisibility;
  return typeof check === "function" ? check.call(el) : true;
}

/** The elements Tab would stop on inside `root`, in document order. */
export function focusableIn(root: HTMLElement): HTMLElement[] {
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(visible);
}

export function useDialog<T extends HTMLElement>(open: boolean, onClose: () => void,
                                                 opts: { closeOnEscape?: boolean } = {}) {
  const ref = useRef<T>(null);
  // Through refs, so a parent re-rendering with a new closure does not re-run the focus effect and yank
  // focus back to the first field while somebody is typing in the second.
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  const escRef = useRef(opts.closeOnEscape !== false);
  escRef.current = opts.closeOnEscape !== false;

  useEffect(() => {
    const root = ref.current;
    if (!open || !root) return;
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    stack.push(root);

    // An element that asked for focus keeps it (React has already honoured autoFocus by now).
    if (!root.contains(document.activeElement)) {
      const wanted = root.querySelector<HTMLElement>("[data-autofocus]") ?? focusableIn(root)[0] ?? root;
      wanted.focus({ preventScroll: true });
    }

    const onKey = (e: KeyboardEvent) => {
      if (stack[stack.length - 1] !== root) return;
      if (e.key === "Escape") {
        if (!escRef.current) return;
        e.stopPropagation();
        closeRef.current();
        return;
      }
      if (e.key !== "Tab") return;
      const items = focusableIn(root);
      if (items.length === 0) { e.preventDefault(); root.focus({ preventScroll: true }); return; }
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      if (e.shiftKey && (active === first || !root.contains(active))) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && (active === last || !root.contains(active))) { e.preventDefault(); first.focus(); }
    };
    // Capture, so the dialog decides before a page-level shortcut handler sees the key.
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("keydown", onKey, true);
      const at = stack.lastIndexOf(root);
      if (at >= 0) stack.splice(at, 1);
      // Only if focus is still inside what is closing (or was lost with it): a close that already sent the
      // user somewhere, like the palette navigating, must not pull them back.
      const active = document.activeElement;
      if (previous && previous.isConnected && (!active || active === document.body || root.contains(active))) {
        previous.focus({ preventScroll: true });
      }
    };
  }, [open]);

  return ref;
}
