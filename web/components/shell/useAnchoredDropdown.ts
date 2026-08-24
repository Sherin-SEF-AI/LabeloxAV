"use client";

// A dropdown anchored to a button that lives inside a horizontally scrollable bar.
//
// THE BUG THIS EXISTS FOR. The frame editor's top bar is `h-[46px] ... overflow-x-auto`, so the row of
// controls can be scrolled to on a narrow viewport. CSS does not allow one axis to be `visible` while the
// other is not: with `overflow-x: auto`, a computed `overflow-y: visible` becomes `auto` too. So the header
// clips its own descendants to 46 pixels, and an `absolute` dropdown anchored inside it renders inside that
// clip box. Measured: the notifications panel was 416x132 at y=41 inside a header of 0,0 1600x46, which
// reached the screen as a five-pixel sliver under the bell and read as "the dropdown opens behind the
// properties panel".
//
// `position: fixed` escapes an ancestor's overflow, which `absolute` does not. It only works while no
// ancestor establishes a containing block for fixed descendants (a transform, filter, perspective, contain
// or will-change); that was verified by walking the chain from the dropdown to <body>, all `transform:
// none`. If someone later animates the top bar with a transform, these dropdowns move with it and this
// stops working, which is the tradeoff against portalling into <body>.
//
// Coordinates are taken at open time rather than in a layout effect, so the first painted frame is already
// in the right place instead of flashing at the top-left corner.

import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";

/** Distance between the trigger and the panel, matching the `mt-1` these dropdowns used to carry. */
const GAP = 4;

export function useAnchoredDropdown<T extends HTMLElement = HTMLButtonElement>(open: boolean) {
  const anchorRef = useRef<T | null>(null);
  const [style, setStyle] = useState<CSSProperties>({ position: "fixed", top: -9999, right: 0 });

  const place = useCallback(() => {
    const el = anchorRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // Right-aligned to the trigger, which is what `right-0` meant when this was absolutely positioned.
    setStyle({
      position: "fixed",
      top: Math.round(r.bottom + GAP),
      right: Math.round(window.innerWidth - r.right),
    });
  }, []);

  // While open, follow the trigger. The capture phase on scroll is deliberate: the trigger sits in a
  // scrollable bar, so the event that moves it does not bubble to window.
  useEffect(() => {
    if (!open) return;
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [open, place]);

  return { anchorRef, style, place };
}
