"use client";

import { useEffect, useState } from "react";
import { subscribe } from "@/lib/progress";

// A thin indeterminate bar pinned to the top of the window whenever any API request is in flight. One
// component gives every page loading feedback (the app has slow endpoints under load), so a slow fetch reads
// as "working" instead of a frozen screen. Debounced so sub-150ms calls never flicker it.
export default function GlobalLoadingBar() {
  const [active, setActive] = useState(0);
  const [show, setShow] = useState(false);

  useEffect(() => subscribe(setActive), []);
  useEffect(() => {
    if (active > 0) {
      const t = setTimeout(() => setShow(true), 150);
      return () => clearTimeout(t);
    }
    setShow(false);
    return undefined;
  }, [active]);

  if (!show) return null;
  return (
    <div className="fixed top-0 left-0 right-0 z-[200] h-0.5 overflow-hidden pointer-events-none"
      role="progressbar" aria-label="loading">
      <div className="lbx-progress-bar" />
    </div>
  );
}
