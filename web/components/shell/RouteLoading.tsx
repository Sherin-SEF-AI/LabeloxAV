"use client";

import PageShell from "./PageShell";
import { useLabel, useT } from "@/lib/i18n";

// What a heavy route shows while its code is still arriving.
//
// The routes that use this pull in a rendering engine (three, maplibre, konva, uplot) or a very large page
// module. Without a loading.tsx a click on "LiDAR" did nothing visible until the chunk had downloaded and
// parsed: the old page sat there, apparently ignoring the click. With one, the navigation commits at once
// to the same chrome every page has, with the sidebar already showing where you are going.
export default function RouteLoading({ name }: { name: string }) {
  const t = useT();
  const label = useLabel();
  return (
    <PageShell active={name}>
      {/* Placeholder shapes in the layout most pages have (a row of tiles, then a list), so the page does
          not jump when it arrives; the sentence is for screen readers. */}
      <div className="page-wrap space-y-4" role="status" aria-live="polite">
        <span className="sr-only">{t("load.route", "loading {what}", { what: label(name) })}</span>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4" aria-hidden>
          {[0, 1, 2, 3].map((i) => (
            <div key={i} className="panel px-4 py-3.5 space-y-2.5">
              <div className="skeleton h-2.5 w-20" />
              <div className="skeleton h-5 w-24" />
            </div>
          ))}
        </div>
        <div className="panel p-4 space-y-3" aria-hidden>
          {[46, 70, 58, 64, 40, 52].map((w, i) => (
            <div key={i} className="flex items-center gap-3">
              <div className="skeleton h-3" style={{ width: `${w}%` }} />
              <div className="skeleton h-3 w-14 ml-auto" />
            </div>
          ))}
        </div>
      </div>
    </PageShell>
  );
}
