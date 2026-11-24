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
      <div className="p-6" role="status" aria-live="polite">
        <div className="panel px-3 py-2 text-[12px] text-ink-3 inline-flex items-center gap-2">
          <span className="inline-block w-2 h-2 rounded-full bg-accent animate-pulse" aria-hidden />
          {t("load.route", "loading {what}", { what: label(name) })}
        </div>
      </div>
    </PageShell>
  );
}
