"use client";

// Platform launcher: the home-of-homes for the LabeloxAV data engine. One tile per platform (annotation core
// + the six folded subsystems), in flywheel order. Operational Materialism: matte graphite tiles, monospace,
// binary hover (border on / off), color earned only by state. A tile opens that platform's home route.

import Link from "next/link";
import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { PLATFORMS_ORDERED } from "@/platforms/registry";

export default function PlatformLauncher() {
  // Reconcile against the backend registry so a drift between web/platforms/registry.ts and
  // platforms/registry.py is visible, not silent.
  const [backendIds, setBackendIds] = useState<string[] | null>(null);
  useEffect(() => {
    fetch("/api/platforms")
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then((d) => setBackendIds((d.platforms ?? []).map((p: { id: string }) => p.id)))
      .catch(() => setBackendIds([]));
  }, []);

  const drift =
    backendIds && backendIds.length
      ? PLATFORMS_ORDERED.filter((p) => !backendIds.includes(p.id)).map((p) => p.id)
      : [];

  return (
    <PageShell
      active="PLATFORMS"
      title="Platforms"
      right={
        <span className="font-mono text-[11px] text-ink-3">
          one engine, one spine ·{" "}
          {backendIds == null ? (
            "checking backend"
          ) : drift.length ? (
            <span className="text-warn">registry drift: {drift.join(", ")}</span>
          ) : (
            <span className="text-pass">registry in sync</span>
          )}
        </span>
      }
    >
      <div className="p-6">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-w-5xl">
          {PLATFORMS_ORDERED.map((p) => (
            <Link
              key={p.id}
              href={p.home}
              className="group block border border-line bg-panel p-4 hover:border-accent"
            >
              <div className="flex items-start justify-between">
                <span className="font-mono text-[11px] text-ink-3 tracking-wider">{p.glyph}</span>
                {p.flywheelStage != null && (
                  <span className="font-mono text-[10px] text-ink-3">stage {p.flywheelStage}</span>
                )}
              </div>
              <div className="mt-3 font-mono text-sm text-ink group-hover:text-accent">{p.label}</div>
              <div className="mt-1 font-mono text-[11px] leading-snug text-ink-3">{p.role}</div>
              <div className="mt-3 flex items-center gap-2 font-mono text-[10px] text-ink-3">
                <span className="border border-line px-1.5 py-0.5 rounded">{p.home}</span>
                {p.gate && (
                  <span className="border border-line px-1.5 py-0.5 rounded uppercase">gate: {p.gate}</span>
                )}
              </div>
            </Link>
          ))}
        </div>
      </div>
    </PageShell>
  );
}
