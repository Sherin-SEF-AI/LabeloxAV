"use client";

// Platform launcher: the home-of-homes for the data engine. Tiles in flywheel order, each showing the
// platform's role, a live-state badge where the plane exposes one, and its destinations. Operational
// Materialism: matte graphite tiles, monospace, binary hover, color earned only by live state.

import Link from "next/link";
import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { PLATFORMS_ORDERED, type Platform } from "@/platforms/registry";

// a lightweight live-state probe per platform: [label, tone] or null
async function probe(id: string): Promise<{ text: string; tone: string } | null> {
  try {
    if (id === "sanyx") {
      const d = await fetch("/api/sanyx/board?limit=500").then((r) => r.json());
      const q = (d.sessions ?? []).filter((s: { decision: string }) => s.decision === "quarantine").length;
      const run = (d.sessions ?? []).filter((s: { decision: string | null }) => s.decision).length;
      return { text: `${run} scored · ${q} quarantined`, tone: q ? "warn" : "pass" };
    }
    if (id === "sievyx") {
      const d = await fetch("/api/sievyx/composition?top_n=200").then((r) => r.json());
      return { text: `${d.n ?? 0} in priority window`, tone: "neutral" };
    }
    if (id === "oraclyx") {
      const d = await fetch("/api/oraclyx/board").then((r) => r.json());
      return { text: `${d.total ?? 0} pseudo-labels`, tone: "neutral" };
    }
    if (id === "forgyx") {
      const d = await fetch("/api/forgyx/benchmarks").then((r) => r.json());
      return { text: `${(d.benchmarks ?? []).length} benchmarks`, tone: "neutral" };
    }
    if (id === "verdyx") {
      const d = await fetch("/api/verdyx/pairs").then((r) => r.json());
      const rej = (d.pairs ?? []).filter((p: { verdict: string }) => p.verdict === "reject").length;
      return { text: `${(d.pairs ?? []).length} verdicts · ${rej} reject`, tone: rej ? "block" : "neutral" };
    }
  } catch {
    return null;
  }
  return null;
}

const TONE: Record<string, string> = { pass: "text-pass", warn: "text-warn", block: "text-block", neutral: "text-ink-3" };

function Tile({ p }: { p: Platform }) {
  const [live, setLive] = useState<{ text: string; tone: string } | null>(null);
  useEffect(() => { probe(p.id).then(setLive); }, [p.id]);
  return (
    <Link href={p.home} className="group block border border-line bg-panel p-4 hover:border-accent">
      <div className="flex items-start justify-between">
        <span className="font-mono text-[11px] text-ink-3 tracking-wider">{p.glyph}</span>
        {p.flywheelStage != null && <span className="font-mono text-[10px] text-ink-3">stage {p.flywheelStage}</span>}
      </div>
      <div className="mt-3 font-mono text-sm text-ink group-hover:text-accent">{p.label}</div>
      <div className="mt-1 font-mono text-[11px] leading-snug text-ink-3">{p.role}</div>
      {live && <div className={`mt-2 font-mono text-[10px] ${TONE[live.tone]}`}>{live.text}</div>}
      <div className="mt-3 flex flex-wrap gap-1">
        {p.nav.slice(0, 4).map((n) => (
          <span key={n.href} className="font-mono text-[9px] text-ink-3 border border-line rounded px-1 py-0.5">{n.label}</span>
        ))}
        {p.nav.length > 4 && <span className="font-mono text-[9px] text-ink-3">+{p.nav.length - 4}</span>}
      </div>
    </Link>
  );
}

export default function PlatformLauncher() {
  const [inSync, setInSync] = useState<boolean | null>(null);
  useEffect(() => {
    fetch("/api/platforms").then((r) => r.json()).then((d) => {
      const ids = (d.platforms ?? []).map((p: { id: string }) => p.id);
      setInSync(PLATFORMS_ORDERED.every((p) => ids.includes(p.id)));
    }).catch(() => setInSync(false));
  }, []);

  return (
    <PageShell active="PLATFORMS" title="Platforms"
      right={<span className="font-mono text-[11px] text-ink-3">one engine, one spine · {inSync == null ? "checking" : inSync ? <span className="text-pass">registry in sync</span> : <span className="text-warn">registry drift</span>}</span>}>
      <div className="p-6">
        {/* flywheel flow */}
        <div className="mb-5 flex flex-wrap items-center gap-1 font-mono text-[10px] text-ink-3">
          <span className="uppercase mr-1">flywheel</span>
          {PLATFORMS_ORDERED.filter((p) => p.flywheelStage != null).map((p, i, a) => (
            <span key={p.id} className="flex items-center gap-1">
              <Link href={p.home} className="hover:text-accent">{p.glyph}</Link>
              {i < a.length - 1 && <span className="text-line">&rarr;</span>}
            </span>
          ))}
          <span className="text-line">&rarr;</span>
          <span className="text-ink-3">deploy</span>
          <span className="text-line">&#8630;</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 max-w-5xl">
          {PLATFORMS_ORDERED.map((p) => <Tile key={p.id} p={p} />)}
        </div>
      </div>
    </PageShell>
  );
}
