"use client";

// Platform launcher: the home-of-homes for the data engine. Tiles in flywheel order, each explaining the
// plane's job in plain language, a live-state badge where the plane exposes one, and its destinations.
// Blender-style: rounded panels, blue hover/active, a header hint and gate legend so a first-time user
// understands what each plane does and how data flows between them.

import Link from "next/link";
import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { HintBar } from "@/components/ui/kit";
import { PLATFORMS_ORDERED, type Platform } from "@/platforms/registry";
import { apiGet } from "@/lib/api";

// a lightweight live-state probe per platform: {text, tone} or null
async function probe(id: string): Promise<{ text: string; tone: string } | null> {
  try {
    if (id === "sanyx") {
      const d = await apiGet<{ sessions?: { decision: string | null }[] }>("/api/sanyx/board?limit=500");
      const q = (d.sessions ?? []).filter((s) => s.decision === "quarantine").length;
      const run = (d.sessions ?? []).filter((s: { decision: string | null }) => s.decision).length;
      return { text: `${run} scored · ${q} quarantined`, tone: q ? "warn" : "pass" };
    }
    if (id === "sievyx") {
      const d = await apiGet<{ n?: number }>("/api/sievyx/composition?top_n=200");
      return { text: `${d.n ?? 0} in priority window`, tone: "neutral" };
    }
    if (id === "oraclyx") {
      const d = await apiGet<{ total?: number }>("/api/oraclyx/board");
      return { text: `${d.total ?? 0} pseudo-labels`, tone: "neutral" };
    }
    if (id === "forgyx") {
      const d = await apiGet<{ benchmarks?: unknown[] }>("/api/forgyx/benchmarks");
      return { text: `${(d.benchmarks ?? []).length} benchmarks`, tone: "neutral" };
    }
    if (id === "verdyx") {
      const d = await apiGet<{ pairs?: { verdict: string }[] }>("/api/verdyx/pairs");
      const rej = (d.pairs ?? []).filter((p: { verdict: string }) => p.verdict === "reject").length;
      return { text: `${(d.pairs ?? []).length} verdicts · ${rej} reject`, tone: rej ? "block" : "neutral" };
    }
  } catch {
    return null;
  }
  return null;
}

const TONE: Record<string, string> = { pass: "text-pass", warn: "text-warn", block: "text-block", neutral: "text-info" };
const GATE_DOT: Record<string, string> = { health: "bg-pass", calibration: "bg-info", eval: "bg-warn", benchmark: "bg-accent" };

function Tile({ p }: { p: Platform }) {
  const [live, setLive] = useState<{ text: string; tone: string } | null>(null);
  useEffect(() => { probe(p.id).then(setLive); }, [p.id]);
  return (
    <Link href={p.home} data-tip={p.role}
      className="group block bg-panel border border-line rounded p-4 hover:border-accent hover:bg-head/40 transition-colors">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-ink-3 tracking-wider font-mono">{p.glyph}</span>
        <div className="flex items-center gap-2">
          {p.gate && <span className={`w-1.5 h-1.5 rounded-full ${GATE_DOT[p.gate] ?? "bg-ink-3"}`} data-tip={`gate: ${p.gate}`} />}
          {p.flywheelStage != null && <span className="text-[10px] text-ink-3">stage {p.flywheelStage}</span>}
        </div>
      </div>
      <div className="mt-3 text-[15px] text-ink font-display font-semibold group-hover:text-accent-2">{p.label}</div>
      <div className="mt-1 text-[12px] leading-snug text-ink-3">{p.role}</div>
      {live && <div className={`mt-2 text-[11px] font-mono ${TONE[live.tone]}`}>{live.text}</div>}
      <div className="mt-3 flex flex-wrap gap-1">
        {p.nav.slice(0, 5).map((n) => (
          <span key={n.href} className="text-[10px] text-ink-2 bg-bg-2 border border-line rounded px-1.5 py-0.5">{n.label}</span>
        ))}
        {p.nav.length > 5 && <span className="text-[10px] text-ink-3 px-1 py-0.5">+{p.nav.length - 5}</span>}
      </div>
    </Link>
  );
}

export default function PlatformLauncher() {
  const [inSync, setInSync] = useState<boolean | null>(null);
  useEffect(() => {
    apiGet<{ platforms?: { id: string }[] }>("/api/platforms").then((d) => {
      const ids = (d.platforms ?? []).map((p: { id: string }) => p.id);
      setInSync(PLATFORMS_ORDERED.every((p) => ids.includes(p.id)));
    }).catch(() => setInSync(false));
  }, []);

  return (
    <PageShell active="PLATFORMS" title="Platforms"
      subtitle="one data engine, seven planes, one spine"
      right={<span className="text-[12px] text-ink-3">{inSync == null ? "checking registry" : inSync ? <span className="text-pass">registry in sync</span> : <span className="text-warn">registry drift</span>}</span>}>
      <div className="p-6 max-w-6xl">
        <HintBar>
          Each platform is one plane of the India AV data engine. Data flows through them in flywheel order:
          ingest QA, calibration, curation, annotation, pseudo-truth, evaluation, then edge deploy. Click a tile
          to open that plane, or use the platform switcher (top-left) from anywhere.
        </HintBar>

        {/* flywheel flow, labeled */}
        <div className="mb-5 flex flex-wrap items-center gap-1.5 text-[11px] text-ink-3 bg-bg-2 border border-line rounded px-3 py-2">
          <span className="uppercase mr-1 text-ink-2 tracking-wide">Flywheel</span>
          {PLATFORMS_ORDERED.filter((p) => p.flywheelStage != null).sort((a, b) => (a.flywheelStage ?? 0) - (b.flywheelStage ?? 0)).map((p, i, a) => (
            <span key={p.id} className="flex items-center gap-1.5">
              <Link href={p.home} data-tip={p.role} className="hover:text-accent-2 text-ink-2">{p.label}</Link>
              {i < a.length - 1 && <span className="text-line-2">&rarr;</span>}
            </span>
          ))}
          <span className="text-line-2">&rarr;</span>
          <span className="text-ink-3">deploy</span>
          <span className="text-accent-2 ml-1" data-tip="the loop closes: failures and gaps feed the next collection + labeling cycle">&#8630; loop</span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {PLATFORMS_ORDERED.map((p) => <Tile key={p.id} p={p} />)}
        </div>

        {/* gate legend, self-explanatory */}
        <div className="mt-5 flex flex-wrap items-center gap-4 text-[11px] text-ink-3">
          <span className="uppercase tracking-wide">Gates</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-pass" /> health</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-info" /> calibration</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-warn" /> eval</span>
          <span className="flex items-center gap-1.5"><span className="w-1.5 h-1.5 rounded-full bg-accent" /> benchmark</span>
          <span className="text-ink-3">a gate can block a session or a model from advancing</span>
        </div>
      </div>
    </PageShell>
  );
}
