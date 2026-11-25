"use client";

// Platform launcher: the home-of-homes for the data engine. Tiles in flywheel order, each explaining the
// plane's job in plain language, a live-state badge where the plane exposes one, and its destinations.
// Blender-style: rounded panels, blue hover/active, a header hint and gate legend so a first-time user
// understands what each plane does and how data flows between them.

import Link from "next/link";
import { useEffect, useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { PLATFORMS_ORDERED, type Platform } from "@/platforms/registry";
import { titleFor } from "@/lib/sidebar";
import { apiGet } from "@/lib/api";

// a lightweight live-state probe per platform: {text, tone} or null
async function probe(id: string): Promise<{ text: string; tone: string } | null> {
  try {
    if (id === "sanyx") {
      const d = await apiGet<{ sessions?: { decision: string | null }[] }>("/api/sanyx/board?limit=500");
      const q = (d.sessions ?? []).filter((s) => s.decision === "quarantine").length;
      const run = (d.sessions ?? []).filter((s: { decision: string | null }) => s.decision).length;
      return { text: `${run} recordings scored · ${q} set aside`, tone: q ? "warn" : "pass" };
    }
    if (id === "sievyx") {
      const d = await apiGet<{ n?: number }>("/api/sievyx/composition?top_n=200");
      return { text: `${d.n ?? 0} in the labelling priority window`, tone: "neutral" };
    }
    if (id === "oraclyx") {
      const d = await apiGet<{ total?: number }>("/api/oraclyx/board");
      return { text: `${d.total ?? 0} machine labels`, tone: "neutral" };
    }
    if (id === "forgyx") {
      const d = await apiGet<{ benchmarks?: unknown[] }>("/api/forgyx/benchmarks");
      return { text: `${(d.benchmarks ?? []).length} device benchmarks`, tone: "neutral" };
    }
    if (id === "verdyx") {
      const d = await apiGet<{ pairs?: { verdict: string }[] }>("/api/verdyx/pairs");
      const rej = (d.pairs ?? []).filter((p: { verdict: string }) => p.verdict === "reject").length;
      return { text: `${(d.pairs ?? []).length === 1 ? "1 comparison" : `${(d.pairs ?? []).length} comparisons`} · ${rej} rejected`, tone: rej ? "block" : "neutral" };
    }
  } catch {
    return null;
  }
  return null;
}

const TONE: Record<string, string> = { pass: "text-pass", warn: "text-warn", block: "text-block", neutral: "text-info" };
const GATE_DOT: Record<string, string> = { health: "bg-pass", calibration: "bg-info", eval: "bg-warn", benchmark: "bg-accent" };
const GATE_NAME: Record<string, string> = { health: "Health", calibration: "Calibration", eval: "Evaluation", benchmark: "Benchmark" };

// The registry still carries the retired code names (it mirrors the server's copy); this page shows the
// names the sidebar uses, and says what each part does in plain words.
const NAME: Record<string, string> = {
  labelox: "Labelling", sanyx: "Ingest health", calyx: "Calibration", sievyx: "Curation",
  oraclyx: "Pseudo ground truth", verdyx: "Slice evaluation", forgyx: "Edge deployment",
};
const ROLE: Record<string, string> = {
  labelox: "Auto-labelling and the human workspace where labels are checked and fixed.",
  sanyx: "Scores every recording's quality and sets aside the bad ones before anyone labels them.",
  calyx: "Watches each camera rig for calibration drift over time.",
  sievyx: "Decides what is worth labelling next: rare scenes, the long tail, near-duplicates.",
  oraclyx: "Combines several models offline to label the easy majority and route disagreements to people.",
  verdyx: "Compares a new model against the current one, slice by slice, before it is promoted.",
  forgyx: "Shrinks, compiles and benchmarks models for the computers in the vehicles.",
};
const nameOf = (p: Platform) => NAME[p.id] ?? p.label;
const navLabel = (href: string, fallback: string) => titleFor(href)?.title ?? fallback;

function Tile({ p, step }: { p: Platform; step: number | null }) {
  const [live, setLive] = useState<{ text: string; tone: string } | null>(null);
  useEffect(() => { probe(p.id).then(setLive); }, [p.id]);
  return (
    <div className="panel lift p-4 flex flex-col">
      <div className="flex items-center gap-2">
        {step != null && <span className="label-caps">Step {step}</span>}
        {p.gate && (
          <span className="ml-auto flex items-center gap-1.5 text-[11px] text-ink-3" data-tip="Can stop a recording or a model from moving on">
            <span className={`w-1.5 h-1.5 rounded-full ${GATE_DOT[p.gate] ?? "bg-ink-3"}`} aria-hidden />
            {GATE_NAME[p.gate] ?? p.gate} gate
          </span>
        )}
      </div>
      <Link href={p.home} className="mt-2 text-[15px] text-ink font-semibold hover:text-accent-2">{nameOf(p)}</Link>
      <p className="mt-1 text-[12.5px] leading-snug text-ink-3">{ROLE[p.id] ?? p.role}</p>
      {live && <div className={`mt-2 text-[12px] tabular-nums ${TONE[live.tone]}`}>{live.text}</div>}
      <div className="mt-auto pt-3 flex flex-wrap gap-1.5">
        {p.nav.slice(0, 5).map((n) => (
          <Link key={n.href} href={n.href} data-tip={n.hint} className="chip">{navLabel(n.href, n.label)}</Link>
        ))}
        {p.nav.length > 5 && <span className="text-[12px] text-ink-3 px-1 self-center">+{p.nav.length - 5} more</span>}
      </div>
    </div>
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

  const stages = PLATFORMS_ORDERED.filter((p) => p.flywheelStage != null)
    .sort((a, b) => (a.flywheelStage ?? 0) - (b.flywheelStage ?? 0));

  return (
    <PageShell active="Platforms" title="Platforms"
      subtitle="The parts of the data engine, in the order data flows through them."
      meta={inSync === false
        ? <span className="text-warn" data-tip="The server's list of platforms differs from this page's; an administrator should check the deployment">Platform list out of date</span>
        : undefined}>
      <div className="page-wrap space-y-5">
        {/* flywheel flow, labeled */}
        <section className="panel px-4 py-3">
          <div className="label-caps mb-2">How data flows</div>
          <ol className="flex flex-wrap items-center gap-x-1.5 gap-y-2 text-[12.5px]">
            {stages.map((p, i) => (
              <li key={p.id} className="flex items-center gap-1.5">
                <Link href={p.home} className="chip">{nameOf(p)}</Link>
                {i < stages.length - 1 && <span className="text-ink-3" aria-hidden>&rarr;</span>}
              </li>
            ))}
            <li className="flex items-center gap-1.5">
              <span className="text-ink-3" aria-hidden>&rarr;</span>
              <span className="text-ink-2">in the vehicle</span>
              <span className="text-accent-2 ml-1" data-tip="Failures and gaps found in the field feed the next round of recording and labelling">&#8630; and back to the start</span>
            </li>
          </ol>
        </section>

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
          {/* In flow order, numbered the way the strip above reads, so step 4 is the fourth tile. */}
          {[...stages, ...PLATFORMS_ORDERED.filter((p) => p.flywheelStage == null)].map((p) => {
            const i = stages.indexOf(p);
            return <Tile key={p.id} p={p} step={i >= 0 ? i + 1 : null} />;
          })}
        </div>

        {/* gate legend, self-explanatory */}
        <div className="flex flex-wrap items-center gap-4 text-[12px] text-ink-3">
          <span className="label-caps">Gates</span>
          {Object.entries(GATE_NAME).map(([g, name]) => (
            <span key={g} className="flex items-center gap-1.5"><span className={`w-1.5 h-1.5 rounded-full ${GATE_DOT[g]}`} aria-hidden /> {name}</span>
          ))}
          <span>A gate can stop a recording or a model from moving to the next step.</span>
        </div>
      </div>
    </PageShell>
  );
}
