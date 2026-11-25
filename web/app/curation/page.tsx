"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useJobStream } from "@/lib/useEventStream";
import { api , humanizeError } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";
import type { CurationSummary } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";
import EmptyState from "@/components/ui/EmptyState";
import Thumb from "@/components/engine/Thumb";
import { Stat } from "@/components/ui/kit";
import { num, pct } from "@/lib/format";

// Active-learning curation: label the RIGHT frames. Novel frames (far from everything = coverage gaps)
// are worth labeling; near-duplicates are worth skipping. Powered by DINOv2 frame embeddings.

function Section({ title, desc, children }: { title: string; desc?: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2 className="section-title">{title}</h2>
        {desc && <span className="text-[12px] text-ink-3">{desc}</span>}
      </div>
      <div className="p-3.5">{children}</div>
    </section>
  );
}

export default function CurationPage() {
  const router = useRouter();
  // A failed read is ignored, as before: the last summary stays up.
  const sumQ = useQuery(["curationSummary"], () => api.curationSummary());
  const sum: CurationSummary | null = sumQ.data ?? null;
  const refresh = sumQ.reload;
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // The curation summary changes when embedding or autolabel work lands, both of which are on the shared
  // job stream, so this follows the stream instead of running its own three-second timer.
  const jobStream = useJobStream();
  useEffect(() => { if (jobStream.data) void refresh(); }, [jobStream.data, refresh]);

  async function embed() {
    setBusy(true);
    setMsg(null);
    try {
      await api.curationEmbed();
      setMsg("Comparing frames in the background. The counts below climb as it runs.");
    } catch (e) {
      setMsg(humanizeError(e).includes("503") ? "The GPU is busy with training. Try again when it finishes." : humanizeError(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <PageShell
      active="CURATION"
      title="Curation"
      subtitle="Which frames are worth labelling next, and which are near-copies you can skip"
      primaryAction={
        <button type="button" onClick={embed} disabled={busy} className="btn btn-primary">
          {busy ? "Starting..." : "Compare frames"}
        </button>
      }
    >
      <div className="page-wrap space-y-4">
        {msg && <div role="status" className="panel px-3.5 py-2.5 text-[12.5px] text-ink-2">{msg}</div>}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Frames compared" value={sum ? `${num(sum.embedded)} / ${num(sum.total_frames)}` : "-"}
            sub={sum ? `${pct(sum.total_frames ? sum.embedded / sum.total_frames : 0)} of the corpus` : undefined} />
          <Stat label="Similarity to nearest" value={sum?.mean_nn_sim != null ? pct(sum.mean_nn_sim) : "-"}
            sub="Lower means a more varied corpus" />
          <Stat label="Near-copies" value={sum ? num(sum.duplicate_frames) : "-"} sub="Frames you can skip" />
          <Stat label="Unusual frames" value={sum ? num(sum.novel.length) : "-"} sub="Unlike anything labelled so far" tone={sum?.novel.length ? "accent" : undefined} />
        </div>

        {!sum || sum.embedded < 2 ? (
          <div className="panel">
            <EmptyState icon="layers" title="No frames compared yet"
              action={<button type="button" onClick={embed} disabled={busy} className="btn btn-primary">{busy ? "Starting..." : "Compare frames"}</button>}>
              Comparing every frame with every other finds the unusual ones worth labelling first and the
              near-copies worth skipping. It runs on the GPU in the background.
            </EmptyState>
          </div>
        ) : (
          <>
            <Section title={`Label these next (${num(sum.novel.length)})`} desc="Frames least like anything already in the corpus">
              <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2.5">
                {sum.novel.map((f) => (
                  <button key={f.frame_id} type="button" onClick={() => router.push(`/frame/${f.frame_id}`)}
                    className="rounded overflow-hidden border border-line hover:border-accent text-left"
                    data-tip={`How unusual: ${f.novelty}`} aria-label="open this frame in the editor">
                    <Thumb src={f.image_url} className="w-full h-20" />
                    <div className="text-[11px] text-ink-3 px-1.5 py-1">Unusual {f.novelty}</div>
                  </button>
                ))}
              </div>
            </Section>

            <Section title={`Near-copies (${num(sum.duplicates.length)})`} desc="Pairs that look almost the same: label one of each">
              {sum.duplicates.length ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                  {sum.duplicates.map((d, i) => (
                    <div key={i} className="relative flex items-center gap-1 rounded border border-line p-1">
                      <button type="button" className="w-1/2 h-16" aria-label="open the first frame of this duplicate pair"
                        onClick={() => router.push(`/frame/${d.a}`)}>
                        <Thumb src={d.a_url} className="w-full h-full rounded-sm" />
                      </button>
                      <button type="button" className="w-1/2 h-16" aria-label="open the second frame of this duplicate pair"
                        onClick={() => router.push(`/frame/${d.b}`)}>
                        <Thumb src={d.b_url} className="w-full h-full rounded-sm" />
                      </button>
                      <span className="absolute top-1.5 left-1.5 text-[11px] text-ink bg-bg/80 rounded px-1">{pct(d.sim)} alike</span>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState compact icon="check" title="No near-copies found">
                  No two compared frames are similar enough to skip one.
                </EmptyState>
              )}
            </Section>
          </>
        )}
      </div>
    </PageShell>
  );
}
