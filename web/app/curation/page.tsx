"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useJobStream } from "@/lib/useEventStream";
import { api , humanizeError } from "@/lib/api";
import type { CurationSummary } from "@/lib/types";
import PageShell from "@/components/shell/PageShell";

// Active-learning curation: label the RIGHT frames. Novel frames (far from everything = coverage gaps)
// are worth labeling; near-duplicates are worth skipping. Powered by DINOv2 frame embeddings.

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="panel px-3 py-3">
      <div className="text-[10.5px] font-medium uppercase tracking-[0.06em] text-ink-3">{label}</div>
      <div className="text-2xl font-semibold text-ink mt-1">{value}</div>
      {sub && <div className="text-[12px] text-ink-3 mt-0.5">{sub}</div>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="text-[10.5px] font-medium uppercase tracking-[0.06em] text-ink-3 border-b hairline px-3 py-2">{title}</div>
      <div className="p-3">{children}</div>
    </section>
  );
}

export default function CurationPage() {
  const router = useRouter();
  const [sum, setSum] = useState<CurationSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  async function refresh() {
    try {
      setSum(await api.curationSummary());
    } catch {
      /* ignore */
    }
  }

  // The curation summary changes when embedding or autolabel work lands, both of which are on the shared
  // job stream, so this follows the stream instead of running its own three-second timer.
  const jobStream = useJobStream();
  useEffect(() => { refresh(); }, []);
  useEffect(() => { if (jobStream.data) refresh(); }, [jobStream.data]);

  async function embed() {
    setBusy(true);
    setMsg(null);
    try {
      await api.curationEmbed();
      setMsg("embedding frames (DINOv2) in the background - the count climbs as it runs");
    } catch (e) {
      setMsg(humanizeError(e).includes("503") ? "GPU busy (training). Try after it finishes." : humanizeError(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <PageShell
      active="CURATION"
      title="Curation"
      primaryAction={
        <button onClick={embed} disabled={busy} className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-50">
          {busy ? "..." : "compute embeddings"}
        </button>
      }
    >
      <div className="p-4 space-y-4">
        {msg && <div className="panel px-3 py-1.5 text-[12px] text-warn">{msg}</div>}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="frames embedded" value={sum ? `${sum.embedded}/${sum.total_frames}` : "-"} sub={sum ? `${sum.embedded_pct}%` : undefined} />
          <Stat label="mean nn similarity" value={sum?.mean_nn_sim != null ? sum.mean_nn_sim.toFixed(3) : "-"} sub="lower = more diverse corpus" />
          <Stat label="near-duplicate frames" value={sum ? String(sum.duplicate_frames) : "-"} sub="candidates to skip" />
          <Stat label="coverage gaps" value={sum ? String(sum.novel.length) : "-"} sub="novel frames to label" />
        </div>

        {!sum || sum.embedded < 2 ? (
          <div className="panel px-3 py-8 text-center text-[12.5px] text-ink-3">
            no frame embeddings yet. Click <span className="text-ink-2">compute embeddings</span> (or run{" "}
            <span className="text-ink-2">make embed-frames ARGS=&quot;--all&quot;</span>).
          </div>
        ) : (
          <>
            <Section title={`label these next - coverage gaps (${sum.novel.length})`}>
              <div className="grid grid-cols-3 md:grid-cols-6 lg:grid-cols-8 gap-2">
                {sum.novel.map((f) => (
                  <button key={f.frame_id} onClick={() => router.push(`/frame/${f.frame_id}`)}
                    className="border border-line hover:border-accent" title={`novelty ${f.novelty}`}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={f.image_url} alt="" className="w-full h-16 object-cover bg-bg-2" />
                    <div className="text-[10.5px] text-ink-3 px-1 py-0.5">nov {f.novelty}</div>
                  </button>
                ))}
              </div>
            </Section>

            <Section title={`near-duplicates - skip one of each (${sum.duplicates.length})`}>
              {sum.duplicates.length ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {sum.duplicates.map((d, i) => (
                    <div key={i} className="flex items-center gap-1 border border-line p-1">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={d.a_url} alt="" className="w-1/2 h-14 object-cover bg-bg-2 cursor-pointer" onClick={() => router.push(`/frame/${d.a}`)} />
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={d.b_url} alt="" className="w-1/2 h-14 object-cover bg-bg-2 cursor-pointer" onClick={() => router.push(`/frame/${d.b}`)} />
                      <span className="text-[10.5px] text-block absolute">{d.sim}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-[12.5px] text-ink-3 py-3 text-center">no near-duplicates above threshold</div>
              )}
            </Section>
          </>
        )}
      </div>
    </PageShell>
  );
}
