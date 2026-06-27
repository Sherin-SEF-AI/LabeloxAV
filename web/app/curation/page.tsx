"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import type { CurationSummary } from "@/lib/types";
import TopNav from "@/components/TopNav";

// Active-learning curation: label the RIGHT frames. Novel frames (far from everything = coverage gaps)
// are worth labeling; near-duplicates are worth skipping. Powered by DINOv2 frame embeddings.

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="panel px-3 py-3">
      <div className="font-mono text-[11px] uppercase text-ink-3">{label}</div>
      <div className="font-mono text-2xl text-ink mt-1">{value}</div>
      {sub && <div className="font-mono text-[11px] text-ink-3 mt-0.5">{sub}</div>}
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="font-mono text-[11px] uppercase text-ink-3 border-b hairline px-3 py-2">{title}</div>
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

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  async function embed() {
    setBusy(true);
    setMsg(null);
    try {
      await api.curationEmbed();
      setMsg("embedding frames (DINOv2) in the background - the count climbs as it runs");
    } catch (e) {
      setMsg(String(e).includes("503") ? "GPU busy (training). Try after it finishes." : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen flex flex-col">
      <TopNav active="CURATION" right={
        <button onClick={embed} disabled={busy} className="border border-line px-2 py-0.5 hover:border-accent disabled:opacity-50">
          {busy ? "..." : "compute embeddings"}
        </button>
      } />
      <main className="flex-1 overflow-auto p-4 space-y-4">
        {msg && <div className="panel px-3 py-1.5 font-mono text-[11px] text-warn">{msg}</div>}

        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="frames embedded" value={sum ? `${sum.embedded}/${sum.total_frames}` : "-"} sub={sum ? `${sum.embedded_pct}%` : undefined} />
          <Stat label="mean nn similarity" value={sum?.mean_nn_sim != null ? sum.mean_nn_sim.toFixed(3) : "-"} sub="lower = more diverse corpus" />
          <Stat label="near-duplicate frames" value={sum ? String(sum.duplicate_frames) : "-"} sub="candidates to skip" />
          <Stat label="coverage gaps" value={sum ? String(sum.novel.length) : "-"} sub="novel frames to label" />
        </div>

        {!sum || sum.embedded < 2 ? (
          <div className="panel px-3 py-8 text-center font-mono text-xs text-ink-3">
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
                    <div className="font-mono text-[9px] text-ink-3 px-1 py-0.5">nov {f.novelty}</div>
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
                      <span className="font-mono text-[9px] text-block absolute">{d.sim}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="font-mono text-xs text-ink-3 py-3 text-center">no near-duplicates above threshold</div>
              )}
            </Section>
          </>
        )}
      </main>
    </div>
  );
}
