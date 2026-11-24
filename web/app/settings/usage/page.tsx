"use client";

import { useState } from "react";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { apiGet, apiPost } from "@/lib/api";
import { summarize, type UsageRow } from "@/lib/llmUsage";
import { useQuery } from "@/lib/useQuery";

// What the language and vision models were asked and what it cost, and what this server has on its GPU.
//
// Two cloud providers are wired in and, before the ledger (services/llm/ledger.py), nothing counted a
// token. This page is that ledger read back: one line per model and purpose, local and cloud alike. Cost
// is tokens times the price an administrator configured, so a cloud model with no price shows as
// "unpriced" and is named at the top rather than added in as zero.
//
// The second card is the API process's model registry (core/model_manager.py). On a Docker deployment
// auto-label runs in the GPU worker, a separate process, so this is the editor's models, not the worker's.

type Usage = {
  since: string; days: number; rows: UsageRow[];
  totals: { calls: number; failures: number; input_tokens: number; output_tokens: number; images: number;
            cost_usd: number; cloud_cost_usd: number };
  unpriced: string[];
  budget: { daily_usd: number | null; spent_today_usd: number; exhausted: boolean };
};

type Loaded = {
  device: string; vram_free_mb: number | null; vram_total_mb: number | null;
  models: { key: string; est_mb: number; device: string | null; leases: number; pinned: boolean;
            uses: number; load_s: number; idle_s: number; age_s: number }[];
};

const RANGES = [7, 30, 90] as const;
const n = (v: number) => v.toLocaleString();
const usd = (v: number | null) => (v === null ? "unpriced" : `$${v.toFixed(v < 1 ? 4 : 2)}`);

function ModelUsage() {
  const [days, setDays] = useState<(typeof RANGES)[number]>(30);
  const q = useQuery([`/api/system/llm-usage?days=${days}`], () => apiGet<Usage>(`/api/system/llm-usage?days=${days}`));
  const data = q.data ?? null;
  const groups = data ? summarize(data.rows) : [];

  return (
    <section className="panel">
      <header className="flex flex-wrap items-start justify-between gap-3 px-4 pt-4 pb-3">
        <div className="min-w-0">
          <h2 className="text-[14px] font-semibold text-ink">Model usage</h2>
          <p className="mt-0.5 text-[12.5px] text-ink-3 max-w-[65ch]">
            Every call to a language or vision model, by model and by what it was for. Local models cost nothing;
            a cloud model costs what its configured price says.
          </p>
        </div>
        <div className="flex gap-1" role="group" aria-label="Period">
          {RANGES.map((d) => (
            <button key={d} className={`btn ${d === days ? "btn-primary" : ""}`} aria-pressed={d === days}
              onClick={() => setDays(d)}>{d} days</button>
          ))}
        </div>
      </header>
      <div className="px-4 pb-4 flex flex-col gap-3">
        <LoadState {...q.loadState} />
        {data && (
          <>
            <dl className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-[12.5px]">
              <div><dt className="text-ink-3">Calls</dt><dd className="text-ink text-[15px] tabular-nums">{n(data.totals.calls)}</dd></div>
              <div><dt className="text-ink-3">Failed</dt><dd className="text-ink text-[15px] tabular-nums">{n(data.totals.failures)}</dd></div>
              <div><dt className="text-ink-3">Tokens in / out</dt><dd className="text-ink text-[15px] tabular-nums">{n(data.totals.input_tokens)} / {n(data.totals.output_tokens)}</dd></div>
              <div><dt className="text-ink-3">Cloud cost</dt><dd className="text-ink text-[15px] tabular-nums">${data.totals.cloud_cost_usd.toFixed(2)}</dd></div>
            </dl>

            {data.unpriced.length > 0 && (
              <p className="rounded-md border hairline bg-bg-2 px-3 py-2 text-[12.5px] text-warn" role="status">
                No price is configured for {data.unpriced.join(", ")}, so the cost above leaves {data.unpriced.length === 1 ? "it" : "them"} out.
                Set <code className="font-mono text-ink-2">LBX_LLM__PRICES</code> to include {data.unpriced.length === 1 ? "it" : "them"}.
              </p>
            )}

            <p className="text-[12.5px] text-ink-2">
              {data.budget.daily_usd === null
                ? "No daily cap on cloud spend is set (LBX_LLM__DAILY_BUDGET_USD)."
                : <>Today: ${data.budget.spent_today_usd.toFixed(2)} of a ${data.budget.daily_usd.toFixed(2)} daily cap.
                    {data.budget.exhausted && <span className="text-warn"> The cap is reached, so calls are going to the local model.</span>}</>}
            </p>

            {groups.length === 0
              ? <p className="text-[12.5px] text-ink-3">No model calls were recorded in this period.</p>
              : (
                <div className="overflow-x-auto -mx-1">
                  <table className="w-full text-[12.5px]">
                    <thead>
                      <tr className="text-left text-ink-3 border-b hairline">
                        <th className="font-medium px-1 py-2">Model</th>
                        <th className="font-medium px-1 py-2">For</th>
                        <th className="font-medium px-1 py-2 text-right">Calls</th>
                        <th className="font-medium px-1 py-2 text-right">Failed</th>
                        <th className="font-medium px-1 py-2 text-right">Tokens in</th>
                        <th className="font-medium px-1 py-2 text-right">Tokens out</th>
                        <th className="font-medium px-1 py-2 text-right">Images</th>
                        <th className="font-medium px-1 py-2 text-right">Avg time</th>
                        <th className="font-medium px-1 py-2 text-right">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {groups.map((g) => (
                        <tr key={`${g.provider}:${g.model}:${g.purpose}`} className="border-b hairline last:border-0">
                          <td className="px-1 py-2 text-ink"><span className="text-ink-3">{g.provider}</span> {g.model}</td>
                          <td className="px-1 py-2 text-ink-2">{g.purpose}</td>
                          <td className="px-1 py-2 text-right tabular-nums">{n(g.calls)}</td>
                          <td className={`px-1 py-2 text-right tabular-nums ${g.failures ? "text-warn" : ""}`}>{n(g.failures)}</td>
                          <td className="px-1 py-2 text-right tabular-nums">{n(g.input_tokens)}</td>
                          <td className="px-1 py-2 text-right tabular-nums">{n(g.output_tokens)}</td>
                          <td className="px-1 py-2 text-right tabular-nums">{n(g.images)}</td>
                          <td className="px-1 py-2 text-right tabular-nums">{g.avg_latency_ms === null ? "" : `${n(g.avg_latency_ms)} ms`}</td>
                          <td className={`px-1 py-2 text-right tabular-nums ${g.cost_usd === null ? "text-warn" : ""}`}>{usd(g.cost_usd)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
          </>
        )}
      </div>
    </section>
  );
}

function LoadedModels() {
  const q = useQuery(["/api/system/models"], () => apiGet<Loaded>("/api/system/models"));
  const data = q.data ?? null;
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  async function release(key?: string) {
    setBusy(true);
    setNote(null);
    try {
      const path = key ? `/api/system/models/release?key=${encodeURIComponent(key)}` : "/api/system/models/release";
      const res = await apiPost<{ released: string[]; still_loaded: string[] }>(path, {});
      setNote(res.released.length
        ? `Freed ${res.released.join(", ")}.`
        : "Nothing was freed: every loaded model is in use right now.");
      await q.reload();
    } catch (e) {
      setNote(e instanceof Error ? e.message : "Could not free the models.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <header className="flex flex-wrap items-start justify-between gap-3 px-4 pt-4 pb-3">
        <div className="min-w-0">
          <h2 className="text-[14px] font-semibold text-ink">Models on the GPU</h2>
          <p className="mt-0.5 text-[12.5px] text-ink-3 max-w-[65ch]">
            What the application server has loaded. Idle models are freed automatically when a job needs the room;
            this frees them now. Auto-label and training run in the GPU worker, which is not shown here.
          </p>
        </div>
        <button className="btn shrink-0" disabled={busy || !data?.models.length} onClick={() => void release()}>
          {busy ? "Freeing" : "Free idle models"}
        </button>
      </header>
      <div className="px-4 pb-4 flex flex-col gap-3">
        <LoadState {...q.loadState} />
        {note && <p className="text-[12.5px] text-ink-2" role="status">{note}</p>}
        {data && (
          <>
            <p className="text-[12.5px] text-ink-2">
              {data.vram_total_mb === null
                ? "No GPU is visible to the application server."
                : `${data.device}: ${n(Math.round(data.vram_free_mb ?? 0))} MB free of ${n(Math.round(data.vram_total_mb))} MB.`}
            </p>
            {data.models.length === 0
              ? <p className="text-[12.5px] text-ink-3">Nothing is loaded.</p>
              : (
                <ul className="rounded-md border hairline divide-y divide-line overflow-hidden">
                  {data.models.map((m) => (
                    <li key={m.key} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2.5 bg-bg/40 text-[12.5px]">
                      <span className="font-mono text-[12px] text-ink min-w-0 break-all">{m.key}</span>
                      <span className="text-ink-3">{m.leases > 0 ? "in use" : `idle ${Math.round(m.idle_s)} s`}</span>
                      <span className="text-ink-3">used {n(m.uses)} times</span>
                      <button className="btn ml-auto" disabled={busy || m.leases > 0} onClick={() => void release(m.key)}>Free</button>
                    </li>
                  ))}
                </ul>
              )}
          </>
        )}
      </div>
    </section>
  );
}

export default function UsagePage() {
  return (
    <PageShell active="Settings" title="Models and usage" subtitle="what the models were asked, what it cost, what is loaded">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6 flex flex-col gap-5">
        <ModelUsage />
        <LoadedModels />
      </div>
    </PageShell>
  );
}
