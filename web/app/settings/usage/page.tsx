"use client";

import { useState } from "react";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { apiGet, apiPost } from "@/lib/api";
import { summarize, type UsageRow } from "@/lib/llmUsage";
import { useQuery } from "@/lib/useQuery";
import { duration, humanize } from "@/lib/format";
import { Segmented, Stat } from "@/components/ui/kit";
import EmptyState from "@/components/ui/EmptyState";

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
const usd = (v: number | null) => (v === null ? "Unpriced" : `$${v.toFixed(v < 1 ? 4 : 2)}`);

function ModelUsage() {
  const [days, setDays] = useState<(typeof RANGES)[number]>(30);
  const q = useQuery([`/api/system/llm-usage?days=${days}`], () => apiGet<Usage>(`/api/system/llm-usage?days=${days}`));
  const data = q.data ?? null;
  const groups = data ? summarize(data.rows) : [];

  return (
    <section className="panel">
      <header className="flex flex-wrap items-start justify-between gap-3 px-4 pt-4 pb-3">
        <div className="min-w-0">
          <h2 className="section-title text-[14px]">Model usage</h2>
          <p className="mt-0.5 text-[12.5px] text-ink-3 max-w-[65ch]">
            Every call to a language or vision model, by model and by what it was for. Local models cost nothing;
            a cloud model costs what its configured price says.
          </p>
        </div>
        <div role="group" aria-label="Period">
          <Segmented value={String(days)} onChange={(v) => setDays(Number(v) as (typeof RANGES)[number])}
            options={RANGES.map((d) => ({ value: String(d), label: `${d} days` }))} />
        </div>
      </header>
      <div className="px-4 pb-4 flex flex-col gap-3">
        <LoadState {...q.loadState} />
        {data && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <Stat label="Calls" value={n(data.totals.calls)} />
              <Stat label="Failed" value={n(data.totals.failures)} tone={data.totals.failures ? "warn" : undefined} />
              <Stat label="Tokens in / out" value={<span className="text-[18px]">{n(data.totals.input_tokens)} / {n(data.totals.output_tokens)}</span>} />
              <Stat label="Cloud cost" value={`$${data.totals.cloud_cost_usd.toFixed(2)}`} />
            </div>

            {data.unpriced.length > 0 && (
              <p className="rounded-md border hairline bg-bg-2 px-3 py-2 text-[12.5px] text-warn" role="status">
                No price is configured for {data.unpriced.join(", ")}, so the cost above leaves {data.unpriced.length === 1 ? "it" : "them"} out.
                Set <code className="font-mono text-ink-2">LBX_LLM__PRICES</code> to include {data.unpriced.length === 1 ? "it" : "them"}.
              </p>
            )}

            <p className="text-[12.5px] text-ink-2">
              {data.budget.daily_usd === null
                ? <span title="Set LBX_LLM__DAILY_BUDGET_USD on the server to add one">No daily cap on cloud spend is set.</span>
                : <>Today: ${data.budget.spent_today_usd.toFixed(2)} of a ${data.budget.daily_usd.toFixed(2)} daily cap.
                    {data.budget.exhausted && <span className="text-warn"> The cap is reached, so calls are going to the local model.</span>}</>}
            </p>

            {groups.length === 0
              ? (
                <div className="rounded-md border hairline">
                  <EmptyState compact icon="chart" title="No model calls in this period">
                    The assistant, the vision reviewer and label checks are counted here each time they ask a model something.
                  </EmptyState>
                </div>
              )
              : (
                <div className="overflow-x-auto rounded-md border hairline">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>For</th>
                        <th className="num">Calls</th>
                        <th className="num">Failed</th>
                        <th className="num">Tokens in</th>
                        <th className="num">Tokens out</th>
                        <th className="num">Images</th>
                        <th className="num">Avg time</th>
                        <th className="num">Cost</th>
                      </tr>
                    </thead>
                    <tbody>
                      {groups.map((g) => (
                        <tr key={`${g.provider}:${g.model}:${g.purpose}`}>
                          <td className="text-ink"><span className="text-ink-3">{g.provider}</span> {g.model}</td>
                          <td>{humanize(g.purpose)}</td>
                          <td className="num">{n(g.calls)}</td>
                          <td className={`num ${g.failures ? "text-warn" : ""}`}>{n(g.failures)}</td>
                          <td className="num">{n(g.input_tokens)}</td>
                          <td className="num">{n(g.output_tokens)}</td>
                          <td className="num">{n(g.images)}</td>
                          <td className="num">{g.avg_latency_ms === null ? "" : `${n(g.avg_latency_ms)} ms`}</td>
                          <td className={`num ${g.cost_usd === null ? "text-warn" : ""}`}>{usd(g.cost_usd)}</td>
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
          <h2 className="section-title text-[14px]">Models on the GPU</h2>
          <p className="mt-0.5 text-[12.5px] text-ink-3 max-w-[65ch]">
            What the application server has loaded. Idle models are freed automatically when a job needs the room;
            this frees them now. Auto-label and training run in the GPU worker, which is not shown here.
          </p>
        </div>
        <button className="btn shrink-0" disabled={busy || !data?.models.length} onClick={() => void release()}>
          {busy ? "Freeing…" : "Free idle models"}
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
              ? (
                <div className="rounded-md border hairline">
                  <EmptyState compact icon="server" title="No models loaded">
                    Models load the first time the editor or the assistant needs one, and are freed when idle.
                  </EmptyState>
                </div>
              )
              : (
                <ul className="rounded-md border hairline divide-y divide-line overflow-hidden">
                  {data.models.map((m) => (
                    <li key={m.key} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2.5 bg-bg/40 text-[12.5px]">
                      <span className="font-mono text-[12px] text-ink min-w-0 break-all">{m.key}</span>
                      <span className={m.leases > 0 ? "text-accent-2" : "text-ink-3"}>{m.leases > 0 ? "In use" : `Idle for ${duration(m.idle_s)}`}</span>
                      <span className="text-ink-3">Used {n(m.uses)} {m.uses === 1 ? "time" : "times"}</span>
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
    <PageShell active="Settings" title="Models and usage" subtitle="What the models were asked, what it cost, and what is loaded">
      <div className="page-wrap-narrow flex flex-col gap-4">
        <ModelUsage />
        <LoadedModels />
      </div>
    </PageShell>
  );
}
