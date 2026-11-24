// Folding the usage ledger's daily rows into one line per model and purpose, for the usage page.
//
// The API returns a row per day. A table of ninety days times every model is not what a person is asking
// when they ask what the judge cost, so the page shows the period's total per (provider, model, purpose).

export type UsageRow = {
  day: string; provider: string; model: string; purpose: string;
  calls: number; failures: number; input_tokens: number; output_tokens: number; images: number;
  avg_latency_ms: number | null; unmetered_calls: number; cost_usd: number | null; cloud: boolean;
};

export type UsageGroup = Omit<UsageRow, "day" | "unmetered_calls">;

/**
 * One group per (provider, model, purpose), costliest first, then busiest.
 *
 * A group's cost is null only when none of its days had a known cost. Latency is weighted by calls, since
 * averaging the daily averages would let a day with three calls count as much as a day with three thousand.
 */
export function summarize(rows: UsageRow[]): UsageGroup[] {
  const groups = new Map<string, UsageGroup & { latencyTotal: number; latencyCalls: number }>();
  for (const r of rows) {
    const key = `${r.provider}\u0000${r.model}\u0000${r.purpose}`;
    let g = groups.get(key);
    if (!g) {
      g = { provider: r.provider, model: r.model, purpose: r.purpose, calls: 0, failures: 0, input_tokens: 0,
            output_tokens: 0, images: 0, avg_latency_ms: null, cost_usd: null, cloud: r.cloud,
            latencyTotal: 0, latencyCalls: 0 };
      groups.set(key, g);
    }
    g.calls += r.calls;
    g.failures += r.failures;
    g.input_tokens += r.input_tokens;
    g.output_tokens += r.output_tokens;
    g.images += r.images;
    if (r.cost_usd !== null) g.cost_usd = (g.cost_usd ?? 0) + r.cost_usd;
    if (r.avg_latency_ms !== null && r.calls > 0) {
      g.latencyTotal += r.avg_latency_ms * r.calls;
      g.latencyCalls += r.calls;
    }
  }
  return [...groups.values()]
    .map(({ latencyTotal, latencyCalls, ...g }) => ({
      ...g, avg_latency_ms: latencyCalls ? Math.round(latencyTotal / latencyCalls) : null,
    }))
    .sort((a, b) => (b.cost_usd ?? -1) - (a.cost_usd ?? -1) || b.calls - a.calls);
}
