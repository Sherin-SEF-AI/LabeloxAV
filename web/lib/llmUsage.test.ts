import { describe, expect, it } from "vitest";

import { summarize, type UsageRow } from "./llmUsage";

const row = (over: Partial<UsageRow>): UsageRow => ({
  day: "2026-11-24", provider: "ollama", model: "qwen2.5vl:7b", purpose: "vlm_verify", calls: 0, failures: 0,
  input_tokens: 0, output_tokens: 0, images: 0, avg_latency_ms: null, unmetered_calls: 0, cost_usd: 0,
  cloud: false, ...over,
});

describe("summarize", () => {
  it("folds days into one line per model and purpose", () => {
    const out = summarize([
      row({ day: "2026-11-23", calls: 10, input_tokens: 100, output_tokens: 5, images: 10 }),
      row({ day: "2026-11-24", calls: 30, input_tokens: 300, output_tokens: 15, images: 30, failures: 2 }),
      row({ purpose: "sign_ocr", calls: 4 }),
    ]);
    expect(out).toHaveLength(2);
    const verify = out.find((g) => g.purpose === "vlm_verify")!;
    expect(verify).toMatchObject({ calls: 40, failures: 2, input_tokens: 400, output_tokens: 20, images: 40 });
  });

  it("weights latency by calls rather than averaging the daily averages", () => {
    const [g] = summarize([
      row({ day: "2026-11-23", calls: 3, avg_latency_ms: 1000 }),
      row({ day: "2026-11-24", calls: 3000, avg_latency_ms: 100 }),
    ]);
    expect(g.avg_latency_ms).toBe(101);
  });

  it("keeps an unpriced model's cost unknown instead of calling it free", () => {
    const [g] = summarize([
      row({ provider: "groq", model: "m", cloud: true, calls: 5, cost_usd: null }),
      row({ day: "2026-11-23", provider: "groq", model: "m", cloud: true, calls: 5, cost_usd: null }),
    ]);
    expect(g.cost_usd).toBeNull();
  });

  it("adds the days that were priced when only some were", () => {
    const [g] = summarize([
      row({ provider: "groq", model: "m", cloud: true, cost_usd: null }),
      row({ day: "2026-11-23", provider: "groq", model: "m", cloud: true, cost_usd: 1.25 }),
    ]);
    expect(g.cost_usd).toBe(1.25);
  });

  it("puts what cost the most first, then what was called the most", () => {
    const out = summarize([
      row({ purpose: "busy-local", calls: 900 }),
      row({ provider: "anthropic", model: "judge", purpose: "judge", cloud: true, calls: 2, cost_usd: 4 }),
      row({ provider: "groq", model: "m", purpose: "unpriced", cloud: true, calls: 5000, cost_usd: null }),
      row({ purpose: "quiet-local", calls: 1 }),
    ]);
    expect(out.map((g) => g.purpose)).toEqual(["judge", "busy-local", "quiet-local", "unpriced"]);
  });

  it("has no latency for a group that never reported one", () => {
    expect(summarize([row({ calls: 2 })])[0].avg_latency_ms).toBeNull();
  });

  it("returns nothing for nothing", () => {
    expect(summarize([])).toEqual([]);
  });
});
