/**
 * The dev proxy must outlast the slowest endpoint behind it.
 *
 * Next's rewrite proxy gives up at 30s by default and answers with a bare `500 Internal Server Error`. The
 * backend never sees a failure: it keeps working and finishes. A promotion that took 84 seconds was recorded
 * as a rejection while the operator was shown an error, which is the one outcome a governance control must
 * never produce, because the obvious response is to retry a decision that already stands.
 *
 * Pinned as a test rather than left as a comment because the failure is invisible from the frontend side:
 * nothing throws, nothing logs, and the symptom only appears against a cold GPU cache.
 */

import { describe, expect, it } from "vitest";

import nextConfig from "../next.config.mjs";

// The longest observed real request: a cold promotion evaluating challenger and champion against gold.
const SLOWEST_ENDPOINT_MS = 90_000;
// What Next falls back to when proxyTimeout is unset. Restoring this is the regression being guarded.
const NEXT_DEFAULT_MS = 30_000;

describe("rewrite proxy timeout", () => {
  it("is set at all", () => {
    expect(nextConfig.experimental?.proxyTimeout).toBeTypeOf("number");
  });

  it("outlasts the slowest endpoint it proxies, with headroom", () => {
    expect(nextConfig.experimental!.proxyTimeout).toBeGreaterThan(SLOWEST_ENDPOINT_MS);
  });

  it("is not the default that truncated a completed promotion", () => {
    expect(nextConfig.experimental!.proxyTimeout).toBeGreaterThan(NEXT_DEFAULT_MS);
  });

  it("still bounds the request, so a hung backend does not leak a connection forever", () => {
    expect(nextConfig.experimental!.proxyTimeout).toBeLessThanOrEqual(10 * 60 * 1000);
  });

  it("does not disturb the rewrite it applies to", async () => {
    const rules = await nextConfig.rewrites!();
    expect(rules).toEqual([
      { source: "/api/:path*", destination: expect.stringContaining("/api/:path*") },
    ]);
  });
});
