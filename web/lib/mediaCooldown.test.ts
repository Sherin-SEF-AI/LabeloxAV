/**
 * A failed media-cookie mint must not be retried on every request.
 *
 * `refreshTokenIfNeeded` awaits `ensureMediaCookie` on every single API call, and the freshness guard keyed
 * only on the last *success*. So while the backend was unreachable nothing was ever recorded, the guard
 * never held, and each poll fired an extra failed POST to /api/auth/media-token. That is why a three-minute
 * API restart produced media-token failures interleaved between every cloud/status and cloud/orphans line in
 * the console rather than a handful.
 *
 * The cooldown has to key on failure specifically. Recording an attempt regardless would delay the first
 * successful mint after a genuine expiry, which is the case the two-thirds-of-life rule exists to protect.
 */

import { describe, expect, it } from "vitest";

import { MEDIA_RETRY_COOLDOWN_MS, shouldMintMedia } from "./user";

const TTL = 900_000;      // the server's usual 15 minute media token
const NOW = 1_000_000_000;

describe("shouldMintMedia", () => {
  it("mints when nothing has ever been minted", () => {
    expect(shouldMintMedia(NOW, 0, TTL, 0)).toBe(true);
  });

  it("does not re-mint while the cookie is still comfortably alive", () => {
    expect(shouldMintMedia(NOW, NOW - TTL * 0.1, TTL, 0)).toBe(false);
  });

  it("re-mints at two thirds of life, before an image request can race the expiry", () => {
    expect(shouldMintMedia(NOW, NOW - TTL * 0.65, TTL, 0)).toBe(false);
    expect(shouldMintMedia(NOW, NOW - TTL * 0.67, TTL, 0)).toBe(true);
  });
});

describe("the failure cooldown", () => {
  it("does not retry immediately after a failure", () => {
    /* The defect: with no record of the failure this returned true on the very next API request. */
    expect(shouldMintMedia(NOW, 0, TTL, NOW - 1_000)).toBe(false);
  });

  it("retries once the cooldown has passed", () => {
    expect(shouldMintMedia(NOW, 0, TTL, NOW - MEDIA_RETRY_COOLDOWN_MS - 1)).toBe(true);
  });

  it("collapses a burst of requests during an outage into one attempt per cooldown", () => {
    /* 60 API calls over 30 seconds while the backend is down. */
    let lastFail = 0;
    let attempts = 0;
    for (let i = 0; i < 60; i++) {
      const now = NOW + i * 500;
      if (shouldMintMedia(now, 0, TTL, lastFail)) { attempts++; lastFail = now; }
    }
    expect(attempts).toBe(2);   // one at t=0, one after the 15s cooldown
  });

  it("was unbounded before the cooldown existed", () => {
    /* The same burst with the old rule, expressed as a cooldown of zero. */
    let attempts = 0;
    for (let i = 0; i < 60; i++) {
      if (shouldMintMedia(NOW + i * 500, 0, TTL, 0, 0)) attempts++;
    }
    expect(attempts).toBe(60);
  });

  it("keeps the cooldown short enough that a restart heals without a reload", () => {
    expect(MEDIA_RETRY_COOLDOWN_MS).toBeLessThanOrEqual(60_000);
  });

  it("lets a live cookie win over a recent failure", () => {
    /* A failed re-mint attempt must not stop a still-valid cookie being used; the answer is the same
       either way (do not mint), but for the freshness reason rather than the cooldown. */
    expect(shouldMintMedia(NOW, NOW - 1000, TTL, NOW - 1000)).toBe(false);
  });

  it("mints again as soon as the cooldown lapses even if failures keep coming", () => {
    const lastFail = NOW - MEDIA_RETRY_COOLDOWN_MS;
    expect(shouldMintMedia(NOW, 0, TTL, lastFail)).toBe(true);
  });
});
