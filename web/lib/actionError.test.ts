/**
 * A failure message must not assert a cause the status contradicts.
 *
 * The agent console carried thirteen catch blocks that appended "(needs reviewer role)" to every failure of
 * an action, unconditionally. `ApiError.status` was there the whole time and nothing read it. The result was
 * lines like "sweep failed (needs reviewer role): The service is busy (the GPU may be in use)", which
 * contradicts itself and sends an operator to fix permissions while the real fix is to wait.
 */

import { describe, expect, it } from "vitest";

import { ApiError } from "./api";
import { describeFailure, failureLine } from "./actionError";

const err = (status: number, detail = "") => new ApiError(status, "POST", "/api/agent/error-sweep", detail);

describe("describeFailure", () => {
  it("claims a permission problem only when the status says so", () => {
    expect(describeFailure("sweep", err(403)).hint).toBe("this needs a higher role");
    expect(describeFailure("sweep", err(401)).hint).toBe("you are signed out");
  });

  it("does not claim a permission problem for a busy GPU", () => {
    const f = describeFailure("sweep", err(503));
    expect(f.hint).toBeUndefined();
    expect(f.message).toContain("busy");
  });

  it.each([500, 404, 409, 422, 502])("does not claim a permission problem for %i", (status) => {
    expect(describeFailure("sweep", err(status)).hint).toBeUndefined();
  });

  it("never produces the self-contradicting line the console used to show", () => {
    const line = failureLine("sweep", err(503));
    expect(line).not.toMatch(/role/i);
  });

  it("keeps the server's own explanation as the reason", () => {
    expect(describeFailure("dispatch", err(409, "that order is already dispatched")).message)
      .toContain("already dispatched");
  });

  it("names the action, so a transcript line stands alone", () => {
    expect(describeFailure("ontology scan", err(500)).message).toMatch(/^ontology scan failed:/);
  });
});

describe("retryability", () => {
  it.each([408, 429, 500, 502, 503, 504])("marks %i retryable", (status) => {
    expect(describeFailure("x", err(status)).retryable).toBe(true);
  });

  it.each([400, 403, 404, 409, 422])("marks %i not retryable", (status) => {
    expect(describeFailure("x", err(status)).retryable).toBe(false);
  });

  it("treats a request that never reached the server as retryable", () => {
    /* A dev server restarting mid-click looks exactly like a network drop, and both are worth retrying. */
    const f = describeFailure("x", new TypeError("Failed to fetch"));
    expect(f.status).toBeNull();
    expect(f.retryable).toBe(true);
  });
});

describe("non-ApiError inputs", () => {
  it("uses a plain Error's message", () => {
    expect(describeFailure("x", new Error("boom")).message).toBe("x failed: boom");
  });

  it("stringifies anything else rather than throwing", () => {
    expect(describeFailure("x", "just a string").message).toBe("x failed: just a string");
    expect(describeFailure("x", null).message).toBe("x failed: null");
  });
});
