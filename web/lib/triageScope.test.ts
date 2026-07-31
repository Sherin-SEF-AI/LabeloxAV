import { describe, expect, it } from "vitest";
import {
  DEFAULT_TRIAGE_LIMIT, SCOPED_TRIAGE_LIMIT, describeScope, triageQuery,
} from "./triageScope";

const params = (s: string) => new URLSearchParams(s);

describe("triageQuery", () => {
  it("always asks for a bounded page", () => {
    expect(triageQuery(params(""))).toEqual({ limit: String(DEFAULT_TRIAGE_LIMIT) });
  });

  it("forwards a flywheel batch, which is how a mined batch is opened at all", () => {
    // The reason this module exists: the page previously dropped this parameter, so a batch stamped with a
    // cycle id had no URL that would show it even though the API could filter on it.
    expect(triageQuery(params("flywheel=review-batch-cb3551d0"))).toEqual({
      limit: String(SCOPED_TRIAGE_LIMIT),
      flywheel: "review-batch-cb3551d0",
    });
  });

  it("renames url parameters to what the api calls them", () => {
    expect(triageQuery(params("session=abc&class=rider&city=BLR"))).toEqual({
      limit: String(SCOPED_TRIAGE_LIMIT),
      session_id: "abc",
      klass: "rider",
      city: "BLR",
    });
  });

  it("combines scopes, so one class inside one batch is reachable", () => {
    const q = triageQuery(params("flywheel=batch-1&class=cattle"));
    expect(q.flywheel).toBe("batch-1");
    expect(q.klass).toBe("cattle");
  });

  it("drops empty and whitespace-only values rather than filtering on them", () => {
    // `?class=` would otherwise filter on a class named the empty string and return nothing, which a
    // reviewer reads as "no work left" instead of as a broken link.
    expect(triageQuery(params("class=&session=%20%20"))).toEqual({ limit: "200" });
  });

  it("ignores parameters that are not scopes", () => {
    expect(triageQuery(params("sort=conf&evil=1"))).toEqual({ limit: "200" });
  });

  it("honours an explicit limit", () => {
    expect(triageQuery(params(""), 50).limit).toBe("50");
  });

  it("asks for the whole batch when scoped, not one page of it", () => {
    // A 600-object batch fetched 200 at a time hands the reviewer a third of the work and no sign the rest
    // exists, which reads as a finished batch.
    expect(triageQuery(params("flywheel=batch-1")).limit).toBe(String(SCOPED_TRIAGE_LIMIT));
    expect(triageQuery(params("class=rider")).limit).toBe(String(SCOPED_TRIAGE_LIMIT));
    expect(triageQuery(params("")).limit).toBe(String(DEFAULT_TRIAGE_LIMIT));
  });

  it("does not treat a state filter alone as a finite scope", () => {
    // Narrowing to unreviewed objects says which verdicts are outstanding, not which objects are in scope,
    // so the queue is still endless and a page is still the right ask.
    expect(triageQuery(params("states=review")).limit).toBe(String(DEFAULT_TRIAGE_LIMIT));
  });
});

describe("the behaviour this replaced", () => {
  // What the rapid review page did inline before this module existed. Kept executable rather than described
  // in a comment, so the regression it fixes stays demonstrable instead of becoming folklore.
  function oldPageQuery(p: URLSearchParams): Record<string, string> {
    const q: Record<string, string> = { limit: "200" };
    const sessionId = p.get("session") || undefined;
    if (sessionId) q.session_id = sessionId;
    return q;
  }

  it("dropped every scope except the session", () => {
    const url = params("flywheel=review-batch-cb3551d0&class=rider&city=BLR");
    expect(oldPageQuery(url)).toEqual({ limit: "200" });     // batch silently ignored
    expect(triageQuery(url)).toEqual({
      limit: String(SCOPED_TRIAGE_LIMIT), flywheel: "review-batch-cb3551d0",
      klass: "rider", city: "BLR",
    });
  });

  it("meant a mined batch returned the whole corpus instead of the batch", () => {
    // The failure is quiet, which is the worst part: the reviewer gets a full queue of unrelated objects
    // and no indication that the batch they were sent to was never applied.
    const url = params("flywheel=review-batch-cb3551d0");
    expect(oldPageQuery(url).flywheel).toBeUndefined();
    expect(triageQuery(url).flywheel).toBe("review-batch-cb3551d0");
  });
});

describe("describeScope", () => {
  it("says nothing when the queue is the whole corpus", () => {
    expect(describeScope(params(""))).toBeNull();
  });

  it("names the batch, so a short queue is explicable", () => {
    expect(describeScope(params("flywheel=review-batch-cb3551d0")))
      .toBe("batch review-batch-cb3551d0");
  });

  it("reads as a sentence when several scopes combine", () => {
    expect(describeScope(params("class=rider&city=BLR&flywheel=b1")))
      .toBe("rider, in BLR, batch b1");
  });

  it("abbreviates a session id, which is a uuid nobody reads in full", () => {
    expect(describeScope(params("session=40693d14-020b-4eac-bac4-cd103270e85c")))
      .toBe("session 40693d14");
  });
});
