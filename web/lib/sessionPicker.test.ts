import { describe, expect, it } from "vitest";

import { matchesSession, orderSessions, sessionDate, sessionDetail, sessionLabel } from "./sessionPicker";
import type { SessionRow } from "./types";

// The fallbacks are the point. 42 sessions in this corpus carry no city and some carry no route, so a
// label built by joining fields renders an empty row that looks like a bug rather than like a drive.

const s = (over: Partial<SessionRow> = {}): SessionRow => ({
  session_id: "aaaaaaaa-1111-2222-3333-444444444444",
  vehicle_id: "DASHCAM-01", city: "BLR", route: null,
  start_ts_ns: Date.UTC(2026, 4, 31) * 1_000_000,
  end_ts_ns: Date.UTC(2026, 4, 31) * 1_000_000,
  ...over,
});

describe("sessionLabel", () => {
  it("names a drive by vehicle and city", () => {
    expect(sessionLabel(s())).toBe("DASHCAM-01 · BLR");
  });

  it("still says something when the city is missing", () => {
    expect(sessionLabel(s({ city: null }))).toBe("DASHCAM-01");
  });

  it("falls back to the id rather than rendering an empty row", () => {
    expect(sessionLabel(s({ vehicle_id: "", city: null }))).toBe("aaaaaaaa");
  });
});

describe("sessionDetail", () => {
  it("keeps a capture label as it is, because it already identifies the clip", () => {
    // And it already carries a date, a different one from start_ts_ns: filmed against ingested. Printing
    // both gave "2026-06-06 10:01 · 043849F · 2026-07-01".
    expect(sessionDetail(s({ route: "2026-06-06 10:01 · 043849F" }))).toBe("2026-06-06 10:01 · 043849F");
  });

  it("adds the id when the route is a category rather than an identifier", () => {
    // 37 sessions carry `import:video`, so without the id the picker showed thirty-seven identical rows.
    expect(sessionDetail(s({ route: "import:video" }))).toBe("aaaaaaaa · import:video");
  });

  it("falls back to id and date when there is no route at all, as 147 sessions have", () => {
    expect(sessionDetail(s())).toBe("aaaaaaaa · 2026-05-31");
  });

  it("is still identifying when the date is unusable", () => {
    expect(sessionDetail(s({ start_ts_ns: 0 }))).toBe("aaaaaaaa");
    expect(sessionDate(0)).toBe("");
  });

  it("never renders the same line for two different drives with the same route", () => {
    const a = s({ session_id: "11111111-a", route: "import:video" });
    const b = s({ session_id: "22222222-b", route: "import:video" });
    expect(sessionDetail(a)).not.toBe(sessionDetail(b));
  });
});

describe("matchesSession", () => {
  it("matches on city and vehicle, case-insensitively", () => {
    expect(matchesSession(s(), "blr")).toBe(true);
    expect(matchesSession(s(), "dashcam")).toBe(true);
    expect(matchesSession(s(), "chennai")).toBe(false);
  });

  it("matches on the session id, which is what people paste in", () => {
    expect(matchesSession(s(), "aaaaaaaa")).toBe(true);
  });

  it("matches on the date, so a drive can be found by when it happened", () => {
    expect(matchesSession(s(), "2026-05")).toBe(true);
  });

  it("an empty query matches everything", () => {
    expect(matchesSession(s(), "   ")).toBe(true);
  });
});

describe("orderSessions", () => {
  it("puts the session you are in first, then the newest", () => {
    const old = s({ session_id: "old", start_ts_ns: 1_000_000_000_000 });
    const mid = s({ session_id: "mid", start_ts_ns: 3_000_000_000_000 });
    const cur = s({ session_id: "cur", start_ts_ns: 2_000_000_000_000 });
    expect(orderSessions([old, mid, cur], "cur").map((x) => x.session_id)).toEqual(["cur", "mid", "old"]);
  });

  it("is just newest-first when the current session is not in the list", () => {
    const a = s({ session_id: "a", start_ts_ns: 1 });
    const b = s({ session_id: "b", start_ts_ns: 2 });
    expect(orderSessions([a, b], null).map((x) => x.session_id)).toEqual(["b", "a"]);
  });

  it("does not mutate its input", () => {
    const list = [s({ session_id: "a", start_ts_ns: 1 }), s({ session_id: "b", start_ts_ns: 2 })];
    orderSessions(list, "b");
    expect(list.map((x) => x.session_id)).toEqual(["a", "b"]);
  });
});
