import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  DRIVE_STATUS, canOpen, driveStatus, matchesSession, orderByVisit, orderSessions,
  previousSession, recentSessions, recordVisit, sessionDate, sessionDetail, sessionLabel,
} from "./sessionPicker";
import type { SessionState } from "./sessionPicker";
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

describe("driveStatus", () => {
  const st = (over: Partial<SessionState> = {}): SessionState =>
    ({ session_id: "s", frames: 100, objects: 500, reviewed_objects: 0, ...over });

  it("calls a drive with no camera frames empty, because the editor cannot open it", () => {
    // 126 sessions here are LiDAR and 3D captures with point clouds and no frames. Clicking one 404s.
    expect(driveStatus(st({ frames: 0 }))).toBe("empty");
    expect(canOpen(st({ frames: 0 }))).toBe(false);
  });

  it("separates frames-with-nothing-on-them from labelled-and-waiting", () => {
    // 42 sessions have never been through detection; opening one shows a blank frame on every frame.
    expect(driveStatus(st({ objects: 0 }))).toBe("unlabelled");
    expect(driveStatus(st())).toBe("ready");
  });

  it("calls a drive in progress once a person has ruled on any of it", () => {
    expect(driveStatus(st({ reviewed_objects: 1 }))).toBe("working");
  });

  it("every status has a label and an explanation", () => {
    for (const k of ["empty", "unlabelled", "ready", "working"] as const) {
      expect(DRIVE_STATUS[k].label).toBeTruthy();
      expect(DRIVE_STATUS[k].tip.length).toBeGreaterThan(20);
    }
  });
});

describe("where you have been", () => {
  // The node tier has no localStorage, so it is stubbed rather than the file being promoted to jsdom:
  // this is pure logic and belongs in the fast tier, the same call this repo makes in panelPrefs.test.ts.
  beforeEach(() => {
    const store = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
      clear: () => store.clear(),
    } as unknown as Storage);
  });
  afterEach(() => vi.unstubAllGlobals());

  it("remembers drives newest first, without duplicating one you return to", () => {
    recordVisit("a"); recordVisit("b"); recordVisit("a");
    expect(recentSessions()).toEqual(["a", "b"]);
  });

  it("names the drive you were in before this one", () => {
    recordVisit("a"); recordVisit("b");
    expect(previousSession("b")).toBe("a");
  });

  it("has no previous drive on a first visit", () => {
    recordVisit("a");
    expect(previousSession("a")).toBeNull();
  });

  it("survives a localStorage that throws, because a picker must not be the thing that breaks", () => {
    const orig = globalThis.localStorage;
    try {
      Object.defineProperty(globalThis, "localStorage", {
        configurable: true,
        get() { throw new DOMException("denied", "SecurityError"); },
      });
      expect(recentSessions()).toEqual([]);
      expect(() => recordVisit("a")).not.toThrow();
    } finally {
      Object.defineProperty(globalThis, "localStorage", { configurable: true, value: orig });
    }
  });
});

describe("orderByVisit", () => {
  const s2 = (id: string, ts: number): SessionRow => ({
    session_id: id, vehicle_id: "V", city: "BLR", route: null, start_ts_ns: ts, end_ts_ns: ts,
  });

  it("puts the current drive first, then ones you have been in, then the rest by date", () => {
    const list = [s2("old", 1), s2("new", 9), s2("seen", 2), s2("cur", 3)];
    expect(orderByVisit(list, "cur", ["cur", "seen"]).map((x) => x.session_id))
      .toEqual(["cur", "seen", "new", "old"]);
  });

  it("is date order when nothing has been visited", () => {
    const list = [s2("a", 1), s2("b", 2)];
    expect(orderByVisit(list, null, []).map((x) => x.session_id)).toEqual(["b", "a"]);
  });
});

describe("a missing drive state", () => {
  it("is unknown, not empty, so a failed request does not report the corpus as blank", () => {
    // The first version answered "empty" here, which also means "cannot be opened", so one failed
    // aggregate marked all 377 drives as having no frames and disabled every row in the picker.
    expect(driveStatus(undefined)).toBe("unknown");
  });

  it("still lets you open the drive, because not knowing is not evidence of absence", () => {
    expect(canOpen(undefined)).toBe(true);
  });

  it("renders no label rather than a guessed one", () => {
    expect(DRIVE_STATUS.unknown.label).toBe("");
  });
});
