import { describe, expect, it } from "vitest";

import {
  MODES,
  groupsForMode,
  hotkeyCollisions,
  modeByKey,
  toolForHotkey,
  toolsForMode,
} from "./registry";

// The editor kept two tool registries: this one, which drove the mode rail, and a second copy inside
// the frame editor's page, which drove the strip and the shortcuts. They had drifted, and the drift had
// two costs that a person could see. The whole-extent tool was declared here with a shortcut and was
// absent from the page's copy, so it had a documented key that did nothing; and the page's shortcut
// handler wrote `k` twice, so the second branch was unreachable. There is one registry now.

describe("the registry is the only source of tools", () => {
  it("gives every mode at least a select tool", () => {
    for (const m of MODES) {
      expect(toolsForMode(m.key), m.key).toContain("select");
    }
  });

  it("has no two tools sharing a hotkey inside one mode", () => {
    // The invariant the old hardcoded chain broke. Across modes a letter may repeat, which is the point
    // of resolving per mode: `k` is the whole-extent tool in Objects and the keypoint tool in Pose.
    expect(hotkeyCollisions()).toEqual([]);
  });

  it("resolves a hotkey differently in two modes rather than letting one win everywhere", () => {
    expect(toolForHotkey("objects", "k")).toBe("amodal");
    expect(toolForHotkey("pose", "k")).toBe("keypoint");
  });

  it("returns null for a key the mode does not claim, so the editor can use it for something else", () => {
    // `f` is fit-to-view and `a` is accept-frame. A mode that quietly claimed either would take a
    // shortcut away from every annotator without anything failing.
    for (const m of MODES) {
      expect(toolForHotkey(m.key, "f"), `${m.key} claims f`).toBeNull();
      expect(toolForHotkey(m.key, "a"), `${m.key} claims a`).toBeNull();
    }
  });

  it("is case insensitive, because a shortcut is not a different one with caps lock on", () => {
    expect(toolForHotkey("objects", "B")).toBe(toolForHotkey("objects", "b"));
  });

  it("falls back to the Objects strip for an unknown mode rather than rendering nothing", () => {
    expect(groupsForMode("no-such-mode")).toEqual(groupsForMode("objects"));
  });

  it("keeps the whole-extent tool reachable", () => {
    expect(toolsForMode("objects")).toContain("amodal");
  });

  it("offers only tools the canvas dispatches", () => {
    // Lanes, 3D and Events are driven by their own panels rather than by the canvas tool. They used to
    // list lane types, surfaces and event marks here; none of those reached the canvas, so each was an
    // inert button with a hotkey that did nothing.
    for (const key of ["lanes", "lidar3d", "events"]) {
      expect(toolsForMode(key), key).toEqual(["select"]);
    }
  });

  it("names every mode it lists", () => {
    for (const m of MODES) {
      expect(modeByKey(m.key)?.label).toBeTruthy();
      expect(m.rail.length).toBeLessThanOrEqual(4);
    }
  });
});
