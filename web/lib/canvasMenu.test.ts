import { describe, expect, it } from "vitest";

import { canvasMenu, enabledKeys, menuPosition, type CanvasTarget } from "./canvasMenu";

// The cases that matter are the ones where the menu would act on something other than what was aimed at,
// and the ones where a row is offered that cannot possibly work.

const target = (over: Partial<CanvasTarget> = {}): CanvasTarget => ({
  object: null, selectedCount: 0, targetInSelection: false, ...over,
});
const obj = (over: Partial<NonNullable<CanvasTarget["object"]>> = {}) => ({
  id: "o1", class_name: "sedan", visible: true, state: "review", ...over,
});
const keys = (t: CanvasTarget) => canvasMenu(t).flatMap((s) => s.items.map((i) => i.key));
const item = (t: CanvasTarget, k: string) =>
  canvasMenu(t).flatMap((s) => s.items).find((i) => i.key === k);

describe("which menu you get", () => {
  it("right-clicking empty canvas acts on the frame, not on a hidden selection", () => {
    expect(keys(target())).toContain("selectAll");
    expect(keys(target())).not.toContain("delete");
  });

  it("right-clicking one object acts on that object", () => {
    const k = keys(target({ object: obj() }));
    expect(k).toContain("delete");
    expect(k).not.toContain("deleteMany");
  });

  it("right-clicking inside a multi-selection acts on all of it", () => {
    const k = keys(target({ object: obj(), selectedCount: 12, targetInSelection: true }));
    expect(k).toContain("deleteMany");
    expect(item(target({ object: obj(), selectedCount: 12, targetInSelection: true }), "deleteMany")!.label)
      .toBe("delete 12");
  });

  it("right-clicking an object OUTSIDE the selection acts on that object alone", () => {
    // Otherwise aiming at one box and getting an action on twelve others is a destructive surprise.
    const k = keys(target({ object: obj(), selectedCount: 12, targetInSelection: false }));
    expect(k).toContain("delete");
    expect(k).not.toContain("deleteMany");
  });

  it("offers reclassifying a multi-selection, which had no control anywhere", () => {
    // relabelSelected acts on the primary selection only and the bulk toolbar offered hide/show/lock/
    // delete, so marquee-selecting twelve boxes and pressing a class key changed one of them.
    expect(item(target({ object: obj(), selectedCount: 12, targetInSelection: true }), "class")!.submenu)
      .toBe("class");
  });
});

describe("rows that cannot work are shown and explained, not hidden", () => {
  it("greys the editing rows on a locked object and says why", () => {
    const t = target({ object: obj({ locked: true }) });
    for (const k of ["class", "accept", "reject", "delete"]) {
      expect(item(t, k)!.disabled, k).toBe(true);
      expect(item(t, k)!.why, k).toMatch(/locked/);
    }
  });

  it("still lets you unlock it, or the lock is unreachable from the only place it shows", () => {
    const t = target({ object: obj({ locked: true }) });
    expect(item(t, "lock")!.disabled).toBeFalsy();
    expect(item(t, "lock")!.label).toBe("unlock");
  });

  it("greys auto-track on an unsaved object, because the server has never seen it", () => {
    expect(item(target({ object: obj({ isNew: true }) }), "propagate")!.disabled).toBe(true);
  });

  it("greys open-track when the object is not in one", () => {
    expect(item(target({ object: obj() }), "viewTrack")!.disabled).toBe(true);
    expect(item(target({ object: obj({ track_id: "t1" }) }), "viewTrack")!.disabled).toBeFalsy();
  });

  it("greys clear-selection when nothing is selected", () => {
    expect(item(target(), "selectNone")!.disabled).toBe(true);
    expect(item(target({ selectedCount: 3 }), "selectNone")!.disabled).toBeFalsy();
  });

  it("every disabled row explains itself", () => {
    for (const t of [target(), target({ object: obj({ locked: true, isNew: true }) })]) {
      for (const s of canvasMenu(t)) {
        for (const i of s.items) {
          if (i.disabled) expect(i.why, i.key).toBeTruthy();
        }
      }
    }
  });

  it("keyboard walking skips the disabled rows", () => {
    const k = enabledKeys(canvasMenu(target({ object: obj({ locked: true }) })));
    expect(k).not.toContain("delete");
    expect(k).toContain("lock");
  });
});

describe("menuPosition", () => {
  const size = { w: 220, h: 300 };
  const vp = { w: 1000, h: 800 };

  it("opens at the pointer when there is room", () => {
    expect(menuPosition({ x: 100, y: 100 }, size, vp)).toEqual({ left: 100, top: 100 });
  });

  it("flips left rather than running off the right edge", () => {
    // The canvas fills the window, so right-clicking near an edge is the normal case.
    expect(menuPosition({ x: 950, y: 100 }, size, vp).left).toBe(730);
  });

  it("lifts up rather than running off the bottom", () => {
    expect(menuPosition({ x: 100, y: 780 }, size, vp).top).toBe(492);
  });

  it("never positions off the top-left when the menu is bigger than the viewport", () => {
    const pos = menuPosition({ x: 10, y: 10 }, { w: 2000, h: 2000 }, vp);
    expect(pos.left).toBeGreaterThanOrEqual(0);
    expect(pos.top).toBeGreaterThanOrEqual(0);
  });
});
