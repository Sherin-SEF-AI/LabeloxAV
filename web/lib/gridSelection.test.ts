import { describe, expect, it } from "vitest";
import {
  cursorAfterRemoval, moveCursor, rangeBetween, targetIndices, toggle,
} from "./gridSelection";

// A 4-wide grid holding 10 tiles, so the last row is partial:
//   0 1 2 3
//   4 5 6 7
//   8 9
const COLS = 4;
const N = 10;

describe("moveCursor", () => {
  it("reads left and right as one sequence, not per row", () => {
    expect(moveCursor(3, "right", N, COLS)).toBe(4);   // over the row end
    expect(moveCursor(4, "left", N, COLS)).toBe(3);    // and back
  });

  it("never leaves the grid", () => {
    expect(moveCursor(0, "left", N, COLS)).toBe(0);
    expect(moveCursor(N - 1, "right", N, COLS)).toBe(N - 1);
  });

  it("clamps up at the top row rather than wrapping a screen away", () => {
    expect(moveCursor(2, "up", N, COLS)).toBe(2);
    expect(moveCursor(6, "up", N, COLS)).toBe(2);
  });

  it("reaches a partial last row from the row above", () => {
    expect(moveCursor(8, "down", N, COLS)).toBe(8);    // already on the last row
    expect(moveCursor(5, "down", N, COLS)).toBe(9);    // 5 + 4 = 9, which exists
    // 7 + 4 = 11 is past the end, but the last row still has tile 9 in it.
    expect(moveCursor(7, "down", N, COLS)).toBe(9);
  });

  it("does not move down when already on the last row", () => {
    expect(moveCursor(9, "down", N, COLS)).toBe(9);
  });

  it("survives an empty grid", () => {
    expect(moveCursor(0, "right", 0, COLS)).toBe(0);
  });
});

describe("rangeBetween", () => {
  it("covers both directions inclusively", () => {
    expect(rangeBetween(2, 5)).toEqual([2, 3, 4, 5]);
    expect(rangeBetween(5, 2)).toEqual([2, 3, 4, 5]);
  });

  it("a range of one is one", () => {
    expect(rangeBetween(4, 4)).toEqual([4]);
  });
});

describe("toggle", () => {
  it("adds then removes", () => {
    const a = toggle(new Set<number>(), 3);
    expect([...a]).toEqual([3]);
    expect([...toggle(a, 3)]).toEqual([]);
  });

  it("does not mutate what it was given", () => {
    const original = new Set([1]);
    toggle(original, 2);
    expect([...original]).toEqual([1]);
  });
});

describe("targetIndices", () => {
  it("acts on the cursor when nothing is selected", () => {
    // The common case is a correct tile taking one keystroke. Requiring a selection first would put two
    // keystrokes on it to save one on the rare run of wrong tiles.
    expect(targetIndices(new Set(), 7)).toEqual([7]);
  });

  it("acts on the selection when there is one, in order", () => {
    expect(targetIndices(new Set([5, 1, 3]), 9)).toEqual([1, 3, 5]);
  });
});

describe("cursorAfterRemoval", () => {
  it("shifts back by the number of decided tiles before it", () => {
    // Tiles 1 and 2 decided while the cursor sat on 5: it should still point at the same crop, now at 3.
    expect(cursorAfterRemoval(5, [1, 2], 8)).toBe(3);
  });

  it("does not move for tiles decided after it", () => {
    expect(cursorAfterRemoval(2, [5, 6], 8)).toBe(2);
  });

  it("lands on the new last tile when the last one was decided", () => {
    // Returning 0 here would send somebody who just finished the queue back to the beginning.
    expect(cursorAfterRemoval(9, [9], 9)).toBe(8);
  });

  it("goes to zero only when the queue is empty", () => {
    expect(cursorAfterRemoval(3, [0, 1, 2, 3], 0)).toBe(0);
  });

  it("handles a whole selection being decided at once", () => {
    expect(cursorAfterRemoval(6, [2, 3, 4], 7)).toBe(3);
  });
});
