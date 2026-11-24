// A brush stroke, as the stamps it is made of.
//
// A stroke is a list of pointer positions, and what is committed is a circle of the brush radius at each
// one. The preview used to draw those as one Konva node per stamp, so a two-second stroke was a few
// hundred nodes created, reconciled and redrawn on every pointer move. It is one shape now, tracing the
// same circles into a single path.
//
// The same circles, not a thick line through the points. A line would look better on a fast stroke and
// would be a preview of something the server does not do: the mask is composed from stamps, and where the
// pointer outran them the result has gaps. The preview's job is to show what will be committed.

export type BrushOp = { op: string; center: number[]; radius: number };

/** The compose-mask operations for a stroke: one stamp per sampled point, added or erased. */
export function brushOps(stamps: number[][], tool: string, radius: number): BrushOp[] {
  const op = tool === "eraser" ? "erase" : "add";
  return stamps.map((c) => ({ op, center: c, radius }));
}

type PathLike = {
  moveTo: (x: number, y: number) => void;
  arc: (x: number, y: number, r: number, a0: number, a1: number) => void;
};

/**
 * Add every stamp to the current path as its own closed circle.
 *
 * The moveTo before each arc matters: without it the path runs a straight edge from the end of one
 * circle to the start of the next, and the fill covers the sliver between them.
 */
export function traceStamps(ctx: PathLike, stamps: number[][], radius: number): void {
  for (const c of stamps) {
    ctx.moveTo(c[0] + radius, c[1]);
    ctx.arc(c[0], c[1], radius, 0, Math.PI * 2);
  }
}
