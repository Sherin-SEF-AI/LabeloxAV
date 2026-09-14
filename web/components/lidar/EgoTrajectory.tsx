"use client";

import { useMemo } from "react";

import type { EgoPosePoint, EgoTrajectory as Trajectory } from "@/lib/types";

// The trajectory plan view: where the vehicle went, drawn in the session's own ENU frame, with the
// current cloud's position marked. It exists because a cuboid is only comparable across frames if the
// frames have a common origin, and until migration 0112 nothing in this engine knew where the car was.
//
// Two things this drawing refuses to smooth over. A step with no recovered scale does not advance the
// position, so the path has a visible stall rather than an invented straight line; and a pose that was
// inferred from pixels is drawn differently from one an instrument measured, because a trajectory made
// of guesses that looks like a measured one is the failure mode this whole table is shaped against.

const PAD = 12;

function bounds(points: EgoPosePoint[]) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  // A stationary session collapses to a point; give it a metre of span so it still draws.
  const spanX = Math.max(maxX - minX, 1);
  const spanY = Math.max(maxY - minY, 1);
  return { minX, minY, spanX, spanY };
}

export default function EgoTrajectory(
  { trajectory, currentTs, height = 200 }:
  { trajectory: Trajectory | null; currentTs?: number | null; height?: number },
) {
  const pts = trajectory?.points ?? [];
  const geom = useMemo(() => {
    if (pts.length < 2) return null;
    const { minX, minY, spanX, spanY } = bounds(pts);
    const scale = Math.min((100 - 2 * PAD) / spanX, (100 - 2 * PAD) / spanY);
    // North up: SVG y grows downward, so the north coordinate is subtracted rather than added.
    const project = (p: EgoPosePoint) => ({
      x: PAD + (p.x - minX) * scale,
      y: 100 - PAD - (p.y - minY) * scale,
    });
    return { project, path: pts.map(project) };
  }, [pts]);

  if (!trajectory) {
    return (
      <div className="px-3 py-2 text-[11px] text-ink-3">
        no trajectory has been built for this session
      </div>
    );
  }
  if (!geom) {
    return (
      <div className="px-3 py-2 text-[11px] text-ink-3">
        {trajectory.poses} pose{trajectory.poses === 1 ? "" : "s"}, too few to draw a path
      </div>
    );
  }

  const inferred = trajectory.poses - trajectory.measured;
  const currentIdx = currentTs == null ? -1
    : pts.reduce((best, p, i) => (Math.abs(p.ts_ns - currentTs)
        < Math.abs(pts[best].ts_ns - currentTs) ? i : best), 0);
  const d = geom.path.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");

  return (
    <div className="flex flex-col gap-1">
      <svg viewBox="0 0 100 100" style={{ height }} className="w-full"
        preserveAspectRatio="xMidYMid meet" role="img" aria-label="ego trajectory">
        <path d={d} fill="none" stroke="#22d3ee" strokeWidth={0.8} strokeLinejoin="round" />
        {pts.map((p, i) => (
          // A pose with no speed is a step whose scale was never recovered. Marked, so the stall in the
          // path reads as missing information rather than as the vehicle having stopped.
          p.speed_mps == null ? (
            <circle key={p.ts_ns} cx={geom.path[i].x} cy={geom.path[i].y} r={0.7}
              fill="none" stroke="#f59e0b" strokeWidth={0.3} />
          ) : null
        ))}
        <circle cx={geom.path[0].x} cy={geom.path[0].y} r={1.4} fill="#22d3ee" />
        {currentIdx >= 0 && (
          <circle cx={geom.path[currentIdx].x} cy={geom.path[currentIdx].y} r={2}
            fill="none" stroke="#ffffff" strokeWidth={0.9} />
        )}
      </svg>
      <div className="px-1 text-[10px] text-ink-3">
        {trajectory.poses} poses ·{" "}
        {trajectory.measured > 0
          ? `${trajectory.measured} measured`
          : `${inferred} inferred from the images, none measured`}
        {trajectory.source ? ` · ${trajectory.source}` : ""}
      </div>
    </div>
  );
}
