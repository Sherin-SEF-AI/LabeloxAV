"use client";

// Lift the oriented boxes drawn on the bird's-eye view into metric 3D cuboids, using the points each box
// encloses. Only rendered when the frame actually carries a point cloud.

export default function LidarBevSection({ onLift }: { onLift: () => void | Promise<void> }) {
  return (
    <button
      onClick={() => void onLift()}
      title="draw oriented boxes (select + rotate handle), then lift each to a metric 3D cuboid using the enclosed points"
      className="w-full font-mono text-[10px] border border-line rounded text-ink-2 px-1.5 py-1 hover:border-accent">
      compute 3D cuboids from boxes &rarr;
    </button>
  );
}
