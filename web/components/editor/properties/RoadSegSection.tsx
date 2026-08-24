"use client";

// Road surface and lane geometry for this frame. Three buttons, because the editing itself happens on the
// dedicated lane surface where there is room to drag a boundary onto a kerb.

import { useRouter } from "next/navigation";

export default function RoadSegSection({ frameId, onSegRoad, onProposeLanes }: {
  frameId: string;
  onSegRoad: () => void;
  onProposeLanes: () => void;
}) {
  const router = useRouter();
  return (
    <div className="grid grid-cols-2 gap-1 font-mono text-[10px]">
      <button onClick={onSegRoad}
        className="border border-line rounded text-ink-2 px-1.5 py-1 hover:border-accent">segment road</button>
      <button onClick={onProposeLanes}
        className="border border-line rounded text-ink-2 px-1.5 py-1 hover:border-accent">propose lanes</button>
      <button onClick={() => router.push(`/annotate/lane/${frameId}`)}
        className="border border-line rounded text-ink-2 px-1.5 py-1 hover:border-accent col-span-2">
        edit lanes + drivable &rarr;
      </button>
    </div>
  );
}
