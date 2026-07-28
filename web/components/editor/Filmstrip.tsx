"use client";

import { useEffect, useRef, useState } from "react";

import { api } from "@/lib/api";

// A strip of neighbouring frames under the canvas.
//
// The editor was strictly frame-at-a-time: [ and ] stepped one frame, and there was no way to see what was
// coming or to jump back several frames to where an object first appeared. Every temporal judgement (is this
// the same vehicle, when did the occlusion start, did the tracker drop it) was a sequence of blind single
// steps. A real scrubber existed only in the inspector and in the separate track workspace, neither of which
// is where annotation happens.

type Tile = {
  frame_id: string;
  ts_ns: number;
  n_objects: number;
  image_url: string;
  current: boolean;
};

export default function Filmstrip({
  frameId,
  onPick,
  span = 12,
}: {
  frameId: string;
  onPick: (frameId: string) => void;
  span?: number;
}) {
  const [tiles, setTiles] = useState<Tile[]>([]);
  const currentRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .frameFilmstrip(frameId, span)
      .then((r) => {
        if (!cancelled) setTiles(r.frames);
      })
      .catch(() => {
        // The strip is an aid, not the editor. A failure here must not take the canvas down with it.
        if (!cancelled) setTiles([]);
      });
    return () => {
      cancelled = true;
    };
  }, [frameId, span]);

  useEffect(() => {
    // Keep the current frame centred as the reviewer steps, so the strip stays a window on "here" rather
    // than something they have to scroll manually every time.
    currentRef.current?.scrollIntoView({ block: "nearest", inline: "center" });
  }, [tiles]);

  if (tiles.length <= 1) return null;

  return (
    <div
      className="flex items-center gap-1 overflow-x-auto no-scrollbar border-t hairline bg-bg px-2 py-1.5"
      role="listbox"
      aria-label="nearby frames"
    >
      {tiles.map((t) => (
        <button
          key={t.frame_id}
          ref={t.current ? currentRef : undefined}
          onClick={() => !t.current && onPick(t.frame_id)}
          role="option"
          aria-selected={t.current}
          aria-label={`frame with ${t.n_objects} objects`}
          title={`${t.n_objects} objects`}
          className={`relative shrink-0 rounded overflow-hidden border ${
            t.current ? "border-accent" : "border-line hover:border-ink-3"
          }`}
        >
          {/* eslint-disable-next-line @next/next/no-img-element -- the frame proxy streams raw jpeg; the
              Next image loader would re-encode every tile for no benefit on a strip this small. */}
          <img src={t.image_url} alt="" width={72} height={48}
               className="w-[72px] h-[48px] object-cover bg-bg-2" loading="lazy" />
          <span className="absolute bottom-0 right-0 px-1 bg-bg/80 font-mono text-[9px] text-ink-3">
            {t.n_objects}
          </span>
        </button>
      ))}
    </div>
  );
}
