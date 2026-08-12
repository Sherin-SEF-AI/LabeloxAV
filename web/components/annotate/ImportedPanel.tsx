"use client";

import { useEffect, useState } from "react";
import { api, humanizeError } from "@/lib/api";
import { type QueueItem, humanSize } from "@/lib/uploadQueue";

// What a finished clip actually became, shown beside the queue instead of instead of it.
//
// The row's "open" used to be an <a href> to the frame editor. Clicking it navigated, which unmounted the
// upload page and abandoned every transfer still in flight, so the one control offered on a completed item
// destroyed the rest of the batch. During a 186-file import that is an hour of work lost to a link that
// looked informational.
//
// So it opens here: the session's first frame, its counts, and an explicit link out for when the user really
// does want to leave. Making the destructive option a deliberate second click rather than the only click is
// the whole point.

type Preview = { frames: number; objects: number; imageUrl: string | null; done: number };

export default function ImportedPanel({ item, onClose }: { item: QueueItem; onClose: () => void }) {
  const [preview, setPreview] = useState<Preview | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let alive = true;
    setPreview(null);
    setErr(null);
    setLoaded(false);
    (async () => {
      try {
        // sessionStats already answers exactly this question, so the panel reuses it rather than adding
        // an endpoint that would return the same numbers under a different name.
        const s = await api.sessionStats(item.sessionId!);
        if (!alive) return;
        setPreview({
          frames: s.frames ?? 0,
          objects: s.objects ?? 0,
          done: s.done ?? 0,
          imageUrl: item.frameId ? `/api/frames/${item.frameId}/image` : null,
        });
      } catch (e) {
        if (alive) setErr(humanizeError(e));
      } finally {
        if (alive) setLoaded(true);
      }
    })();
    return () => { alive = false; };
  }, [item.sessionId, item.frameId]);

  return (
    <aside className="reveal w-80 shrink-0 border hairline rounded bg-panel flex flex-col max-h-[36rem]">
      <div className="flex items-center gap-2 px-3 py-2 border-b hairline">
        <span className="font-mono text-[11px] uppercase text-ink-3">imported</span>
        <button onClick={onClose} title="close"
          className="ml-auto font-mono text-[11px] text-ink-3 hover:text-ink px-1">x</button>
      </div>

      <div className="p-3 space-y-3 overflow-y-auto">
        <div className="font-mono text-[11px] text-ink break-all">{item.name}</div>

        {preview?.imageUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={preview.imageUrl} alt="" className="w-full rounded border hairline object-cover" />
        ) : (
          <div className={`w-full h-32 rounded border hairline ${loaded ? "" : "skeleton"}`}>
            {loaded && (
              <div className="h-full flex items-center justify-center font-mono text-[10px] text-ink-3">
                {/* An import with no frames is a real outcome, and saying so beats an endless placeholder. */}
                no frame to preview
              </div>
            )}
          </div>
        )}

        <dl className="grid grid-cols-2 gap-y-1 font-mono text-[10px]">
          <dt className="text-ink-3">frames</dt>
          <dd className="text-ink text-right">{loaded ? (preview?.frames ?? 0).toLocaleString() : "..."}</dd>
          <dt className="text-ink-3">objects</dt>
          <dd className="text-ink text-right">{loaded ? (preview?.objects ?? 0).toLocaleString() : "..."}</dd>
          <dt className="text-ink-3">size</dt>
          {/* The same formatter the queue row uses. Dividing by 1e6 rendered a 25 KB clip as "0 MB". */}
          <dd className="text-ink text-right">{humanSize(item.size)}</dd>
          <dt className="text-ink-3">reviewed</dt>
          <dd className="text-ink text-right">{loaded ? (preview?.done ?? 0).toLocaleString() : "..."}</dd>
        </dl>

        {err && <div className="font-mono text-[10px] text-block">{err}</div>}
      </div>

      <div className="border-t hairline p-2 flex items-center gap-2">
        {item.frameId ? (
          <a href={`/frame/${item.frameId}`}
            // Deliberately a real navigation, and deliberately not the only control on the row. Leaving is
            // now a choice somebody makes on purpose rather than the default action on a finished clip.
            className="font-mono text-[10px] border border-line rounded px-2 py-1 text-accent hover:border-accent">
            open in editor
          </a>
        ) : (
          <span className="font-mono text-[10px] text-ink-3">nothing to open</span>
        )}
        <a href={`/annotations?session=${item.sessionId}`}
          className="font-mono text-[10px] text-ink-3 hover:text-ink ml-auto">session -&gt;</a>
      </div>
    </aside>
  );
}
