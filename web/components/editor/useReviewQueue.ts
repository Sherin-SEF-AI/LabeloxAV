"use client";

import { useEffect, useRef, useState } from "react";

import { api, humanizeError } from "@/lib/api";
import type { AlItem, ErrorCandidateRow, FrameMeta } from "@/lib/types";

// Review mode's queues for one frame in the editor: the value queue (this frame's items first) and the
// pending error candidates, loaded once when review mode opens. Moved out of app/frame/[id]/page.tsx.
//
// A failed load is reported, not turned into an empty list: "value queue empty" and "could not load the
// value queue" send a reviewer in opposite directions, and the old code said the first when it meant the
// second.

export function useReviewQueue({ mode, meta, frameId: id }: { mode: string; meta: FrameMeta | null; frameId: string }) {
  const [alItems, setAlItems] = useState<AlItem[]>([]);
  const [errItems, setErrItems] = useState<ErrorCandidateRow[]>([]);
  // `reviewLoaded` means the load FINISHED. It used to be set as the load started, so the panel said "value
  // queue empty" for the several seconds the scoring takes. `requested` stops a second fetch meanwhile.
  const [reviewLoaded, setReviewLoaded] = useState(false);
  const [reviewError, setReviewError] = useState<string | null>(null);
  const requested = useRef(false);

  useEffect(() => {
    if (mode !== "review" || !meta || requested.current) return;
    requested.current = true;
    (async () => {
      const [al, ec] = await Promise.allSettled([
        api.alScore(meta.session_id, 80).then((r) => r.items),
        api.errorCandidates("pending", 80),
      ]);
      const problems: string[] = [];
      if (al.status === "fulfilled") {
        const items = [...al.value];
        items.sort((a, b) => (a.frame_id === id ? 0 : 1) - (b.frame_id === id ? 0 : 1) || b.value - a.value);
        setAlItems(items);
      } else problems.push(`value queue: ${humanizeError(al.reason)}`);
      if (ec.status === "fulfilled") setErrItems(ec.value);
      else problems.push(`error candidates: ${humanizeError(ec.reason)}`);
      setReviewError(problems.length ? `could not load the ${problems.join("; ")}` : null);
      setReviewLoaded(true);
    })();
  }, [mode, meta, id]);

  return { alItems, setAlItems, errItems, setErrItems, reviewLoaded, reviewError };
}
