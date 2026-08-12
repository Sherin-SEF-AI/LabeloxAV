"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { subscribeUploads, type UploadState } from "@/lib/uploadManager";
import { summarize } from "@/lib/uploadQueue";

// What the upload queue is doing, in the top bar, from anywhere in the app.
//
// The queue survives navigation now, which solved the important half of the problem and created a smaller
// one: work continues with nothing on screen to say so. Somebody who starts a 186-clip import and moves on
// has no way to tell whether it is still going, how far it has got, or how to get back to it, short of
// remembering the URL.
//
// So it lives beside the cloud-GPU chip, which is the same kind of claim: a thing the machine is doing that
// you did not ask about again. It renders nothing at all when nothing is running, because a permanent
// zero-state chip in a top bar is furniture, and the bar is already dense.

export default function UploadIndicator() {
  const router = useRouter();
  const [state, setState] = useState<UploadState | null>(null);

  useEffect(() => subscribeUploads(setState), []);

  if (!state?.running) return null;
  const s = summarize(state.items);
  const pct = Math.round(s.progress * 100);
  const verb = state.phase === "autolabeling" ? "labelling" : "importing";

  return (
    <button
      onClick={() => router.push("/annotate/new")}
      title={`${verb} ${s.done + s.failed} of ${s.total}, click to open the queue`}
      className="btn text-[11px] gap-1.5 h-7 items-center">
      <span className="running-dot" />
      <span className="hidden sm:inline text-ink-2">{verb}</span>
      <span className="text-ink-3">{s.done + s.failed}/{s.total}</span>
      {/* A bar rather than only a count: at 186 files the counter moves once every few minutes, and a
          number that still reads 12/186 says nothing about whether it is progressing or wedged. */}
      <span className="hidden md:block w-12 h-1 bg-line rounded overflow-hidden">
        <span className="block h-full bg-accent transition-[width] duration-500 ease-out"
          style={{ width: `${pct}%` }} />
      </span>
      {s.failed > 0 && <span className="text-block">{s.failed}!</span>}
    </button>
  );
}
