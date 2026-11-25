"use client";

import { useState } from "react";
import Icon from "@/components/shell/Icon";

// A frame thumbnail that is never a bare black tile. While the image is on its way the tile shimmers like
// the rest of the loading states; if it cannot be fetched (an expired link, a frame whose image was never
// stored) it says so, instead of a dark rectangle that reads as a night-time frame or a broken page.
// Lazy, because the curation and discovery grids ask for a few hundred of these at once.

export default function Thumb({ src, className = "", alt = "" }: { src: string | null | undefined; className?: string; alt?: string }) {
  const [state, setState] = useState<"loading" | "ok" | "failed">(src ? "loading" : "failed");
  return (
    <span className={`relative block overflow-hidden bg-bg-2 ${className}`}>
      {state === "loading" && <span className="skeleton absolute inset-0 rounded-none" aria-hidden />}
      {state === "failed" ? (
        <span className="absolute inset-0 flex flex-col items-center justify-center gap-1 text-ink-3 text-[11px]">
          <Icon name="eye" size={14} />
          <span>Image unavailable</span>
        </span>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={src ?? undefined} alt={alt} loading="lazy" decoding="async"
          // A cached image can finish before React attaches onLoad; the ref catches that case.
          ref={(el) => { if (el?.complete && el.naturalWidth > 0 && state === "loading") setState("ok"); }}
          onLoad={() => setState("ok")} onError={() => setState("failed")}
          className={`w-full h-full object-cover transition-opacity ${state === "ok" ? "opacity-100" : "opacity-0"}`} />
      )}
    </span>
  );
}
