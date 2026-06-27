"use client";

import { useRouter } from "next/navigation";

// In-app back navigation: returns to wherever you came from (triage, curation, a track, ...), falling
// back to a sensible route when there is no history (deep link / fresh tab). Operational Materialism:
// grey until hovered, no motion.
export default function BackButton({ fallback = "/", label = "back" }: { fallback?: string; label?: string }) {
  const router = useRouter();
  const goBack = () => {
    if (typeof window !== "undefined" && window.history.length > 1) router.back();
    else router.push(fallback);
  };
  return (
    <button
      onClick={goBack}
      title="back (Alt+Left)"
      className="font-mono text-[11px] text-ink-3 hover:text-accent border border-line hover:border-accent px-2 py-1 shrink-0"
    >
      &larr; {label}
    </button>
  );
}
