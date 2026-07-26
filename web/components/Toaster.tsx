"use client";

import { useEffect, useState } from "react";
import { subscribeToasts, type Toast } from "@/lib/toast";

// Renders the app-wide toast stack, bottom-right, auto-dismissing. Mounted once in the root layout so any
// module can raise a toast without a provider prop drilling through the tree. Accessible: the region is a
// polite live region so a screen reader announces each message.

const STYLE: Record<Toast["kind"], string> = {
  info: "border-line text-ink-2",
  success: "border-pass text-pass",
  warn: "border-warn text-warn",
  error: "border-block text-block",
};

export default function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  useEffect(() => {
    return subscribeToasts((t) => {
      setToasts((cur) => [...cur, t]);
      setTimeout(() => setToasts((cur) => cur.filter((x) => x.id !== t.id)), t.ttl);
    });
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div aria-live="polite" aria-atomic="false"
      className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm pointer-events-none">
      {toasts.map((t) => (
        <div key={t.id}
          className={`pointer-events-auto bg-bg-2 border ${STYLE[t.kind]} px-3 py-2 font-mono text-[11px] shadow-lg flex items-start gap-2`}>
          <span className="flex-1 break-words">{t.message}</span>
          <button onClick={() => setToasts((cur) => cur.filter((x) => x.id !== t.id))}
            aria-label="dismiss" className="text-ink-3 hover:text-ink shrink-0">✕</button>
        </div>
      ))}
    </div>
  );
}
