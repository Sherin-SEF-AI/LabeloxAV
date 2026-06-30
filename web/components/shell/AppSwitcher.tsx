"use client";

import { useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { APP_GROUPS } from "@/lib/editor/registry";

// The grouped application menu. Replaces the flat nav scroll: destinations are organized by area, so the
// list grows by adding an item to a group, never by widening a row. Opened from the "apps" button.

export default function AppSwitcher() {
  const router = useRouter();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const go = (href: string) => { setOpen(false); router.push(href); };

  return (
    <div className="relative">
      <button onClick={() => setOpen((o) => !o)} title="all apps"
        className="font-mono text-xs border border-line px-2 py-1 text-ink-2 hover:border-accent">
        apps
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div className="absolute left-0 mt-1 z-50 panel p-3 w-[34rem] grid grid-cols-2 gap-x-4 gap-y-3">
            {APP_GROUPS.map((g) => (
              <div key={g.key}>
                <div className="font-mono text-[10px] uppercase tracking-wide text-ink-3 mb-1">{g.label}</div>
                <div className="space-y-0.5">
                  {g.items.map((it) => (
                    <button key={it.href} onClick={() => go(it.href)} title={it.hint}
                      className={`block w-full text-left font-mono text-xs px-1.5 py-1 hover:bg-bg-2 ${pathname === it.href ? "text-accent" : "text-ink-2"}`}>
                      {it.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
