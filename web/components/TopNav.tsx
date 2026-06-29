"use client";

import { useRouter } from "next/navigation";
import UserPicker from "./UserPicker";
import AppSwitcher from "./shell/AppSwitcher";
import CommandPalette from "./shell/CommandPalette";

// Shared app navigation. The flat link row (which overflowed once the platform had ~20 destinations) is
// replaced by a grouped app switcher plus a Cmd+K command palette, so nav grows by organization. The
// {active, right} interface is preserved, so every page that renders TopNav keeps working unchanged; the
// active label now reads as a quiet breadcrumb.

export default function TopNav({ active, right }: { active: string; right?: React.ReactNode }) {
  const router = useRouter();
  return (
    <header className="flex items-center justify-between gap-4 px-4 h-12 border-b hairline shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        <button onClick={() => router.push("/")} className="font-display font-bold shrink-0" title="home (triage)">
          Labelox<span className="text-accent">AV</span>
        </button>
        <AppSwitcher />
        <button onClick={() => window.dispatchEvent(new Event("lbx:palette"))} title="command palette (Cmd+K)"
          className="font-mono text-[11px] text-ink-3 border border-line px-2 py-1 hover:border-accent">
          go to <span className="text-ink-2">Cmd K</span>
        </button>
        <span className="font-mono text-xs text-ink-3 truncate">/ {active}</span>
      </div>
      <div className="flex items-center gap-3 font-mono text-xs shrink-0">
        {right}
        <UserPicker />
      </div>
      <CommandPalette />
    </header>
  );
}
