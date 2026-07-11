"use client";

import { useRouter } from "next/navigation";
import UserPicker from "./UserPicker";
import AppSwitcher from "./shell/AppSwitcher";
import PlatformSwitcher from "./shell/PlatformSwitcher";
import CommandPalette from "./shell/CommandPalette";
import ShortcutOverlay from "./shell/ShortcutOverlay";
import CloudControl from "./shell/CloudControl";

// Shared app navigation, Blender-style: a raised header strip with the platform launcher, the platform and
// app switchers, and self-explanatory controls (every button carries a tooltip). The {active, right}
// interface is preserved, so every page keeps working; the active label reads as a quiet breadcrumb.

export default function TopNav({ active, right }: { active: string; right?: React.ReactNode }) {
  const router = useRouter();
  return (
    <header className="flex items-center justify-between gap-4 px-3 h-12 bg-head border-b border-[#262626] shrink-0">
      <div className="flex items-center gap-2 min-w-0">
        <button
          onClick={() => { if (typeof window !== "undefined" && window.history.length > 1) router.back(); else router.push("/"); }}
          data-tip="Back (Alt+Left)"
          className="flex items-center justify-center w-7 h-7 rounded text-ink-2 hover:text-ink hover:bg-panel border border-line shrink-0"
        >
          <span className="text-[15px] leading-none">&larr;</span>
        </button>
        <button onClick={() => router.push("/platforms")} data-tip="Platform launcher: pick a plane of the data engine"
          className="font-display font-bold shrink-0 px-1 text-ink">
          Labelox<span className="text-accent-2">AV</span>
        </button>
        <PlatformSwitcher />
        <AppSwitcher />
        <button onClick={() => window.dispatchEvent(new Event("lbx:palette"))} data-tip="Command palette: jump to any page"
          className="btn text-[11px] gap-1.5">
          <span className="text-ink-3">go to</span> <kbd className="text-ink-2">Cmd K</kbd>
        </button>
        <button onClick={() => window.dispatchEvent(new Event("lbx:shortcuts"))} data-tip="Keyboard shortcuts (?)"
          className="btn w-7 px-0 justify-center">?</button>
        <span className="text-xs text-ink-3 truncate pl-1">/ {active}</span>
      </div>
      <div className="flex items-center gap-2 text-xs shrink-0">
        {right}
        <CloudControl />
        <UserPicker />
      </div>
      <CommandPalette />
      <ShortcutOverlay />
    </header>
  );
}
