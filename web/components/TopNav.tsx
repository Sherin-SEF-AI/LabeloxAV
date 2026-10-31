"use client";

import { usePathname, useRouter } from "next/navigation";
import UserPicker from "./UserPicker";
import CommandPalette from "./shell/CommandPalette";
import ShortcutOverlay from "./shell/ShortcutOverlay";
import CloudControl from "./shell/CloudControl";
import UploadIndicator from "./shell/UploadIndicator";
import Icon from "./shell/Icon";
import NotificationBell from "./shell/NotificationBell";
import { MENU_DESTINATIONS } from "@/lib/menus";
import { NAV_DESTINATIONS, activeKey } from "@/lib/sidebar";

// The top bar over every page: where you are, then search and the session controls. Navigation itself lives
// in the sidebar (components/shell/Sidebar.tsx); below the tablet breakpoint the sidebar is a drawer and the
// menu button here opens it.
//
// The {active, right} interface is unchanged, so every page keeps working.

function Crumb({ active }: { active: string }) {
  const pathname = usePathname();
  // Prefer the real destination label for the current path, so the crumb matches what the menu called it
  // rather than whatever string each page happened to pass.
  const key = activeKey(pathname ?? "/");
  const label = NAV_DESTINATIONS.find((d) => d.key === key && d.href === pathname)?.label
    ?? MENU_DESTINATIONS.find((d) => d.href === pathname)?.label ?? active;
  return <span className="truncate text-[13px] font-medium text-ink">{label}</span>;
}

export default function TopNav({ active, right, showCrumb = true }: {
  active: string; right?: React.ReactNode; showCrumb?: boolean;
}) {
  const router = useRouter();
  return (
    <header className="flex items-center gap-2 px-3 h-12 bg-bg border-b hairline shrink-0">
      {/* Below the tablet breakpoint the sidebar is a drawer; this opens it. */}
      <button onClick={() => window.dispatchEvent(new Event("lbx:sidebar"))}
        aria-label="Open navigation" className="side-icon-btn flex md:hidden">
        <Icon name="list" size={16} />
      </button>
      <button
        onClick={() => { if (typeof window !== "undefined" && window.history.length > 1) router.back(); else router.push("/"); }}
        data-tip="Back (Alt+Left)" aria-label="Back"
        className="side-icon-btn hidden md:flex">
        <Icon name="chevL" size={15} />
      </button>

      {showCrumb && <Crumb active={active} />}

      {/* right side: page actions, then session controls.
          The whole cluster used to be shrink-0 with {right} injected raw, so a page passing a long status
          pushed the session controls past the viewport instead of the status giving way: /training's
          "worker required: make train-worker" put the page at 1323px inside a 1280px window and the user
          chip off the edge. The page's own node is now the only part allowed to shrink, and it truncates. */}
      <div className="flex items-center gap-2 text-xs ml-auto min-w-0">
        {right && <div className="min-w-0 truncate hidden lg:block">{right}</div>}
        <div className="flex items-center gap-2 shrink-0">
        <button onClick={() => window.dispatchEvent(new Event("lbx:palette"))}
          data-tip="Jump to any page"
          className="hidden sm:flex items-center gap-2 h-8 w-56 lg:w-64 px-2.5 rounded-md bg-bg-2 border hairline text-[12px] text-ink-3 hover:border-[#383b42] hover:text-ink-2 transition-colors">
          <Icon name="search" size={13} />
          <span className="flex-1 text-left">Go to...</span>
          <kbd className="font-mono text-[10.5px] px-1.5 py-0.5 rounded border hairline text-ink-3">Ctrl K</kbd>
        </button>
        <button onClick={() => window.dispatchEvent(new Event("lbx:palette"))}
          aria-label="Jump to any page" className="side-icon-btn flex sm:hidden">
          <Icon name="search" size={15} />
        </button>
        {/* Before the cloud chip: an upload is something the user started and is waiting on, so it
            outranks a standing status light. Renders nothing when nothing is running. */}
        <UploadIndicator />
        <CloudControl />
        <NotificationBell />
        <UserPicker />
        </div>
      </div>

      <CommandPalette />
      <ShortcutOverlay />
    </header>
  );
}
