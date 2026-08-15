"use client";

import { usePathname, useRouter } from "next/navigation";
import UserPicker from "./UserPicker";
import PlatformSwitcher from "./shell/PlatformSwitcher";
import CommandPalette from "./shell/CommandPalette";
import ShortcutOverlay from "./shell/ShortcutOverlay";
import CloudControl from "./shell/CloudControl";
import UploadIndicator from "./shell/UploadIndicator";
import MenuBar from "./shell/MenuBar";
import Icon from "./shell/Icon";
import NotificationBell from "./shell/NotificationBell";
import { MENU_DESTINATIONS } from "@/lib/menus";

// The application header, Blender-style: a menu bar carrying every destination, then a quiet breadcrumb, then
// the session controls on the right.
//
// A menu bar rather than a row of buttons because the header must not grow with the app. Buttons compete for
// horizontal space and eventually force a scroll or an overflow chevron; menus stay one row at any number of
// destinations, which is the same reason the editor's tool strip groups its tools into flyouts.
//
// The {active, right} interface is unchanged, so every page keeps working.

function Crumb({ active }: { active: string }) {
  const pathname = usePathname();
  // Prefer the real destination label for the current path, so the crumb matches what the menu called it
  // rather than whatever string each page happened to pass.
  const match = MENU_DESTINATIONS.find((d) => d.href === pathname);
  const label = match?.label ?? active;
  return (
    <span className="flex items-center gap-1.5 min-w-0 pl-1 text-[11.5px] text-ink-3">
      <span className="opacity-50">/</span>
      <span className="truncate text-ink-2">{label}</span>
    </span>
  );
}

export default function TopNav({ active, right, showCrumb = true }: {
  active: string; right?: React.ReactNode; showCrumb?: boolean;
}) {
  const router = useRouter();
  return (
    <header className="flex items-center gap-2 pr-3 h-11 bg-head border-b border-[#262626] shrink-0">
      {/* brand + back */}
      <div className="flex items-center gap-1 pl-2 shrink-0">
        <button
          onClick={() => { if (typeof window !== "undefined" && window.history.length > 1) router.back(); else router.push("/"); }}
          data-tip="Back (Alt+Left)"
          className="w-6 h-6 flex items-center justify-center rounded text-ink-3 hover:text-ink hover:bg-panel"
        >
          <Icon name="chevL" size={14} />
        </button>
        <button onClick={() => router.push("/platforms")}
          data-tip="Platform launcher: pick a plane of the data engine"
          className="font-display font-bold px-1 text-[13px] text-ink">
          Labelox<span className="text-accent-2">AV</span>
        </button>
      </div>

      {/* The menu bar: every destination lives here, so the header never grows. Hidden below the tablet
          breakpoint, where nine top-level menus would wrap the header onto three lines; the command
          palette beside it reaches every one of the same destinations by name. */}
      <div className="hidden md:contents"><MenuBar /></div>

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
          data-tip="Command palette: jump to any page"
          className="btn text-[11px] gap-1.5 h-7">
          <Icon name="search" size={12} />
          <kbd className="text-ink-3 hidden sm:inline">Cmd K</kbd>
        </button>
        {/* Before the cloud chip: an upload is something the user started and is waiting on, so it
            outranks a standing status light. Renders nothing when nothing is running. */}
        <UploadIndicator />
        <PlatformSwitcher />
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
