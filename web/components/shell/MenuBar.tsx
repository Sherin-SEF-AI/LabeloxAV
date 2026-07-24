"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import Icon from "./Icon";
import { MENUS, type MenuItem } from "@/lib/menus";

// The application menu bar, Blender-style.
//
// The behaviour that makes a menu bar feel native, and that a naive dropdown gets wrong:
//
//  - Menu tracking. Once one menu is open, HOVERING another title switches to it without a second click.
//    Without this you have to click, read, close, click again, which is why ad-hoc dropdowns feel clumsy.
//  - Submenus open on hover and close on a timer, so travelling diagonally toward a submenu does not
//    dismiss it the instant the pointer leaves the parent row.
//  - Full keyboard control: Escape closes, arrows move, Enter activates, Left/Right walk submenus.
//  - Disabled items are shown and greyed rather than hidden, so the menu is a stable map of what exists.

const CLOSE_DELAY_MS = 180;

export default function MenuBar() {
  const router = useRouter();
  const [open, setOpen] = useState<string | null>(null);
  const [sub, setSub] = useState<string | null>(null);
  const [cursor, setCursor] = useState(0);
  const barRef = useRef<HTMLDivElement | null>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isEnabled = useCallback((it: MenuItem) => !it.disabled, []);

  const close = useCallback(() => { setOpen(null); setSub(null); setCursor(0); }, []);

  // Click outside and Escape both close, the two things every user tries.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (barRef.current && !barRef.current.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") { e.preventDefault(); close(); } };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open, close]);

  useEffect(() => () => { if (closeTimer.current) clearTimeout(closeTimer.current); }, []);

  const run = useCallback((it: MenuItem) => {
    if (!isEnabled(it) || it.items) return;
    close();
    if (it.href) router.push(it.href);
    else if (it.event) window.dispatchEvent(new Event(it.event));
  }, [isEnabled, close, router]);

  const menu = MENUS.find((m) => m.key === open) ?? null;

  // Keyboard navigation within the open menu.
  useEffect(() => {
    if (!menu) return;
    const onKey = (e: KeyboardEvent) => {
      const items = menu.items;
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        const dir = e.key === "ArrowDown" ? 1 : -1;
        // skip past anything not selectable so the cursor never lands on a dead row
        let i = cursor;
        for (let n = 0; n < items.length; n++) {
          i = (i + dir + items.length) % items.length;
          if (isEnabled(items[i])) break;
        }
        setCursor(i);
      } else if (e.key === "ArrowRight") {
        const it = items[cursor];
        if (it?.items) { e.preventDefault(); setSub(it.key); }
        else {
          e.preventDefault();
          const idx = MENUS.findIndex((m) => m.key === menu.key);
          setOpen(MENUS[(idx + 1) % MENUS.length].key); setCursor(0); setSub(null);
        }
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        if (sub) setSub(null);
        else {
          const idx = MENUS.findIndex((m) => m.key === menu.key);
          setOpen(MENUS[(idx - 1 + MENUS.length) % MENUS.length].key); setCursor(0);
        }
      } else if (e.key === "Enter") {
        e.preventDefault();
        const it = items[cursor];
        if (it?.items) setSub(it.key); else if (it) run(it);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [menu, cursor, sub, isEnabled, run]);

  const scheduleClose = () => {
    if (closeTimer.current) clearTimeout(closeTimer.current);
    closeTimer.current = setTimeout(() => setSub(null), CLOSE_DELAY_MS);
  };
  const cancelClose = () => { if (closeTimer.current) clearTimeout(closeTimer.current); };

  const Row = ({ it, i }: { it: MenuItem; i: number }) => {
    const enabled = isEnabled(it);
    const active = i === cursor;
    return (
      <div key={it.key} className="relative"
        onMouseEnter={() => { setCursor(i); cancelClose(); if (it.items) setSub(it.key); else scheduleClose(); }}>
        {it.separatorBefore && <div className="my-1 border-t border-line/70" />}
        <button
          disabled={!enabled}
          onClick={() => run(it)}
          title={it.hint}
          className={`w-full flex items-center gap-2 pl-2 pr-2 py-[5px] text-left text-[12.5px] rounded-[3px]
            ${!enabled ? "text-ink-3/45 cursor-default"
              : active ? "bg-accent text-bg" : "text-ink-2 hover:bg-accent hover:text-bg"}`}
        >
          <span className="w-4 shrink-0 flex items-center justify-center opacity-80">
            {it.icon ? <Icon name={it.icon} size={13} /> : null}
          </span>
          <span className="flex-1 truncate">{it.label}</span>
          {it.shortcut && (
            <span className={`font-mono text-[10.5px] shrink-0 ${active && enabled ? "text-bg/70" : "text-ink-3"}`}>
              {it.shortcut}
            </span>
          )}
          {it.items && <span className="shrink-0 text-[10px] leading-none opacity-70">&#9656;</span>}
        </button>

        {/* submenu */}
        {it.items && sub === it.key && (
          <div
            onMouseEnter={cancelClose}
            onMouseLeave={scheduleClose}
            className="absolute left-full top-0 -ml-1 min-w-[190px] panel py-1 z-[60] shadow-xl border border-line"
          >
            {it.items.map((s) => (
              <button key={s.key} onClick={() => run(s)} title={s.hint}
                className="w-full flex items-center gap-2 pl-2 pr-3 py-[5px] text-left text-[12.5px]
                  text-ink-2 hover:bg-accent hover:text-bg rounded-[3px]">
                <span className="w-4 shrink-0" />
                <span className="flex-1 truncate">{s.label}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div ref={barRef} className="flex items-center h-full">
      {MENUS.map((m) => (
        <div key={m.key} className="relative h-full">
          <button
            onClick={() => { setOpen(open === m.key ? null : m.key); setCursor(0); setSub(null); }}
            // Menu tracking: with one menu open, hovering a sibling title switches to it.
            onMouseEnter={() => { if (open && open !== m.key) { setOpen(m.key); setCursor(0); setSub(null); } }}
            className={`h-full px-2.5 text-[12.5px] font-display ${
              open === m.key ? "bg-panel text-ink" : "text-ink-2 hover:text-ink"}`}
          >
            {m.label}
          </button>

          {open === m.key && (
            <div className="absolute left-0 top-full mt-px min-w-[232px] panel py-1 z-[60] shadow-xl border border-line">
              {m.items.map((it, i) => <Row key={it.key} it={it} i={i} />)}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
