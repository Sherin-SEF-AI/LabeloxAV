"use client";

// The right-click menu on the annotation canvas.
//
// Every action it offers already existed and lived somewhere else: the properties rail, the tool strip, a
// keyboard shortcut, or a page you had to navigate to. That is fine when you are setting up and wrong when
// you are working, because the thing you want to act on is under the cursor and the control for it is four
// hundred pixels away in a panel that may be collapsed.
//
// The behaviours that make a context menu feel native, from the same list components/shell/MenuBar.tsx
// records for the app menu: Escape closes, arrows walk, Enter fires, click-outside dismisses, and disabled
// rows are shown and greyed rather than hidden so the menu is a stable map of what exists.
//
// Escape is handled in the CAPTURE phase with propagation stopped, for the reason
// components/editor/properties/Popover.tsx documents: the editor binds its keymap on window, and its
// Escape branch clears the annotator's selection. A menu that closes and also wipes a twelve-object
// selection is worse than no menu.

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

import { classColor } from "@/lib/colors";
import { canvasMenu, enabledKeys, menuPosition, type CanvasTarget } from "@/lib/canvasMenu";
import type { OntologyClass } from "@/lib/types";

export type CanvasMenuProps = {
  at: { x: number; y: number };
  target: CanvasTarget;
  classes: OntologyClass[];
  onClose: () => void;
  /** Fired with the item key, and the class when the class submenu was used. */
  onAction: (key: string, cls?: OntologyClass) => void;
};

const WIDTH = 232;

export default function CanvasMenu({ at, target, classes, onClose, onAction }: CanvasMenuProps) {
  const boxRef = useRef<HTMLDivElement | null>(null);
  const [pos, setPos] = useState<{ left: number; top: number }>({ left: at.x, top: at.y });
  const [sub, setSub] = useState(false);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);

  const sections = canvasMenu(target);
  const keys = enabledKeys(sections);

  // Measured after paint, because the height depends on which menu this is and a guess would flip the
  // menu upward on a short one and off the bottom on a long one.
  useLayoutEffect(() => {
    const h = boxRef.current?.offsetHeight ?? 280;
    setPos(menuPosition(at, { w: WIDTH, h }, { w: window.innerWidth, h: window.innerHeight }));
  }, [at, sub]);

  const fire = useCallback((key: string) => {
    const it = sections.flatMap((s) => s.items).find((i) => i.key === key);
    if (!it || it.disabled) return;
    if (it.submenu === "class") { setSub(true); setQuery(""); return; }
    onAction(key);
    onClose();
  }, [sections, onAction, onClose]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!boxRef.current?.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault(); e.stopPropagation();
        if (sub) setSub(false); else onClose();
        return;
      }
      if (sub) return;   // the submenu's own input owns the keyboard while it is open
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault(); e.stopPropagation();
        setCursor((c) => {
          const n = keys.length;
          if (!n) return 0;
          return e.key === "ArrowDown" ? (c + 1) % n : (c - 1 + n) % n;
        });
      } else if (e.key === "Enter") {
        e.preventDefault(); e.stopPropagation();
        if (keys[cursor]) fire(keys[cursor]);
      }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey, true);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [onClose, sub, keys, cursor, fire]);

  const norm = query.trim().toLowerCase().replace(/[\s-]+/g, "_");
  const shown = classes.filter((c) => c.name.includes(norm)).slice(0, 40);

  return (
    <div ref={boxRef} role="menu" aria-label="canvas actions"
      style={{ position: "fixed", left: pos.left, top: pos.top, width: WIDTH }}
      className="z-[70] panel border border-line rounded shadow-xl py-1 select-none">
      {sub ? (
        <>
          <div className="px-2 pb-1 pt-0.5 border-b hairline">
            <input autoFocus value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="class..." aria-label="choose a class"
              onKeyDown={(e) => {
                if (e.key !== "Enter") return;
                const exact = classes.find((c) => c.name === norm) ?? shown[0];
                if (exact) { onAction("class", exact); onClose(); }
              }}
              className="w-full bg-bg-2 border border-line rounded px-1.5 py-1 font-mono text-[11px] text-ink placeholder:text-ink-3/70 focus:border-accent outline-none" />
          </div>
          <div className="max-h-64 overflow-auto">
            {shown.map((c) => (
              <button key={c.id} role="menuitem"
                onClick={() => { onAction("class", c); onClose(); }}
                className="w-full flex items-center gap-1.5 px-2 py-1 text-left font-mono text-[11px] text-ink-2 hover:bg-line/50 hover:text-ink">
                <span className="w-2.5 h-2.5 shrink-0 rounded-sm" style={{ background: classColor(c.id) }} />
                <span className="truncate">{c.name}</span>
                {c.india && <span className="text-accent shrink-0">*</span>}
              </button>
            ))}
            {!shown.length && <div className="px-2 py-2 font-mono text-[11px] text-ink-3">no class matches that.</div>}
          </div>
        </>
      ) : (
        sections.map((sec, si) => (
          <div key={sec.key} className={si ? "border-t hairline pt-1 mt-1" : ""}>
            {sec.items.map((it) => {
              const idx = keys.indexOf(it.key);
              const active = !it.disabled && idx === cursor;
              return (
                <button key={it.key} role="menuitem" disabled={it.disabled}
                  title={it.why}
                  onMouseEnter={() => { if (!it.disabled) setCursor(idx); }}
                  onClick={() => fire(it.key)}
                  className={`w-full flex items-center gap-2 px-2 py-1 text-left font-mono text-[11px]
                    ${it.disabled ? "text-ink-3/50 cursor-not-allowed"
                      : it.danger ? "text-block hover:bg-block/15"
                      : "text-ink-2 hover:bg-line/50 hover:text-ink"}
                    ${active ? (it.danger ? "bg-block/15" : "bg-line/50 text-ink") : ""}`}>
                  <span className="truncate flex-1">{it.label}</span>
                  {it.submenu && <span className="text-ink-3 shrink-0" aria-hidden>›</span>}
                  {it.hint && !it.submenu && (
                    <span className="text-ink-3/70 shrink-0 text-[10px]">{it.hint}</span>
                  )}
                </button>
              );
            })}
          </div>
        ))
      )}
    </div>
  );
}
