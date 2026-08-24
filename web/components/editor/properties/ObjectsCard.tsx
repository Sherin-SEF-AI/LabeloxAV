"use client";

// The object list, and the single biggest change in this redesign: it is no longer behind a tab.
//
// The panel used to be two exclusive tabs, "objects" and "tools", so an annotator could see the object
// they were labelling or the agent that was labelling it, never both. This is a collapsible card that is
// always present, and the tools moved into their own sub-tabs underneath it.
//
// Group headers stick while their rows scroll. A dense junction frame runs to forty objects across six
// classes, and scrolling into the middle of one used to lose the only thing that said which class the
// rows belonged to.

import { useState, type Dispatch } from "react";

import { classColor } from "@/lib/colors";
import { ConfBar } from "@/components/StateBadge";

import { groupObjects } from "./objectGroups";
import { PREF_OBJECTS_OPEN, usePanelFlag } from "./panelPrefs";
import { lowConfCount, qualityTone } from "./panelStats";
import type { Action, EdObject } from "../useEditor";

const QUALITY_DOT = { good: "bg-pass", weak: "bg-warn", bad: "bg-block" } as const;

export default function ObjectsCard({ objects, selectedIds, dispatch, filters }: {
  objects: EdObject[];
  selectedIds: string[];
  dispatch: Dispatch<Action>;
  /** The selection chips, passed in so this file does not need to know what a SelectHow is. */
  filters?: React.ReactNode;
}) {
  const [open, setOpen] = usePanelFlag(PREF_OBJECTS_OPEN, true);
  const [query, setQuery] = useState("");
  // Not persisted: a group collapsed on a junction frame means nothing on the next frame, which may not
  // even contain that class, and a restored collapse hides objects with no visible cause.
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const groups = groupObjects(objects, query);
  const low = lowConfCount(objects);

  return (
    <section className="border-b hairline">
      <button onClick={() => setOpen(!open)} aria-expanded={open}
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-line/30 group transition-colors">
        <span aria-hidden
          className={`text-ink-3 group-hover:text-ink w-3 text-center text-[9px] leading-none transition-transform duration-200 ${open ? "rotate-90" : ""}`}>▸</span>
        <span className="font-mono text-[10px] uppercase tracking-wide text-ink-2">objects</span>
        <span className="font-mono text-[10px] text-ink-3 bg-line/50 rounded px-1.5">{objects.length}</span>
        {low > 0 && (
          <span className="ml-auto font-mono text-[10px] text-warn"
            title="objects the model was unsure about, below the 0.5 threshold the selection chip uses">
            {low} low conf
          </span>
        )}
      </button>

      {open && (
        <div className="reveal">
          <div className="px-2 pb-1.5">
            <input value={query} onChange={(e) => setQuery(e.target.value)}
              placeholder="search objects..." aria-label="search objects"
              className="w-full bg-bg-2 border border-line rounded px-1.5 py-1 font-mono text-[11px] text-ink placeholder:text-ink-3/70 focus:border-accent outline-none" />
          </div>

          {!groups.length ? (
            <div className="text-ink-3 text-center py-4 font-mono text-[11px]">
              {objects.length ? "no object matches that." : "no objects. draw a box (B)."}
            </div>
          ) : (
            // Capped so the tool tabs below stay on screen without scrolling. A junction frame runs to
            // forty objects and an uncapped list pushes the agent off the bottom of a 340px rail, which is
            // the exact problem the tab split was supposed to solve.
            <div className="max-h-[30vh] overflow-y-auto">
              {groups.map((g) => {
                const shut = collapsed.has(g.name);
                return (
                  <div key={g.name}>
                    <button
                      onClick={() => setCollapsed((s) => {
                        const n = new Set(s);
                        if (n.has(g.name)) n.delete(g.name); else n.add(g.name);
                        return n;
                      })}
                      aria-expanded={!shut}
                      className="sticky top-0 z-10 flex items-center gap-1.5 w-full bg-head/95 backdrop-blur-sm border-y border-line-2 px-2 py-1 font-mono text-[10px] text-ink-3 hover:text-ink-2">
                      <span className="w-2.5 h-2.5 inline-block shrink-0 rounded-sm"
                        style={{ background: classColor(g.classId) }} />
                      <span className="flex-1 text-left truncate uppercase tracking-wide">{g.name}</span>
                      <span className="tabular-nums">{g.objects.length}</span>
                      <span className="w-3 text-right" aria-hidden>{shut ? "+" : "−"}</span>
                    </button>

                    {!shut && g.objects.map((o) => (
                      <div key={o.id}
                        onClick={(e) => {
                          // Ctrl/Cmd or Shift extends the selection, matching every other list in every
                          // other tool; a plain click still selects exactly one.
                          if (e.ctrlKey || e.metaKey || e.shiftKey) dispatch({ t: "toggleSelect", id: o.id });
                          else dispatch({ t: "select", id: o.id });
                        }}
                        className={`flex items-center gap-1.5 pl-3 pr-1.5 py-0.5 cursor-pointer font-mono text-[11px] border-l-2 ${
                          selectedIds.includes(o.id)
                            ? "bg-line text-ink border-accent"
                            : "text-ink-3 hover:text-ink-2 hover:bg-line/25 border-transparent"}`}>
                        <button title={o.visible ? "hide" : "show"}
                          aria-label={o.visible ? "hide object" : "show object"}
                          onClick={(e) => { e.stopPropagation(); dispatch({ t: "setVisible", ids: [o.id], visible: !o.visible }); }}
                          className={o.visible ? "text-ink-2" : "text-ink-3"}>{o.visible ? "●" : "○"}</button>
                        <button title={o.locked ? "unlock" : "lock"}
                          aria-label={o.locked ? "unlock object" : "lock object"}
                          onClick={(e) => { e.stopPropagation(); dispatch({ t: "setLocked", ids: [o.id], locked: !o.locked }); }}
                          className={o.locked ? "text-warn" : "text-ink-3 hover:text-ink-2"}>{o.locked ? "L" : "l"}</button>
                        <span className="truncate flex-1">
                          {o.id.startsWith("tmp-") ? "new" : o.id.slice(0, 8)}{o.isNew ? " *" : ""}
                        </span>
                        {o.quality_score != null && (
                          <span title={`label quality ${o.quality_score.toFixed(2)}`}
                            className={`w-1.5 h-1.5 rounded-full shrink-0 ${QUALITY_DOT[qualityTone(o.quality_score)]}`} />
                        )}
                        <ConfBar conf={o.conf} />
                        {o.mask.length > 0 && <span className="text-info" title="has mask">&#9670;</span>}
                        <button onClick={(e) => { e.stopPropagation(); dispatch({ t: "delete", id: o.id }); }}
                          disabled={o.locked} aria-label="delete object"
                          className={o.locked ? "text-line cursor-not-allowed" : "text-ink-3 hover:text-block"}>x</button>
                      </div>
                    ))}
                  </div>
                );
              })}
            </div>
          )}

          {filters}
        </div>
      )}
    </section>
  );
}
