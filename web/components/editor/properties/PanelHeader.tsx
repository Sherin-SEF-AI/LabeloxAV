"use client";

// The panel's fixed head: how much of the frame is done, and what you are painting with.
//
// The class picker used to be a permanently expanded block here, roughly 180px of search box and scrolling
// list that an annotator working one class for an hour never touched. What they did read was the one line
// naming the current class, so that line stays and the list moves behind "change".

import { useRef, useState } from "react";

import { classColor } from "@/lib/colors";
import type { OntologyClass } from "@/lib/types";
import Icon from "@/components/shell/Icon";

import ClassPopover from "./ClassPopover";
import ReviewProgress from "./ReviewProgress";
import { reviewCounts } from "./panelStats";
import type { EdObject } from "../useEditor";

export default function PanelHeader({
  objects, dirty, selectedName, currentClass, classes, onPickClass, onAddClass, onCollapse,
}: {
  objects: EdObject[];
  dirty: boolean;
  /** The selected object's class, which the title used to show in place of "Properties". */
  selectedName: string | null;
  currentClass: OntologyClass | null;
  classes: OntologyClass[];
  onPickClass: (c: OntologyClass) => void;
  onAddClass: (raw: string) => void;
  onCollapse: () => void;
}) {
  const anchorRef = useRef<HTMLButtonElement | null>(null);
  const [open, setOpen] = useState(false);
  const counts = reviewCounts(objects);

  return (
    <div className="shrink-0 border-b hairline bg-head/40 px-3 py-2.5 flex flex-col gap-2.5">
      <div className="flex items-center gap-2">
        <span className="font-display font-semibold text-[12.5px] text-ink truncate">
          {selectedName ?? "Properties"}
        </span>
        <span className="font-mono text-[10px] text-ink-3 bg-line/50 rounded px-1.5 shrink-0">{objects.length}</span>
        <span className="ml-auto flex items-center gap-1.5 font-mono text-[10px] text-ink-3 shrink-0">
          <span className={`w-1.5 h-1.5 rounded-full ${dirty ? "bg-warn" : "bg-pass"}`} />
          {dirty ? "unsaved" : "saved"}
        </span>
        <button onClick={onCollapse} title="collapse panel" aria-label="collapse panel"
          className="w-6 h-6 shrink-0 flex items-center justify-center rounded text-ink-3 hover:bg-line/50 hover:text-ink">
          <Icon name="chevR" size={14} />
        </button>
      </div>

      <ReviewProgress counts={counts} />

      {/* `relative` is what the popover anchors to. It is here in the header rather than in the scroll
          body below, because that body is overflow-y-auto and would clip it. */}
      <div className="relative flex items-center gap-2 bg-bg-2 border border-line rounded px-2 py-1.5">
        <span className="w-4 h-4 shrink-0 rounded-sm border border-line/60"
          style={{ background: currentClass ? classColor(currentClass.id) : "#333" }} />
        <div className="flex flex-col min-w-0">
          <span className="font-mono text-[9px] uppercase tracking-wider text-ink-3">painting as</span>
          <span className="font-mono text-[11.5px] text-ink truncate">{currentClass?.name ?? "-"}</span>
        </div>
        <button ref={anchorRef} onClick={() => setOpen((o) => !o)}
          aria-haspopup="dialog" aria-expanded={open}
          title="pick the class for new objects and for the selection (1-9 also work)"
          className="ml-auto shrink-0 font-mono text-[10px] border border-line rounded px-2 py-1 text-ink-2 hover:border-accent hover:text-accent-2">
          change
        </button>
        <ClassPopover anchorRef={anchorRef} open={open} onClose={() => setOpen(false)}
          classes={classes} currentId={currentClass?.id ?? null}
          onPick={onPickClass} onAdd={onAddClass} />
      </div>
    </div>
  );
}
