"use client";

// Ways to pick a set that are not "drag a box round them". A dense frame holds forty vehicles and the
// useful selections are almost never contiguous: every autorickshaw, everything the model was unsure
// about, everything nobody has looked at yet.
//
// The chips sit directly above the bulk actions they feed. The note below them is not decoration: a bulk
// action skipping a locked object is otherwise a silent no-op, and this is the only place the panel says
// why.

import type { Dispatch } from "react";

import { LOW_CONF } from "./panelStats";
import type { Action, SelectHow } from "../useEditor";

// `lowConf` passes LOW_CONF rather than a second 0.5 literal, so this chip and the "N low conf" figure in
// the panel header can never select different sets while claiming to mean the same thing.
export const SELECTIONS: {
  how: SelectHow; label: string; value?: string | number; hint: string; key?: string;
}[] = [
  { how: "all", label: "All", hint: "every visible, unlocked object", key: "⌘A" },
  { how: "none", label: "None", hint: "clear the selection", key: "Esc" },
  { how: "invert", label: "Invert", hint: "everything not currently selected", key: "⌘I" },
  { how: "sameClass", label: "Same class", hint: "everything of the selected object's class", key: "⌘⇧A" },
  { how: "unreviewed", label: "Unreviewed", hint: "still in review: the queue you are working" },
  { how: "new", label: "New", hint: "drawn here and not yet saved" },
  { how: "lowConf", label: `Conf < ${LOW_CONF}`, value: LOW_CONF, hint: "the model was unsure about these" },
  { how: "state", label: "Rejected", value: "rejected", hint: "already rejected" },
];

export default function SelectionFilters({ dispatch }: { dispatch: Dispatch<Action> }) {
  return (
    <div className="px-2 py-1.5 border-t hairline">
      <div className="flex flex-wrap items-center gap-1">
        <span className="text-[11px] text-ink-3 mr-0.5">Select</span>
        {SELECTIONS.map((sel) => (
          <button
            key={sel.how + String(sel.value ?? "")}
            onClick={() => dispatch({ t: "selectBy", how: sel.how, value: sel.value })}
            title={sel.hint + (sel.key ? ` (${sel.key})` : "")}
            className="border border-line text-ink-3 px-1.5 py-0.5 rounded hover:border-accent hover:text-ink-2 text-[11px]"
          >
            {sel.label}
          </button>
        ))}
      </div>
      <p className="text-[11px] text-ink-3 mt-1.5 leading-snug">
        Locked and hidden objects are never picked: a bulk action must not reach the thing somebody
        locked to protect it.
      </p>
    </div>
  );
}
