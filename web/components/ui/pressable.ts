import type { KeyboardEvent } from "react";

// A non-button element that acts like one: reachable by Tab, announced as a button, activated by Enter or
// Space as well as by a click.
//
// A real <button> is better and is what new code should use. This is for the rows and tiles that are a
// flex or grid container already, where swapping the element would change the layout: spreading these
// props gives them the keyboard half of the contract that their onClick alone did not.
//
//   <div {...pressable(() => select(id))} aria-pressed={selected}> ... </div>
//
// Not for an element with another control inside it: a button within a button has no defined behaviour.
// Split those into a row with two real buttons instead.
export function pressable(onPress: () => void, opts: { disabled?: boolean } = {}) {
  return {
    role: "button" as const,
    tabIndex: opts.disabled ? -1 : 0,
    "aria-disabled": opts.disabled || undefined,
    onClick: () => { if (!opts.disabled) onPress(); },
    onKeyDown: (e: KeyboardEvent) => {
      if (opts.disabled || e.target !== e.currentTarget) return;
      if (e.key === "Enter" || e.key === " ") {
        // Space would otherwise scroll the page.
        e.preventDefault();
        onPress();
      }
    },
  };
}
