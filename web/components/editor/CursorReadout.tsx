"use client";

import { useSyncExternalStore } from "react";

import { getCursor, getCursorServerSnapshot, subscribeCursor } from "@/lib/editor/cursorStore";

// The pixel coordinate under the pointer, as its own leaf.
//
// Split out so that the one piece of the editor which genuinely has to change on every pointer move is the
// only piece that does. Previously this text lived inline in the frame metadata bar and was fed by page
// state, so each mouse move re-rendered the editor page and the whole canvas scene graph to update two
// numbers.
export default function CursorReadout() {
  const xy = useSyncExternalStore(subscribeCursor, getCursor, getCursorServerSnapshot);
  if (!xy || xy.length < 2) return null;
  return <>{`  ·  ${Math.round(xy[0])}, ${Math.round(xy[1])}`}</>;
}
