"use client";

import { useCallback, useRef } from "react";

/**
 * A function whose identity never changes and which always runs the latest version of `fn`.
 *
 * The canvas is memoized, and a memoized component is only as still as its props. Most of what the frame
 * page hands it are handlers that read the selection, the current class or the mode, so a plain
 * `useCallback` would list those as dependencies and change identity whenever they do, which is most
 * renders, and the canvas would re-render for a flash message. This gives the canvas one function for the
 * life of the page and lets the function see current state through a ref.
 *
 * The ref is written during render, as `saveRef` and `stRef` on the page already are: a handler can only
 * be called after the render that defined it has committed.
 */
export function useStableCallback<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useRef(fn);
  ref.current = fn;
  return useCallback((...args: A) => ref.current(...args), []);
}
