// The pointer position, kept out of React state on purpose.
//
// It used to be `useState` on the frame editor page, set from the canvas on every pointer move. The page is
// large and the canvas below it is not memoized, so moving the mouse across the image re-rendered the whole
// editor and rebuilt the entire Konva element tree at pointer rate. The value it did all that for is read in
// exactly one place: a coordinate readout in the frame metadata bar.
//
// A store with an explicit subscription lets that one readout re-render on its own while nothing else
// listens. `useSyncExternalStore` is the supported way to read it, so the readout stays correct under
// concurrent rendering rather than relying on a mutable ref that React cannot see change.

type Listener = () => void;

let current: readonly number[] | null = null;
const listeners = new Set<Listener>();

export function setCursor(xy: readonly number[] | null): void {
  // Integer compare before notifying. A pointer move within the same image pixel is not a change anybody can
  // see, and at 3 to 5 events per pixel of travel this drops most notifications for free.
  if (current === xy) return;
  if (current && xy && Math.round(current[0]) === Math.round(xy[0]) && Math.round(current[1]) === Math.round(xy[1])) return;
  current = xy;
  for (const fn of listeners) fn();
}

export function getCursor(): readonly number[] | null {
  return current;
}

export function subscribeCursor(fn: Listener): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

// The server render has no pointer. Returning a stable null keeps useSyncExternalStore from looping on a
// fresh value every call.
export function getCursorServerSnapshot(): readonly number[] | null {
  return null;
}
