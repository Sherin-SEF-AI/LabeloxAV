// A tiny inflight-request tracker so a single global top bar can show whenever the app is talking to the
// backend, without every page wiring its own loading state. api.ts increments/decrements around each fetch.

type Listener = (active: number) => void;

let active = 0;
const listeners = new Set<Listener>();

function emit() {
  for (const l of listeners) l(active);
}

export function begin(): void {
  active += 1;
  emit();
}

export function end(): void {
  active = Math.max(0, active - 1);
  emit();
}

export function subscribe(l: Listener): () => void {
  listeners.add(l);
  l(active);
  return () => {
    listeners.delete(l);
  };
}
