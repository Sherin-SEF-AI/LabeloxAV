// One notification channel for the whole app. A module-level pub/sub (not a React context) so the plain
// api-client module can raise a toast on a failed request exactly the same way a component can, which is what
// lets error feedback be consistent everywhere instead of each page inventing its own note/flash/alert.

export type ToastKind = "info" | "success" | "warn" | "error";
export type Toast = { id: number; kind: ToastKind; message: string; ttl: number };

type Listener = (t: Toast) => void;

const listeners = new Set<Listener>();
let seq = 0;

export function subscribeToasts(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function toast(message: string, kind: ToastKind = "info", ttl = 5000): void {
  // errors linger a little longer than a success flash, since the user may need to read them
  const t: Toast = { id: ++seq, kind, message, ttl: kind === "error" ? Math.max(ttl, 7000) : ttl };
  listeners.forEach((fn) => fn(t));
}

export const toastSuccess = (m: string) => toast(m, "success");
export const toastError = (m: string) => toast(m, "error");
export const toastWarn = (m: string) => toast(m, "warn");
