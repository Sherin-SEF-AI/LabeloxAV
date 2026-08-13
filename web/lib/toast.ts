// One notification channel for the whole app. A module-level pub/sub (not a React context) so the plain
// api-client module can raise a toast on a failed request exactly the same way a component can, which is what
// lets error feedback be consistent everywhere instead of each page inventing its own note/flash/alert.

export type ToastKind = "info" | "success" | "warn" | "error";
/**
 * `action` is what makes a bulk operation safe to attempt. A batch that changed fifty objects and offers no
 * way back asks the operator to be certain before acting; one that can be undone from the confirmation
 * itself asks them only to look. Optional, so every existing caller is unchanged.
 */
export type ToastAction = { label: string; run: () => void | Promise<void> };
export type Toast = { id: number; kind: ToastKind; message: string; ttl: number; action?: ToastAction };

type Listener = (t: Toast) => void;

const listeners = new Set<Listener>();
let seq = 0;

export function subscribeToasts(fn: Listener): () => void {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function toast(message: string, kind: ToastKind = "info", ttl = 5000, action?: ToastAction): void {
  // errors linger a little longer than a success flash, since the user may need to read them. A toast
  // carrying an action lingers longer still: it is the only offer of that action, and five seconds is not
  // long enough to read a sentence and decide to undo something.
  const base = kind === "error" ? Math.max(ttl, 7000) : ttl;
  const t: Toast = { id: ++seq, kind, message, ttl: action ? Math.max(base, 12000) : base, action };
  listeners.forEach((fn) => fn(t));
}

export const toastSuccess = (m: string) => toast(m, "success");
export const toastError = (m: string) => toast(m, "error");
export const toastWarn = (m: string) => toast(m, "warn");
