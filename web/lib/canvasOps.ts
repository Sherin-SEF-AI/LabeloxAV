// What the canvas is doing, as a module-level store rather than a component's state.
//
// The editor runs real work on every interaction: SAM segmentation, mask composition, propagation across
// frames, drivable-surface inference, an auto-classify on each new box, an autosave behind all of it. All of
// it was fire-and-forget with a one-line `flash()` on the way out, so while a call was in flight the canvas
// looked idle, and when one was slow there was nothing to say whether it was slow, stuck, or never sent.
//
// Module scope for the same reason `toast.ts` and `uploadManager.ts` are: an operation started by the canvas
// has to remain visible to a console that is a sibling component, and it must survive that console being
// closed and reopened. React subscribes; the store owns.
//
// Finished operations are kept for a short while on purpose. "It succeeded and vanished" and "it never ran"
// look identical at the moment you look up, and the question people ask about a slow editor is what just
// happened, not what is happening.

export type OpStatus = "running" | "ok" | "failed";

export type CanvasOp = {
  id: number;
  /** Machine-readable, for grouping: sam, propagate, save, classify, mask, drivable. */
  kind: string;
  /** What a person would call it. */
  label: string;
  status: OpStatus;
  startedAt: number;
  endedAt?: number;
  /** One line of outcome: how many objects, which class, or why it failed. */
  detail?: string;
  /** 0..1 when the operation reports it. Most do not, and a made-up bar is worse than none. */
  progress?: number;
};

type Listener = (ops: CanvasOp[]) => void;

const listeners = new Set<Listener>();
let ops: CanvasOp[] = [];
let seq = 0;

// Long enough to answer "what just happened", short enough that the list is not a log.
const KEEP_FINISHED_MS = 20_000;
const MAX_OPS = 40;

function emit() {
  const snapshot = ops.slice();
  listeners.forEach((fn) => fn(snapshot));
}

function prune() {
  const cutoff = Date.now() - KEEP_FINISHED_MS;
  ops = ops.filter((o) => o.status === "running" || (o.endedAt ?? 0) > cutoff).slice(-MAX_OPS);
}

/** Current operations, newest last. */
export function getOps(): CanvasOp[] {
  prune();
  return ops.slice();
}

export function subscribeOps(fn: Listener): () => void {
  listeners.add(fn);
  fn(getOps());
  return () => { listeners.delete(fn); };
}

/** Start tracking an operation. Returns its id, which `endOp` needs. */
export function beginOp(kind: string, label: string): number {
  const id = ++seq;
  ops = [...ops, { id, kind, label, status: "running", startedAt: Date.now() }];
  prune();
  emit();
  return id;
}

/** Report progress for an operation that knows its own. */
export function updateOp(id: number, patch: Partial<Pick<CanvasOp, "progress" | "detail">>): void {
  ops = ops.map((o) => (o.id === id ? { ...o, ...patch } : o));
  emit();
}

/** Finish an operation. `detail` is the one line worth keeping: a count, a class, or a reason. */
export function endOp(id: number, status: Exclude<OpStatus, "running">, detail?: string): void {
  ops = ops.map((o) => (o.id === id ? { ...o, status, detail, endedAt: Date.now() } : o));
  prune();
  emit();
}

/**
 * Run something and track it, which is the form every call site actually wants.
 *
 * The failure path is the reason this exists as a wrapper: hand-written begin/end pairs leak a `running`
 * entry every time somebody adds an early return or forgets a try, and an operation stuck at running forever
 * is exactly the lie the console is meant to remove.
 */
export async function trackOp<T>(
  kind: string, label: string, run: (report: (p: Partial<CanvasOp>) => void) => Promise<T>,
  describe?: (result: T) => string,
): Promise<T> {
  const id = beginOp(kind, label);
  try {
    const result = await run((p) => updateOp(id, p));
    endOp(id, "ok", describe ? describe(result) : undefined);
    return result;
  } catch (e) {
    endOp(id, "failed", e instanceof Error ? e.message : String(e));
    throw e;
  }
}

/** For tests, and for leaving a frame: the next frame's canvas is not still doing the last one's work. */
export function resetOps(): void {
  ops = [];
  emit();
}

export type OpsSummary = { running: number; failed: number; label: string };

/**
 * The one line for the toggle button, so a closed console still says whether to open it.
 *
 * A failure outranks progress: something that went wrong and was never read is the state this whole surface
 * exists to prevent.
 */
export function summarizeOps(list: readonly CanvasOp[]): OpsSummary {
  const running = list.filter((o) => o.status === "running");
  const failed = list.filter((o) => o.status === "failed");
  let label = "idle";
  if (failed.length) label = `${failed.length} failed`;
  else if (running.length === 1) label = running[0].label;
  else if (running.length > 1) label = `${running.length} running`;
  return { running: running.length, failed: failed.length, label };
}
