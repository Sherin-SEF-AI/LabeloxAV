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

/** What a poll of a background run has to tell this store. Anything else about the run is the run's business. */
export type RunSnapshot = {
  status: string;
  /** 0..1 when the run recorded a total. Null when it did not, which is a different statement from zero. */
  fraction?: number | null;
  detail?: string;
};

/** Statuses that mean the run is over, matching the set `JobWatcher` uses so the two cannot disagree. */
export const TERMINAL_RUN_STATUSES: ReadonlySet<string> =
  new Set(["committed", "reverted", "error", "interrupted", "skipped"]);

/**
 * Follow a background run started from the canvas, so it appears here rather than only in the console.
 *
 * An action taken in the editor that hands back a `run_id` used to vanish from the canvas the moment it
 * returned: the work carried on for minutes in the API process, and the surface that had just launched it
 * showed nothing at all. That is the same gap `trackOp` closes for in-flight requests, one level up.
 *
 * The poll is injected rather than imported so this module stays a store with no network of its own, which
 * is also what lets the loop be tested without a server.
 */
export async function trackRun(
  kind: string, label: string, runId: string,
  poll: (runId: string) => Promise<RunSnapshot>,
  { intervalMs = 2_500, maxMs = 6 * 60 * 60 * 1000 }: { intervalMs?: number; maxMs?: number } = {},
): Promise<RunSnapshot | null> {
  const id = beginOp(kind, label);
  const deadline = Date.now() + maxMs;
  let consecutiveErrors = 0;
  try {
    for (;;) {
      let snap: RunSnapshot;
      try {
        snap = await poll(runId);
        consecutiveErrors = 0;
      } catch (e) {
        // A restart or a blip must not report a running job as failed. Three in a row is the backend being
        // gone, which is worth saying; one is not.
        if (++consecutiveErrors < 3) {
          await new Promise((r) => setTimeout(r, intervalMs));
          continue;
        }
        endOp(id, "failed", e instanceof Error ? e.message : String(e));
        return null;
      }
      if (typeof snap.fraction === "number") updateOp(id, { progress: snap.fraction });
      if (snap.detail) updateOp(id, { detail: snap.detail });
      if (TERMINAL_RUN_STATUSES.has(snap.status)) {
        endOp(id, snap.status === "error" ? "failed" : "ok", snap.detail);
        return snap;
      }
      if (Date.now() > deadline) {
        // Giving up on watching is not the same as the run failing, and saying otherwise would be the lie
        // this console exists to remove. The run is still in the console's Background panel.
        endOp(id, "ok", "still running; follow it in the console");
        return snap;
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
  } catch (e) {
    endOp(id, "failed", e instanceof Error ? e.message : String(e));
    return null;
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
