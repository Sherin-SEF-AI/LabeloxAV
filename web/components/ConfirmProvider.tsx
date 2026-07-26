"use client";

import { createContext, useCallback, useContext, useState } from "react";

// A promise-based confirm dialog for destructive actions. Delete a webhook, remove a cuboid, discard a
// storage source: these were one un-guarded click with no undo. useConfirm() returns an async function that
// resolves true only when the user explicitly confirms, so a caller writes:
//     if (await confirm({ title: "Remove webhook?", danger: true })) await api.deleteWebhook(id);
// The provider is mounted once at the root; the browser-native confirm() it replaces blocked the thread and
// looked unpolished, and could not be styled or made keyboard-consistent.

type ConfirmOpts = { title: string; body?: string; confirmLabel?: string; cancelLabel?: string; danger?: boolean };
type Resolver = (ok: boolean) => void;

const Ctx = createContext<(o: ConfirmOpts) => Promise<boolean>>(async () => false);

export function useConfirm() {
  return useContext(Ctx);
}

export default function ConfirmProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<{ opts: ConfirmOpts; resolve: Resolver } | null>(null);

  const confirm = useCallback((opts: ConfirmOpts) => {
    return new Promise<boolean>((resolve) => setState({ opts, resolve }));
  }, []);

  const close = (ok: boolean) => {
    state?.resolve(ok);
    setState(null);
  };

  return (
    <Ctx.Provider value={confirm}>
      {children}
      {state && (
        <div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/50"
          role="dialog" aria-modal="true" aria-label={state.opts.title}
          onKeyDown={(e) => { if (e.key === "Escape") close(false); if (e.key === "Enter") close(true); }}>
          <div className="bg-bg-2 border border-line shadow-2xl w-[min(28rem,92vw)] p-4 font-mono">
            <div className="text-sm text-ink mb-1">{state.opts.title}</div>
            {state.opts.body && <div className="text-[12px] text-ink-3 mb-3">{state.opts.body}</div>}
            <div className="flex justify-end gap-2 mt-3 text-[12px]">
              <button onClick={() => close(false)} autoFocus
                className="border border-line text-ink-2 px-3 py-1 hover:text-ink">
                {state.opts.cancelLabel ?? "Cancel"}
              </button>
              <button onClick={() => close(true)}
                className={`border px-3 py-1 ${state.opts.danger
                  ? "border-block text-block hover:bg-block/10" : "border-accent text-accent hover:bg-accent/10"}`}>
                {state.opts.confirmLabel ?? "Confirm"}
              </button>
            </div>
          </div>
        </div>
      )}
    </Ctx.Provider>
  );
}
