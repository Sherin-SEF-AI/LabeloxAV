"use client";

// A boundary around the annotation canvas, and nothing else.
//
// The canvas is the one part of the editor that runs somebody else's renderer over data the editor did not
// validate: a polygon with a NaN in it, an image the decoder rejects, a zero-sized buffer. Any of those
// thrown during render unmounted the whole page, and the page is where the unsaved work lives. One bad
// object on a frame cost the annotator every edit they had made to the other forty.
//
// The editor's state is in the page's reducer, above this boundary, so it is untouched by a crash here.
// Retry remounts the canvas over that same state. If the same object crashes it again the panel comes
// back, and the object list, undo and save are all still there to deal with it.

import { Component, Fragment, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null; attempt: number };

export default class CanvasErrorBoundary extends Component<Props, State> {
  state: State = { error: null, attempt: 0 };

  static getDerivedStateFromError(error: unknown): Partial<State> {
    return { error: error instanceof Error ? error : new Error(String(error)) };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // The stack is the only record of which shape did it, and the panel has no room for one.
    console.error("editor canvas crashed", error, info.componentStack);
  }

  // A new key, so the canvas mounts fresh rather than resuming the tree that threw.
  retry = (): void => this.setState((s) => ({ error: null, attempt: s.attempt + 1 }));

  render(): ReactNode {
    const { error, attempt } = this.state;
    if (!error) return <Fragment key={attempt}>{this.props.children}</Fragment>;
    return (
      <div role="alert" className="w-full h-full bg-bg-2 flex items-center justify-center p-4">
        <div className="panel px-5 py-4 text-center space-y-3 max-w-sm">
          <div className="text-ink font-medium">canvas crashed — reload canvas</div>
          <div className="text-[12px] text-ink-3">
            Your edits are intact: they are held outside the canvas, and undo, the object list and save
            still work.
          </div>
          <div className="text-[11px] font-mono text-block break-words">{error.message || "unknown error"}</div>
          <button onClick={this.retry}
            className="border border-accent text-accent px-3 py-1.5 text-[12.5px] hover:bg-accent/10">
            Reload canvas
          </button>
        </div>
      </div>
    );
  }
}
