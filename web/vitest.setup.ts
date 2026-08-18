import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// Unmount between tests. Without it a component from the previous test is still in the document and a
// getByText finds the wrong one - which fails in a way that looks like the component under test.
afterEach(() => cleanup());

// jsdom implements neither of these, and both are used by components that have nothing to do with them:
// the canvas measures, the shell observes resize. Stubbed rather than mocked per-test so a render does not
// fail for a reason unrelated to what is being asserted.
if (typeof window !== "undefined") {
  window.matchMedia =
    window.matchMedia ||
    ((query: string) =>
      ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      }) as unknown as MediaQueryList);

  globalThis.ResizeObserver =
    globalThis.ResizeObserver ||
    (class {
      observe() {}
      unobserve() {}
      disconnect() {}
    } as unknown as typeof ResizeObserver);

  globalThis.IntersectionObserver =
    globalThis.IntersectionObserver ||
    (class {
      root = null;
      rootMargin = "";
      thresholds = [];
      observe() {}
      unobserve() {}
      disconnect() {}
      takeRecords() {
        return [];
      }
    } as unknown as typeof IntersectionObserver);
}
