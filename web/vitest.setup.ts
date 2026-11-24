import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

import { clearQueries, configureQueries } from "@/lib/useQuery";

// Unmount between tests. Without it a component from the previous test is still in the document and a
// getByText finds the wrong one - which fails in a way that looks like the component under test.
//
// Then empty the query cache. It is process-wide, which is the point of it in the app and a trap here:
// without this a page test would be answered by the previous test's response. One hook, in this order,
// because emptying the cache under a still-mounted component is a state update outside act().
afterEach(async () => {
  cleanup();
  await clearQueries();
});

// The dedupe window goes to zero for the same reason: inside it, a second mount of the same key is handed
// the first one's request.
configureQueries({ dedupeMs: 0 });

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

  // jsdom implements no layout, so it ships no scrollIntoView at all. Any component that keeps a cursor
  // on screen calls it and throws here, which reads as a component bug and is not one.
  if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = function () {};

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
