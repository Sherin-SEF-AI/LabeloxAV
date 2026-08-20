import { resolve } from "path";
import { defineConfig } from "vitest/config";

// Two tiers, because they answer different questions and have different costs.
//
// The node tier is the pure logic that has no business breaking silently: token-expiry decoding, menu
// wiring, the editor reducer, and the source-tree scanners that guard the auth header and swallowed
// failures. Fast, no DOM.
//
// The jsdom tier renders components. There was none: `environment: "node"` with an `include` of
// `*.test.ts` meant a `.test.tsx` was not merely failing, it was never collected - so 79 components had
// zero render coverage and adding a test for one would have looked like it passed by doing nothing. That
// is the worst kind of gap, because it is invisible from the inside.
//
// Selected per file by extension rather than by directory, so a component test lives next to its
// component and gets a DOM by virtue of being a .tsx, while the pure modules keep their speed and cannot
// quietly acquire a DOM dependency.
export default defineConfig({
  resolve: { alias: { "@": resolve(__dirname, ".") } },
  // tsconfig says jsx: "preserve" because Next does its own transform. esbuild honours that and hands
  // vitest untransformed JSX, which fails at runtime with "React is not defined". The automatic runtime
  // is what Next itself compiles to, so this matches the app rather than diverging from it.
  esbuild: { jsx: "automatic" },
  test: {
    globals: true,
    environment: "node",
    environmentMatchGlobs: [["**/*.test.tsx", "jsdom"]],
    setupFiles: ["./vitest.setup.ts"],
    include: ["lib/**/*.test.ts", "components/**/*.test.ts", "**/*.test.tsx"],
    exclude: ["node_modules/**", ".next/**"],
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary"],
      // Scoped to what is actually under test. Reporting coverage over the whole app when the suite is a
      // logic floor plus a handful of render tests produces a number nobody can act on, and a threshold
      // set against it would either be 3% (meaningless) or permanently red (ignored).
      include: ["lib/**/*.ts", "components/shell/LoadState.tsx", "components/Toaster.tsx",
                "components/editor/useEditor.ts", "components/editor/properties/**/*.{ts,tsx}"],
      exclude: ["**/*.test.ts", "**/*.test.tsx", "lib/types.ts", "lib/analytics-api.ts"],
      // A ratchet, not an aspiration: set just under what the suite achieves today so it fails when
      // coverage drops rather than sitting red until somebody turns it off. Raise it as tests land.
      // Measured 2026-08-20, after the properties panel came out of app/frame/[id]/page.tsx with tests:
      // 66.58% lines/statements, 87.19% branches, 29.81% functions. Functions is much lower than lines
      // because lib/api.ts is ~300 thin request wrappers that no unit test calls; the logic in that file
      // that matters (humanizeError, the 401 path, token refresh) is covered. Set just under each measured
      // value so a drop fails rather than a gap sitting red until somebody removes it.
      //
      // Branches went DOWN, 88.0 to 87.19, while the other three rose several points. That is the panel's
      // conditional rendering: the selected-object band, the LiDAR section and the empty states are
      // branches, and the suite exercises most but not all of them. Recorded rather than smoothed over,
      // because a ratchet whose history is edited is not evidence of anything.
      thresholds: { lines: 65, functions: 28, statements: 65, branches: 85 },
    },
  },
});
