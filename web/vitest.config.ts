import { resolve } from "path";
import { defineConfig } from "vitest/config";

// A small unit floor for the pure logic that has no business breaking silently: token-expiry decoding,
// navigation menu wiring. These run in the node environment (no jsdom) because the modules under test are
// pure and never touch the DOM. The @ alias mirrors tsconfig so test imports resolve like app imports.
export default defineConfig({
  resolve: { alias: { "@": resolve(__dirname, ".") } },
  test: {
    environment: "node",
    include: ["lib/**/*.test.ts", "components/**/*.test.ts"],
  },
});
