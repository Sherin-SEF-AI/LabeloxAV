import type { Config } from "tailwindcss";

// Operational Materialism design tokens. Color is earned: grey by default, color only encodes
// state. Depth is a single tone step, not shadows.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0B0C0E",
        "bg-2": "#0F1113",
        panel: "#131519",
        line: "#23262B",
        ink: "#E7E9EB",
        "ink-2": "#A0A6AD",
        "ink-3": "#6C727A",
        accent: "#FF7A2F",
        pass: "#56D364",
        warn: "#E3B341",
        block: "#F85149",
        info: "#58A6FF",
      },
      fontFamily: {
        display: ["var(--font-display)", "system-ui", "sans-serif"],
        body: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
