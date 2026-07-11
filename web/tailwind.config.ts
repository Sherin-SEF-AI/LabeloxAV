import type { Config } from "tailwindcss";

// Blender-style dark theme. Neutral greys for surfaces, the signature Blender blue for active/accent state,
// rounded controls, and state colors (pass/warn/block) kept for status. The palette maps the old token names
// so the whole app re-themes without per-page edits: bg = editor background, panel = a panel surface, head =
// a raised header/tab strip, bg-2 = a recessed input field, line = panel outlines, ink* = text tiers.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#1d1d1d",         // editor / window background
        "bg-2": "#161616",     // recessed input / number field (reads inset)
        panel: "#2d2d2d",      // panel surface (card)
        head: "#383838",       // raised header / tab strip / toolbar
        line: "#3c3c3c",       // panel outline / separator
        "line-2": "#242424",   // darker inset separator
        ink: "#e6e6e6",        // primary text
        "ink-2": "#b0b0b0",    // secondary text
        "ink-3": "#808080",    // tertiary / dim text
        accent: "#4772b3",     // Blender blue: active, selection, primary action
        "accent-2": "#5b87c9", // brighter blue: hover / focus
        pass: "#62c25a",       // healthy / accept
        warn: "#e0a63f",       // degraded / caution
        block: "#e0524b",      // quarantine / fail
        info: "#6a9ee0",       // informational blue
        btn: "#4a4a4a",        // tool-button base
        "btn-2": "#565656",    // tool-button hover
      },
      borderRadius: {
        DEFAULT: "4px",        // Blender's moderate control rounding
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
