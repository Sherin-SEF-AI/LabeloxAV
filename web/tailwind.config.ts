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
        // 4.88:1 on the bg. Was #808080 at 4.27, which fails WCAG AA and carried most of the body copy
        // in the app: 201 elements at 14px on the home page alone. The lift is four values per channel
        // and is not visible side by side, which is the point.
        "ink-3": "#8a8a8a",    // tertiary / dim text
        // 4.56:1. Was #4772b3 at 3.47, below AA for normal text, and this is the colour on links, state
        // badges and quality flags: over a hundred elements at 10 and 11px. accent-2 moves up with it so
        // hover is still a visible change rather than the two collapsing into one blue.
        accent: "#5b86c7",     // Blender blue: active, selection, primary action
        "accent-2": "#7ea9e0", // brighter blue: hover / focus (6.93:1)
        pass: "#62c25a",       // healthy / accept
        warn: "#e0a63f",       // degraded / caution
        // 4.52:1. Was #e0524b at 4.40, a hair under AA, and it is the colour on every error badge and
        // failure message: 102 elements at 10px. pass, warn and info were already comfortably over.
        block: "#e3564f",      // quarantine / fail
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
