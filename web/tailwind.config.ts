import type { Config } from "tailwindcss";

// Graphite dark theme for a labelling workstation. Imagery is judged against the interface around it, so the
// surfaces are near-neutral (a slight cool bias, nothing saturated) and colour is reserved for meaning: one
// blue for selection and action, and the pass / warn / block / info states. Controls are flat; depth comes from
// three surface steps (rail < bg < panel < head) and hairlines, not gradients.
//
// The token names are the ones every page already uses, so the whole app re-themes from here. Text tokens are
// held to WCAG AA (4.5:1) on every surface they sit on, measured rather than eyeballed:
//   ink 13.4-15.9, ink-2 8.2-9.7, ink-3 5.3-6.3, accent 5.8-6.9, accent-2 7.7-9.1,
//   pass 7.0-8.3, warn 7.6-9.0, block 5.1-6.0, info 6.5-7.7   (range over rail, bg, panel and head)
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        rail: "#0f1013",       // sidebar
        bg: "#121316",         // page background
        "bg-2": "#0d0e10",     // recessed: input fields, wells
        panel: "#18191d",      // card / panel surface
        head: "#1f2025",       // raised: panel headers, hovered rows, active nav item
        line: "#2a2c32",       // hairline borders and separators
        "line-2": "#1a1b1f",   // faint inner separator
        ink: "#e8e9ec",        // primary text
        "ink-2": "#b4b8c0",    // secondary text
        "ink-3": "#8e939c",    // tertiary text, labels, captions
        accent: "#5f9bf5",     // selection, links, active state
        "accent-2": "#86b4fa", // hover and focus
        pass: "#4fbf7a",       // accepted / healthy
        warn: "#e2a640",       // degraded / needs a look
        block: "#ee6259",      // failed / rejected
        info: "#6aa6f0",       // informational
        btn: "#24262c",        // button surface
        "btn-2": "#2c2f36",    // button hover
      },
      borderRadius: {
        DEFAULT: "6px",
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
