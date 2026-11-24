import type { Config } from "tailwindcss";

// Graphite theme for a labelling workstation, dark by default with a light palette beside it. Imagery is judged against the interface around it, so the
// surfaces are near-neutral (a slight cool bias, nothing saturated) and colour is reserved for meaning: one
// blue for selection and action, and the pass / warn / block / info states. Controls are flat; depth comes from
// three surface steps (rail < bg < panel < head) and hairlines, not gradients.
//
// The token names are the ones every page already uses, so the whole app re-themes from here. Text tokens are
// held to WCAG AA (4.5:1) on every surface they sit on, measured rather than eyeballed:
//   ink 13.4-15.9, ink-2 8.2-9.7, ink-3 5.3-6.3, accent 5.8-6.9, accent-2 7.7-9.1,
//   pass 7.0-8.3, warn 7.6-9.0, block 5.1-6.0, info 6.5-7.7   (range over rail, bg, panel and head)
// Those figures are the dark palette's. lib/themePalette.test.ts measures both palettes from globals.css on
// every run, so the claim is checked rather than remembered.
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Each token is a CSS variable holding RGB channels (app/globals.css), not a hex. That is what lets
        // the same class resolve to a different colour under [data-theme="light"], and the
        // `<alpha-value>` slot is what keeps opacity modifiers (bg-accent/20) working through a variable.
        // The dark values are the ones that were written here as hex; the comments still describe them.
        rail: "rgb(var(--c-rail) / <alpha-value>)",         // sidebar
        bg: "rgb(var(--c-bg) / <alpha-value>)",             // page background
        "bg-2": "rgb(var(--c-bg-2) / <alpha-value>)",       // recessed: input fields, wells
        panel: "rgb(var(--c-panel) / <alpha-value>)",       // card / panel surface
        head: "rgb(var(--c-head) / <alpha-value>)",         // raised: panel headers, hovered rows, active nav item
        line: "rgb(var(--c-line) / <alpha-value>)",         // hairline borders and separators
        "line-2": "rgb(var(--c-line-2) / <alpha-value>)",   // faint inner separator
        "line-3": "rgb(var(--c-line-3) / <alpha-value>)",   // a hairline under the pointer: hovered field or card
        ink: "rgb(var(--c-ink) / <alpha-value>)",           // primary text
        "ink-2": "rgb(var(--c-ink-2) / <alpha-value>)",     // secondary text
        "ink-3": "rgb(var(--c-ink-3) / <alpha-value>)",     // tertiary text, labels, captions
        accent: "rgb(var(--c-accent) / <alpha-value>)",     // selection, links, active state
        "accent-2": "rgb(var(--c-accent-2) / <alpha-value>)", // hover and focus
        pass: "rgb(var(--c-pass) / <alpha-value>)",         // accepted / healthy
        warn: "rgb(var(--c-warn) / <alpha-value>)",         // degraded / needs a look
        block: "rgb(var(--c-block) / <alpha-value>)",       // failed / rejected
        info: "rgb(var(--c-info) / <alpha-value>)",         // informational
        btn: "rgb(var(--c-btn) / <alpha-value>)",           // button surface
        "btn-2": "rgb(var(--c-btn-2) / <alpha-value>)",     // button hover
        primary: "rgb(var(--c-primary) / <alpha-value>)",     // the solid action blue of .btn-primary, white text on it
        "primary-2": "rgb(var(--c-primary-2) / <alpha-value>)", // its hover
        "primary-3": "rgb(var(--c-primary-3) / <alpha-value>)", // its pressed state
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
