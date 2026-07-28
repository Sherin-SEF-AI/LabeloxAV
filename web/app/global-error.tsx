"use client";

// The last-resort boundary: catches a throw in the root layout itself, which route-level error.tsx cannot.
// It replaces the whole document, so it must render its own html and body and cannot rely on app chrome,
// providers, or the stylesheet being mounted. Styles are inline for exactly that reason.
export default function GlobalError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <html lang="en">
      <body style={{ background: "#0b0d10", color: "#e6e8eb", fontFamily: "ui-monospace, monospace", padding: 32 }}>
        <h1 style={{ fontSize: 18, marginBottom: 12 }}>LabeloxAV failed to start this page</h1>
        <p style={{ fontSize: 12, color: "#9aa3ad", marginBottom: 12 }}>
          The application shell itself threw, so nothing else could render. Reloading is usually enough; if not,
          the message below is what to report.
        </p>
        <pre style={{ fontSize: 11, color: "#e0a458", whiteSpace: "pre-wrap", marginBottom: 16 }}>
          {error.message || "unknown error"}
          {error.digest ? `\n\ndigest: ${error.digest}` : ""}
        </pre>
        <button
          onClick={reset}
          style={{ fontSize: 12, padding: "6px 12px", background: "transparent",
                   border: "1px solid #4a9eff", color: "#4a9eff", cursor: "pointer" }}
        >
          reload
        </button>
      </body>
    </html>
  );
}
