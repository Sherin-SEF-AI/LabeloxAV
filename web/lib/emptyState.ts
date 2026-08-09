// Telling "nothing here" apart from "not asked yet".
//
// The analytics page rendered "no objects yet" and "class distribution (0 classes present)" for about ten
// seconds on a corpus holding 570,505 objects, because its panels branch on `count === 0` and the fetch had
// not returned. The KPI row did the same thing with "-", which reads as a measured zero rather than an
// absent measurement.
//
// That is the same class of error this codebase keeps finding in its own metrics: a number computed over
// nothing, presented with the confidence of a number computed over everything. It matters more here than it
// looks, because "no objects yet" is a plausible state for a fresh deployment, so nobody reads it as a bug.
// They read it as the truth and go looking for why the ingest failed.
//
// Pure so it can be tested without a DOM, which is how the rest of web/lib is tested.

export type Placeholder = { kind: "loading" | "empty"; text: string } | null;

/** Marker for a count that cannot be stated yet. Chosen over "-" and "0", both of which read as measured. */
export const PENDING = "...";

/**
 * What a panel should draw in place of its data.
 *
 * `null` means draw the data. Note that a non-zero count wins over `loading`: a page that loads a dozen
 * endpoints is still loading long after this panel has what it needs, and blanking it would be a second lie
 * in the other direction.
 */
export function placeholder(loading: boolean, count: number, emptyText: string,
                            loadingText = "loading..."): Placeholder {
  if (count > 0) return null;
  if (loading) return { kind: "loading", text: loadingText };
  return { kind: "empty", text: emptyText };
}

/**
 * A scalar for a stat card: the value once known, `PENDING` while in flight, and the caller's `absent` only
 * once the answer is genuinely nothing.
 */
export function statValue(loading: boolean, value: string | number | null | undefined,
                          absent = "-"): string {
  if (value !== null && value !== undefined) return String(value);
  return loading ? PENDING : absent;
}

/**
 * A count to interpolate into a panel title, such as "class distribution (N classes present)".
 *
 * A title is read as a fact about the corpus, so it must not say zero while the request is open.
 */
export function titleCount(loading: boolean, count: number, known: boolean): string {
  if (known) return String(count);
  return loading ? PENDING : String(count);
}
