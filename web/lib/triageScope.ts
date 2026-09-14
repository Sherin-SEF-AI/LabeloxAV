// What subset of the corpus a review session is working through.
//
// `GET /api/triage` already accepts session, class, city, state and flywheel-cycle filters, but the rapid
// review page only ever forwarded the session, so every other way of aiming a review was reachable from the
// API and not from the UI. A batch mined for one class, or dispatched by a flywheel cycle, had no URL that
// would open it.
//
// The mapping lives here rather than in the page because it is the part with rules: which parameters are
// allowed through, what the URL calls them versus what the API calls them, and what a missing value means.
// A component that reads `params.get` inline cannot be tested without rendering it.

// URL parameter -> API query parameter. The names differ because the URL should read as a person would
// write it (`?session=`, `?class=`) while the API keeps its own vocabulary (`session_id`, `klass`).
const SCOPES: Record<string, string> = {
  session: "session_id",
  class: "klass",
  city: "city",
  flywheel: "flywheel",
  states: "states",
  // The control queue: gate-audited samples whose verdicts move the measured auto-accept precision.
  control: "control",
};

// An unscoped queue is effectively endless, so a page of 200 is a page. A scoped one is a finite piece of
// work somebody was sent to finish, and asking for 200 of a 600-object batch silently hands back a third of
// it with no indication that the rest exists. Scoped queues therefore ask for as much as the API will
// return, which is its own cap of 1000.
export const DEFAULT_TRIAGE_LIMIT = 200;
export const SCOPED_TRIAGE_LIMIT = 1000;

/** The triage query for a review page, given its URL parameters. */
export function triageQuery(
  params: { get(key: string): string | null },
  limit?: number,
): Record<string, string> {
  const scoped: Record<string, string> = {};
  for (const [urlKey, apiKey] of Object.entries(SCOPES)) {
    const raw = params.get(urlKey);
    // An empty parameter is a parameter nobody set. Forwarding `klass=` would filter on a class named the
    // empty string and return nothing, which reads as "no work left" rather than as a malformed link.
    const value = raw?.trim();
    if (value) scoped[apiKey] = value;
  }
  // `states` narrows which verdicts are outstanding rather than which objects are in scope, so it does not
  // on its own make the queue finite.
  const isScoped = Object.keys(scoped).some((k) => k !== "states");
  const effective = limit ?? (isScoped ? SCOPED_TRIAGE_LIMIT : DEFAULT_TRIAGE_LIMIT);
  return { limit: String(effective), ...scoped };
}

/** How to describe the current scope to the reviewer, so a short queue is explicable rather than alarming. */
export function describeScope(params: { get(key: string): string | null }): string | null {
  const parts: string[] = [];
  const cls = params.get("class")?.trim();
  const session = params.get("session")?.trim();
  const city = params.get("city")?.trim();
  const batch = params.get("flywheel")?.trim();
  const control = params.get("control")?.trim();
  if (control) parts.push("control samples (each verdict scores the auto-accept gate)");
  if (cls) parts.push(cls);
  if (city) parts.push(`in ${city}`);
  if (session) parts.push(`session ${session.slice(0, 8)}`);
  if (batch) parts.push(`batch ${batch}`);
  return parts.length ? parts.join(", ") : null;
}
