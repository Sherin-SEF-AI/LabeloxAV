// The generated API types against the hand-written ones the app actually uses.
//
// `lib/api.generated.ts` is derived from the server's OpenAPI schema; `lib/types.ts` and the inline types in
// `lib/api.ts` were written by hand from memory of what a route returned. This file is where the two are
// held against each other, for the routes the app leans on most: for each one, either the generated
// response is assignable to the hand-written type (the app can take `Result<"...">` today and nothing
// changes), or the exact fields where they part are written down.
//
// Nearly all of it is checked by the compiler, not at run time: `npx tsc --noEmit` is what fails when a
// pin stops being true. Both directions fail on purpose. A route listed as agreeing that drifts apart
// fails, and so does a recorded difference that somebody fixes, because a list of known differences that
// is allowed to go stale stops being read. Fix the type, then delete its entry here.
//
// The differences recorded below are of three kinds, and only the first is somebody's bug:
//   WRONG    the hand-written type says something the server does not do;
//   LOOSER   the server passes a JSON column through, so the generated type is an open object or
//            `unknown` where the hand-written type names a shape nothing on the server enforces;
//   MISSING  the server sends a field the hand-written type does not know about (never an error to
//            compile, which is exactly why it goes unnoticed).

import { readFileSync } from "node:fs";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import type { api, ImportJob, ModelLine, ModelRunRow, TrainingJob } from "./api";
import { API_SCHEMA_STATS } from "./api.generated";
import type { Result } from "./api.generated";
import type {
  DatasetDetail, DatasetRow, FrameMeta, FrameObject, JobRow, NotificationRow, Ontology, Profile,
  SessionRow, UserRow,
} from "./types";

// ---------------------------------------------------------------------------------------- the instruments

type Same<A, B> = [A] extends [B] ? ([B] extends [A] ? true : false) : false;
/** What an `api.*` call resolves to: the hand-written type at that call, wherever it was declared. */
type Returned<F extends (...args: never[]) => unknown> = Awaited<ReturnType<F>>;
/** The fields of `Hand` that a `Generated` value cannot fill: the type differs, or it is required by the
 *  hand-written type and the server's model has no such field. `never` means `Generated` is assignable. */
type Differ<Generated, Hand> = {
  [K in keyof Hand]-?: K extends keyof Generated
    ? ([Generated[K]] extends [Hand[K]] ? never : K)
    : (object extends Pick<Hand, K> ? never : K);
}[keyof Hand];
/** The fields the server declares that the hand-written type has never heard of. */
type Unknown<Generated, Hand> = Exclude<keyof Generated, keyof Hand>;

/** Compiles only when `T` is `true`. */
function holds<T extends true>(..._proof: T[]): void {}

/** `Generated` is assignable to `Hand`, field for field. */
type Agrees<Generated, Hand> = Same<Differ<Generated, Hand>, never>;
/** `Generated` and `Hand` part at exactly these fields and agree on every other. */
type PartsAt<Generated, Hand, Fields> = Same<Differ<Generated, Hand>, Fields>;

// Each operation checked below, so the run-time half can confirm the server still declares it. A type
// that has silently become `unknown` would fail the assertions too, but with an error about fields; this
// says what actually happened.
const CHECKED = [
  "GET /api/sessions", "GET /api/sessions/page", "GET /api/sessions/states",
  "GET /api/sessions/{session_id}/stats", "GET /api/sessions/{session_id}/first-frame",
  "GET /api/frames/{frame_id}", "GET /api/frames/{frame_id}/objects", "GET /api/frames/{frame_id}/filmstrip",
  "PATCH /api/frames/{frame_id}/context", "PUT /api/objects/{object_id}/mask",
  "DELETE /api/objects/{object_id}", "POST /api/objects/{object_id}/review", "POST /api/objects/bulk-review",
  "GET /api/jobs", "GET /api/imports", "GET /api/imports/formats", "GET /api/training",
  "GET /api/training/tasks", "GET /api/training/registry", "GET /api/training/runs",
  "GET /api/datasets", "GET /api/datasets/{commit_id}", "GET /api/ontology", "POST /api/ontology/classes",
  "GET /api/users", "GET /api/users/me", "GET /api/auth/profile", "GET /api/notifications",
  "GET /api/notifications/count", "POST /api/notifications/{notification_id}/read",
  "POST /api/notifications/read-all",
] as const;

// ------------------------------------------------------------------------- where the two already agree
//
// Twenty-two operations. For each, the app could declare the call with the generated type and no caller
// would notice.

function agreements(): void {
  holds<Agrees<Result<"GET /api/jobs">[number], JobRow>>();
  holds<Agrees<Result<"GET /api/datasets">[number], DatasetRow>>();
  holds<Agrees<Result<"GET /api/users/me">, UserRow>>();
  holds<Agrees<Result<"GET /api/users">["users"][number], UserRow>>();
  holds<Agrees<Result<"GET /api/auth/profile">, Profile>>();
  holds<Agrees<Result<"GET /api/notifications">["notifications"][number], NotificationRow>>();
  holds<Agrees<Result<"GET /api/notifications">, Returned<typeof api.notifications>>>();
  holds<Agrees<Result<"GET /api/notifications/count">, Returned<typeof api.notificationCount>>>();
  holds<Agrees<Result<"POST /api/notifications/{notification_id}/read">, Returned<typeof api.markNotificationRead>>>();
  holds<Agrees<Result<"POST /api/notifications/read-all">, Returned<typeof api.markAllNotificationsRead>>>();
  holds<Agrees<Result<"GET /api/ontology">, Ontology>>();
  holds<Agrees<Result<"POST /api/ontology/classes">, Returned<typeof api.addClass>>>();
  holds<Agrees<Result<"GET /api/sessions/states">[number], Returned<typeof api.sessionStates>[number]>>();
  holds<Agrees<Result<"GET /api/sessions/{session_id}/stats">, Returned<typeof api.sessionStats>>>();
  holds<Agrees<Result<"GET /api/sessions/{session_id}/first-frame">, Returned<typeof api.firstFrame>>>();
  holds<Agrees<Result<"GET /api/frames/{frame_id}/filmstrip">, Returned<typeof api.frameFilmstrip>>>();
  holds<Agrees<Result<"GET /api/frames/{frame_id}/filmstrip">["frames"][number],
               Returned<typeof api.frameFilmstrip>["frames"][number]>>();
  holds<Agrees<Result<"PUT /api/objects/{object_id}/mask">, Returned<typeof api.updateMask>>>();
  holds<Agrees<Result<"DELETE /api/objects/{object_id}">, Returned<typeof api.deleteObject>>>();
  holds<Agrees<Result<"POST /api/objects/bulk-review">, Returned<typeof api.bulkReview>>>();
  holds<Agrees<Result<"GET /api/imports/formats">, Returned<typeof api.importFormats>>>();
  holds<Agrees<Result<"GET /api/training/tasks">[number], Returned<typeof api.trainingTasks>[number]>>();
  holds<Agrees<Result<"GET /api/training/registry">[number], ModelLine>>();
  // The run's three metric dictionaries are JSON on the server and named optional fields here; an open
  // object is assignable to a type whose every field is optional, so this agrees without the server
  // having promised any of the names.
  holds<Agrees<Result<"GET /api/training/runs">[number], ModelRunRow>>();
  // The whole-value form of the same claim, for the ones a caller is most likely to switch first.
  const jobs: JobRow[] = null as unknown as Result<"GET /api/jobs">;
  const datasets: DatasetRow[] = null as unknown as Result<"GET /api/datasets">;
  const me: UserRow = null as unknown as Result<"GET /api/users/me">;
  const profile: Profile = null as unknown as Result<"GET /api/auth/profile">;
  const ontology: Ontology = null as unknown as Result<"GET /api/ontology">;
  const lines: ModelLine[] = null as unknown as Result<"GET /api/training/registry">;
  void [jobs, datasets, me, profile, ontology, lines];
}

// ------------------------------------------------------------------------------- where they part, and how

function differences(): void {
  type Session = Result<"GET /api/sessions">[number];
  // WRONG (nullability). `session.vehicle_id` is a nullable column and the server types it so; SessionRow
  // says `string`. Every session in the working corpus has one, which is how `s.vehicle_id.startsWith(...)`
  // has gone unpunished. Not changed here: the pages that would then need a null check are not this
  // file's to edit.
  holds<PartsAt<Session, SessionRow, "vehicle_id">>();
  holds<PartsAt<Result<"GET /api/sessions/page">["sessions"][number], SessionRow, "vehicle_id">>();
  // MISSING, fixed: SessionRow had no `ontology_version` or `project_id`, which every row carries.
  holds<Same<Unknown<Session, SessionRow>, never>>();

  type Frame = Result<"GET /api/frames/{frame_id}">;
  // LOOSER. `context_provenance` is the frame's scene_provenance JSON passed through; FrameMeta types
  // each value as { by, ts_ns }, which is what the one writer stores and nothing validates on the way out.
  holds<PartsAt<Frame, FrameMeta, "context_provenance">>();
  holds<PartsAt<Result<"PATCH /api/frames/{frame_id}/context">, Returned<typeof api.setFrameContext>,
                "context_provenance">>();
  holds<Same<Unknown<Frame, FrameMeta>, never>>();

  type Obj = Result<"GET /api/frames/{frame_id}/objects">[number];
  // LOOSER. `keypoints` and `cuboid_3d` are JSON columns the server declares as open objects (as the
  // existing ObjectDetail model already did); FrameObject names { skeleton, points } and
  // { center, size, yaw }.
  holds<PartsAt<Obj, FrameObject, "keypoints" | "cuboid_3d">>();
  holds<Same<Unknown<Obj, FrameObject>, never>>();

  type Detail = Result<"GET /api/datasets/{commit_id}">;
  // LOOSER. `slice_spec` is a JSON column passed through, so the server allows null; DatasetDetail does not.
  // (WRONG, fixed: DatasetDetail extended the list row and so promised an `n_files` the detail never
  // sends. MISSING, fixed: each file's `missing` flag.)
  holds<PartsAt<Detail, DatasetDetail, "slice_spec">>();
  holds<Agrees<Detail["files"][number], DatasetDetail["files"][number]>>();
  holds<Same<Unknown<Detail["files"][number], DatasetDetail["files"][number]>, never>>();

  // LOOSER. `counts` is a JSON column: ImportJob says every value is a number and non-null.
  holds<PartsAt<Result<"GET /api/imports">[number], ImportJob, "counts">>();

  type Training = Result<"GET /api/training">[number];
  // LOOSER, the same four JSON columns, each non-null and (for counts) all-numeric in TrainingJob.
  holds<PartsAt<Training, TrainingJob, "counts" | "metrics" | "result" | "config">>();

  // Was the one that mattered: api.review was declared to return an ObjectDetail while the server
  // returns eleven fields, and components/inspect/EvidencePanel.tsx stored the response as the object it
  // was showing, losing the frame and the image. It is declared with the generated type now and the panel
  // merges (mergeReview), so the two agree exactly.
  holds<Agrees<Result<"POST /api/objects/{object_id}/review">, Returned<typeof api.review>>>();
}

// ---------------------------------------------------------------------------------------- the run-time half

/** `"METHOD /path": { ... response: T; }` for one operation, read out of the generated file's text. */
function declaredResponse(source: string, operation: string): string | null {
  const start = source.indexOf(`  ${JSON.stringify(operation)}: {`);
  if (start < 0) return null;
  const block = source.slice(start, source.indexOf("\n  };", start));
  const m = block.match(/\n {4}response: ([^\n]+);$/m);
  return m ? m[1] : null;
}

describe("the generated API types", () => {
  const source = readFileSync(join(__dirname, "api.generated.ts"), "utf8");

  it("declare a response for every operation compared here", () => {
    const undeclared = CHECKED.filter((op) => {
      const response = declaredResponse(source, op);
      return response === null || response === "unknown";
    });
    expect(undeclared).toEqual([]);
    expect(CHECKED.length).toBeGreaterThanOrEqual(30);
  });

  it("still cover at least the operations that were typed when this was written", () => {
    // 41 before the response models were declared, 90 after. A regeneration that drops below this lost a
    // declaration somewhere, and the type-level checks above only notice the ones they name.
    expect(API_SCHEMA_STATS.typedResponses).toBeGreaterThanOrEqual(90);
    expect(API_SCHEMA_STATS.operations).toBeGreaterThan(API_SCHEMA_STATS.typedResponses);
  });

  it("are compared with the hand-written types by the compiler", () => {
    // Nothing to run: the two functions hold only type-level assertions, and `npx tsc --noEmit` is what
    // checks them. Referenced so they are not dead code, and so a reader lands on this sentence.
    expect([agreements, differences].every((f) => typeof f === "function")).toBe(true);
  });
});
