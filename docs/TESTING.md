# Testing

How the suite is organised, and the two things about it that are not obvious and will bite you.

---

## The two that will bite you

**`pytest.mark.db` is what arms the production-database guard.** `tests/conftest.py` refuses to run against
a database whose name does not contain "test". It also skips provisioning when no selected test carries the
`db` marker, so the fast tier needs no Postgres. Those two facts used to be in the wrong order: the refusal
sat *after* the skip, so a run selecting only unmarked tests wrote rows with the guard never evaluated. 162
files touch the database and 45 said so. That is the mechanism behind the 1,730 fixture sessions that had to
be purged from the real corpus on 2026-07-30.

The guard now runs first, unconditionally. But the marker still matters: without it your test lands in
`make test-unit`, which is documented as needing no Postgres, and it will fail there for a reason that
looks like your test rather than like a missing marker. `tests/test_db_markers.py` fails if you forget.

**Nothing rolls back between individual tests.** The truncate is session-scoped, so a test sees everything
the tests before it committed. Two consequences: seed what you assert on rather than relying on what is
there, and be suspicious of a test that passes in a full run and fails alone — that is usually order
dependence, not flake. `tests/test_m44_govern.py::test_governance_end_to_end` is a known instance.

---

## Tiers

```bash
make test        # everything; needs infra (make up)
make test-unit   # -m "not db and not gpu and not infra"; no Postgres, no GPU, no Redis
```

The unit tier is real: with Postgres stopped it completes 1,449 passed in about 60 seconds. It was
unrunnable before the marker pass, which is why nobody used it.

| Marker | Means |
| --- | --- |
| `db` | needs a real Postgres, and usually MinIO |
| `gpu` | needs a CUDA device or downloaded weights |
| `infra` | needs Redis or another running service |

`--strict-markers` is on, so a typo is a collection error rather than a test that is silently never
selected.

---

## Floors, so green means something

Most of this suite *skips* rather than fails when infra is down — about 206 tests gate on a Redis ping. So
"1495 passed, 215 skipped" and "2580 passed, 3 skipped" are both green, and only the counts tell them
apart. `make test` therefore sets `LBX_MIN_PASSED` and `LBX_MAX_SKIPPED`, enforced in
`pytest_sessionfinish`, and fails a run that passed everything it ran but ran too little of the suite.

`make up` likewise exits non-zero when the core services never become healthy, instead of falling through
to a `docker compose ps` that exits 0 regardless.

Update the floor in the Makefile when the suite grows; the number in the comment there is the measured
baseline and the date it was measured.

---

## Web

```bash
cd web
npm test              # both tiers
npm run test:coverage # with the ratchet CI enforces
```

Two environments, selected by file extension:

- **`*.test.ts` → node.** Pure logic: token expiry, menu wiring, the editor reducer, and the source-tree
  scanners.
- **`*.test.tsx` → jsdom.** Rendered components, with Testing Library.

The second did not exist. The config was node-only with an include of `*.test.ts`, so a `.test.tsx` was not
failing — it was never collected. Anyone adding one would have watched the suite go green without it ever
running.

Coverage has a threshold set just under the measured numbers, so a drop fails rather than a gap sitting red
until somebody deletes the gate. Raise it as tests land.

---

## End-to-end

```bash
make e2e     # needs the app running (make app-up, or api + web)
```

A smoke over the golden path, not a regression suite: it asserts the journey is walkable and that the
specific failures this remediation fixed stay fixed — the queue not calling a dropped request a finished
shift, the driving-events pages being reachable, the skip link being first in the tab order. A broad e2e
suite over 71 pages would be slow and flaky, and a flaky gate gets switched off.

It **skips** when nothing is serving rather than failing, because a suite that goes red on absent infra
teaches people to ignore red.

Two things learned writing it, both worth keeping:

- **Assert the status, not just that the body has content.** The first version checked
  `expect(body).not.toBeEmpty()`, which a Next error page satisfies perfectly — so a page serving 500 was
  green. It was found exactly that way.
- **Do not run `npm run build` while `next dev` is running.** They share `.next`, and the build leaves the
  dev server with a chunk manifest pointing at files that no longer exist; every route then 500s with
  `Cannot find module './NNNN.js'` until the dev server is restarted. The source is fine — it looks like a
  code failure and is not one.

## Source-tree invariants

Several tests assert things about the source rather than behaviour. They are cheap and they catch a class
of regression no unit test can: a rule that was followed once and then quietly stopped being followed.

| Test | Invariant |
| --- | --- |
| `tests/test_route_auth.py` | no unapproved public read; no write route without a role floor; the privileged writes stay above the annotator floor |
| `tests/test_db_markers.py` | every DB-touching test file carries the marker |
| `tests/test_deploy_composition.py` | the deployment still contains the daemon, the weights job and the workers |
| `tests/test_known_failures.py` | every xfail is documented and every documented test still exists |
| `web/lib/nofetch.test.ts` | no bare `fetch("/api/…")` bypassing the authed client |
| `web/lib/noswallow.test.ts` | swallowed failures cannot grow beyond the recorded baseline |
| `web/lib/routeGaps.test.ts` | the route-gap list agrees with the tree in both directions |

Each has a **non-triviality floor** — an assertion that the scan found a plausible number of things. Without
one, a broken glob turns the whole file into a statement about the empty set, which passes.

---

## Writing one

Match what is already there:

- **No mocks of the thing under test.** There is not a single `MagicMock` in 339 files. Substitution is
  `monkeypatch.setattr` with a real function, and assertions land on real database rows.
- **Assert the property, not the implementation.** `test_containment_not_iou` asserts the IoU is ~0.05 and
  the region still counts as covering, which is the design decision; a test of the threshold constant would
  pass forever and mean nothing.
- **A test that cannot fail is worse than no test.** `test_mono_depth_recovers_known_size` passed
  `class_id=0`, read the height table, and confirmed the value it had just read — so it could never detect
  that the whole table was keyed against the wrong id space, which is exactly what had happened.
- **Say what broke in the failure message.** `assert offenders == [], f"...{offenders}"` beats
  `assert not offenders`.

Known failures live in `tests/KNOWN_FAILURES.md`, and `tests/test_known_failures.py` keeps that file honest.
A run failing only what is listed there is at baseline; anything else is a regression.
