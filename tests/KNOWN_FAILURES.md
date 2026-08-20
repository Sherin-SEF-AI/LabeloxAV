# Known failing tests

This file is the recorded baseline so "is the build broken?" has a mechanical answer instead of a judgement
call: a run that fails only tests listed here is at baseline, and anything else is a regression.
`tests/test_known_failures.py` enforces that the list stays accurate.

Every entry states why it fails and what would fix it. An entry with no route to a fix does not belong here;
it belongs in the code as an xfail or in the backlog as work.

The suite currently has no failures outside the strict xfails below: 2488 pass, 3 skip, 4 xfail, measured
2026-08-18 against full infra (`make up`) on `labeloxav_test`. (The two GPU tests below also failed in that
run, on a card already holding 8.7 GB of 16.3 GB — see that section; both pass in isolation.)

Run it with infra down and ~206 of those tests skip rather than fail, because `_infra_up()` pings Redis. A run
reporting "1495 passed, 215 skipped" is not a smaller green, it is most of the suite never executing, and the
pass count is the only thing that distinguishes the two.

That is no longer only a note. `make test` now sets `LBX_MIN_PASSED` / `LBX_MAX_SKIPPED`, and
`pytest_sessionfinish` in `tests/conftest.py` fails a run that passed everything it ran but ran too little of
the suite. `make up` likewise exits non-zero when the core services never become healthy, instead of falling
through to a `docker compose ps` that exits 0 regardless.

Two categories were removed rather than fixed in place, because the diagnosis recorded here was wrong in a
way worth keeping a note of. Both had been filed under a plausible cause that no amount of work on that
cause would have addressed.

- **"Requires a local model server"** named two `test_m4_vlm.py` tests as needing Ollama. Neither touches a
  model server. `needs_vlm` is pure and the one test that does want Ollama is `skipif`-guarded and skips
  cleanly. They failed because they hardcoded a confidence of 0.72 as "review band" and calibration had
  since moved `auto_accept` from 0.95 to 0.45, making 0.72 an auto-accept.
- **"Order-dependent"** was the right family for its three entries but not the right cause for two of the
  other failures grouped with them. `test_m42_relabel.py` and `test_m43_collaborate.py` were failing because
  lakeFS was not running, and died on a raw urllib3 `ConnectionRefusedError` rather than a skip that named
  the service.

The lesson those two share: a failure filed under a cause nobody has tested is a failure nobody will fix.
Reproduce a listed test in isolation before adding it here.

## Environmental: synthetic frames rejected by the quality gate

These are `xfail(strict=True)` in the code, so they do not appear as failures. Listed for completeness.

| Test | Why |
| --- | --- |
| `test_import.py::test_export_import_roundtrip` | The fixture generates random-noise frames. The ingest quality gate correctly rejects them as noise (variance-of-Laplacian far above `noise_blur_threshold`), so no frame is ingested and the roundtrip has nothing to export. |
| `test_m1_ingest.py::test_video_ingest` | Same: synthetic video frames are rejected. |
| `test_gate_a_pii.py::test_ingest_writes_pii_audit_per_frame` | Same, surfacing as an FK violation because no frame row exists for the audit. |
| `test_p2_export_targets.py::test_export_openlabel_and_nuscenes_through_driver` | Same. |

**Fix:** give these fixtures real frames (a small committed sample) instead of `np.random`. Until then the
xfail is honest: the gate is doing its job and the test data is wrong.

## Environmental: two GPU tests fail under memory pressure, and pass in isolation

| Test | |
| --- | --- |
| `test_m2_autolabel.py::test_stage1_runs_under_vram_ceiling` | loads YOLO11l and asserts a VRAM ceiling |
| `test_m3_fusion.py::test_autolabel_pipeline_writes_gated_objects` | runs the same pipeline |

Both pass alone and pass as a pair. They fail in a full run only when something else already holds most of
the card. Measured on 2026-08-08: 12.3 GB of 16.3 GB was in use before the suite started, 5.2 GB by an Ollama
model still resident and 7.1 GB by the long-lived API server, which runs Ultralytics val in-process for the
promotion gate and does not release it between requests.

So a full-suite result is only meaningful once the card is quiet. Before reading a failure here as a
regression, check `nvidia-smi`, unload the Ollama model with a `keep_alive: 0` request, and restart the API.
The same run is otherwise clean: 1746 passed, 2 skipped, 4 xfailed.

**Fix:** the API server holding GPU memory across requests is the real defect. The promotion gate doing
minutes of Ultralytics work inside an HTTP request is already recorded as wanting a job record it can poll;
moving that work out of the API process would take this with it.

## Still true, and not a test failure: the shared test database is never cleaned

The three tests that were listed here as order-dependent now assert against rows they seeded themselves, so
they no longer care what else is in the database. The condition that broke them has not gone away.

The suite seeds sessions, frames and objects into one database and commits. A session-scoped autouse fixture
(`_reset_corpus`) now truncates at the start of every run, so the cross-run accumulation described here is
gone: a run no longer inherits the 6,865 sessions and 12,262 frames that had piled up.

What remains is the within-run half. Nothing rolls back between individual tests, so a test still sees
everything the tests before it committed. Any new assertion about a corpus-wide statistic, a count, an
availability figure, a nearest neighbour, is written against whatever the earlier tests happened to leave and
will expire quietly later. That is how all three of the previous entries were written. The order is
deterministic (no `pytest-randomly`), so this shows up as a stable pass that silently stops meaning what it
says, rather than as flake.

Two habits avoid it, and both are cheaper than the isolation work:

- Assert about rows the test seeded, not the first row the corpus returns.
- Where a corpus-wide number is genuinely the subject, measure it first and assert the relationship, not a
  literal. `test_buyer_agent.py` reads `available` and asks for `available + 500`.

**Fix:** per-test isolation, a transaction rolled back around each test so the database never accumulates.
The obstacle is that tests open their own sessions through `get_sessionmaker()` and commit, so this means
binding every session in a test to one connection holding an outer transaction, and the autouse fixture in
`conftest.py` that clears the engine cache is where that would go. Until then the residue is harmless to a
correctly written test and fatal to a carelessly written one.
