# Known failing tests

The suite is not green. This file is the recorded baseline so "is the build broken?" has a mechanical answer
instead of a judgement call: a run that fails only tests listed here is at baseline, and anything else is a
regression. `tests/test_known_failures.py` enforces that the list stays accurate.

Every entry states why it fails and what would fix it. An entry with no route to a fix does not belong here;
it belongs in the code as an xfail or in the backlog as work.

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

## Order-dependent: shared test database accumulates state

These pass in isolation and fail depending on what ran before them, because they assert on corpus-wide
statistics (counts, availability, nearest neighbours) against a database the whole suite writes into. Which
subset fails varies between runs.

| Test | Assertion that depends on accumulated state |
| --- | --- |
| `test_buyer_agent.py::test_analyze_spec_reports_availability_and_shortfall` | Asserts a shortfall above a fixed threshold; the shortfall shrinks as earlier tests add objects. |
| `test_annotation_copilot.py::test_pattern_similar_batch_revert` | Asserts a corrected object returns to a specific prior state; other tests mutate objects of the same class. |
| `test_copilot.py::test_answer_finds_matching_frame` | Asserts a semantic search returns a specific frame; other tests add competing frames. |

**Fix:** these need per-test isolation (a transaction rolled back per test, or a uniquely-scoped session id
filter on every assertion) rather than sharing one database. That is a test-infrastructure change, tracked
separately from the production code.

## Requires a local model server

| Test | Why |
| --- | --- |
| `test_m4_vlm.py::test_duty_cycle_only_uncertain_objects` | Needs an Ollama VLM on `models.vlm.ollama_url`. |
| `test_m4_vlm.py::test_vlm_runs_on_subset_and_populates_validated_attrs` | Same. |

**Fix:** mark these `gpu`/`infra` so they deselect cleanly instead of failing, or stand up Ollama in CI.
