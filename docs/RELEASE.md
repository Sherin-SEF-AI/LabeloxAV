# Releasing

What has to be true before a version is tagged. Most of it runs in CI on every pull request; the rest needs a
GPU or a person, so it is done once per release on the GPU box, from a clean checkout of the commit to be
tagged.

## In CI, on every pull request

- Python: lint (ruff on the gated paths), the strict-typing allowlist, the import contract, and the full test
  suite against a fresh PostgreSQL with every migration applied.
- Web: type check, unit tests, lint, and a production build (`next build`), which is what catches a page
  that only fails when prerendered.
- Container images: every image a compose file names either builds here or is pinned to a published tag.

A release is not tagged off a red pull request.

## On the GPU box, once per release

1. **First-run end to end.** `make e2e-first-run`. It installs the tagged commit into a throwaway worktree
   with the real installer (its own Compose project, ports and volumes), signs in with the printed link,
   loads a sample drive, auto-labels it on the GPU worker, accepts a label, exports a dataset, checks the
   home checklist reaches 4 of 5, and removes the install. About three minutes once images are cached. It
   needs network access for the sample drive.
2. **Migrations both ways.** On a copy of a real database: `alembic upgrade head`, then
   `alembic downgrade -1` and `upgrade head` again for each migration added since the last release. Every
   migration here has a working downgrade; this checks the new ones still do.
3. **System checks.** `make health` and `GET /api/system/checks` on the installed release: privacy blur
   models present, plate text verification present, GPU visible, worker running.
4. **Measurements still hold.** Rerun the ones the release could move and compare with
   [Measurement](MEASUREMENT.md#is-each-assistive-feature-earning-its-place):
   - `GET /api/triage/ordering` (queue order on gold),
   - `POST /api/errordetect/benchmark` (planted errors),
   - `POST /api/assistant/eval` (assistant tool choice),
   and update the table if a number moved. A feature whose number got worse is a release note, not a
   footnote.
5. **Licensing.** Settings > About shows the release's version and links the exact commit; if the release
   changes a bundled component, update the table in [Licensing](LICENSING.md) and the About page together.

## Tagging

Bump `version` in `pyproject.toml`, tag `vX.Y.Z` on main, and write the release notes from the merged pull
requests since the last tag: what a team will notice, then what changed underneath, then anything a
deployment has to do (a new setting, a model to fetch, a migration that takes a while).
