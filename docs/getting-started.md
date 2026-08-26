# Getting started

## Install

One command on any machine with Docker:

```bash
git clone https://github.com/Sherin-SEF-AI/LabeloxAV.git
cd LabeloxAV
./scripts/install.sh
```

The installer generates secrets, builds images, brings up Postgres and MinIO, applies the schema, seeds the
ontology, starts the API and web app, creates the first administrator, and prints the token to sign in with.
Open `http://localhost:3000`. Re-running is safe: it never rotates a secret that already exists and never
creates a second administrator.

Secrets are generated rather than requested because the application refuses to boot on built-in defaults
anywhere that is not a local dev box. A first run would otherwise fail listing seven variables the operator
has never seen, and asking a person to invent seven high-entropy strings is how you get seven weak ones.

**No GPU is needed to install.** Annotation, review, governance, export and search all work without one. The
model paths that need CUDA refuse rather than fabricating a result.

## Develop

```bash
make up            # Postgres, MinIO, Redis, Redpanda, lakeFS
make install       # dependencies, migrations, ontology seed
make api           # FastAPI on :8000
make web           # Next.js on :3000
```

### Tests

```bash
make test-unit     # fast tier: no Postgres, MinIO, GPU or Redis. Green on a fresh clone.
make test          # everything, needs `make up` first
```

The suite is tiered by marker - `db`, `gpu`, `infra` - so the pure-unit tier runs anywhere. A test that
exercises a torch-only kernel calls `pytest.importorskip("torch")` rather than carrying the `gpu` marker,
because those kernels run on CPU torch and the marker would deselect them from every machine that could
run them.

### Gates that must stay green

```bash
.venv/bin/lint-imports                    # core/services/db must not import packs.av or packs.sec
.venv/bin/python -m scripts.generate_sdk  # the SDK must match the API schema
pytest tests/test_golden_av_pack.py       # pack behaviour is frozen; change it deliberately
```

The golden pack test is the one worth understanding: any change to a pack's ontology, safety definition,
auto-label profile, eval strata, quality profile, forge targets, privacy plane, relations or confusion
cliques changes its digest and fails. Regenerate with `LBX_REGEN_GOLDEN=1` only after reviewing the diff.

## Getting a token

Every route is behind fail-closed bearer auth with three roles.

```bash
.venv/bin/python -m scripts.mint_token --name you --role admin --create
```

This runs on the box, so possession of the signing key is the authority - which is the same authority the
API itself has. That is why it is a script and not an endpoint.

## First labelling run

1. Ingest a drive (`POST /api/ingest`, or the Import menu in the web app).
2. Auto-label it (`POST /api/autolabel/start`). It takes the GPU slot, so it will not run beside a training
   job or a corpus relabel.
3. Review in the frame editor. The triage queue ranks by uncertainty times class rarity.
4. Export (`POST /api/export`). Every release ships a coverage datasheet stating what it does *not* know.
