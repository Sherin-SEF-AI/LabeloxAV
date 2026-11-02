# Getting started

## Install

One command on any machine with Docker:

```bash
git clone https://github.com/Sherin-SEF-AI/LabeloxAV.git
cd LabeloxAV
./scripts/install.sh
```

The installer picks free ports, generates secrets, builds the images, starts the infrastructure (Postgres,
MinIO, Redis, Redpanda), applies the schema, seeds the ontology, starts the API, the web app and the GPU
worker, and ends with a sign-in link for the administrator: open it and choose a password. Re-running is
safe: it never rotates a secret that already exists and never creates a second administrator.

`./scripts/doctor.sh` checks the installation, from Docker and the GPU runtime to whether the GPU worker is
running, and says how to fix whatever is not right. [DEPLOY.md](DEPLOY.md) covers GPUs, optional services,
TLS and backups.

Secrets are generated rather than requested because the application refuses to boot on built-in defaults
anywhere that is not a local dev box. A first run would otherwise fail listing seven variables the operator
has never seen, and asking a person to invent seven high-entropy strings is how you get seven weak ones.

**No GPU is needed to install.** Annotation, review, governance, export and search all work without one. The
model paths that need CUDA refuse rather than fabricating a result.

## Develop

```bash
make up            # Postgres, MinIO, Redis, Redpanda, plus lakeFS, Lichtblick and MLflow
make install       # the virtualenv and dependencies
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

## Signing in and adding people

People sign in with a name and a password. An administrator adds a teammate in **Settings > Team** and
sends them the link it produces; it opens "choose a password", works once and expires. A locked-out
administrator gets a fresh link from the server, where database access is the authority:

```bash
.venv/bin/python -m scripts.sign_in_link --name admin --url http://localhost:3000
```

Scripts, notebooks and CI use a long-lived key from **Settings > API keys** with the Python client:

```bash
pip install ./sdk                       # labelox-client: httpx and click only
export LBX_URL=http://localhost:3000 LBX_API_KEY=lbxk_...
labelox health
```

## Who sees which data

Every session belongs to a project, and every project starts visible to the whole team. When some data
should stay with a few people (a client's drives, a partner's recordings), an administrator opens
**Settings > Projects**, makes that project **Members only**, adds its members and moves its sessions in.
Everyone else then stops seeing those sessions, their frames and their labels, both in lists and by direct
link (the server answers 404, so nothing confirms they exist). Administrators see everything. An upload or
import can name its project (`project_id`); without one it goes to the default project.

Any list the server returns leaves out rows that name a hidden session, frame, object or track, including
pages that were never written with projects in mind. What is not scoped are aggregate numbers: the charts
on analytics and similar overview pages still count across every project.

## First labelling run

The home page's "Get started" checklist walks through this and ticks each step off from the data:

1. **Bring in data.** Upload video or images, import a labelled dataset (COCO, YOLO, KITTI, nuScenes, CVAT
   and more), or load a sample drive. Faces and plates are blurred as frames are stored.
   API: `POST /api/upload/init` then `POST /api/imports/start`.
2. **Auto-label a session.** Three detectors propose every object; confident ones are accepted, the rest wait
   for a person. The GPU worker runs one job at a time, so it never runs beside a training job.
   API: `POST /api/autolabel/start`.
3. **Review.** Accept, fix or reject in the review queue or the frame editor. The most valuable objects come
   first.
4. **Export.** A versioned dataset in the format your training code reads, with a datasheet stating what it
   does *not* know. API: `POST /api/datasets/export`.
5. **Train.** Fine-tune a detector on the reviewed labels and compare it class by class with the current one.
