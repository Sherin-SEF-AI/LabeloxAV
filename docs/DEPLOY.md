# Deploying LabeloxAV

One command on a machine with Docker:

```bash
git clone https://github.com/Sherin-SEF-AI/LabeloxAV.git
cd LabeloxAV
./scripts/install.sh
```

It generates the secrets, builds the images, brings up the database and object store, applies the schema,
seeds the ontology, starts the API and the web app, waits for readiness, creates the first administrator, and
prints the token to sign in with. Open `http://localhost:3000`.

Re-running is safe. It never rotates a secret that already exists and never creates a second administrator.

---

## What it needs

- Docker with the Compose v2 plugin, and a running daemon.
- ~20 GB of disk. Images, model weights, and the corpus add up, and running out during an import corrupts it.
- ~8 GB of RAM for the services. The GPU paths (auto-labeling, embeddings, training) need more and a CUDA
  device; see [GPU](#workers-and-gpu) below.

No GPU is required to install. Without one, the annotation, review, governance, export, and search surfaces
all work; the model paths that need CUDA refuse rather than producing a fabricated result.

Measured on a clean Ubuntu 24.04 with no GPU available and an empty image cache, installing the v0.1.1
package: the first install took about six and a half minutes and left about 4.5 GB of images (api 2.7 GB,
web 1.05 GB, database 794 MB). The API reported Postgres, Redis and MinIO healthy, the web app answered, and
authentication refused a request without a token and accepted one with the generated admin token. The
~20 GB above is for the corpus and model weights on top of that.

MinIO is built from source (`docker/minio/Dockerfile`, pinned to the release the stack has always run).
MinIO no longer publishes images: its Docker Hub repositories were withdrawn and quay.io refuses anonymous
pulls, so an install that pulled one failed at "Starting infrastructure". The first install compiles it
(about two minutes); later ones use the cached image. Backups no longer need MinIO's client image either:
`make backup` and `make restore` mirror the bucket with `scripts/bucket_mirror.py`.
`python scripts/check_images.py` checks that every image a fresh install still pulls resolves; CI runs it
on every pull request.

---

## What the installer does, and why

**It generates the secrets rather than asking for them.** `core/config.py` refuses the built-in dev defaults
on any deployment that is not an explicit local dev box, and refuses to *boot* rather than run insecurely,
because a known signing key lets anyone mint an admin token. That is the right behaviour, but it means a
first run would otherwise fail listing seven variables the operator has never seen. Asking a human to invent
seven high-entropy strings is also how you get seven weak ones. They are written to `.env` with mode 600.

**It waits on `/api/readyz`, not `/api/health`.** Health returns 200 with a degraded body so an operator can
see *which* dependency is down. Readiness returns 503 until every dependency answers. Waiting on health would
declare success while Postgres was still starting.

**Migrations run as their own one-shot container.** A migration failure shows up as a failed container rather
than an API crash loop, and two API replicas can never race the same migration.

**It refuses to overwrite a local-development `.env`.** The dev file carries the well-known credentials on
purpose. Converting it in place would leave a machine that neither runs as a dev box nor boots as a
deployment, with the reason buried in a config validator.

---

## Everyday operation

The installer records which compose files and optional services this machine runs in `.env`
(`COMPOSE_FILE` and `COMPOSE_PROFILES`), and Docker Compose reads both from there. So every command is plain
`docker compose`, and it always means exactly what was installed, including the GPU overlay when Docker can
use the GPU. `make` and the packaged `labeloxav` command use the same:

```bash
make app-down     # stop, keeping all data
make app-up       # start again
make app-logs     # follow the API logs
make doctor       # check the installation and say what to fix

# the same thing without make
docker compose up -d
docker compose ps
./scripts/doctor.sh

# stop and DESTROY the corpus, the object store, and every annotation
docker compose down -v
```

On a development checkout there is no `COMPOSE_FILE` in `.env`, so plain `docker compose up` starts the
infrastructure alone: services in Docker, code on the host. That is why the application lives in an overlay
rather than behind a profile: the overlay demands the deployment secrets, and requiring them in the base file
would break every compose command for a developer who has none.

### When something is wrong

`./scripts/doctor.sh` (or `labeloxav doctor`) checks the host (Docker, the GPU runtime, disk, memory, every
configured port and who holds it, each container's state) and then the application from inside the API
container (database and schema version, storage, event bus, torch and the GPU, model weights, and whether the
GPU worker is running). Every problem it reports comes with the command or setting that fixes it. An
administrator sees the same application checks in the web app.

### Signing in, inviting the team, and recovering an account

The installer ends by printing a sign-in link for the administrator: open it, choose a password, and you
are in. People sign in with a name and password from then on; nobody pastes tokens.

**Adding people.** An administrator opens **Settings > Team**, enters a name and a role, and gets a link to
send. It opens "choose a password", works once, and expires after `LBX_AUTH__INVITE_TTL_HOURS` (72 by
default). Nothing is emailed: the link is handed over however the team talks, which also keeps this working
on an offline lab network.

**A lost password.** On the Team page, **New sign-in link** issues a fresh link for that person. Their old
password keeps working until the link is used; **Sign out everywhere** ends every session they have open.

**A locked-out administrator.** On the server, where database access is the authority:

```bash
labeloxav sign-in-link admin                    # packaged install
make sign-in-link NAME=admin                    # from a checkout
docker compose exec api python -m scripts.sign_in_link --name admin --url http://<host>:3000
```

Roles are `annotator`, `reviewer`, `admin`, and a higher role satisfies any floor below it.

### Scripts, notebooks and CI: API keys and the Python client

Sign-in tokens expire after hours and belong to a person, so nothing automated should borrow one. An
administrator creates a long-lived key per script or pipeline in **Settings > API keys** (or
`POST /api/service-accounts`), with a role and an optional expiry. The key is shown once; the server keeps
only a hash. Rotate it to replace a leaked key without changing what it is allowed to do, or revoke it to
stop it immediately.

The Python client is its own small package (httpx and click only, no server code):

```bash
pip install ./sdk                      # from a checkout; the package is labelox-client
export LBX_URL=http://labelox.lab:3000 # the web app's address; it proxies /api
export LBX_API_KEY=lbxk_...
labelox health
labelox facets --predicate '{"states":["accepted"]}'
```

```python
from labelox import Client
lbx = Client()                        # reads LBX_URL and LBX_API_KEY
lbx.health()
```

Every endpoint is also available as a method on `labelox.generated.LabeloxClient`, generated from the API's
own schema by `python -m scripts.generate_sdk` (CI fails if it falls behind the API).

---

## Serving to more than one machine

The defaults assume you are sitting at the machine. Two things must change when you are not.

**`LBX_MINIO__PUBLIC_ENDPOINT`** is what a browser is handed for a presigned download. Inside the compose
network the object store is `minio:9000`, which a browser cannot resolve, so this has to be an address the
browser can actually reach. Getting it wrong produces download links that 404 in a way that looks like
missing data rather than a misconfiguration:

```bash
LBX_MINIO__PUBLIC_ENDPOINT=https://labelox.example.com/s3
```

**What listens where.** Only two ports face the network: the web app (`WEB_PORT`, default 3000) and MinIO's
S3 port (`MINIO_PORT`), which presigned downloads use. Everything else, the API included, listens on this
machine only (`127.0.0.1`): people and the SDK reach the API through the web app's `/api` proxy. Change
`LBX_BIND_PUBLIC` / `LBX_BIND_LOCAL` in `.env` to bind differently.

**TLS.** Nothing here terminates TLS, and the app is served over plain HTTP. Put a
reverse proxy in front of it for anything beyond a trusted network: tokens live in browser storage and ride
on every request, so an unencrypted hop is an exposed credential. If the proxy is nginx, keep
`proxy_buffering off` on `/api/events/` or the server-sent event streams arrive in bursts instead of live
(the app already sends `X-Accel-Buffering: no`, which nginx honours).

---

## Serving behind a hostname

`LBX_CORS__ORIGINS` is a comma-separated list of the browser origins allowed to call the API, defaulting to
the two localhost dev origins. This was hardcoded in `services/api/main.py`, so a deployment behind any
other hostname had every browser call blocked until somebody edited that line on the host.

```bash
LBX_CORS__ORIGINS=https://labelox.example.com
```

`*` is accepted and is deliberately not the default. This API is credentialed, and an allow-all origin on a
credentialed API is how a browser gets talked into making authenticated requests on somebody else's behalf;
setting it turns credentialed CORS off rather than combining the two, which no browser would honour anyway.

Note that the web app proxies `/api` server-side, so a standard deployment where the browser only ever
talks to the Next.js origin needs no CORS entry at all.

## What a default `up` starts

The overlay brings up, in order: `migrate` (schema + ontology, one-shot) and `pii-models` (fetches the face
and plate detector weights, one-shot), then `api`, `web`, `gpu-worker` and `govern-daemon`.

The two one-shots are ordering, not decoration. `api` waits for both to complete, because an API that
starts before the schema is migrated crash-loops, and an API that starts before the PII weights exist is an
API that accepts an ingest it cannot redact - the anonymizer refuses to construct without them by design.

Every Python service runs one image, `labeloxav/app`, built from the Dockerfile's `gpu` target: CUDA 12.8,
torch, Ultralytics, SAM and the embedding models, with the auto-label weights baked in so a lab without
internet access can still label. The CUDA wheels run on the CPU when there is no GPU, so a CPU-only host uses
the same image, more slowly.

`gpu-worker` runs every queued GPU job one at a time: auto-label jobs first, since somebody is usually
waiting on one, then training. The API only queues auto-label jobs (`LBX_JOBS__AUTOLABEL_EXECUTOR=worker`),
so a job survives an API restart and the API stays responsive while the GPU works. On start the worker
requeues whatever a crash left running; an auto-label job resumes, skipping frames already labelled, and a
job that has crashed the worker three times is failed with a message saying so. Stopping the worker
(`docker compose restart gpu-worker`) requeues its current job without counting that against it. It holds a
Postgres advisory lock for its lifetime, so a second copy exits instead of sharing the card.

`govern-daemon` is the loop's driver: it ticks the controller, which scans drift, gates a registered
challenger and schedules an off-hours retrain. It holds a Postgres advisory lock, so a second copy exits
cleanly instead of double-ticking, which is what makes `restart: unless-stopped` safe on it.

## GPU and optional services

**GPU.** The NVIDIA device reservations live in `docker-compose.gpu.yml`. The installer adds it to
`COMPOSE_FILE` only when Docker's NVIDIA runtime is present, because a reservation on a host without it stops
compose from starting anything. Installed the NVIDIA Container Toolkit later? Append
`:docker-compose.gpu.yml` to `COMPOSE_FILE` in `.env` and run `docker compose up -d`.

**Optional services.** Off by default, switched on by naming them in `COMPOSE_PROFILES` in `.env`
(comma-separated), then `docker compose up -d`:

| Profile   | Starts                | For                                                       |
|-----------|-----------------------|-----------------------------------------------------------|
| `workers` | `embed-worker`        | embedding new frames as they land: search, dedup, active-learning diversity |
| `lakefs`  | lakeFS                | dataset branches and merge requests                       |
| `viewer`  | Lichtblick            | the MCAP session viewer                                   |
| `mlflow`  | MLflow                | experiment tracking                                       |
| `extras`  | lakeFS, Lichtblick, MLflow | all three                                            |

The vision-language reviewer (auto-label's third path) talks to Ollama, which is not part of the compose
stack. Install it on the host; the containers reach it as `host.docker.internal:11434`. Ollama listens on
127.0.0.1 by default, which a container cannot reach, so set `OLLAMA_HOST=0.0.0.0:11434` (or the Docker
bridge address, usually `172.17.0.1:11434`, to keep it off the LAN) in its systemd service and restart it.
Or turn the reviewer off with `LBX_MODELS__VLM__ENABLED=false` in `.env`. `doctor` reports which.

---

## Upgrading

```bash
git pull
./scripts/install.sh          # rebuilds, migrates, restarts; secrets and data are untouched
```

Every migration in this repo has a working `downgrade`, so `alembic downgrade -1` steps back if an upgrade
misbehaves. Take a backup first regardless.

---

## Backups

```bash
make backup                                    # -> .scratch/backups/<timestamp>/
make restore DIR=.scratch/backups/<timestamp>  # destructive; asks for confirmation
```

Both halves, together, because they are one unit: Postgres holds the labels and the object store holds
every frame, mask, point cloud and export. Restoring one without the other is not a restore - the app comes
up, the counts look right, and every image 404s. `restore.sh` refuses a directory missing either half for
exactly that reason, and warns if it ends up with frames in the database and no blobs behind them.

This replaces a `make backup` that had three faults, all silent. It piped `pg_dump` into `gzip`, so the
recipe's exit status was gzip's: a dump that failed on a wrong password or a stopped container produced a
small, valid `.gz` and a green target, and you found out at restore time. It hardcoded the database name
`labeloxav`, ignoring `POSTGRES_DB`. And it *printed* the MinIO mirror command rather than running it, so
the half the warning above is about was the half nobody took.

Store the whole timestamped directory off the machine. `MANIFEST` inside it records what was taken.

---

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| Installer stops at "the API did not become ready" | `make app-logs`. Usually a dependency still starting, or a port already in use. |
| `required variable ... is missing a value` | `.env` is incomplete. Re-run the installer; it fills what is absent and leaves what is present. |
| Reads return 401 | Working as intended: authentication is deny-by-default for reads as well as writes. Sign in. |
| Download links 404 | `LBX_MINIO__PUBLIC_ENDPOINT` points somewhere the browser cannot reach. See above. |
| Ports already in use | Set `WEB_PORT`, `API_PORT`, `POSTGRES_PORT`, `MINIO_PORT` in `.env` and re-run. |
| A model path refuses instead of running | Expected without a GPU. The refusal names what is missing; it never fabricates a result. |

More in [`RUNBOOK.md`](RUNBOOK.md).

## The Debian package

Each tagged release ships `labeloxav_<version>_all.deb`. It installs the source tree under
`/opt/labeloxav` and a `labeloxav` command; the product still runs as Docker containers built from that
tree, so Docker remains a dependency. The package is a predictable way to get the tree onto a server
and drive it, not a set of native services.

```bash
sudo apt install ./labeloxav_0.1.1_all.deb
sudo labeloxav install     # first time only
labeloxav up | down | status | logs [service] | token <user> | url
```

`install` writes `.env` and `.admin-token` into `/opt/labeloxav`, which is why it needs root. Removing
the package keeps that configuration; purging removes it. The containers' data lives in Docker volumes
and is never touched by the package; the removal message says how to delete it deliberately.

The package is built by `scripts/build_deb.sh` from the committed tree at `HEAD`, never from the working
directory, so a release is reproducible from its tag. The `Release` workflow builds it, installs it into a
clean Debian container to prove the maintainer scripts and launcher work, and attaches it to the GitHub
release when a `v*` tag is pushed. The tag must match the version in `pyproject.toml`.
