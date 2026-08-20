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
  device; see [GPU](#gpu) below.

No GPU is required to install. Without one, the annotation, review, governance, export, and search surfaces
all work; the model paths that need CUDA refuse rather than producing a fabricated result.

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

The deployment is the base compose file plus the application overlay. `make` wraps the pair:

```bash
make app-down     # stop, keeping all data
make app-up       # start again
make app-logs     # follow the API logs

# the same thing without make
docker compose -f docker-compose.yml -f docker-compose.app.yml up -d

# stop and DESTROY the corpus, the object store, and every annotation
docker compose -f docker-compose.yml -f docker-compose.app.yml down -v
```

Plain `docker compose up` uses only the base file and starts infrastructure alone, which is the development
workflow: services in Docker, code on the host. That is why the application lives in an overlay rather than
behind a profile: the overlay demands the deployment secrets, and requiring them in the base file would break
every compose command for a developer who has none.

### Signing in when the token is lost

Every credential is issued through the API, issuing one needs an admin token, and the bootstrap that let the
first user be created without one closes as soon as that user exists. The recovery path runs on the server,
where possession of the signing key is the authority:

```bash
make token                       # mints one for `admin`
make token NAME=alice            # or for anyone else

# the same thing without make, plus the two other modes
docker compose -f docker-compose.yml -f docker-compose.app.yml exec api \
  python -m scripts.mint_token --list

# a token leaked: invalidate every one already issued to that user, then mint a fresh one
docker compose -f docker-compose.yml -f docker-compose.app.yml exec api \
  python -m scripts.mint_token --name admin --revoke-existing
```

### Adding people

```bash
docker compose -f docker-compose.yml -f docker-compose.app.yml exec api \
  python -m scripts.mint_token --name alice --role reviewer --create
```

Roles are `annotator`, `reviewer`, `admin`, and a higher role satisfies any floor below it.

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

**TLS.** Nothing here terminates TLS, and the app is served over plain HTTP on ports 3000 and 8000. Put a
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

The overlay brings up five things, in order: `migrate` (schema + ontology, one-shot), `pii-models` (fetches
the face and plate detector weights, one-shot), then `api`, `web` and `govern-daemon`.

The two one-shots are ordering, not decoration. `api` waits for both to complete, because an API that
starts before the schema is migrated crash-loops, and an API that starts before the PII weights exist is an
API that accepts an ingest it cannot redact - the anonymizer refuses to construct without them by design,
so the first ingest on a fresh box used to fail until someone ran `make pii-models` by hand.

`govern-daemon` is the loop's driver: it ticks the controller, which scans drift, gates a registered
challenger and schedules an off-hours retrain. It holds a Postgres advisory lock, so a second copy exits
cleanly instead of double-ticking, which is what makes `restart: unless-stopped` safe on it.

## Workers and GPU

Two workers are defined but profiled off, so a CPU-only host is never asked to pull the CUDA image:

```bash
docker compose -f docker-compose.yml -f docker-compose.app.yml --profile workers up -d   # embed-worker
docker compose -f docker-compose.yml -f docker-compose.app.yml --profile gpu     up -d   # train-worker
```

`embed-worker` consumes `frame.ready` from Redpanda and embeds new frames, which is what keeps search,
dedup and active-learning diversity current. `train-worker` builds from the `gpu` target and drains local
training jobs; it needs the NVIDIA container toolkit on the host.

The scheduling model assumes one GPU: the training worker takes a Postgres advisory lock as a global mutex,
so a second worker refuses to start rather than two runs contending for the same device.

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
