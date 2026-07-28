# LabeloxAV API server image.
#
# Two build targets:
#   - `api`  (default): the serving layer on a slim CPU base. Runs the FastAPI app, DB, MinIO, and every
#     non-GPU route. The autolabel/embedding/training paths that need CUDA are not exercised here; those run
#     on the GPU worker (target `gpu`), matching the single-box-or-split deployment the app supports.
#   - `gpu`: the full stack on a CUDA base with the ml extra, for the box that actually runs inference.
#
# Build:   docker build -t labeloxav-api .                     (api, CPU)
#          docker build -t labeloxav-gpu --target gpu .        (full, CUDA)

# ---------- api (CPU serving layer) ----------
FROM python:3.11-slim AS api

# libGL + glib for opencv (imported widely), curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app
# Install base deps first (cached until pyproject changes), then the source.
COPY pyproject.toml README.md ./
# --no-cache: uv otherwise leaves its downloaded wheels in /root/.cache, which was 2.3GB of an
# image that never reads them again.
RUN uv venv && uv pip install --no-cache -e "."
COPY . .

# PYTHONPATH: running a script by path (`python scripts/x.py`) puts scripts/ on sys.path rather than the
# working directory, so `import core` fails even though the package is right there. Setting it explicitly
# makes every entry form work: -m, by path, and an interactive shell.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    LBX_ENV=production

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- gpu (full stack, CUDA) ----------
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS gpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml README.md ./
# The ml extra pins the cu128 wheels this base matches.
RUN uv venv --python 3.11 && uv pip install --no-cache -e ".[ml]"
COPY . .
ENV PATH="/app/.venv/bin:$PATH" PYTHONPATH=/app PYTHONUNBUFFERED=1 LBX_ENV=production
EXPOSE 8000
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
