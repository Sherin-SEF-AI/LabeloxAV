# LabeloxAV developer commands. Single-box edition.
# Python runs on the host via uv; infra runs in docker-compose.

SHELL := /bin/bash
UV := uv
# --no-sync: run in the existing venv without re-locking the whole project (the ml extra pulls
# torch from the PyTorch CUDA index, which we only resolve at `make install-ml` time).
RUN := $(UV) run --no-sync

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## Create the 3.11 venv and install base deps (no GPU wheels)
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"

.PHONY: install-ml
install-ml: ## Install the ml extra (torch cu128, ultralytics, transformers)
	$(UV) pip install -e ".[ml]" --index-strategy unsafe-best-match

.PHONY: up
up: ## Start infra (postgres, minio, redis, redpanda) and wait for healthy
	docker compose up -d
	@echo "waiting for services to become healthy..."
	@for i in $$(seq 1 40); do \
		unhealthy=$$(docker compose ps --format '{{.Service}} {{.Health}}' | grep -Ev 'healthy|minio-init' | grep -c . || true); \
		if [ "$$unhealthy" = "0" ]; then echo "all healthy"; break; fi; \
		sleep 3; \
	done
	docker compose ps

.PHONY: down
down: ## Stop infra (keep volumes)
	docker compose down

.PHONY: nuke
nuke: ## Stop infra and delete volumes (destroys local data)
	docker compose down -v

.PHONY: backup
backup: ## Back up Postgres (pg_dump.gz) and note the MinIO mirror command, into .scratch/backups/
	@mkdir -p .scratch/backups
	docker compose exec -T postgres pg_dump -U labelox labeloxav | gzip > .scratch/backups/pg_$$(date +%Y%m%d_%H%M%S).sql.gz
	@echo "Postgres dumped to .scratch/backups/"
	@echo "MinIO: configure an mc alias, then 'mc mirror local/labeloxav .scratch/backups/minio' to mirror blobs"

.PHONY: migrate
migrate: ## Apply Alembic migrations
	$(RUN) alembic upgrade head

.PHONY: seed
seed: ## Load and validate the ontology into Postgres
	$(RUN) python scripts/seed_ontology.py

.PHONY: health
health: ## Verify every infra dependency is reachable
	$(RUN) python scripts/healthcheck.py

.PHONY: bootstrap
bootstrap: up migrate seed health ## Full M0 bring-up: infra + schema + ontology + healthcheck

.PHONY: models
models: ## Pre-fetch perception model weights (needs ml extra)
	$(RUN) python scripts/download_models.py

.PHONY: pii-models
pii-models: ## Fetch/verify Gate A PII detector weights (YuNet face; plate optional)
	$(RUN) python scripts/download_pii_models.py

.PHONY: minio-cors
minio-cors: ## Set MinIO bucket CORS for browser direct-to-storage multipart uploads
	$(RUN) python scripts/setup_minio_cors.py

.PHONY: import
import: ## Import an external dataset. Usage: make import ARGS="--format coco --source s3://... --vehicle IMPORT-01"
	$(RUN) python -m services.imports.run $(ARGS)

.PHONY: ingest
ingest: ## Ingest a clip. Usage: make ingest ARGS="--video path.mp4 --vehicle TIGOR-07 --city BLR"
	$(RUN) python -m services.ingest.run $(ARGS)

.PHONY: label
label: ## Auto-label a session. Usage: make label ARGS="--session <uuid>"
	$(RUN) python -m services.autolabel.runner $(ARGS)

.PHONY: mine
mine: ## Scenario-mine a session. Usage: make mine ARGS="--session <uuid>"
	$(RUN) python -m services.intelligence.run $(ARGS)

.PHONY: embed
embed: ## Compute CLIP embeddings for a session. Usage: make embed ARGS="--session <uuid>"
	$(RUN) python -m services.intelligence.embeddings $(ARGS)

.PHONY: embed-frames
embed-frames: ## Compute DINOv2 frame embeddings for curation. Usage: make embed-frames ARGS="--all"
	$(RUN) python -m services.intelligence.frame_embeddings $(ARGS)

.PHONY: backfill-embeddings
backfill-embeddings: ## Backfill DINOv3 + SigLIP2 embeddings (frames + crops). Usage: make backfill-embeddings ARGS="--all"
	$(RUN) python -m scripts.backfill_embeddings $(ARGS)

.PHONY: embed-worker
embed-worker: ## Run the frame.ready embedding consumer (embeds new frames automatically)
	$(RUN) python -m services.intelligence.embed.consumer

.PHONY: cloud-perception
cloud-perception: ## Drivable segmentation on the pod, then auto-stop. Usage: make cloud-perception ARGS="--corpus --limit 200 --batches 12"
	$(RUN) python -m services.perception.cloud $(ARGS)

.PHONY: cloud-autolabel
cloud-autolabel: ## Heavy autolabel (SAM3.1+Qwen3-VL+YOLO26) on the A100 pod. Needs the pod up.
	@echo "Cloud autolabel dispatch (contract: services/autolabel/cloud.py):"
	@echo "  1) make cloud-provision   # start the pod"
	@echo "  2) push session frames + manifest to /workspace/in"
	@echo "  3) ssh pod: python cloud/autolabel_pod.py --manifest in/manifest.jsonl --out out/labels.jsonl --masks"
	@echo "  4) pull out/labels.jsonl and ingest via services.autolabel.persist"
	@echo "  5) make cloud-stop        # cap billing"

.PHONY: cloud-mapfusion
cloud-mapfusion: ## Multi-drive HD-map fusion (GTSAM) on the A100 pod. Needs the pod up. Usage: make cloud-mapfusion JOB=<id>
	@echo "Cloud map-fusion dispatch (contract: services/hdmap/cloud.py):"
	@echo "  1) make cloud-provision   # start the pod"
	@echo "  2) push per-drive map_elements (GeoJSON) + trajectories + manifest to /workspace/in"
	@echo "  3) ssh pod: python cloud/mapfusion_pod.py --manifest in/manifest.json --out out/fused.json"
	@echo "  4) pull out/fused.json; seal the map_commit + export lanelet2/opendrive via services.hdmap.run"
	@echo "  5) make cloud-stop        # cap billing"

.PHONY: cloud-relabel
cloud-relabel: ## Bulk champion-model re-inference (relabel) on the A100 pod. Needs the pod up. Usage: make cloud-relabel JOB=<id>
	@echo "Cloud relabel dispatch (contract: services/relabel/cloud.py):"
	@echo "  1) make cloud-provision   # start the pod"
	@echo "  2) push champion weights + frames + manifest to /workspace/in"
	@echo "  3) ssh pod: python cloud/relabel_pod.py --weights in/champion.pt --manifest in/manifest.jsonl --out out/relabeled.jsonl"
	@echo "  4) pull out/relabeled.jsonl; POST /api/relabel/ingest applies safe improvements on a new lakeFS branch"
	@echo "  5) make cloud-stop        # cap billing"

.PHONY: idd-inspect
idd-inspect: ## Inspect an extracted IDD dir before converting. Usage: make idd-inspect ARGS="--idd-root /data/IDD_Detection"
	$(RUN) python scripts/inspect_idd.py $(ARGS)

.PHONY: idd
idd: ## Convert IDD-Detection (VOC XML) to YOLO. Usage: make idd ARGS="--idd-root /data/IDD_Detection --out .scratch/idd_yolo"
	$(RUN) python scripts/idd_to_yolo.py $(ARGS)

.PHONY: train
train: ## Close the loop: build trainset + fine-tune + eval-gate. Usage: make train ARGS="--route-prefix 202606 --epochs 20 --idd-dir .scratch/idd_yolo"
	$(RUN) python -m services.training.finetune $(ARGS)

.PHONY: train-worker
train-worker: ## Run the training worker (drains LOCAL training jobs serially on the GPU)
	$(RUN) python -m services.training.worker

.PHONY: govern-daemon
govern-daemon: ## Run the autonomy controller daemon (ticks the closed loop: drift, promotion, retrain)
	$(RUN) python -m services.govern.daemon

.PHONY: cloud-provision
cloud-provision: ## Provision a RunPod A100, verify the heavy stack, smoke test, then stop the pod
	bash cloud/provision_runpod.sh

.PHONY: cloud-smoke
cloud-smoke: ## Run the cloud smoke test on the current pod (assumes the stack is installed)
	$(RUN) python cloud/smoke_test.py

.PHONY: gold
gold: ## Seal a gold set from human-accepted objects. Usage: make gold ARGS="--name fleet-v1 --city BLR"
	$(RUN) python -m services.training.gold $(ARGS)

.PHONY: calibrate
calibrate: ## Fit isotonic calibration from reviewed auto-labels. Usage: make calibrate ARGS="--gold <id>"
	$(RUN) python -m services.autolabel.isotonic $(ARGS)

.PHONY: m9
m9: ## Measure label quality on a gold set (per-class P/R + Safe-mIoU). Usage: make m9 ARGS="--gold <id>"
	$(RUN) python -m services.analytics.quality $(ARGS)

.PHONY: export
export: ## Seal + export a dataset. Usage: make export ARGS="--name demo --state accepted"
	$(RUN) python -m services.export.dataset $(ARGS)

.PHONY: api
api: ## Run the FastAPI backend
	$(RUN) uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload

.PHONY: web
web: ## Run the Next.js review UI
	cd web && npm install && npm run dev

.PHONY: test
test: ## Run pytest
	$(RUN) pytest -q

.PHONY: fmt
fmt: ## Format and lint
	$(RUN) ruff check --fix .
	$(RUN) ruff format .
