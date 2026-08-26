<p align="center">
  <img src="docs/logo.png" alt="LabeloxAV" width="420">
</p>

<p align="center"><b>A data engine for autonomous driving, built for Indian roads.</b></p>

---

LabeloxAV takes raw fleet footage, machine-labels it through a three-path fusion pipeline, routes every
label through a confidence gate to human review, mines the rare and risky moments, and retrains its own
models in a closed loop. One ontology, 178 governed classes, tuned for what global datasets never saw:
autorickshaws, cattle on the carriageway, overloaded two-wheelers, hand carts, potholes.

<img width="1920" alt="Home dashboard" src="https://github.com/user-attachments/assets/b68e1a19-94fe-4e9c-a5a0-5b47599e356f" />

## Install

One command on any machine with Docker:

```bash
git clone https://github.com/Sherin-SEF-AI/LabeloxAV.git
cd LabeloxAV
./scripts/install.sh
```

Generates secrets, migrates the schema, seeds the ontology, creates the first admin, and prints the token.
Open `http://localhost:3000`. No GPU needed to install: annotation, review, governance, export, and search
all work without one; the model paths that need CUDA refuse rather than fabricate. GPU, TLS, and backups:
[docs/DEPLOY.md](docs/DEPLOY.md).

## What actually runs

| Path | Role | Model in this build |
|---|---|---|
| `path_a_detect` | Closed-set detector | `yolo11l.pt` (target: YOLO26; weights swap by config) |
| `path_b_openvocab` | Open-vocabulary + segmentation | YOLO-World + `sam_b.pt` |
| `path_c_vlm` | VLM verifier | `qwen2.5vl:7b` via Ollama |

The filenames say what runs today; the identity strings in stored provenance keep their historical
spellings. Fused proposals are calibrated (isotonic, fit against a judge whose own error is measured and
corrected for), then gated: `auto_accept` at 0.45 / safety classes at 0.47 on a calibrated scale.
**Those thresholds are configured constants, not measured precision floors** - a per-class fit replaces
them where one exists, and the gate logs which it used.

## Honest numbers (2026-08-26)

- 716k objects across 377 sessions; **~1,900 human-verified** - every downstream number inherits that limit.
- Per-class label precision, measured by a calibrated VLM judge on random 80-crop samples: `pedestrian`
  0.85, `motorcycle` 0.81 ... `traffic_signal` **0.02**. Full table:
  [reports/class_precision_2026-08-26.json](reports/class_precision_2026-08-26.json).
- The pooled auto-accepted subset judges at 0.93 strict - machine-judged, not yet measured against humans.
- A blind capture-recapture audit is seeded and unscored, so recall numbers are against labels somebody
  already found, and the coverage datasheet shipped with every export says so.

## Documentation

**[sherin-sef-ai.github.io/LabeloxAV](https://sherin-sef-ai.github.io/LabeloxAV/)** - full docs, an
interactive REST reference generated from the running app, and a Python reference for the stable seams.

The full history of what was built, measured, broken, and fixed - including everything that did not work -
is the [engineering log](docs/ENGINEERING_LOG.md).

## Develop

```bash
make up            # infrastructure: Postgres, MinIO, Redis, Redpanda, lakeFS
make install       # deps + migrations + ontology seed
make api           # FastAPI backend on :8000
make web           # Next.js frontend on :3000
make test-unit     # fast tier - green on a fresh clone, no GPU/infra needed
```

Python 3.11, FastAPI + SQLAlchemy async, Postgres 16 + PostGIS + pgvector, Next.js 14. Import contracts
enforced by `lint-imports`; domain logic lives in swappable packs (`packs/av`, `packs/sec`).

## Author

**Sherin Joseph Roy** - building an India-native, self-improving data engine for autonomous driving.

## License

Copyright (c) 2026 Sherin Joseph Roy. All rights reserved. (A proper open-source license is under
consideration; until one lands, this code is source-visible but not licensed for reuse.)
