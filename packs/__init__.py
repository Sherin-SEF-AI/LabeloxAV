"""Domain packs.

A pack is the domain-specific half of LabeloxAV: its ontology, safety definition, auto-label profile,
eval strata, quality profile, forge targets, privacy plane, and (from SEC-M2) its scene model and
ingestion adapters. The engine core in `core/`, `services/`, and `db/` holds the domain-neutral kernels
and calls into a pack only through the contract in `packs.base`, resolved by `packs.registry`.

Import direction is one-way and enforced in CI (see `.importlinter`): a pack may import the engine, the
engine may not statically import a concrete pack (`packs.av`, `packs.sec`, ...). The only bridge is
`packs.registry`, which discovers and dynamically loads concrete packs.
"""
