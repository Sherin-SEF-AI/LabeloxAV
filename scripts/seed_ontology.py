"""Load the ontology YAML, validate it, and upsert it into Postgres.

Idempotent: re-running upserts the active version's classes. Run after migrations.
    uv run python scripts/seed_ontology.py
"""

from __future__ import annotations

import asyncio

from core.config import get_settings
from core.logging import get_logger, setup_logging
from db.models import OntologyClass, OntologyVersion
from db.session import get_sessionmaker
from services.autolabel.ontology import load_ontology

log = get_logger("seed_ontology")


async def seed() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    onto = load_ontology()
    log.info("ontology.loaded", version=onto.version, classes=len(onto.classes), attributes=len(onto.attributes))

    maker = get_sessionmaker()
    async with maker() as session:
        existing = await session.get(OntologyVersion, onto.version)
        attributes_payload = {
            name: {
                "type": a.type,
                **({"values": a.values} if a.values is not None else {}),
                **({"range": list(a.range)} if a.range is not None else {}),
            }
            for name, a in onto.attributes.items()
        }

        if existing is None:
            session.add(
                OntologyVersion(
                    version=onto.version,
                    hierarchy_levels=onto.hierarchy_levels,
                    attributes=attributes_payload,
                )
            )
        else:
            existing.hierarchy_levels = onto.hierarchy_levels
            existing.attributes = attributes_payload
        await session.flush()

        # Upsert classes (merge by primary key). Additive expansion is safe even once objects
        # reference class ids, since nothing is deleted (the ontology_class PK is id; multi-version
        # storage is a later seam).
        for c in onto.classes:
            await session.merge(
                OntologyClass(
                    id=c.id, version=onto.version, name=c.name, l0=c.l0, l1=c.l1, india=c.india, map_to={}
                )
            )

        await session.commit()

    log.info("ontology.seeded", version=onto.version, classes=len(onto.classes))


def main() -> None:
    asyncio.run(seed())


if __name__ == "__main__":
    main()
