"""Ontology evolution recorded rather than performed by hand: parentage, changelog, and class migrations.

An ontology changes. A class turns out to be two things, two classes turn out to be one, a name was wrong.
Today the only trace of any of that is a new `ontology_version` row and whatever somebody remembers: there
is no record of which class became which, so a model trained under one version and a gold set sealed under
another can only be compared by a person who knows the history.

Three additions, and one deliberate non-addition.

`ontology_version.parent_version` makes the versions a chain rather than a set, so "the latest version
descended from the one this gold set was sealed under" is a query. `changelog` holds what the bump was for.

`class_migration` records each individual change between two versions: which class became which, under
which kind (split, merge, rename, retire), and the rule that decided membership for a split. A split's
rule is the only part that cannot be inferred from the rows afterwards, which is exactly why it is stored.

**The primary key of `ontology_class` is deliberately left alone.** The plan for this program called for
it to become `(version, id)`. That would let one id mean different things in two versions, and it would
also force a version column onto all eight tables that reference a class, including `object`, whose
578,436 rows carry a `class_id` that every existing query reads without one. A `class_id` that is only
meaningful alongside a version is a schema where every one of those queries is silently wrong. Instead the
id stays globally unique, which it already is in practice (the two versions in this corpus occupy 1 to 200
and 209 to 225 with no overlap), and a UNIQUE(version, id) makes that a rule rather than a convention.

Revision ID: 0114_ontology_versions
Revises: 0113_occupancy
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0114_ontology_versions"
down_revision = "0113_occupancy"
branch_labels = None
depends_on = None

KINDS = ("split", "merge", "rename", "retire")


def upgrade() -> None:
    op.add_column("ontology_version", sa.Column("parent_version", sa.String(64), nullable=True))
    op.add_column("ontology_version", sa.Column("changelog", postgresql.JSONB(), nullable=False,
                                                server_default=sa.text("'{}'::jsonb")))
    # A rule rather than a convention: an id already unique across versions in this corpus stays that way.
    op.create_unique_constraint("uq_ontology_class_version_id", "ontology_class", ["version", "id"])

    op.create_table(
        "class_migration",
        sa.Column("migration_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("from_version", sa.String(64), nullable=False),
        sa.Column("to_version", sa.String(64), nullable=False),
        sa.Column("from_id", sa.Integer(), nullable=True),
        sa.Column("to_id", sa.Integer(), nullable=True),
        sa.Column("kind", sa.String(16), nullable=False),
        # For a split: the predicate that decided which side an object went to. For the others it is
        # empty, because a rename or a retire needs no rule to be replayable.
        sa.Column("rule", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("run_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("agent_run.run_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_by", sa.String(64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("kind in ('" + "','".join(KINDS) + "')", name="ck_class_migration_kind"),
        # A migration that names neither end names nothing. A retire has no `to_id` and a split from
        # nowhere is not a split, so at least one side must be present.
        sa.CheckConstraint("from_id is not null or to_id is not null",
                           name="ck_class_migration_has_an_end"),
        sa.CheckConstraint("from_version <> to_version", name="ck_class_migration_across_versions"),
    )
    op.create_index("ix_class_migration_from", "class_migration", ["from_version", "from_id"])
    op.create_index("ix_class_migration_to", "class_migration", ["to_version", "to_id"])


def downgrade() -> None:
    op.drop_index("ix_class_migration_to", table_name="class_migration")
    op.drop_index("ix_class_migration_from", table_name="class_migration")
    op.drop_table("class_migration")
    op.drop_constraint("uq_ontology_class_version_id", "ontology_class", type_="unique")
    op.drop_column("ontology_version", "changelog")
    op.drop_column("ontology_version", "parent_version")
