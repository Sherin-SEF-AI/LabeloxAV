"""The corpus had no tenant boundary anywhere, and it gets harder to add with every million objects.

Five tables carried `project_id` and all five sit in the labelling spine (`LabelTask`, `Asset`, `Webhook`,
`StorageSource`, and `LabelProject` itself). The corpus spine, `Session -> Frame -> Object`, carried none, so
nothing scoped the frames or the labels themselves and no query could be shown to stay inside one customer's
data. `Session.pack_id` is the domain pack, not a tenant.

**Only `session` gets the column.** Frames, tracks and objects reach their tenant through `session_id`, which
every one of them already has, so three of the four tables in the spine need no column, no backfill and no
rewrite of 576,393 rows. The join is one hop and it is already indexed.

**Every existing session goes to one default project**, created here if it does not exist. That is the honest
shape of the current state: there is exactly one tenant, and saying so in a row is better than a null that
each reader interprets for itself.

Nullable, because a session ingested by a path that does not yet know about projects must not fail to
ingest. The enforcement that makes this a boundary rather than a label lives in the query layer, and while
there is one tenant it is a convention. That is the stated cost of adding the column before it is needed
rather than after, and adding it later means backfilling a corpus several times this size.
"""

import sqlalchemy as sa
from alembic import op

revision = "0092_session_project"
down_revision = "0091_job_attribution"
branch_labels = None
depends_on = None

DEFAULT_PROJECT = "default"


def upgrade() -> None:
    op.add_column("session", sa.Column("project_id", sa.UUID(), nullable=True))
    op.create_foreign_key("fk_session_project", "session", "label_project",
                          ["project_id"], ["project_id"], ondelete="SET NULL")
    op.create_index("ix_session_project", "session", ["project_id"])

    # The default project, created only if the name is free. A deployment that already has one keeps it.
    op.execute(f"""
        INSERT INTO label_project (project_id, name, description, modality, label_config,
                                   honeypot_frac, min_honeypot_accuracy)
        SELECT gen_random_uuid(), '{DEFAULT_PROJECT}',
               'Everything that existed before projects scoped the corpus', 'image', '{{}}'::jsonb, 0, 0.9
        WHERE NOT EXISTS (SELECT 1 FROM label_project WHERE name = '{DEFAULT_PROJECT}')
    """)
    op.execute(f"""
        UPDATE session SET project_id = (SELECT project_id FROM label_project WHERE name = '{DEFAULT_PROJECT}')
        WHERE project_id IS NULL
    """)


def downgrade() -> None:
    # The default project row is left behind on purpose. It may have acquired tasks, assets or webhooks of
    # its own by now, and deleting a project cascades to all of them: an undo of a column addition must not
    # take somebody's labelling work with it.
    op.drop_index("ix_session_project", table_name="session")
    op.drop_constraint("fk_session_project", "session", type_="foreignkey")
    op.drop_column("session", "project_id")
