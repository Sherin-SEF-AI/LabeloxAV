"""Work parked for a GPU pod said `pending`, which is the same word as work about to run.

Four subsystems park a job for the A100: autolabel, training, relabel and HD-map fusion. All four wrote
`status = "pending"` and recorded the real state somewhere else, or nowhere. The autolabel start route
answered `{"status": "queued-cloud"}`, its own docstring promised the job would "surface on the Jobs
dashboard as queued-cloud", and the jobs list translated `pending` back to `queued-cloud` at read time for
display. So the one place the answer agreed was the dashboard, and every consumer reading the database
directly, including the server-sent-events waiting totals and the top-bar activity chip, saw sixty-eight
parked jobs as ordinary queued work that some runner was about to pick up.

Nothing was going to pick them up. The executors raise NotImplementedError and the Makefile targets are echo
blocks, which is honest about needing a pod, but it makes the distinction the status column was hiding the
only one that matters: this is waiting for hardware that is not here, not waiting for a turn.

`queued-cloud` becomes the status, and this migration moves the rows that already exist. It also repairs
`map_fusion_job.stage`, which wrote `queued_cloud` with an underscore while every other site used a hyphen,
so a parked fusion job did not match the events stream's own list of waiting statuses and was counted as
nothing at all.

The downgrade puts `pending` back only where the row is still marked as a cloud job, so a job that has since
been re-run locally is not dragged backwards into a state it left.
"""

from alembic import op

revision = "0089_queued_cloud_status"
down_revision = "0088_service_account"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Autolabel records its compute target inside counts, the other three carry a column.
    op.execute("""
        UPDATE autolabel_job SET status = 'queued-cloud'
        WHERE status = 'pending' AND counts ->> 'compute_target' = 'cloud'
    """)
    for table in ("training_job", "relabel_job", "map_fusion_job"):
        op.execute(f"""
            UPDATE {table} SET status = 'queued-cloud'
            WHERE status = 'pending' AND compute_target = 'cloud'
        """)
    # The underscore spelling matched nothing that read it.
    op.execute("UPDATE map_fusion_job SET stage = 'queued-cloud' WHERE stage = 'queued_cloud'")


def downgrade() -> None:
    op.execute("""
        UPDATE autolabel_job SET status = 'pending'
        WHERE status = 'queued-cloud' AND counts ->> 'compute_target' = 'cloud'
    """)
    for table in ("training_job", "relabel_job", "map_fusion_job"):
        op.execute(f"""
            UPDATE {table} SET status = 'pending'
            WHERE status = 'queued-cloud' AND compute_target = 'cloud'
        """)
    op.execute("UPDATE map_fusion_job SET stage = 'queued_cloud' WHERE stage = 'queued-cloud'")
