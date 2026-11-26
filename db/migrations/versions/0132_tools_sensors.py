"""Schema for the sensors tools: radar sweeps, event-camera streams, and 3D reconstructions of a drive.

- `radar_sweep` (#10). One radar scan: when, which sensor, where its arrays are (an npz in the object store,
  like a point cloud: positions in the ego frame, ego-motion compensated velocity, radial velocity, RCS), and
  the sensor -> ego transform it was placed with. A sweep belongs to a session so it lines up with that
  session's camera frames and LiDAR by ts_ns.
- `event_stream` (#10). One event camera's recording: the raw (x, y, t, polarity) events as an npz in the
  object store, the sensor size and time span, and how it was last accumulated into frames. The frames
  themselves are ordinary `frame` rows (origin real, tagged with the sensor) so the editor needs nothing new.
- `reconstruction` (#15). A session window rebuilt in 3D, by trained Gaussian splats (`gsplat`) or a
  colourised LiDAR point map (`pointmap`): which frames, what it produced (the splats, a preview cloud, the
  rendered views), and how well it fits (PSNR on held-out frames).

Revision ID: 0132_tools_sensors
Revises: 0131_tools_driving
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0132_tools_sensors"
down_revision = "0131_tools_driving"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "radar_sweep",
        sa.Column("sweep_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("sensor", sa.String(48), nullable=False),
        sa.Column("source", sa.String(24), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("n_points", sa.Integer(), nullable=False),
        sa.Column("has_velocity", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("sensor_to_ego", JSONB, nullable=True),
        sa.Column("stats", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("n_points >= 0", name="ck_radar_sweep_points"),
    )
    op.create_index("ix_radar_sweep_session_ts", "radar_sweep", ["session_id", "ts_ns"])
    op.create_index("ux_radar_sweep_sensor_ts", "radar_sweep", ["session_id", "sensor", "ts_ns"], unique=True)

    op.create_table(
        "event_stream",
        sa.Column("stream_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("sensor", sa.String(32), nullable=False),
        sa.Column("source", sa.String(24), nullable=False),
        sa.Column("uri", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("n_events", sa.BigInteger(), nullable=False),
        sa.Column("t0_ns", sa.BigInteger(), nullable=False),
        sa.Column("t1_ns", sa.BigInteger(), nullable=False),
        sa.Column("stats", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("accumulation", JSONB, nullable=True),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("t1_ns >= t0_ns", name="ck_event_stream_span"),
    )
    op.create_index("ix_event_stream_session", "event_stream", ["session_id"])

    op.create_table(
        "reconstruction",
        sa.Column("recon_id", UUID(as_uuid=True), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("session.session_id", ondelete="CASCADE"),
                  nullable=False),
        sa.Column("method", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("cam_id", sa.String(32), nullable=True),
        sa.Column("first_ts_ns", sa.BigInteger(), nullable=True),
        sa.Column("last_ts_ns", sa.BigInteger(), nullable=True),
        sa.Column("params", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("artifacts", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("views", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("metrics", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("job_id", UUID(as_uuid=True), sa.ForeignKey("tool_job.job_id", ondelete="SET NULL"), nullable=True),
        sa.Column("created_by", UUID(as_uuid=True), sa.ForeignKey("app_user.user_id", ondelete="SET NULL"),
                  nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("method in ('gsplat','pointmap')", name="ck_reconstruction_method"),
        sa.CheckConstraint("status in ('pending','running','done','error','canceled')",
                           name="ck_reconstruction_status"),
    )
    op.create_index("ix_reconstruction_session", "reconstruction", ["session_id", "created_at"])
    op.create_index("ix_reconstruction_job", "reconstruction", ["job_id"])


def downgrade() -> None:
    op.drop_index("ix_reconstruction_job", table_name="reconstruction")
    op.drop_index("ix_reconstruction_session", table_name="reconstruction")
    op.drop_table("reconstruction")
    op.drop_index("ix_event_stream_session", table_name="event_stream")
    op.drop_table("event_stream")
    op.drop_index("ux_radar_sweep_sensor_ts", table_name="radar_sweep")
    op.drop_index("ix_radar_sweep_session_ts", table_name="radar_sweep")
    op.drop_table("radar_sweep")
