"""Campaigns, security scene analytics, edge telemetry, and resumable exports.

Four capabilities whose ingredients all existed and whose orchestration did not.

The improvement loop had every stage built and a person between each pair of them: read the gate's per-class
deficit, build a batch, judge it, retrain, promote, decide whether to go again. A campaign is that
orchestration, with a budget and a stopping condition, because an autonomous loop that cannot stop is a way
to spend a GPU and a labelling team on a class that is not moving.

The Sec pack had a static-camera scene model and no way to say where anything was. A fixed camera's frame
does not move, which makes a polygon on it a permanent statement about the world; without zones every
detection was a detection somewhere in the picture.

FORGYX gated on bench numbers. A bench does not thermally throttle in a parked vehicle or share its GPU with
a video encoder, so artifacts have been passing on figures true in a room nobody deploys in.

Exports were all-or-nothing: a failure at 90% restarted from zero.

Revision ID: 0077_campaigns_secv2_edge
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0077_campaigns_secv2_edge"
down_revision = "0076_tracks_experiments"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ---- campaigns ------------------------------------------------------------------------------
    op.create_table(
        "campaign",
        sa.Column("campaign_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("class_name", sa.String(64), nullable=False),
        sa.Column("task_type", sa.String(32), nullable=False, server_default="detection"),
        sa.Column("target_metric", sa.String(32), nullable=False, server_default="recall"),
        sa.Column("target_value", sa.Float(), nullable=False, server_default="0.6"),
        # The hardest limit in the table: review batches are human time, and an autonomous loop must not be
        # able to commission an unbounded amount of it.
        sa.Column("label_budget", sa.Integer(), nullable=False, server_default="2000"),
        sa.Column("labels_spent", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_iterations", sa.Integer(), nullable=False, server_default="6"),
        sa.Column("patience", sa.Integer(), nullable=False, server_default="2"),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("iteration", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("stalled_iterations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("best_value", sa.Float()),
        # True by default: a loop that can promote a model with no human in it is a different product with
        # a different risk profile, and that must be opted into rather than inherited.
        sa.Column("require_approval", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("autopilot_stages", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("created_by", sa.String(64)),
        sa.Column("notes", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_campaign_name", "campaign", ["name"])
    op.create_index("ix_campaign_class", "campaign", ["class_name"])
    op.create_index("ix_campaign_status", "campaign", ["status"])

    op.create_table(
        "campaign_step",
        sa.Column("step_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("campaign_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("campaign.campaign_id", ondelete="CASCADE")),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("stage", sa.String(16), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("detail", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("metrics", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("awaiting", sa.String(64)),
        sa.Column("job_id", postgresql.UUID(as_uuid=True)),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_campaign_step_campaign", "campaign_step", ["campaign_id"])
    # The question asked when a campaign stalls: which stage stopped paying, iteration by iteration.
    op.create_index("ix_campaign_step_campaign_iter", "campaign_step", ["campaign_id", "iteration"])

    # ---- security scene analytics ---------------------------------------------------------------
    op.create_table(
        "camera_zone",
        sa.Column("zone_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("camera_id", sa.String(64), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("name", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False, server_default="area"),
        sa.Column("points", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("rule", sa.String(24), nullable=False, server_default="enter"),
        sa.Column("dwell_seconds", sa.Float()),
        sa.Column("classes", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("severity", sa.String(16), nullable=False, server_default="warn"),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("pack_id", sa.String(32), nullable=False, server_default="sec"),
        sa.Column("created_by", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_camera_zone_camera", "camera_zone", ["camera_id"])
    op.create_index("ix_camera_zone_session", "camera_zone", ["session_id"])

    op.create_table(
        "security_incident",
        sa.Column("incident_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("camera_id", sa.String(64)),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("zone_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("camera_zone.zone_id", ondelete="SET NULL")),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("severity", sa.String(16), nullable=False, server_default="warn"),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text()),
        sa.Column("start_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("end_ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("evidence", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("plate", sa.String(16)),
        sa.Column("person_identity", sa.String(64)),
        sa.Column("status", sa.String(16), nullable=False, server_default="open"),
        sa.Column("acknowledged_by", sa.String(64)),
        sa.Column("pack_id", sa.String(32), nullable=False, server_default="sec"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    for name, cols in (("ix_incident_camera", ["camera_id"]), ("ix_incident_session", ["session_id"]),
                       ("ix_incident_kind", ["kind"]), ("ix_incident_start", ["start_ts_ns"]),
                       ("ix_incident_plate", ["plate"]), ("ix_incident_identity", ["person_identity"]),
                       ("ix_incident_created", ["created_at"]),
                       ("ix_incident_camera_time", ["camera_id", "start_ts_ns"])):
        op.create_index(name, "security_incident", cols)

    op.create_table(
        "person_identity",
        sa.Column("identity_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        # No name column, and that is the design. A signature links tracks to each other and can never
        # say who anybody is; adding a name here would turn re-identification into a face database.
        sa.Column("signature", postgresql.ARRAY(sa.Float())),
        sa.Column("n_tracks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cameras", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("first_ts_ns", sa.BigInteger()),
        sa.Column("last_ts_ns", sa.BigInteger()),
        sa.Column("pack_id", sa.String(32), nullable=False, server_default="sec"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "person_sighting",
        sa.Column("sighting_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("identity_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("person_identity.identity_id", ondelete="CASCADE")),
        sa.Column("track_id", postgresql.UUID(as_uuid=True)),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("session.session_id", ondelete="CASCADE")),
        sa.Column("camera_id", sa.String(64)),
        sa.Column("ts_ns", sa.BigInteger(), nullable=False),
        sa.Column("similarity", sa.Float(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_sighting_identity", "person_sighting", ["identity_id"])
    op.create_index("ix_sighting_track", "person_sighting", ["track_id"])
    op.create_index("ix_sighting_camera", "person_sighting", ["camera_id"])

    # ---- edge telemetry -------------------------------------------------------------------------
    op.create_table(
        "edge_device",
        sa.Column("device_id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(128)),
        sa.Column("hardware", sa.String(64)),
        sa.Column("runtime", sa.String(32)),
        sa.Column("artifact_id", sa.String(128)),
        sa.Column("model_version", sa.String(64)),
        sa.Column("fleet", sa.String(64)),
        sa.Column("last_seen_at", sa.DateTime(timezone=True)),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_edge_device_artifact", "edge_device", ["artifact_id"])
    op.create_index("ix_edge_device_model", "edge_device", ["model_version"])
    op.create_index("ix_edge_device_fleet", "edge_device", ["fleet"])

    op.create_table(
        "edge_telemetry",
        sa.Column("telemetry_id", postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text("gen_random_uuid()")),
        sa.Column("device_id", sa.String(64),
                  sa.ForeignKey("edge_device.device_id", ondelete="CASCADE")),
        sa.Column("artifact_id", sa.String(128)),
        sa.Column("model_version", sa.String(64)),
        sa.Column("window_start_ns", sa.BigInteger(), nullable=False),
        sa.Column("window_end_ns", sa.BigInteger(), nullable=False),
        sa.Column("n_inferences", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_p50_ms", sa.Float()),
        sa.Column("latency_p95_ms", sa.Float()),
        sa.Column("latency_max_ms", sa.Float()),
        sa.Column("fps", sa.Float()),
        sa.Column("temp_c_max", sa.Float()),
        sa.Column("throttled_fraction", sa.Float()),
        sa.Column("power_w_mean", sa.Float()),
        # The field has no ground truth, so a moved confidence distribution is the accuracy-drift signal
        # that is actually available there.
        sa.Column("conf_histogram", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("detections_per_frame", sa.Float()),
        sa.Column("dropped_frames", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("meta", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_edge_telemetry_device", "edge_telemetry", ["device_id"])
    op.create_index("ix_edge_telemetry_artifact", "edge_telemetry", ["artifact_id"])
    op.create_index("ix_edge_telemetry_created", "edge_telemetry", ["created_at"])
    op.create_index("ix_edge_telemetry_artifact_time", "edge_telemetry", ["artifact_id", "created_at"])

    # ---- resumable exports ----------------------------------------------------------------------
    # An export was all or nothing: a failure at ninety percent restarted from zero, which on a large
    # corpus means hours. The checkpoint records what has already been written and verified.
    op.add_column("export_job", sa.Column("checkpoint", postgresql.JSONB(), nullable=False,
                                          server_default="{}"))
    op.add_column("export_job", sa.Column("resumed_from", postgresql.UUID(as_uuid=True)))


def downgrade() -> None:
    op.drop_column("export_job", "resumed_from")
    op.drop_column("export_job", "checkpoint")
    op.drop_table("edge_telemetry")
    op.drop_table("edge_device")
    op.drop_table("person_sighting")
    op.drop_table("person_identity")
    op.drop_table("security_incident")
    op.drop_table("camera_zone")
    op.drop_table("campaign_step")
    op.drop_table("campaign")
