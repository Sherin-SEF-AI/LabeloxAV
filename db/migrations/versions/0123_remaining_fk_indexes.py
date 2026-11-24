"""Index every remaining foreign key a parent delete has to search.

0122 indexed the foreign keys that point at `object`, after an unindexed `eval_patch.object_id` turned each
object delete into a scan of 36 million rows. The same table kept two more of them: `prediction_id` and
`frame_id`, both ON DELETE CASCADE. Retention erases frames one at a time, so a DPDPA erasure of a session
would have repeated that scan once per frame, and deleting an inference run cascades through `prediction`
into the same table. The rest of this list is every other single-column foreign key in the schema that had
no index leading on it; most of those tables are small today and were small when `eval_patch` was too.

Postgres checks the referencing table on a parent delete whatever the ON DELETE action is. The foreign keys
to `ontology_class` are deliberately left out: classes are retired, not deleted, and an index on
`prediction.class_id` would be paid for on every one of a million prediction inserts to speed up a delete
that does not happen. tests/test_fk_indexes.py enforces the rule for anything added later.

Built CONCURRENTLY so writes, eval runs above all, are not blocked for the length of the build.

Revision ID: 0123_remaining_fk_indexes
Revises: 0122_object_fk_indexes
"""

from alembic import op

revision = "0123_remaining_fk_indexes"
down_revision = "0122_object_fk_indexes"
branch_labels = None
depends_on = None

INDEXES = (
    ("ix_eval_patch_prediction_id_fk", "eval_patch", "prediction_id"),
    ("ix_eval_patch_frame_id_fk", "eval_patch", "frame_id"),
    ("ix_object_annotator_id_fk", "object", "annotator_id"),
    ("ix_ego_pose_run_id_fk", "ego_pose", "run_id"),
    ("ix_review_user_id_fk", "review", "user_id"),
    ("ix_scenario_candidate_session_id_fk", "scenario_candidate", "session_id"),
    ("ix_scenario_candidate_frame_id_fk", "scenario_candidate", "frame_id"),
    ("ix_error_candidate_decided_by_fk", "error_candidate", "decided_by"),
    ("ix_shadow_disagreement_task_id_fk", "shadow_disagreement", "task_id"),
    ("ix_shadow_disagreement_challenger_prediction_id_fk", "shadow_disagreement", "challenger_prediction_id"),
    ("ix_shadow_disagreement_champion_prediction_id_fk", "shadow_disagreement", "champion_prediction_id"),
    ("ix_shadow_disagreement_champion_run_id_fk", "shadow_disagreement", "champion_run_id"),
    ("ix_timeline_event_frame_id_fk", "timeline_event", "frame_id"),
    ("ix_track_event_start_frame_id_fk", "track_event", "start_frame_id"),
    ("ix_track_event_end_frame_id_fk", "track_event", "end_frame_id"),
    ("ix_occupancy_grid_run_id_fk", "occupancy_grid", "run_id"),
    ("ix_asset_session_id_fk", "asset", "session_id"),
    ("ix_asset_frame_id_fk", "asset", "frame_id"),
    ("ix_blind_audit_frame_frame_id_fk", "blind_audit_frame", "frame_id"),
    ("ix_issue_created_by_fk", "issue", "created_by"),
    ("ix_issue_resolved_by_fk", "issue", "resolved_by"),
    ("ix_annotation_created_by_fk", "annotation", "created_by"),
    ("ix_project_member_added_by_fk", "project_member", "added_by"),
    ("ix_assistant_proposal_created_by_fk", "assistant_proposal", "created_by"),
    ("ix_assistant_proposal_decided_by_fk", "assistant_proposal", "decided_by"),
    ("ix_quality_flag_3d_cloud_id_fk", "quality_flag_3d", "cloud_id"),
    ("ix_security_incident_zone_id_fk", "security_incident", "zone_id"),
    ("ix_inspector_layout_user_id_fk", "inspector_layout", "user_id"),
    ("ix_vlm_target_session_id_fk", "vlm_target", "session_id"),
    ("ix_evaluation_release_commit_fk", "evaluation", "release_commit"),
    ("ix_evaluation_gold_id_fk", "evaluation", "gold_id"),
    ("ix_benchmark_accuracy_ref_fk", "benchmark", "accuracy_ref"),
    ("ix_deployment_release_commit_fk", "deployment", "release_commit"),
    ("ix_deployment_verdict_ref_fk", "deployment", "verdict_ref"),
    ("ix_deployment_benchmark_ref_fk", "deployment", "benchmark_ref"),
    ("ix_deployment_superseded_by_fk", "deployment", "superseded_by"),
    ("ix_rig_object_session_id_fk", "rig_object", "session_id"),
    ("ix_embedding_projection_session_id_fk", "embedding_projection", "session_id"),
    ("ix_label_project_created_by_fk", "label_project", "created_by"),
    ("ix_label_task_session_id_fk", "label_task", "session_id"),
    ("ix_label_task_slice_id_fk", "label_task", "slice_id"),
    ("ix_issue_comment_author_id_fk", "issue_comment", "author_id"),
    ("ix_notification_read_user_id_fk", "notification_read", "user_id"),
    ("ix_person_sighting_session_id_fk", "person_sighting", "session_id"),
    ("ix_usage_record_user_id_fk", "usage_record", "user_id"),
    ("ix_service_account_user_id_fk", "service_account", "user_id"),
    ("ix_service_account_created_by_fk", "service_account", "created_by"),
    ("ix_job_agreement_job_a_id_fk", "job_agreement", "job_a_id"),
    ("ix_job_agreement_job_b_id_fk", "job_agreement", "job_b_id"),
    ("ix_propagation_conflict_to_frame_id_fk", "propagation_conflict", "to_frame_id"),
    ("ix_class_migration_run_id_fk", "class_migration", "run_id"),
)


def upgrade() -> None:
    # Outside a transaction: Postgres refuses CREATE INDEX CONCURRENTLY inside one.
    with op.get_context().autocommit_block():
        for name, table, column in INDEXES:
            op.execute(f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table} ({column})")


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for name, _table, _column in INDEXES:
            op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
