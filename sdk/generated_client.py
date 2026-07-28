"""LabeloxAV Python client, generated from the API's OpenAPI schema.

Do not edit by hand. Regenerate with:

    python -m scripts.generate_sdk --out sdk/generated_client.py

Every method here corresponds to one route on the server, with its real path, method and parameters. The
hand-written client in `labelox_client.py` remains for the ergonomic helpers that compose several calls;
this is the complete, always-current surface underneath it.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class LabeloxError(RuntimeError):
    """An API call failed. Carries the status and the server's own message."""

    def __init__(self, status: int, method: str, path: str, detail: str) -> None:
        super().__init__(f"{method} {path} -> {status}: {detail}")
        self.status, self.method, self.path, self.detail = status, method, path, detail


class LabeloxClient:
    """A thin, complete client over the REST API.

    The token is required rather than optional. Reads are deny-by-default on the server, so a client
    constructed without one fails on its first call with a 401 that looks like a server problem; refusing
    at construction says what is actually wrong.
    """

    def __init__(self, base_url: str | None = None, token: str | None = None,
                 timeout: float = 60.0) -> None:
        self.base_url = (base_url or os.environ.get("LABELOX_URL")
                         or "http://localhost:8000").rstrip("/")
        self.token = token or os.environ.get("LABELOX_TOKEN")
        if not self.token:
            raise LabeloxError(0, "INIT", "", (
                "no token. Pass token= or set LABELOX_TOKEN; reads are deny-by-default on the server, so "
                "an unauthenticated client fails on its first call with a 401 that looks like an outage."))
        self._client = httpx.Client(timeout=timeout,
                                    headers={"Authorization": f"Bearer {self.token}"})

    def __enter__(self) -> LabeloxClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _call(self, method: str, path: str, *, params: dict | None = None,
              json_body: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        r = self._client.request(method, url, params=_clean(params), json=json_body)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:  # noqa: BLE001
                detail = r.text[:400]
            raise LabeloxError(r.status_code, method, path, str(detail))
        if not r.content:
            return None
        try:
            return r.json()
        except Exception:  # noqa: BLE001 - a binary body (a crop, an export) is returned as bytes
            return r.content

    # ---- generated methods ----

    def get_activelearn_batches(self) -> Any:
        """Batches"""
        return self._call("GET", f"/api/activelearn/batches",
                          params=None, json_body=None)

    def post_activelearn_champion_sweep(self, session_id: Any | None = None, conf_floor: float | None = 0.05, accept_threshold: float | None = 0.5, limit: int | None = 500, top_k: int | None = 100) -> Any:
        """Champion Sweep"""
        return self._call("POST", f"/api/activelearn/champion-sweep",
                          params={"session_id": session_id, "conf_floor": conf_floor, "accept_threshold": accept_threshold, "limit": limit, "top_k": top_k}, json_body=None)

    def get_activelearn_false_negatives(self, session_id: Any | None = None, limit: int | None = 4000, top_k: int | None = 200, accept_threshold: float | None = 0.5) -> Any:
        """False Negatives"""
        return self._call("GET", f"/api/activelearn/false-negatives",
                          params={"session_id": session_id, "limit": limit, "top_k": top_k, "accept_threshold": accept_threshold}, json_body=None)

    def get_activelearn_loop(self) -> Any:
        """Loop Status"""
        return self._call("GET", f"/api/activelearn/loop",
                          params=None, json_body=None)

    def post_activelearn_loop_retrain(self, body: Any = None) -> Any:
        """Loop Retrain"""
        return self._call("POST", f"/api/activelearn/loop/retrain",
                          params=None, json_body=body)

    def post_activelearn_recall_reliability(self, apply: bool | None = True, min_verdicts: int | None = 20) -> Any:
        """Recall Reliability"""
        return self._call("POST", f"/api/activelearn/recall-reliability",
                          params={"apply": apply, "min_verdicts": min_verdicts}, json_body=None)

    def get_activelearn_score(self, session_id: Any | None = None, limit: int | None = 50) -> Any:
        """Score"""
        return self._call("GET", f"/api/activelearn/score",
                          params={"session_id": session_id, "limit": limit}, json_body=None)

    def post_activelearn_select(self, body: Any = None) -> Any:
        """Select Route"""
        return self._call("POST", f"/api/activelearn/select",
                          params=None, json_body=body)

    def get_activity(self, user_id: Any | None = None, verb: Any | None = None, since_hours: Any | None = None, mine: bool | None = False, limit: int | None = 100, offset: int | None = 0) -> Any:
        """List Activity"""
        return self._call("GET", f"/api/activity",
                          params={"user_id": user_id, "verb": verb, "since_hours": since_hours, "mine": mine, "limit": limit, "offset": offset}, json_body=None)

    def get_activity_summary(self, hours: int | None = 24, mine: bool | None = True) -> Any:
        """Activity Summary"""
        return self._call("GET", f"/api/activity/summary",
                          params={"hours": hours, "mine": mine}, json_body=None)

    def delete_adverse_by_region_id(self, region_id: str) -> Any:
        """Delete Adverse"""
        return self._call("DELETE", f"/api/adverse/{region_id}",
                          params=None, json_body=None)

    def post_agent_ask(self, body: Any = None) -> Any:
        """Ask"""
        return self._call("POST", f"/api/agent/ask",
                          params=None, json_body=body)

    def get_agent_audit_latest(self) -> Any:
        """Audit Latest"""
        return self._call("GET", f"/api/agent/audit/latest",
                          params=None, json_body=None)

    def post_agent_audit_run(self, body: Any = None) -> Any:
        """Audit Run"""
        return self._call("POST", f"/api/agent/audit/run",
                          params=None, json_body=body)

    def post_agent_buyer_spec(self, body: Any = None) -> Any:
        """Buyer Spec"""
        return self._call("POST", f"/api/agent/buyer/spec",
                          params=None, json_body=body)

    def post_agent_cleanup_sweep(self, body: Any = None) -> Any:
        """Cleanup Sweep"""
        return self._call("POST", f"/api/agent/cleanup-sweep",
                          params=None, json_body=body)

    def post_agent_command(self, body: Any = None) -> Any:
        """Command"""
        return self._call("POST", f"/api/agent/command",
                          params=None, json_body=body)

    def post_agent_copilot_batch_fix(self, body: Any = None) -> Any:
        """Copilot Batch Fix"""
        return self._call("POST", f"/api/agent/copilot/batch-fix",
                          params=None, json_body=body)

    def get_agent_copilot_pattern(self, mine: bool | None = True) -> Any:
        """Copilot Pattern"""
        return self._call("GET", f"/api/agent/copilot/pattern",
                          params={"mine": mine}, json_body=None)

    def get_agent_coverage(self) -> Any:
        """Coverage"""
        return self._call("GET", f"/api/agent/coverage",
                          params=None, json_body=None)

    def post_agent_disagreements_mine(self, body: Any = None) -> Any:
        """Disagreements Mine"""
        return self._call("POST", f"/api/agent/disagreements/mine",
                          params=None, json_body=body)

    def post_agent_docs_datasheet(self, body: Any = None) -> Any:
        """Docs Datasheet"""
        return self._call("POST", f"/api/agent/docs/datasheet",
                          params=None, json_body=body)

    def post_agent_docs_model_card(self, model_version: str) -> Any:
        """Docs Model Card"""
        return self._call("POST", f"/api/agent/docs/model-card",
                          params={"model_version": model_version}, json_body=None)

    def post_agent_docs_weekly(self) -> Any:
        """Docs Weekly"""
        return self._call("POST", f"/api/agent/docs/weekly",
                          params=None, json_body=None)

    def post_agent_drift_investigate(self, body: Any = None) -> Any:
        """Drift Investigate"""
        return self._call("POST", f"/api/agent/drift/investigate",
                          params=None, json_body=body)

    def get_agent_drift_latest(self) -> Any:
        """Drift Latest"""
        return self._call("GET", f"/api/agent/drift/latest",
                          params=None, json_body=None)

    def get_agent_errors_queue(self, status: str | None = 'pending', limit: int | None = 100) -> Any:
        """Error Queue"""
        return self._call("GET", f"/api/agent/errors/queue",
                          params={"status": status, "limit": limit}, json_body=None)

    def post_agent_errors_sweep(self, body: Any = None) -> Any:
        """Error Sweep"""
        return self._call("POST", f"/api/agent/errors/sweep",
                          params=None, json_body=body)

    def get_agent_fleet_orders(self, status: str | None = 'proposed') -> Any:
        """Fleet Orders"""
        return self._call("GET", f"/api/agent/fleet/orders",
                          params={"status": status}, json_body=None)

    def post_agent_fleet_orders_by_order_id_by_status(self, order_id: str, status: str) -> Any:
        """Fleet Order Status"""
        return self._call("POST", f"/api/agent/fleet/orders/{order_id}/{status}",
                          params=None, json_body=None)

    def post_agent_fleet_plan(self) -> Any:
        """Fleet Plan"""
        return self._call("POST", f"/api/agent/fleet/plan",
                          params=None, json_body=None)

    def post_agent_flywheel(self, body: Any = None) -> Any:
        """Flywheel"""
        return self._call("POST", f"/api/agent/flywheel",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_attributes(self, frame_id: str) -> Any:
        """Attributes Run"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/attributes",
                          params=None, json_body=None)

    def post_agent_frames_by_frame_id_attributes_plan(self, frame_id: str) -> Any:
        """Attributes Plan"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/attributes/plan",
                          params=None, json_body=None)

    def post_agent_frames_by_frame_id_cuboids(self, frame_id: str, body: Any = None) -> Any:
        """Cuboids Run"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/cuboids",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_cuboids_plan(self, frame_id: str, body: Any = None) -> Any:
        """Cuboids Plan"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/cuboids/plan",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_fresh(self, frame_id: str, body: Any = None) -> Any:
        """Fresh"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/fresh",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_plan(self, frame_id: str, body: Any = None) -> Any:
        """Plan"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/plan",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_reconcile(self, frame_id: str, body: Any = None) -> Any:
        """Reconcile"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/reconcile",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_relabel(self, frame_id: str, body: Any = None) -> Any:
        """Relabel Frame"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/relabel",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_relabel_plan(self, frame_id: str, body: Any = None) -> Any:
        """Relabel Plan"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/relabel/plan",
                          params=None, json_body=body)

    def post_agent_frames_by_frame_id_run(self, frame_id: str, body: Any = None) -> Any:
        """Run"""
        return self._call("POST", f"/api/agent/frames/{frame_id}/run",
                          params=None, json_body=body)

    def get_agent_frames_by_frame_id_suggest(self, frame_id: str) -> Any:
        """Suggest"""
        return self._call("GET", f"/api/agent/frames/{frame_id}/suggest",
                          params=None, json_body=None)

    def post_agent_gold_drift(self) -> Any:
        """Gold Drift"""
        return self._call("POST", f"/api/agent/gold-drift",
                          params=None, json_body=None)

    def post_agent_nl_edit_apply(self, body: Any = None) -> Any:
        """Nl Edit Apply"""
        return self._call("POST", f"/api/agent/nl-edit/apply",
                          params=None, json_body=body)

    def post_agent_nl_edit_preview(self, body: Any = None) -> Any:
        """Nl Edit Preview"""
        return self._call("POST", f"/api/agent/nl-edit/preview",
                          params=None, json_body=body)

    def post_agent_objects_by_object_id_crosscam(self, object_id: str, body: Any = None) -> Any:
        """Crosscam Run"""
        return self._call("POST", f"/api/agent/objects/{object_id}/crosscam",
                          params=None, json_body=body)

    def post_agent_objects_by_object_id_crosscam_plan(self, object_id: str, body: Any = None) -> Any:
        """Crosscam Plan"""
        return self._call("POST", f"/api/agent/objects/{object_id}/crosscam/plan",
                          params=None, json_body=body)

    def post_agent_objects_by_object_id_propagate(self, object_id: str, body: Any = None) -> Any:
        """Propagate Run"""
        return self._call("POST", f"/api/agent/objects/{object_id}/propagate",
                          params=None, json_body=body)

    def post_agent_objects_by_object_id_propagate_plan(self, object_id: str, body: Any = None) -> Any:
        """Propagate Plan"""
        return self._call("POST", f"/api/agent/objects/{object_id}/propagate/plan",
                          params=None, json_body=body)

    def get_agent_ontology_proposals(self, status: str | None = 'proposed') -> Any:
        """Ontology Proposals"""
        return self._call("GET", f"/api/agent/ontology/proposals",
                          params={"status": status}, json_body=None)

    def post_agent_ontology_proposals_by_proposal_id_approve(self, proposal_id: str, body: Any = None) -> Any:
        """Ontology Approve"""
        return self._call("POST", f"/api/agent/ontology/proposals/{proposal_id}/approve",
                          params=None, json_body=body)

    def post_agent_ontology_proposals_by_proposal_id_reject(self, proposal_id: str) -> Any:
        """Ontology Reject"""
        return self._call("POST", f"/api/agent/ontology/proposals/{proposal_id}/reject",
                          params=None, json_body=None)

    def post_agent_ontology_scan(self, body: Any = None) -> Any:
        """Ontology Scan"""
        return self._call("POST", f"/api/agent/ontology/scan",
                          params=None, json_body=body)

    def post_agent_ops_ask(self, body: Any = None) -> Any:
        """Ops Ask"""
        return self._call("POST", f"/api/agent/ops/ask",
                          params=None, json_body=body)

    def post_agent_relabel_all(self, body: Any = None) -> Any:
        """Relabel All"""
        return self._call("POST", f"/api/agent/relabel/all",
                          params=None, json_body=body)

    def get_agent_report(self) -> Any:
        """Report"""
        return self._call("GET", f"/api/agent/report",
                          params=None, json_body=None)

    def get_agent_runs(self, limit: int | None = 50) -> Any:
        """Runs"""
        return self._call("GET", f"/api/agent/runs",
                          params={"limit": limit}, json_body=None)

    def get_agent_runs_by_run_id(self, run_id: str) -> Any:
        """Run Detail"""
        return self._call("GET", f"/api/agent/runs/{run_id}",
                          params=None, json_body=None)

    def post_agent_runs_by_run_id_revert(self, run_id: str) -> Any:
        """Revert"""
        return self._call("POST", f"/api/agent/runs/{run_id}/revert",
                          params=None, json_body=None)

    def post_agent_scenarios_mine(self, body: Any = None) -> Any:
        """Scenarios Mine"""
        return self._call("POST", f"/api/agent/scenarios/mine",
                          params=None, json_body=body)

    def post_agent_temporal_repair(self, body: Any = None) -> Any:
        """Temporal Repair Run"""
        return self._call("POST", f"/api/agent/temporal-repair",
                          params=None, json_body=body)

    def post_agent_temporal_repair_plan(self, body: Any = None) -> Any:
        """Temporal Repair Plan"""
        return self._call("POST", f"/api/agent/temporal-repair/plan",
                          params=None, json_body=body)

    def post_agent_training_cycle(self, body: Any = None) -> Any:
        """Training Cycle"""
        return self._call("POST", f"/api/agent/training/cycle",
                          params=None, json_body=body)

    def get_analytics_classes(self, session_id: Any | None = None) -> Any:
        """Classes"""
        return self._call("GET", f"/api/analytics/classes",
                          params={"session_id": session_id}, json_body=None)

    def get_analytics_cluster_map(self, limit: int | None = 1500) -> Any:
        """Cluster Map"""
        return self._call("GET", f"/api/analytics/cluster-map",
                          params={"limit": limit}, json_body=None)

    def get_analytics_dedup_rate(self, session_id: Any | None = None) -> Any:
        """Dedup Rate"""
        return self._call("GET", f"/api/analytics/dedup-rate",
                          params={"session_id": session_id}, json_body=None)

    def get_analytics_geo(self, session_id: Any | None = None, limit: int | None = 2000) -> Any:
        """Geo"""
        return self._call("GET", f"/api/analytics/geo",
                          params={"session_id": session_id, "limit": limit}, json_body=None)

    def get_analytics_growth(self) -> Any:
        """Growth"""
        return self._call("GET", f"/api/analytics/growth",
                          params=None, json_body=None)

    def get_analytics_overview(self, session_id: Any | None = None) -> Any:
        """Overview"""
        return self._call("GET", f"/api/analytics/overview",
                          params={"session_id": session_id}, json_body=None)

    def get_analytics_pii(self, session_id: Any | None = None) -> Any:
        """Pii"""
        return self._call("GET", f"/api/analytics/pii",
                          params={"session_id": session_id}, json_body=None)

    def get_analytics_productivity(self) -> Any:
        """Productivity"""
        return self._call("GET", f"/api/analytics/productivity",
                          params=None, json_body=None)

    def get_analytics_report(self, session_id: Any | None = None) -> Any:
        """Report"""
        return self._call("GET", f"/api/analytics/report",
                          params={"session_id": session_id}, json_body=None)

    def get_analytics_review_agreement(self) -> Any:
        """Review Agreement"""
        return self._call("GET", f"/api/analytics/review-agreement",
                          params=None, json_body=None)

    def get_analytics_scenarios(self, session_id: Any | None = None) -> Any:
        """Scenarios"""
        return self._call("GET", f"/api/analytics/scenarios",
                          params={"session_id": session_id}, json_body=None)

    def get_analytics_scene_splits(self, session_id: Any | None = None) -> Any:
        """Scene Splits"""
        return self._call("GET", f"/api/analytics/scene-splits",
                          params={"session_id": session_id}, json_body=None)

    def get_analytics_source_mix(self, session_id: Any | None = None) -> Any:
        """Source Mix"""
        return self._call("GET", f"/api/analytics/source-mix",
                          params={"session_id": session_id}, json_body=None)

    def delete_annotations_by_annotation_id(self, annotation_id: str) -> Any:
        """Delete Annotation"""
        return self._call("DELETE", f"/api/annotations/{annotation_id}",
                          params=None, json_body=None)

    def patch_annotations_by_annotation_id(self, annotation_id: str, body: Any = None) -> Any:
        """Update Annotation"""
        return self._call("PATCH", f"/api/annotations/{annotation_id}",
                          params=None, json_body=body)

    def post_assets(self, body: Any = None) -> Any:
        """Create Assets"""
        return self._call("POST", f"/api/assets",
                          params=None, json_body=body)

    def get_assets_kinds(self) -> Any:
        """Kinds"""
        return self._call("GET", f"/api/assets/kinds",
                          params=None, json_body=None)

    def get_assets_by_asset_id(self, asset_id: str) -> Any:
        """Get Asset"""
        return self._call("GET", f"/api/assets/{asset_id}",
                          params=None, json_body=None)

    def get_assets_by_asset_id_annotations(self, asset_id: str, kind: Any | None = None) -> Any:
        """List Annotations"""
        return self._call("GET", f"/api/assets/{asset_id}/annotations",
                          params={"kind": kind}, json_body=None)

    def post_assets_by_asset_id_annotations(self, asset_id: str, body: Any = None) -> Any:
        """Create Annotation"""
        return self._call("POST", f"/api/assets/{asset_id}/annotations",
                          params=None, json_body=body)

    def post_assets_by_asset_id_state(self, asset_id: str, state: str) -> Any:
        """Set Asset State"""
        return self._call("POST", f"/api/assets/{asset_id}/state",
                          params={"state": state}, json_body=None)

    def post_auth_login(self, body: Any = None) -> Any:
        """Login"""
        return self._call("POST", f"/api/auth/login",
                          params=None, json_body=body)

    def post_auth_login_mfa(self, body: Any = None) -> Any:
        """Login Mfa"""
        return self._call("POST", f"/api/auth/login/mfa",
                          params=None, json_body=body)

    def post_auth_logout(self) -> Any:
        """Logout"""
        return self._call("POST", f"/api/auth/logout",
                          params=None, json_body=None)

    def post_auth_media_token(self) -> Any:
        """Media Token"""
        return self._call("POST", f"/api/auth/media-token",
                          params=None, json_body=None)

    def get_auth_methods(self) -> Any:
        """Auth Methods"""
        return self._call("GET", f"/api/auth/methods",
                          params=None, json_body=None)

    def post_auth_mfa_confirm(self, body: Any = None) -> Any:
        """Mfa Confirm"""
        return self._call("POST", f"/api/auth/mfa/confirm",
                          params=None, json_body=body)

    def post_auth_mfa_disable(self, body: Any = None) -> Any:
        """Mfa Disable"""
        return self._call("POST", f"/api/auth/mfa/disable",
                          params=None, json_body=body)

    def post_auth_mfa_setup(self) -> Any:
        """Mfa Setup"""
        return self._call("POST", f"/api/auth/mfa/setup",
                          params=None, json_body=None)

    def post_auth_mfa_setup_pending(self, body: Any = None) -> Any:
        """Mfa Setup Pending"""
        return self._call("POST", f"/api/auth/mfa/setup-pending",
                          params=None, json_body=body)

    def post_auth_password_change(self, body: Any = None) -> Any:
        """Change Password"""
        return self._call("POST", f"/api/auth/password/change",
                          params=None, json_body=body)

    def post_auth_password_reset(self, body: Any = None) -> Any:
        """Reset"""
        return self._call("POST", f"/api/auth/password/reset",
                          params=None, json_body=body)

    def post_auth_password_reset_request(self, body: Any = None) -> Any:
        """Reset Request"""
        return self._call("POST", f"/api/auth/password/reset-request",
                          params=None, json_body=body)

    def get_auth_profile(self) -> Any:
        """Profile"""
        return self._call("GET", f"/api/auth/profile",
                          params=None, json_body=None)

    def post_auth_refresh(self) -> Any:
        """Refresh"""
        return self._call("POST", f"/api/auth/refresh",
                          params=None, json_body=None)

    def post_auth_sessions_revoke(self) -> Any:
        """Revoke Sessions"""
        return self._call("POST", f"/api/auth/sessions/revoke",
                          params=None, json_body=None)

    def post_auth_signup(self, body: Any = None) -> Any:
        """Signup"""
        return self._call("POST", f"/api/auth/signup",
                          params=None, json_body=body)

    def post_autolabel_ego_masks_estimate(self, force: bool | None = False) -> Any:
        """Estimate Ego Masks"""
        return self._call("POST", f"/api/autolabel/ego-masks/estimate",
                          params={"force": force}, json_body=None)

    def post_autolabel_pii_backfill(self, limit: int | None = 2000, session_id: Any | None = None) -> Any:
        """Pii Backfill"""
        return self._call("POST", f"/api/autolabel/pii-backfill",
                          params={"limit": limit, "session_id": session_id}, json_body=None)

    def post_autolabel_redetect_all(self, backfill_pii: bool | None = True) -> Any:
        """Redetect All"""
        return self._call("POST", f"/api/autolabel/redetect-all",
                          params={"backfill_pii": backfill_pii}, json_body=None)

    def post_autolabel_start(self, body: Any = None) -> Any:
        """Start"""
        return self._call("POST", f"/api/autolabel/start",
                          params=None, json_body=body)

    def get_autolabel_by_job_id(self, job_id: str) -> Any:
        """Status"""
        return self._call("GET", f"/api/autolabel/{job_id}",
                          params=None, json_body=None)

    def get_calibration_sessions(self) -> Any:
        """List Sessions"""
        return self._call("GET", f"/api/calibration/sessions",
                          params=None, json_body=None)

    def post_calibration_validate(self, session_id: str) -> Any:
        """Validate"""
        return self._call("POST", f"/api/calibration/validate",
                          params={"session_id": session_id}, json_body=None)

    def post_calibration_vehicle_by_vehicle_id_calibrate(self, vehicle_id: str, body: Any = None) -> Any:
        """Calibrate Vehicle"""
        return self._call("POST", f"/api/calibration/vehicle/{vehicle_id}/calibrate",
                          params=None, json_body=body)

    def get_calibration_by_session_id(self, session_id: str) -> Any:
        """Get Validation"""
        return self._call("GET", f"/api/calibration/{session_id}",
                          params=None, json_body=None)

    def post_calibration_by_session_id_calibrate(self, session_id: str, body: Any = None) -> Any:
        """Calibrate Session"""
        return self._call("POST", f"/api/calibration/{session_id}/calibrate",
                          params=None, json_body=body)

    def post_calibration_by_session_id_estimate(self, session_id: str) -> Any:
        """Estimate Session"""
        return self._call("POST", f"/api/calibration/{session_id}/estimate",
                          params=None, json_body=None)

    def post_calibration_by_session_id_extrinsics(self, session_id: str) -> Any:
        """Check Extrinsics"""
        return self._call("POST", f"/api/calibration/{session_id}/extrinsics",
                          params=None, json_body=None)

    def post_calibration_by_session_id_import(self, session_id: str, body: Any = None) -> Any:
        """Import Calib"""
        return self._call("POST", f"/api/calibration/{session_id}/import",
                          params=None, json_body=body)

    def get_calibration_by_session_id_resolved(self, session_id: str) -> Any:
        """Resolved"""
        return self._call("GET", f"/api/calibration/{session_id}/resolved",
                          params=None, json_body=None)

    def get_calyx_rig_by_vehicle_id_consensus(self, vehicle_id: str) -> Any:
        """Rig Consensus"""
        return self._call("GET", f"/api/calyx/rig/{vehicle_id}/consensus",
                          params=None, json_body=None)

    def get_calyx_rig_by_vehicle_id_history(self, vehicle_id: str) -> Any:
        """Rig"""
        return self._call("GET", f"/api/calyx/rig/{vehicle_id}/history",
                          params=None, json_body=None)

    def get_calyx_session_by_session_id(self, session_id: str) -> Any:
        """Session State"""
        return self._call("GET", f"/api/calyx/session/{session_id}",
                          params=None, json_body=None)

    def post_calyx_session_by_session_id_drift(self, session_id: str, body: Any = None) -> Any:
        """Record"""
        return self._call("POST", f"/api/calyx/session/{session_id}/drift",
                          params=None, json_body=body)

    def post_calyx_session_by_session_id_recover(self, session_id: str, body: Any = None) -> Any:
        """Recover Session"""
        return self._call("POST", f"/api/calyx/session/{session_id}/recover",
                          params=None, json_body=body)

    def post_calyx_targetless(self, body: Any = None) -> Any:
        """Targetless"""
        return self._call("POST", f"/api/calyx/targetless",
                          params=None, json_body=body)

    def get_campaigns(self, status: Any | None = None, limit: int | None = 100) -> Any:
        """List Campaigns"""
        return self._call("GET", f"/api/campaigns",
                          params={"status": status, "limit": limit}, json_body=None)

    def post_campaigns(self, body: Any = None) -> Any:
        """Create Campaign"""
        return self._call("POST", f"/api/campaigns",
                          params=None, json_body=body)

    def get_campaigns_by_campaign_id(self, campaign_id: str) -> Any:
        """Campaign Detail"""
        return self._call("GET", f"/api/campaigns/{campaign_id}",
                          params=None, json_body=None)

    def post_campaigns_by_campaign_id_approve(self, campaign_id: str, stage: str) -> Any:
        """Approve Stage"""
        return self._call("POST", f"/api/campaigns/{campaign_id}/approve",
                          params={"stage": stage}, json_body=None)

    def post_campaigns_by_campaign_id_stop(self, campaign_id: str, reason: str | None = 'stopped by an operator') -> Any:
        """Stop"""
        return self._call("POST", f"/api/campaigns/{campaign_id}/stop",
                          params={"reason": reason}, json_body=None)

    def post_campaigns_by_campaign_id_tick(self, campaign_id: str, dry_run: bool | None = False) -> Any:
        """Tick"""
        return self._call("POST", f"/api/campaigns/{campaign_id}/tick",
                          params={"dry_run": dry_run}, json_body=None)

    def post_cloud_connect(self, body: Any = None) -> Any:
        """Cloud Connect"""
        return self._call("POST", f"/api/cloud/connect",
                          params=None, json_body=body)

    def post_cloud_disconnect(self, body: Any = None) -> Any:
        """Cloud Disconnect"""
        return self._call("POST", f"/api/cloud/disconnect",
                          params=None, json_body=body)

    def get_cloud_orphans(self) -> Any:
        """Cloud Orphans"""
        return self._call("GET", f"/api/cloud/orphans",
                          params=None, json_body=None)

    def post_cloud_orphans_terminate(self, body: Any = None) -> Any:
        """Cloud Terminate Orphan"""
        return self._call("POST", f"/api/cloud/orphans/terminate",
                          params=None, json_body=body)

    def get_cloud_status(self) -> Any:
        """Cloud Status"""
        return self._call("GET", f"/api/cloud/status",
                          params=None, json_body=None)

    def post_collaborate_assign(self, body: Any = None) -> Any:
        """Assign"""
        return self._call("POST", f"/api/collaborate/assign",
                          params=None, json_body=body)

    def get_collaborate_assignments(self, user_id: Any | None = None) -> Any:
        """Assignments"""
        return self._call("GET", f"/api/collaborate/assignments",
                          params={"user_id": user_id}, json_body=None)

    def post_collaborate_assignments_by_assignment_id_commit(self, assignment_id: str, body: Any = None) -> Any:
        """Commit Work"""
        return self._call("POST", f"/api/collaborate/assignments/{assignment_id}/commit",
                          params=None, json_body=body)

    def get_collaborate_branches(self) -> Any:
        """Branches"""
        return self._call("GET", f"/api/collaborate/branches",
                          params=None, json_body=None)

    def get_collaborate_merge_requests(self) -> Any:
        """Merge Requests"""
        return self._call("GET", f"/api/collaborate/merge_requests",
                          params=None, json_body=None)

    def post_collaborate_merge_requests_open(self, body: Any = None) -> Any:
        """Open Mr"""
        return self._call("POST", f"/api/collaborate/merge_requests/open",
                          params=None, json_body=body)

    def post_collaborate_merge_requests_by_mr_id_approve(self, mr_id: str, body: Any = None) -> Any:
        """Approve Mr"""
        return self._call("POST", f"/api/collaborate/merge_requests/{mr_id}/approve",
                          params=None, json_body=body)

    def post_collaborate_merge_requests_by_mr_id_merge(self, mr_id: str, body: Any = None) -> Any:
        """Merge Mr"""
        return self._call("POST", f"/api/collaborate/merge_requests/{mr_id}/merge",
                          params=None, json_body=body)

    def post_collaborate_merge_requests_by_mr_id_revert(self, mr_id: str, body: Any = None) -> Any:
        """Revert Mr"""
        return self._call("POST", f"/api/collaborate/merge_requests/{mr_id}/revert",
                          params=None, json_body=body)

    def post_collaborate_tasks_claim(self, user_id: str) -> Any:
        """Claim Task Ep"""
        return self._call("POST", f"/api/collaborate/tasks/claim",
                          params={"user_id": user_id}, json_body=None)

    def post_collaborate_tasks_enqueue(self, body: Any = None) -> Any:
        """Enqueue Tasks Ep"""
        return self._call("POST", f"/api/collaborate/tasks/enqueue",
                          params=None, json_body=body)

    def get_collaborate_tasks_stats(self, user_id: str) -> Any:
        """Task Stats Ep"""
        return self._call("GET", f"/api/collaborate/tasks/stats",
                          params={"user_id": user_id}, json_body=None)

    def post_collaborate_tasks_by_assignment_id_advance(self, assignment_id: str, to_status: str) -> Any:
        """Advance Task Ep"""
        return self._call("POST", f"/api/collaborate/tasks/{assignment_id}/advance",
                          params={"to_status": to_status}, json_body=None)

    def get_corrections_confusions(self, by: str | None = 'class', limit: int | None = 30) -> Any:
        """Confusions"""
        return self._call("GET", f"/api/corrections/confusions",
                          params={"by": by, "limit": limit}, json_body=None)

    def get_corrections_coverage(self) -> Any:
        """Coverage"""
        return self._call("GET", f"/api/corrections/coverage",
                          params=None, json_body=None)

    def post_corrections_embed(self, session_id: Any | None = None) -> Any:
        """Embed"""
        return self._call("POST", f"/api/corrections/embed",
                          params={"session_id": session_id}, json_body=None)

    def post_corrections_suggest(self, body: Any = None) -> Any:
        """Suggest"""
        return self._call("POST", f"/api/corrections/suggest",
                          params=None, json_body=body)

    def post_curation_dedup(self, session_id: str) -> Any:
        """Dedup"""
        return self._call("POST", f"/api/curation/dedup",
                          params={"session_id": session_id}, json_body=None)

    def get_curation_diverse(self, session_id: Any | None = None, k: int | None = 50) -> Any:
        """Diverse"""
        return self._call("GET", f"/api/curation/diverse",
                          params={"session_id": session_id, "k": k}, json_body=None)

    def post_curation_embed(self, session_id: Any | None = None) -> Any:
        """Embed"""
        return self._call("POST", f"/api/curation/embed",
                          params={"session_id": session_id}, json_body=None)

    def post_curation_extract(self, session_id: str) -> Any:
        """Extract"""
        return self._call("POST", f"/api/curation/extract",
                          params={"session_id": session_id}, json_body=None)

    def get_curation_slices(self) -> Any:
        """List Curation Slices"""
        return self._call("GET", f"/api/curation/slices",
                          params=None, json_body=None)

    def post_curation_slices(self, body: Any = None) -> Any:
        """Create Curation Slice"""
        return self._call("POST", f"/api/curation/slices",
                          params=None, json_body=body)

    def get_curation_slices_by_slice_id_materialize(self, slice_id: str, sample: int | None = 20) -> Any:
        """Materialize Curation Slice"""
        return self._call("GET", f"/api/curation/slices/{slice_id}/materialize",
                          params={"sample": sample}, json_body=None)

    def get_curation_summary(self, session_id: Any | None = None) -> Any:
        """Summary"""
        return self._call("GET", f"/api/curation/summary",
                          params={"session_id": session_id}, json_body=None)

    def get_datasets(self, limit: int | None = 100) -> Any:
        """List Datasets"""
        return self._call("GET", f"/api/datasets",
                          params={"limit": limit}, json_body=None)

    def post_datasets_export(self, body: Any = None) -> Any:
        """Start Export"""
        return self._call("POST", f"/api/datasets/export",
                          params=None, json_body=body)

    def get_datasets_versions_by_name(self, name: str, limit: int | None = 20) -> Any:
        """Dataset Versions"""
        return self._call("GET", f"/api/datasets/versions/{name}",
                          params={"limit": limit}, json_body=None)

    def get_datasets_by_a_id_diff_by_b_id(self, a_id: str, b_id: str, deep: bool | None = True, sample: int | None = 20) -> Any:
        """Dataset Diff"""
        return self._call("GET", f"/api/datasets/{a_id}/diff/{b_id}",
                          params={"deep": deep, "sample": sample}, json_body=None)

    def get_datasets_by_commit_id(self, commit_id: str) -> Any:
        """Dataset Detail"""
        return self._call("GET", f"/api/datasets/{commit_id}",
                          params=None, json_body=None)

    def get_datasets_by_commit_id_download(self, commit_id: str) -> Any:
        """Download Dataset"""
        return self._call("GET", f"/api/datasets/{commit_id}/download",
                          params=None, json_body=None)

    def get_datasets_by_commit_id_lineage(self, commit_id: str) -> Any:
        """Dataset Lineage"""
        return self._call("GET", f"/api/datasets/{commit_id}/lineage",
                          params=None, json_body=None)

    def get_discovery_queue(self, state: str | None = 'pending', limit: int | None = 200) -> Any:
        """Queue"""
        return self._call("GET", f"/api/discovery/queue",
                          params={"state": state, "limit": limit}, json_body=None)

    def post_discovery_run(self, session_id: str) -> Any:
        """Run"""
        return self._call("POST", f"/api/discovery/run",
                          params={"session_id": session_id}, json_body=None)

    def post_discovery_by_candidate_id_state(self, candidate_id: str, body: Any = None) -> Any:
        """Set State"""
        return self._call("POST", f"/api/discovery/{candidate_id}/state",
                          params=None, json_body=body)

    def post_driving_events_by_event_id_confirm(self, event_id: str, body: Any = None) -> Any:
        """Confirm"""
        return self._call("POST", f"/api/driving-events/{event_id}/confirm",
                          params=None, json_body=body)

    def post_driving_events_by_event_id_reject(self, event_id: str, body: Any = None) -> Any:
        """Reject"""
        return self._call("POST", f"/api/driving-events/{event_id}/reject",
                          params=None, json_body=body)

    def post_dynamics_compute(self, session_id: str) -> Any:
        """Compute"""
        return self._call("POST", f"/api/dynamics/compute",
                          params={"session_id": session_id}, json_body=None)

    def get_dynamics_frame_by_frame_id(self, frame_id: str) -> Any:
        """Get Frame"""
        return self._call("GET", f"/api/dynamics/frame/{frame_id}",
                          params=None, json_body=None)

    def get_dynamics_object_by_object_id(self, object_id: str) -> Any:
        """Get Object"""
        return self._call("GET", f"/api/dynamics/object/{object_id}",
                          params=None, json_body=None)

    def get_edge_artifacts_by_artifact_id_field(self, artifact_id: str, hours: int | None = 168) -> Any:
        """Field Report"""
        return self._call("GET", f"/api/edge/artifacts/{artifact_id}/field",
                          params={"hours": hours}, json_body=None)

    def get_edge_artifacts_by_artifact_id_gate(self, artifact_id: str, hours: int | None = 168) -> Any:
        """Field Gate"""
        return self._call("GET", f"/api/edge/artifacts/{artifact_id}/gate",
                          params={"hours": hours}, json_body=None)

    def get_edge_devices(self, fleet: Any | None = None, limit: int | None = 200) -> Any:
        """List Devices"""
        return self._call("GET", f"/api/edge/devices",
                          params={"fleet": fleet, "limit": limit}, json_body=None)

    def post_edge_devices(self, body: Any = None) -> Any:
        """Register Device"""
        return self._call("POST", f"/api/edge/devices",
                          params=None, json_body=body)

    def get_edge_fleet(self, hours: int | None = 24) -> Any:
        """Fleet Summary"""
        return self._call("GET", f"/api/edge/fleet",
                          params={"hours": hours}, json_body=None)

    def post_edge_telemetry(self, body: Any = None) -> Any:
        """Ingest Telemetry"""
        return self._call("POST", f"/api/edge/telemetry",
                          params=None, json_body=body)

    def post_embeddings_compute(self, body: Any = None) -> Any:
        """Embeddings Compute"""
        return self._call("POST", f"/api/embeddings/compute",
                          params=None, json_body=body)

    def get_errordetect_candidates(self, status: str | None = 'pending', limit: int | None = 100) -> Any:
        """Candidates"""
        return self._call("GET", f"/api/errordetect/candidates",
                          params={"status": status, "limit": limit}, json_body=None)

    def post_errordetect_candidates_by_candidate_id_confirm(self, candidate_id: str, body: Any = None) -> Any:
        """Confirm"""
        return self._call("POST", f"/api/errordetect/candidates/{candidate_id}/confirm",
                          params=None, json_body=body)

    def post_errordetect_candidates_by_candidate_id_dismiss(self, candidate_id: str) -> Any:
        """Dismiss"""
        return self._call("POST", f"/api/errordetect/candidates/{candidate_id}/dismiss",
                          params=None, json_body=None)

    def post_errordetect_run(self, body: Any = None) -> Any:
        """Run"""
        return self._call("POST", f"/api/errordetect/run",
                          params=None, json_body=body)

    def get_errordetect_summary(self) -> Any:
        """Summary Ep"""
        return self._call("GET", f"/api/errordetect/summary",
                          params=None, json_body=None)

    def post_eval_tracking(self, gold_id: str, run_id: str, iou_thr: float | None = 0.5) -> Any:
        """Score Tracking"""
        return self._call("POST", f"/api/eval/tracking",
                          params={"gold_id": gold_id, "run_id": run_id, "iou_thr": iou_thr}, json_body=None)

    def get_experiments(self, task_type: Any | None = None) -> Any:
        """List Experiments"""
        return self._call("GET", f"/api/experiments",
                          params={"task_type": task_type}, json_body=None)

    def post_experiments(self, body: Any = None) -> Any:
        """Create Experiment"""
        return self._call("POST", f"/api/experiments",
                          params=None, json_body=body)

    def get_experiments_runs_compare(self, a: str, b: str) -> Any:
        """Compare"""
        return self._call("GET", f"/api/experiments/runs/compare",
                          params={"a": a, "b": b}, json_body=None)

    def get_experiments_by_name(self, name: str, metric: str | None = 'map50') -> Any:
        """Experiment Detail"""
        return self._call("GET", f"/api/experiments/{name}",
                          params={"metric": metric}, json_body=None)

    def post_experiments_by_name_runs(self, name: str, body: Any = None) -> Any:
        """Attach Run"""
        return self._call("POST", f"/api/experiments/{name}/runs",
                          params=None, json_body=body)

    def post_explore_eval(self, body: Any = None) -> Any:
        """Run Eval"""
        return self._call("POST", f"/api/explore/eval",
                          params=None, json_body=body)

    def delete_explore_eval_by_eval_id(self, eval_id: str) -> Any:
        """Delete Eval"""
        return self._call("DELETE", f"/api/explore/eval/{eval_id}",
                          params=None, json_body=None)

    def get_explore_eval_by_eval_id_cells(self, eval_id: str) -> Any:
        """Eval Cells"""
        return self._call("GET", f"/api/explore/eval/{eval_id}/cells",
                          params=None, json_body=None)

    def get_explore_eval_by_eval_id_patches(self, eval_id: str, gt_class_id: Any | None = None, pred_class_id: Any | None = None, outcome: Any | None = None, limit: int | None = 120) -> Any:
        """Eval Patches"""
        return self._call("GET", f"/api/explore/eval/{eval_id}/patches",
                          params={"gt_class_id": gt_class_id, "pred_class_id": pred_class_id, "outcome": outcome, "limit": limit}, json_body=None)

    def get_explore_evals(self, limit: int | None = 50) -> Any:
        """List Evals"""
        return self._call("GET", f"/api/explore/evals",
                          params={"limit": limit}, json_body=None)

    def post_explore_facets(self, body: Any = None) -> Any:
        """Facets"""
        return self._call("POST", f"/api/explore/facets",
                          params=None, json_body=body)

    def get_explore_gold_by_gold_id_provenance(self, gold_id: str) -> Any:
        """Gold Provenance"""
        return self._call("GET", f"/api/explore/gold/{gold_id}/provenance",
                          params=None, json_body=None)

    def post_explore_projection(self, body: Any = None) -> Any:
        """Fit Projection"""
        return self._call("POST", f"/api/explore/projection",
                          params=None, json_body=body)

    def delete_explore_projection_by_projection_id(self, projection_id: str) -> Any:
        """Delete Projection"""
        return self._call("DELETE", f"/api/explore/projection/{projection_id}",
                          params=None, json_body=None)

    def get_explore_projection_by_projection_id_points(self, projection_id: str, limit: int | None = 50000) -> Any:
        """Projection Points"""
        return self._call("GET", f"/api/explore/projection/{projection_id}/points",
                          params={"limit": limit}, json_body=None)

    def get_explore_projections(self, limit: int | None = 50) -> Any:
        """List Projections"""
        return self._call("GET", f"/api/explore/projections",
                          params={"limit": limit}, json_body=None)

    def post_explore_select(self, body: Any = None, level: str | None = 'object', limit: int | None = 5000) -> Any:
        """Select Ids"""
        return self._call("POST", f"/api/explore/select",
                          params={"level": level, "limit": limit}, json_body=body)

    def post_explore_tag(self, body: Any = None) -> Any:
        """Apply Tags"""
        return self._call("POST", f"/api/explore/tag",
                          params=None, json_body=body)

    def get_explore_tags(self, level: str | None = 'object') -> Any:
        """Tag Vocabulary"""
        return self._call("GET", f"/api/explore/tags",
                          params={"level": level}, json_body=None)

    def get_explore_views(self) -> Any:
        """List Views"""
        return self._call("GET", f"/api/explore/views",
                          params=None, json_body=None)

    def post_explore_views(self, body: Any = None) -> Any:
        """Save View"""
        return self._call("POST", f"/api/explore/views",
                          params=None, json_body=body)

    def get_explore_views_by_slice_id_export_spec(self, slice_id: str) -> Any:
        """View Export Spec"""
        return self._call("GET", f"/api/explore/views/{slice_id}/export-spec",
                          params=None, json_body=None)

    def post_export(self, body: Any = None) -> Any:
        """Export"""
        return self._call("POST", f"/api/export",
                          params=None, json_body=body)

    def get_exports_resumable(self, limit: int | None = 50) -> Any:
        """List Resumable"""
        return self._call("GET", f"/api/exports/resumable",
                          params={"limit": limit}, json_body=None)

    def get_exports_by_job_id_progress(self, job_id: str) -> Any:
        """Export Progress"""
        return self._call("GET", f"/api/exports/{job_id}/progress",
                          params=None, json_body=None)

    def post_exports_by_job_id_resume(self, job_id: str) -> Any:
        """Resume Export"""
        return self._call("POST", f"/api/exports/{job_id}/resume",
                          params=None, json_body=None)

    def post_flywheel_adaptive_auto(self, total_label_budget: int | None = 2000, safety_floor: int | None = 200, min_share: float | None = 0.003) -> Any:
        """Adaptive Auto"""
        return self._call("POST", f"/api/flywheel/adaptive/auto",
                          params={"total_label_budget": total_label_budget, "safety_floor": safety_floor, "min_share": min_share}, json_body=None)

    def post_flywheel_adaptive_collection_orders(self, cycle_id: Any | None = None) -> Any:
        """Adaptive Collection Orders"""
        return self._call("POST", f"/api/flywheel/adaptive/collection-orders",
                          params={"cycle_id": cycle_id}, json_body=None)

    def post_flywheel_adaptive_cycle(self, body: Any = None) -> Any:
        """Adaptive Cycle Run"""
        return self._call("POST", f"/api/flywheel/adaptive/cycle",
                          params=None, json_body=body)

    def get_flywheel_adaptive_cycles(self, limit: int | None = 20) -> Any:
        """Adaptive Cycles"""
        return self._call("GET", f"/api/flywheel/adaptive/cycles",
                          params={"limit": limit}, json_body=None)

    def post_flywheel_adaptive_dispatch(self, body: Any = None) -> Any:
        """Adaptive Dispatch"""
        return self._call("POST", f"/api/flywheel/adaptive/dispatch",
                          params=None, json_body=body)

    def get_flywheel_gate_directed(self, budget: int | None = 500) -> Any:
        """Gate Directed Latest"""
        return self._call("GET", f"/api/flywheel/gate-directed",
                          params={"budget": budget}, json_body=None)

    def get_flywheel_gate_directed_by_run_id(self, run_id: str, budget: int | None = 500) -> Any:
        """Gate Directed Plan"""
        return self._call("GET", f"/api/flywheel/gate-directed/{run_id}",
                          params={"budget": budget}, json_body=None)

    def post_flywheel_gate_directed_by_run_id_materialize(self, run_id: str, body: Any = None) -> Any:
        """Gate Directed Materialize"""
        return self._call("POST", f"/api/flywheel/gate-directed/{run_id}/materialize",
                          params=None, json_body=body)

    def get_flywheel_lineage_by_deployment_id(self, deployment_id: str) -> Any:
        """Lineage"""
        return self._call("GET", f"/api/flywheel/lineage/{deployment_id}",
                          params=None, json_body=None)

    def post_flywheel_session_by_session_id_run(self, session_id: str) -> Any:
        """Run"""
        return self._call("POST", f"/api/flywheel/session/{session_id}/run",
                          params=None, json_body=None)

    def get_flywheel_stages(self) -> Any:
        """Stages"""
        return self._call("GET", f"/api/flywheel/stages",
                          params=None, json_body=None)

    def post_forgyx_benchmark(self, body: Any = None) -> Any:
        """Ingest Benchmark"""
        return self._call("POST", f"/api/forgyx/benchmark",
                          params=None, json_body=body)

    def get_forgyx_benchmarks(self, model_version: Any | None = None) -> Any:
        """Benchmarks"""
        return self._call("GET", f"/api/forgyx/benchmarks",
                          params={"model_version": model_version}, json_body=None)

    def get_forgyx_capabilities(self) -> Any:
        """Capabilities"""
        return self._call("GET", f"/api/forgyx/capabilities",
                          params=None, json_body=None)

    def post_forgyx_cooptimize(self, body: Any = None) -> Any:
        """Cooptimize"""
        return self._call("POST", f"/api/forgyx/cooptimize",
                          params=None, json_body=body)

    def get_forgyx_deployments(self, model_version: Any | None = None) -> Any:
        """Deployments"""
        return self._call("GET", f"/api/forgyx/deployments",
                          params={"model_version": model_version}, json_body=None)

    def post_forgyx_gate(self, body: Any = None) -> Any:
        """Gate"""
        return self._call("POST", f"/api/forgyx/gate",
                          params=None, json_body=body)

    def post_forgyx_rollout(self, body: Any = None) -> Any:
        """Rollout"""
        return self._call("POST", f"/api/forgyx/rollout",
                          params=None, json_body=body)

    def post_forgyx_thermal(self, body: Any = None) -> Any:
        """Thermal"""
        return self._call("POST", f"/api/forgyx/thermal",
                          params=None, json_body=body)

    def get_frames_by_frame_id(self, frame_id: str) -> Any:
        """Get Frame"""
        return self._call("GET", f"/api/frames/{frame_id}",
                          params=None, json_body=None)

    def get_frames_by_frame_id_adverse(self, frame_id: str) -> Any:
        """List Adverse"""
        return self._call("GET", f"/api/frames/{frame_id}/adverse",
                          params=None, json_body=None)

    def post_frames_by_frame_id_adverse(self, frame_id: str, body: Any = None) -> Any:
        """Create Adverse"""
        return self._call("POST", f"/api/frames/{frame_id}/adverse",
                          params=None, json_body=body)

    def get_frames_by_frame_id_cuboids(self, frame_id: str) -> Any:
        """Frame Cuboids"""
        return self._call("GET", f"/api/frames/{frame_id}/cuboids",
                          params=None, json_body=None)

    def get_frames_by_frame_id_drivable(self, frame_id: str) -> Any:
        """Get Drivable"""
        return self._call("GET", f"/api/frames/{frame_id}/drivable",
                          params=None, json_body=None)

    def post_frames_by_frame_id_drivable(self, frame_id: str) -> Any:
        """Segment Frame"""
        return self._call("POST", f"/api/frames/{frame_id}/drivable",
                          params=None, json_body=None)

    def put_frames_by_frame_id_drivable(self, frame_id: str, body: Any = None) -> Any:
        """Refine Drivable"""
        return self._call("PUT", f"/api/frames/{frame_id}/drivable",
                          params=None, json_body=body)

    def get_frames_by_frame_id_filmstrip(self, frame_id: str, span: int | None = 12) -> Any:
        """Frame Filmstrip"""
        return self._call("GET", f"/api/frames/{frame_id}/filmstrip",
                          params={"span": span}, json_body=None)

    def get_frames_by_frame_id_image(self, frame_id: str) -> Any:
        """Frame Image"""
        return self._call("GET", f"/api/frames/{frame_id}/image",
                          params=None, json_body=None)

    def get_frames_by_frame_id_lanes(self, frame_id: str) -> Any:
        """List Lanes"""
        return self._call("GET", f"/api/frames/{frame_id}/lanes",
                          params=None, json_body=None)

    def post_frames_by_frame_id_lanes(self, frame_id: str, body: Any = None) -> Any:
        """Create Lane"""
        return self._call("POST", f"/api/frames/{frame_id}/lanes",
                          params=None, json_body=body)

    def post_frames_by_frame_id_lanes_propagate(self, frame_id: str, frames: int | None = 8) -> Any:
        """Propagate"""
        return self._call("POST", f"/api/frames/{frame_id}/lanes/propagate",
                          params={"frames": frames}, json_body=None)

    def post_frames_by_frame_id_lanes_propose(self, frame_id: str) -> Any:
        """Propose"""
        return self._call("POST", f"/api/frames/{frame_id}/lanes/propose",
                          params=None, json_body=None)

    def get_frames_by_frame_id_lift_ground(self, frame_id: str, u: float, v: float) -> Any:
        """Lift Ground"""
        return self._call("GET", f"/api/frames/{frame_id}/lift_ground",
                          params={"u": u, "v": v}, json_body=None)

    def get_frames_by_frame_id_objects(self, frame_id: str) -> Any:
        """Frame Objects"""
        return self._call("GET", f"/api/frames/{frame_id}/objects",
                          params=None, json_body=None)

    def post_frames_by_frame_id_objects(self, frame_id: str, body: Any = None) -> Any:
        """Create Object"""
        return self._call("POST", f"/api/frames/{frame_id}/objects",
                          params=None, json_body=body)

    def get_frames_by_frame_id_relations(self, frame_id: str) -> Any:
        """Relations List"""
        return self._call("GET", f"/api/frames/{frame_id}/relations",
                          params=None, json_body=None)

    def post_frames_by_frame_id_relations_propose(self, frame_id: str) -> Any:
        """Relations Propose"""
        return self._call("POST", f"/api/frames/{frame_id}/relations/propose",
                          params=None, json_body=None)

    def get_frames_by_frame_id_relationships(self, frame_id: str) -> Any:
        """Frame Relationships"""
        return self._call("GET", f"/api/frames/{frame_id}/relationships",
                          params=None, json_body=None)

    def get_frames_by_frame_id_segment(self, frame_id: str, kind: str | None = 'semantic') -> Any:
        """Get Segment"""
        return self._call("GET", f"/api/frames/{frame_id}/segment",
                          params={"kind": kind}, json_body=None)

    def post_frames_by_frame_id_segment(self, frame_id: str, kind: str | None = 'semantic') -> Any:
        """Auto Segment"""
        return self._call("POST", f"/api/frames/{frame_id}/segment",
                          params={"kind": kind}, json_body=None)

    def put_frames_by_frame_id_segment(self, frame_id: str, body: Any = None) -> Any:
        """Edit Segment"""
        return self._call("PUT", f"/api/frames/{frame_id}/segment",
                          params=None, json_body=body)

    def get_frames_by_frame_id_segment_labelids_png(self, frame_id: str, kind: str | None = 'semantic') -> Any:
        """Segment Labelids"""
        return self._call("GET", f"/api/frames/{frame_id}/segment/labelids.png",
                          params={"kind": kind}, json_body=None)

    def get_frames_by_frame_id_segment_overlay(self, frame_id: str, kind: str | None = 'semantic') -> Any:
        """Segment Overlay"""
        return self._call("GET", f"/api/frames/{frame_id}/segment/overlay",
                          params={"kind": kind}, json_body=None)

    def get_frames_by_frame_id_segment_panoptic_png(self, frame_id: str) -> Any:
        """Segment Panoptic Png"""
        return self._call("GET", f"/api/frames/{frame_id}/segment/panoptic.png",
                          params=None, json_body=None)

    def post_frames_by_frame_id_vlm_target_generate(self, frame_id: str) -> Any:
        """Vlm Target Generate"""
        return self._call("POST", f"/api/frames/{frame_id}/vlm-target/generate",
                          params=None, json_body=None)

    def get_frames_by_frame_id_vlm_targets(self, frame_id: str) -> Any:
        """Vlm Targets List"""
        return self._call("GET", f"/api/frames/{frame_id}/vlm-targets",
                          params=None, json_body=None)

    def get_govern_audit(self, actor: Any | None = None, limit: int | None = 100) -> Any:
        """Audit"""
        return self._call("GET", f"/api/govern/audit",
                          params={"actor": actor, "limit": limit}, json_body=None)

    def post_govern_consent(self, body: Any = None) -> Any:
        """Consent"""
        return self._call("POST", f"/api/govern/consent",
                          params=None, json_body=body)

    def get_govern_consent_by_consent_status_gate(self, consent_status: str) -> Any:
        """Consent Gate"""
        return self._call("GET", f"/api/govern/consent/{consent_status}/gate",
                          params=None, json_body=None)

    def get_govern_control_precision(self) -> Any:
        """Control Precision"""
        return self._call("GET", f"/api/govern/control/precision",
                          params=None, json_body=None)

    def post_govern_control_seed(self, limit: int | None = 500, rate: Any | None = None) -> Any:
        """Control Seed"""
        return self._call("POST", f"/api/govern/control/seed",
                          params={"limit": limit, "rate": rate}, json_body=None)

    def post_govern_control_by_sample_id_verdict(self, sample_id: str, body: Any = None) -> Any:
        """Control Verdict"""
        return self._call("POST", f"/api/govern/control/{sample_id}/verdict",
                          params=None, json_body=body)

    def post_govern_controller_tick(self, schedule_bursts: bool | None = True) -> Any:
        """Controller Tick"""
        return self._call("POST", f"/api/govern/controller/tick",
                          params={"schedule_bursts": schedule_bursts}, json_body=None)

    def post_govern_cost_gate(self, body: Any = None) -> Any:
        """Cost Ceiling"""
        return self._call("POST", f"/api/govern/cost/gate",
                          params=None, json_body=body)

    def post_govern_drift_scan(self, body: Any = None) -> Any:
        """Drift Scan"""
        return self._call("POST", f"/api/govern/drift/scan",
                          params=None, json_body=body)

    def post_govern_erase(self, body: Any = None) -> Any:
        """Erase"""
        return self._call("POST", f"/api/govern/erase",
                          params=None, json_body=body)

    def post_govern_killswitch_engage(self, body: Any = None) -> Any:
        """Killswitch Engage"""
        return self._call("POST", f"/api/govern/killswitch/engage",
                          params=None, json_body=body)

    def post_govern_killswitch_release(self) -> Any:
        """Killswitch Release"""
        return self._call("POST", f"/api/govern/killswitch/release",
                          params=None, json_body=None)

    def get_govern_lineage_by_subject(self, subject: str) -> Any:
        """Lineage"""
        return self._call("GET", f"/api/govern/lineage/{subject}",
                          params=None, json_body=None)

    def get_govern_pii_access(self, user_id: Any | None = None, subject_id: Any | None = None, session_id: Any | None = None, action: Any | None = None, since_hours: Any | None = None, limit: int | None = 100, offset: int | None = 0) -> Any:
        """Pii Access"""
        return self._call("GET", f"/api/govern/pii-access",
                          params={"user_id": user_id, "subject_id": subject_id, "session_id": session_id, "action": action, "since_hours": since_hours, "limit": limit, "offset": offset}, json_body=None)

    def get_govern_pii_access_summary(self, hours: int | None = 168) -> Any:
        """Pii Access Summary"""
        return self._call("GET", f"/api/govern/pii-access/summary",
                          params={"hours": hours}, json_body=None)

    def post_govern_promote(self, model_version: str, task: str | None = 'detection') -> Any:
        """Promote"""
        return self._call("POST", f"/api/govern/promote",
                          params={"model_version": model_version, "task": task}, json_body=None)

    def post_govern_redaction_proof(self, body: Any = None) -> Any:
        """Redaction Proof"""
        return self._call("POST", f"/api/govern/redaction/proof",
                          params=None, json_body=body)

    def get_govern_registry(self, task: Any | None = None) -> Any:
        """Registry List"""
        return self._call("GET", f"/api/govern/registry",
                          params={"task": task}, json_body=None)

    def post_govern_registry_register(self, body: Any = None) -> Any:
        """Registry Register"""
        return self._call("POST", f"/api/govern/registry/register",
                          params=None, json_body=body)

    def post_govern_registry_register_run(self, run_id: str, task: Any | None = None) -> Any:
        """Registry Register Run"""
        return self._call("POST", f"/api/govern/registry/register_run",
                          params={"run_id": run_id, "task": task}, json_body=None)

    def get_govern_retention_due(self) -> Any:
        """Retention Due"""
        return self._call("GET", f"/api/govern/retention/due",
                          params=None, json_body=None)

    def post_govern_retention_sweep(self, body: Any = None) -> Any:
        """Retention Sweep"""
        return self._call("POST", f"/api/govern/retention/sweep",
                          params=None, json_body=body)

    def get_govern_state(self) -> Any:
        """State"""
        return self._call("GET", f"/api/govern/state",
                          params=None, json_body=None)

    def post_hardening_efficiency(self, body: Any = None) -> Any:
        """Efficiency"""
        return self._call("POST", f"/api/hardening/efficiency",
                          params=None, json_body=body)

    def post_hardening_reproducible(self, body: Any = None) -> Any:
        """Reproducible"""
        return self._call("POST", f"/api/hardening/reproducible",
                          params=None, json_body=body)

    def post_hardening_slo(self, body: Any = None) -> Any:
        """Slo Tick"""
        return self._call("POST", f"/api/hardening/slo",
                          params=None, json_body=body)

    def get_hardening_slo_board(self) -> Any:
        """Slo Board View"""
        return self._call("GET", f"/api/hardening/slo/board",
                          params=None, json_body=None)

    def get_hdmap_commits(self, limit: int | None = 200) -> Any:
        """Commits"""
        return self._call("GET", f"/api/hdmap/commits",
                          params={"limit": limit}, json_body=None)

    def get_hdmap_elements(self, commit_id: Any | None = None, session_id: Any | None = None) -> Any:
        """Elements"""
        return self._call("GET", f"/api/hdmap/elements",
                          params={"commit_id": commit_id, "session_id": session_id}, json_body=None)

    def post_hdmap_elements_metric(self, session_id: str, body: Any = None) -> Any:
        """Create Metric Element Ep"""
        return self._call("POST", f"/api/hdmap/elements/metric",
                          params={"session_id": session_id}, json_body=body)

    def post_hdmap_fuse(self, session_ids: str, region: Any | None = None, compute_target: str | None = 'local') -> Any:
        """Fuse"""
        return self._call("POST", f"/api/hdmap/fuse",
                          params={"session_ids": session_ids, "region": region, "compute_target": compute_target}, json_body=None)

    def post_hdmap_georef(self, session_id: str, height_m: Any | None = None) -> Any:
        """Georef"""
        return self._call("POST", f"/api/hdmap/georef",
                          params={"session_id": session_id, "height_m": height_m}, json_body=None)

    def get_hdmap_provenance(self, element_id: str) -> Any:
        """Provenance"""
        return self._call("GET", f"/api/hdmap/provenance",
                          params={"element_id": element_id}, json_body=None)

    def get_health(self) -> Any:
        """Health"""
        return self._call("GET", f"/api/health",
                          params=None, json_body=None)

    def get_imports(self, limit: int | None = 50) -> Any:
        """List Jobs"""
        return self._call("GET", f"/api/imports",
                          params={"limit": limit}, json_body=None)

    def post_imports_start(self, body: Any = None) -> Any:
        """Start"""
        return self._call("POST", f"/api/imports/start",
                          params=None, json_body=body)

    def get_imports_by_job_id(self, job_id: str) -> Any:
        """Status"""
        return self._call("GET", f"/api/imports/{job_id}",
                          params=None, json_body=None)

    def get_ingest_progress(self) -> Any:
        """Ingest Progress"""
        return self._call("GET", f"/api/ingest/progress",
                          params=None, json_body=None)

    def post_inspector_health_sweep(self, limit: int | None = 500) -> Any:
        """Health Sweep"""
        return self._call("POST", f"/api/inspector/health/sweep",
                          params={"limit": limit}, json_body=None)

    def post_inspector_index_backfill(self, limit: int | None = 500) -> Any:
        """Index Backfill"""
        return self._call("POST", f"/api/inspector/index/backfill",
                          params={"limit": limit}, json_body=None)

    def get_inspector_layouts(self) -> Any:
        """List Layouts"""
        return self._call("GET", f"/api/inspector/layouts",
                          params=None, json_body=None)

    def post_inspector_layouts(self, body: Any = None) -> Any:
        """Save Layout"""
        return self._call("POST", f"/api/inspector/layouts",
                          params=None, json_body=body)

    def delete_inspector_layouts_by_layout_id(self, layout_id: str) -> Any:
        """Delete Layout"""
        return self._call("DELETE", f"/api/inspector/layouts/{layout_id}",
                          params=None, json_body=None)

    def get_inspector_sessions(self, limit: int | None = 100) -> Any:
        """List Sessions"""
        return self._call("GET", f"/api/inspector/sessions",
                          params={"limit": limit}, json_body=None)

    def get_inspector_sessions_by_session_id_annotations_at(self, session_id: str, ts_ns: int) -> Any:
        """Annotations At"""
        return self._call("GET", f"/api/inspector/sessions/{session_id}/annotations-at",
                          params={"ts_ns": ts_ns}, json_body=None)

    def get_inspector_sessions_by_session_id_events(self, session_id: str) -> Any:
        """Session Events"""
        return self._call("GET", f"/api/inspector/sessions/{session_id}/events",
                          params=None, json_body=None)

    def get_inspector_sessions_by_session_id_frame_at(self, session_id: str, ts_ns: int) -> Any:
        """Frame At"""
        return self._call("GET", f"/api/inspector/sessions/{session_id}/frame-at",
                          params={"ts_ns": ts_ns}, json_body=None)

    def get_inspector_sessions_by_session_id_health(self, session_id: str) -> Any:
        """Get Health"""
        return self._call("GET", f"/api/inspector/sessions/{session_id}/health",
                          params=None, json_body=None)

    def post_inspector_sessions_by_session_id_health(self, session_id: str) -> Any:
        """Run Health"""
        return self._call("POST", f"/api/inspector/sessions/{session_id}/health",
                          params=None, json_body=None)

    def get_inspector_sessions_by_session_id_index(self, session_id: str) -> Any:
        """Get Index"""
        return self._call("GET", f"/api/inspector/sessions/{session_id}/index",
                          params=None, json_body=None)

    def post_inspector_sessions_by_session_id_index(self, session_id: str) -> Any:
        """Build Index"""
        return self._call("POST", f"/api/inspector/sessions/{session_id}/index",
                          params=None, json_body=None)

    def get_inspector_sessions_by_session_id_lichtblick(self, session_id: str) -> Any:
        """Lichtblick Link"""
        return self._call("GET", f"/api/inspector/sessions/{session_id}/lichtblick",
                          params=None, json_body=None)

    def get_inspector_sessions_by_session_id_mcap_url(self, session_id: str) -> Any:
        """Mcap Url"""
        return self._call("GET", f"/api/inspector/sessions/{session_id}/mcap-url",
                          params=None, json_body=None)

    def get_integrations_events(self) -> Any:
        """Known Events"""
        return self._call("GET", f"/api/integrations/events",
                          params=None, json_body=None)

    def get_integrations_sources(self, project_id: Any | None = None) -> Any:
        """List Sources"""
        return self._call("GET", f"/api/integrations/sources",
                          params={"project_id": project_id}, json_body=None)

    def post_integrations_sources(self, body: Any = None) -> Any:
        """Register Source"""
        return self._call("POST", f"/api/integrations/sources",
                          params=None, json_body=body)

    def delete_integrations_sources_by_source_id(self, source_id: str) -> Any:
        """Delete Source"""
        return self._call("DELETE", f"/api/integrations/sources/{source_id}",
                          params=None, json_body=None)

    def get_integrations_sources_by_source_id_preview(self, source_id: str, limit: int | None = 50) -> Any:
        """Preview Source"""
        return self._call("GET", f"/api/integrations/sources/{source_id}/preview",
                          params={"limit": limit}, json_body=None)

    def get_integrations_webhooks(self, project_id: Any | None = None) -> Any:
        """List Webhooks"""
        return self._call("GET", f"/api/integrations/webhooks",
                          params={"project_id": project_id}, json_body=None)

    def post_integrations_webhooks(self, body: Any = None) -> Any:
        """Create Webhook"""
        return self._call("POST", f"/api/integrations/webhooks",
                          params=None, json_body=body)

    def delete_integrations_webhooks_by_webhook_id(self, webhook_id: str) -> Any:
        """Delete Webhook"""
        return self._call("DELETE", f"/api/integrations/webhooks/{webhook_id}",
                          params=None, json_body=None)

    def post_intent_propose_session(self, session_id: str) -> Any:
        """Intent Propose Session"""
        return self._call("POST", f"/api/intent/propose-session",
                          params={"session_id": session_id}, json_body=None)

    def get_intent_vocab(self) -> Any:
        """Intent Vocab"""
        return self._call("GET", f"/api/intent/vocab",
                          params=None, json_body=None)

    def get_jobs(self, limit: int | None = 100) -> Any:
        """Jobs"""
        return self._call("GET", f"/api/jobs",
                          params={"limit": limit}, json_body=None)

    def get_labelops_issues(self, frame_id: Any | None = None, job_id: Any | None = None, object_id: Any | None = None, status: Any | None = None) -> Any:
        """List Issues"""
        return self._call("GET", f"/api/labelops/issues",
                          params={"frame_id": frame_id, "job_id": job_id, "object_id": object_id, "status": status}, json_body=None)

    def post_labelops_issues(self, body: Any = None) -> Any:
        """Create Issue"""
        return self._call("POST", f"/api/labelops/issues",
                          params=None, json_body=body)

    def get_labelops_issues_by_issue_id(self, issue_id: str) -> Any:
        """Get Issue"""
        return self._call("GET", f"/api/labelops/issues/{issue_id}",
                          params=None, json_body=None)

    def post_labelops_issues_by_issue_id_comment(self, issue_id: str, body: Any = None) -> Any:
        """Comment"""
        return self._call("POST", f"/api/labelops/issues/{issue_id}/comment",
                          params=None, json_body=body)

    def post_labelops_issues_by_issue_id_resolve(self, issue_id: str, reopen: bool | None = False) -> Any:
        """Resolve Issue"""
        return self._call("POST", f"/api/labelops/issues/{issue_id}/resolve",
                          params={"reopen": reopen}, json_body=None)

    def get_labelops_jobs(self, project_id: Any | None = None, task_id: Any | None = None, assignee_id: Any | None = None, stage: Any | None = None, state: Any | None = None, limit: int | None = 200) -> Any:
        """List Jobs"""
        return self._call("GET", f"/api/labelops/jobs",
                          params={"project_id": project_id, "task_id": task_id, "assignee_id": assignee_id, "stage": stage, "state": state, "limit": limit}, json_body=None)

    def post_labelops_jobs_by_job_id_assign(self, job_id: str, body: Any = None) -> Any:
        """Assign Job"""
        return self._call("POST", f"/api/labelops/jobs/{job_id}/assign",
                          params=None, json_body=body)

    def post_labelops_jobs_by_job_id_state(self, job_id: str, body: Any = None) -> Any:
        """Set State"""
        return self._call("POST", f"/api/labelops/jobs/{job_id}/state",
                          params=None, json_body=body)

    def post_labelops_jobs_by_job_id_submit(self, job_id: str, body: Any = None) -> Any:
        """Submit Job"""
        return self._call("POST", f"/api/labelops/jobs/{job_id}/submit",
                          params=None, json_body=body)

    def get_labelops_my_jobs(self) -> Any:
        """My Jobs"""
        return self._call("GET", f"/api/labelops/my-jobs",
                          params=None, json_body=None)

    def get_labelops_projects(self, limit: int | None = 100) -> Any:
        """List Projects"""
        return self._call("GET", f"/api/labelops/projects",
                          params={"limit": limit}, json_body=None)

    def post_labelops_projects(self, body: Any = None) -> Any:
        """Create Project"""
        return self._call("POST", f"/api/labelops/projects",
                          params=None, json_body=body)

    def get_labelops_projects_by_project_id_board(self, project_id: str) -> Any:
        """Project Board"""
        return self._call("GET", f"/api/labelops/projects/{project_id}/board",
                          params=None, json_body=None)

    def get_labelops_scorecards(self, project_id: Any | None = None) -> Any:
        """Scorecards"""
        return self._call("GET", f"/api/labelops/scorecards",
                          params={"project_id": project_id}, json_body=None)

    def post_labelops_tasks(self, body: Any = None) -> Any:
        """Create Task"""
        return self._call("POST", f"/api/labelops/tasks",
                          params=None, json_body=body)

    def post_labelox_propagate4d(self, body: Any = None) -> Any:
        """Propagate4D"""
        return self._call("POST", f"/api/labelox/propagate4d",
                          params=None, json_body=body)

    def post_labelox_quality_agreement(self, body: Any = None) -> Any:
        """Quality Agreement"""
        return self._call("POST", f"/api/labelox/quality/agreement",
                          params=None, json_body=body)

    def post_labelox_quality_audit(self, body: Any = None) -> Any:
        """Quality Audit"""
        return self._call("POST", f"/api/labelox/quality/audit",
                          params=None, json_body=body)

    def get_labelox_queue(self, limit: int | None = 100) -> Any:
        """Label Queue"""
        return self._call("GET", f"/api/labelox/queue",
                          params={"limit": limit}, json_body=None)

    def post_labelox_reconcile_parity(self, body: Any = None) -> Any:
        """Reconcile Parity Gate"""
        return self._call("POST", f"/api/labelox/reconcile/parity",
                          params=None, json_body=body)

    def get_lanes_type_coverage(self) -> Any:
        """Lane Type Coverage"""
        return self._call("GET", f"/api/lanes/type-coverage",
                          params=None, json_body=None)

    def delete_lanes_by_lane_id(self, lane_id: str) -> Any:
        """Delete Lane"""
        return self._call("DELETE", f"/api/lanes/{lane_id}",
                          params=None, json_body=None)

    def put_lanes_by_lane_id(self, lane_id: str, body: Any = None) -> Any:
        """Update Lane"""
        return self._call("PUT", f"/api/lanes/{lane_id}",
                          params=None, json_body=body)

    def post_lidar_aggregate(self, body: Any = None) -> Any:
        """Aggregate"""
        return self._call("POST", f"/api/lidar/aggregate",
                          params=None, json_body=body)

    def post_lidar_aggregate_by_agg_id_label(self, agg_id: str, body: Any = None) -> Any:
        """Aggregate Label"""
        return self._call("POST", f"/api/lidar/aggregate/{agg_id}/label",
                          params=None, json_body=body)

    def get_lidar_analytics3d(self) -> Any:
        """Analytics3D"""
        return self._call("GET", f"/api/lidar/analytics3d",
                          params=None, json_body=None)

    def get_lidar_clouds_by_cloud_id(self, cloud_id: str) -> Any:
        """Cloud Meta"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}",
                          params=None, json_body=None)

    def get_lidar_clouds_by_cloud_id_bev_labels(self, cloud_id: str) -> Any:
        """Cloud Bev Labels"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}/bev-labels",
                          params=None, json_body=None)

    def post_lidar_clouds_by_cloud_id_extract(self, cloud_id: str) -> Any:
        """Extract Static"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/extract",
                          params=None, json_body=None)

    def get_lidar_clouds_by_cloud_id_ground_qa(self, cloud_id: str) -> Any:
        """Cloud Ground Qa"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}/ground-qa",
                          params=None, json_body=None)

    def post_lidar_clouds_by_cloud_id_lift(self, cloud_id: str) -> Any:
        """Lift Cloud Objects"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/lift",
                          params=None, json_body=None)

    def post_lidar_clouds_by_cloud_id_link(self, cloud_id: str) -> Any:
        """Link Cloud Objects"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/link",
                          params=None, json_body=None)

    def get_lidar_clouds_by_cloud_id_objects3d(self, cloud_id: str) -> Any:
        """List Objects3D"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}/objects3d",
                          params=None, json_body=None)

    def post_lidar_clouds_by_cloud_id_objects3d(self, cloud_id: str, body: Any = None) -> Any:
        """Create Object3D"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/objects3d",
                          params=None, json_body=body)

    def post_lidar_clouds_by_cloud_id_occupancy(self, cloud_id: str, voxel_size: float | None = 0.5) -> Any:
        """Cloud Occupancy"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/occupancy",
                          params={"voxel_size": voxel_size}, json_body=None)

    def get_lidar_clouds_by_cloud_id_points(self, cloud_id: str, variant: Any | None = None, max: int | None = 400000, full: bool | None = False) -> Any:
        """Cloud Points"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}/points",
                          params={"variant": variant, "max": max, "full": full}, json_body=None)

    def get_lidar_clouds_by_cloud_id_quality3d(self, cloud_id: str) -> Any:
        """List Quality3D"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}/quality3d",
                          params=None, json_body=None)

    def post_lidar_clouds_by_cloud_id_quality3d(self, cloud_id: str) -> Any:
        """Quality3D"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/quality3d",
                          params=None, json_body=None)

    def post_lidar_clouds_by_cloud_id_segment(self, cloud_id: str) -> Any:
        """Segment"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/segment",
                          params=None, json_body=None)

    def get_lidar_clouds_by_cloud_id_segmentation(self, cloud_id: str) -> Any:
        """Get Segmentation"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}/segmentation",
                          params=None, json_body=None)

    def get_lidar_clouds_by_cloud_id_segmentation_points(self, cloud_id: str, max: int | None = 300000) -> Any:
        """Segmentation Points"""
        return self._call("GET", f"/api/lidar/clouds/{cloud_id}/segmentation/points",
                          params={"max": max}, json_body=None)

    def post_lidar_clouds_by_cloud_id_traverse(self, cloud_id: str) -> Any:
        """Traverse"""
        return self._call("POST", f"/api/lidar/clouds/{cloud_id}/traverse",
                          params=None, json_body=None)

    def post_lidar_cuboids_by_frame_id(self, frame_id: str) -> Any:
        """Compute Cuboids"""
        return self._call("POST", f"/api/lidar/cuboids/{frame_id}",
                          params=None, json_body=None)

    def post_lidar_export3d(self, body: Any = None) -> Any:
        """Export3D"""
        return self._call("POST", f"/api/lidar/export3d",
                          params=None, json_body=body)

    def post_lidar_frames_by_frame_id_lift(self, frame_id: str) -> Any:
        """Lift Frame Objects"""
        return self._call("POST", f"/api/lidar/frames/{frame_id}/lift",
                          params=None, json_body=None)

    def get_lidar_objects2d_by_object_id_linked3d(self, object_id: str) -> Any:
        """Object2D Linked3D"""
        return self._call("GET", f"/api/lidar/objects2d/{object_id}/linked3d",
                          params=None, json_body=None)

    def post_lidar_objects3d_batch_correct(self, body: Any = None) -> Any:
        """Batch Correct Objects3D"""
        return self._call("POST", f"/api/lidar/objects3d/batch_correct",
                          params=None, json_body=body)

    def delete_lidar_objects3d_by_object_3d_id(self, object_3d_id: str) -> Any:
        """Delete Object3D"""
        return self._call("DELETE", f"/api/lidar/objects3d/{object_3d_id}",
                          params=None, json_body=None)

    def patch_lidar_objects3d_by_object_3d_id(self, object_3d_id: str, body: Any = None) -> Any:
        """Edit Object3D"""
        return self._call("PATCH", f"/api/lidar/objects3d/{object_3d_id}",
                          params=None, json_body=body)

    def post_lidar_objects3d_by_object_3d_id_consistency(self, object_3d_id: str) -> Any:
        """Object3D Consistency"""
        return self._call("POST", f"/api/lidar/objects3d/{object_3d_id}/consistency",
                          params=None, json_body=None)

    def get_lidar_objects3d_by_object_3d_id_linked(self, object_3d_id: str) -> Any:
        """Object3D Linked"""
        return self._call("GET", f"/api/lidar/objects3d/{object_3d_id}/linked",
                          params=None, json_body=None)

    def get_lidar_objects3d_by_object_3d_id_projection(self, object_3d_id: str, cam_id: str | None = 'cam_f', w: int | None = 1280, h: int | None = 960) -> Any:
        """Project Object3D"""
        return self._call("GET", f"/api/lidar/objects3d/{object_3d_id}/projection",
                          params={"cam_id": cam_id, "w": w, "h": h}, json_body=None)

    def post_lidar_objects3d_by_object_3d_id_properties(self, object_3d_id: str) -> Any:
        """Compute Properties Endpoint"""
        return self._call("POST", f"/api/lidar/objects3d/{object_3d_id}/properties",
                          params=None, json_body=None)

    def get_lidar_objects3d_by_object_3d_id_similar(self, object_3d_id: str, k: int | None = 10) -> Any:
        """Similar Objects3D"""
        return self._call("GET", f"/api/lidar/objects3d/{object_3d_id}/similar",
                          params={"k": k}, json_body=None)

    def post_lidar_quality3d_by_flag_id_confirm(self, flag_id: str) -> Any:
        """Confirm Quality3D"""
        return self._call("POST", f"/api/lidar/quality3d/{flag_id}/confirm",
                          params=None, json_body=None)

    def get_lidar_search3d(self, q: str) -> Any:
        """Search3D"""
        return self._call("GET", f"/api/lidar/search3d",
                          params={"q": q}, json_body=None)

    def post_lidar_sessions_by_session_id_build(self, session_id: str, body: Any = None) -> Any:
        """Build Clouds"""
        return self._call("POST", f"/api/lidar/sessions/{session_id}/build",
                          params=None, json_body=body)

    def get_lidar_sessions_by_session_id_clouds(self, session_id: str) -> Any:
        """List Clouds"""
        return self._call("GET", f"/api/lidar/sessions/{session_id}/clouds",
                          params=None, json_body=None)

    def post_lidar_sessions_by_session_id_rare3d(self, session_id: str) -> Any:
        """Rare3D"""
        return self._call("POST", f"/api/lidar/sessions/{session_id}/rare3d",
                          params=None, json_body=None)

    def post_lidar_sessions_by_session_id_scene3d(self, session_id: str) -> Any:
        """Scene3D"""
        return self._call("POST", f"/api/lidar/sessions/{session_id}/scene3d",
                          params=None, json_body=None)

    def get_lidar_sessions_by_session_id_static_elements(self, session_id: str) -> Any:
        """List Static Elements"""
        return self._call("GET", f"/api/lidar/sessions/{session_id}/static_elements",
                          params=None, json_body=None)

    def post_lidar_sessions_by_session_id_track3d(self, session_id: str) -> Any:
        """Run Tracking"""
        return self._call("POST", f"/api/lidar/sessions/{session_id}/track3d",
                          params=None, json_body=None)

    def get_lidar_sessions_by_session_id_tracks3d(self, session_id: str) -> Any:
        """List Tracks3D"""
        return self._call("GET", f"/api/lidar/sessions/{session_id}/tracks3d",
                          params=None, json_body=None)

    def get_lidar_sessions_by_session_id_trajectory(self, session_id: str, ref_ts_ns: Any | None = None) -> Any:
        """Trajectory"""
        return self._call("GET", f"/api/lidar/sessions/{session_id}/trajectory",
                          params={"ref_ts_ns": ref_ts_ns}, json_body=None)

    def post_lidar_sessions_by_session_id_validate(self, session_id: str) -> Any:
        """Validate Lidar"""
        return self._call("POST", f"/api/lidar/sessions/{session_id}/validate",
                          params=None, json_body=None)

    def post_lidar_tracks3d_by_track_3d_id_interpolate(self, track_3d_id: str) -> Any:
        """Interpolate Track"""
        return self._call("POST", f"/api/lidar/tracks3d/{track_3d_id}/interpolate",
                          params=None, json_body=None)

    def get_lineage_dataset_by_commit_id(self, commit_id: str) -> Any:
        """Dataset Lineage"""
        return self._call("GET", f"/api/lineage/dataset/{commit_id}",
                          params=None, json_body=None)

    def get_lineage_model_by_model_version(self, model_version: str, max_sessions: int | None = 40) -> Any:
        """Model Lineage"""
        return self._call("GET", f"/api/lineage/model/{model_version}",
                          params={"max_sessions": max_sessions}, json_body=None)

    def get_lineage_session_by_session_id(self, session_id: str) -> Any:
        """Session Lineage"""
        return self._call("GET", f"/api/lineage/session/{session_id}",
                          params=None, json_body=None)

    def post_mapassist_match(self, session_id: str, max_dist_m: float | None = 30.0) -> Any:
        """Match"""
        return self._call("POST", f"/api/mapassist/match",
                          params={"session_id": session_id, "max_dist_m": max_dist_m}, json_body=None)

    def get_mapassist_priors(self, frame_id: str) -> Any:
        """Priors"""
        return self._call("GET", f"/api/mapassist/priors",
                          params={"frame_id": frame_id}, json_body=None)

    def post_mask_compose(self, body: Any = None) -> Any:
        """Compose"""
        return self._call("POST", f"/api/mask/compose",
                          params=None, json_body=body)

    def get_metrics(self) -> Any:
        """Metrics"""
        return self._call("GET", f"/api/metrics",
                          params=None, json_body=None)

    def post_mine(self, body: Any = None) -> Any:
        """Mine"""
        return self._call("POST", f"/api/mine",
                          params=None, json_body=body)

    def get_models(self, limit: int | None = 200) -> Any:
        """List Models"""
        return self._call("GET", f"/api/models",
                          params={"limit": limit}, json_body=None)

    def post_multicam_associate(self, session_id: str) -> Any:
        """Associate"""
        return self._call("POST", f"/api/multicam/associate",
                          params={"session_id": session_id}, json_body=None)

    def post_multicam_consistency_check(self, session_id: str) -> Any:
        """Consistency Check Ep"""
        return self._call("POST", f"/api/multicam/consistency-check",
                          params={"session_id": session_id}, json_body=None)

    def get_multicam_group_at(self, session_id: str, ts_ns: int) -> Any:
        """Group At"""
        return self._call("GET", f"/api/multicam/group/at",
                          params={"session_id": session_id, "ts_ns": ts_ns}, json_body=None)

    def post_multicam_group_confirm(self, group_id: str, confirmed: bool | None = True) -> Any:
        """Group Confirm"""
        return self._call("POST", f"/api/multicam/group/confirm",
                          params={"group_id": group_id, "confirmed": confirmed}, json_body=None)

    def get_multicam_group_nav(self, session_id: str, group_id: str, direction: str | None = 'next') -> Any:
        """Group Nav"""
        return self._call("GET", f"/api/multicam/group/nav",
                          params={"session_id": session_id, "group_id": group_id, "direction": direction}, json_body=None)

    def get_multicam_groups(self, session_id: str, tol_ms: int | None = 20) -> Any:
        """Groups"""
        return self._call("GET", f"/api/multicam/groups",
                          params={"session_id": session_id, "tol_ms": tol_ms}, json_body=None)

    def post_multicam_groups_build(self, session_id: str, tol_ms: int | None = 20) -> Any:
        """Build Groups"""
        return self._call("POST", f"/api/multicam/groups/build",
                          params={"session_id": session_id, "tol_ms": tol_ms}, json_body=None)

    def get_multicam_groups_persisted(self, session_id: str) -> Any:
        """Persisted Groups"""
        return self._call("GET", f"/api/multicam/groups/persisted",
                          params={"session_id": session_id}, json_body=None)

    def post_multicam_link(self, body: Any = None) -> Any:
        """Link Ep"""
        return self._call("POST", f"/api/multicam/link",
                          params=None, json_body=body)

    def post_multicam_propagate(self, object_id: str, use_sam: bool | None = True) -> Any:
        """Propagate Ep"""
        return self._call("POST", f"/api/multicam/propagate",
                          params={"object_id": object_id, "use_sam": use_sam}, json_body=None)

    def get_multicam_rig_objects(self, session_id: str, group_id: str) -> Any:
        """Rig Objects Ep"""
        return self._call("GET", f"/api/multicam/rig-objects",
                          params={"session_id": session_id, "group_id": group_id}, json_body=None)

    def get_multicam_rig_track_timeline(self, session_id: str, rig_track_id: str) -> Any:
        """Rig Track Timeline Ep"""
        return self._call("GET", f"/api/multicam/rig-track/timeline",
                          params={"session_id": session_id, "rig_track_id": rig_track_id}, json_body=None)

    def get_multicam_rig_tracks(self, session_id: str) -> Any:
        """Rig Tracks Ep"""
        return self._call("GET", f"/api/multicam/rig-tracks",
                          params={"session_id": session_id}, json_body=None)

    def post_multicam_rig_tracks_build(self, session_id: str) -> Any:
        """Rig Tracks Build Ep"""
        return self._call("POST", f"/api/multicam/rig-tracks/build",
                          params={"session_id": session_id}, json_body=None)

    def get_multicam_suggest_links(self, session_id: str, group_id: str, appearance_cos: float | None = 0.55) -> Any:
        """Suggest Links Ep"""
        return self._call("GET", f"/api/multicam/suggest-links",
                          params={"session_id": session_id, "group_id": group_id, "appearance_cos": appearance_cos}, json_body=None)

    def post_multicam_unlink(self, object_id: str) -> Any:
        """Unlink Ep"""
        return self._call("POST", f"/api/multicam/unlink",
                          params={"object_id": object_id}, json_body=None)

    def get_notifications(self, unread_only: bool | None = False, limit: int | None = 50, offset: int | None = 0) -> Any:
        """List Notifications"""
        return self._call("GET", f"/api/notifications",
                          params={"unread_only": unread_only, "limit": limit, "offset": offset}, json_body=None)

    def get_notifications_count(self) -> Any:
        """Count Notifications"""
        return self._call("GET", f"/api/notifications/count",
                          params=None, json_body=None)

    def post_notifications_read_all(self) -> Any:
        """Read All"""
        return self._call("POST", f"/api/notifications/read-all",
                          params=None, json_body=None)

    def post_notifications_by_notification_id_read(self, notification_id: str) -> Any:
        """Read Notification"""
        return self._call("POST", f"/api/notifications/{notification_id}/read",
                          params=None, json_body=None)

    def post_objects_bulk_review(self, body: Any = None) -> Any:
        """Bulk Review"""
        return self._call("POST", f"/api/objects/bulk-review",
                          params=None, json_body=body)

    def post_objects_classify(self, body: Any = None) -> Any:
        """Classify Object"""
        return self._call("POST", f"/api/objects/classify",
                          params=None, json_body=body)

    def post_objects_quality_backfill(self, session_id: Any | None = None) -> Any:
        """Quality Backfill Ep"""
        return self._call("POST", f"/api/objects/quality/backfill",
                          params={"session_id": session_id}, json_body=None)

    def delete_objects_by_object_id(self, object_id: str) -> Any:
        """Delete Object"""
        return self._call("DELETE", f"/api/objects/{object_id}",
                          params=None, json_body=None)

    def get_objects_by_object_id(self, object_id: str) -> Any:
        """Get Object"""
        return self._call("GET", f"/api/objects/{object_id}",
                          params=None, json_body=None)

    def get_objects_by_object_id_crop(self, object_id: str, pad: float | None = 0.15) -> Any:
        """Object Crop"""
        return self._call("GET", f"/api/objects/{object_id}/crop",
                          params={"pad": pad}, json_body=None)

    def get_objects_by_object_id_explain(self, object_id: str) -> Any:
        """Explain Object Ep"""
        return self._call("GET", f"/api/objects/{object_id}/explain",
                          params=None, json_body=None)

    def post_objects_by_object_id_keyframe(self, object_id: str, value: bool | None = True) -> Any:
        """Set Keyframe"""
        return self._call("POST", f"/api/objects/{object_id}/keyframe",
                          params={"value": value}, json_body=None)

    def put_objects_by_object_id_mask(self, object_id: str, body: Any = None) -> Any:
        """Update Mask"""
        return self._call("PUT", f"/api/objects/{object_id}/mask",
                          params=None, json_body=body)

    def post_objects_by_object_id_propagate(self, object_id: str, frames: int | None = 12) -> Any:
        """Propagate Object"""
        return self._call("POST", f"/api/objects/{object_id}/propagate",
                          params={"frames": frames}, json_body=None)

    def get_objects_by_object_id_quality(self, object_id: str) -> Any:
        """Object Quality Ep"""
        return self._call("GET", f"/api/objects/{object_id}/quality",
                          params=None, json_body=None)

    def post_objects_by_object_id_reinterpolate(self, object_id: str, method: str | None = 'linear') -> Any:
        """Reinterpolate"""
        return self._call("POST", f"/api/objects/{object_id}/reinterpolate",
                          params={"method": method}, json_body=None)

    def post_objects_by_object_id_relate(self, object_id: str, body: Any = None) -> Any:
        """Relate Object"""
        return self._call("POST", f"/api/objects/{object_id}/relate",
                          params=None, json_body=body)

    def post_objects_by_object_id_review(self, object_id: str, body: Any = None) -> Any:
        """Review Object"""
        return self._call("POST", f"/api/objects/{object_id}/review",
                          params=None, json_body=body)

    def post_objects_by_object_id_sam_propagate(self, object_id: str, frames: int | None = 12, direction: str | None = 'both', refine: bool | None = True) -> Any:
        """Sam Propagate"""
        return self._call("POST", f"/api/objects/{object_id}/sam_propagate",
                          params={"frames": frames, "direction": direction, "refine": refine}, json_body=None)

    def get_objects_by_object_id_similar(self, object_id: str, limit: int | None = 12) -> Any:
        """Objects Similar"""
        return self._call("GET", f"/api/objects/{object_id}/similar",
                          params={"limit": limit}, json_body=None)

    def post_ocr_region(self, body: Any = None) -> Any:
        """Ocr Region"""
        return self._call("POST", f"/api/ocr/region",
                          params=None, json_body=body)

    def post_ocr_run(self, session_id: str, limit: Any | None = None) -> Any:
        """Run"""
        return self._call("POST", f"/api/ocr/run",
                          params={"session_id": session_id, "limit": limit}, json_body=None)

    def get_ontology(self) -> Any:
        """Ontology"""
        return self._call("GET", f"/api/ontology",
                          params=None, json_body=None)

    def post_ontology_classes(self, body: Any = None) -> Any:
        """Create Class"""
        return self._call("POST", f"/api/ontology/classes",
                          params=None, json_body=body)

    def get_oraclyx_board(self) -> Any:
        """Board"""
        return self._call("GET", f"/api/oraclyx/board",
                          params=None, json_body=None)

    def post_oraclyx_disagreements_rank(self, body: Any = None) -> Any:
        """Disagreements Rank"""
        return self._call("POST", f"/api/oraclyx/disagreements/rank",
                          params=None, json_body=body)

    def get_oraclyx_distillation(self, min_score: float | None = 0.5) -> Any:
        """Distillation"""
        return self._call("GET", f"/api/oraclyx/distillation",
                          params={"min_score": min_score}, json_body=None)

    def post_oraclyx_mono_depth(self, body: Any = None) -> Any:
        """Mono Depth"""
        return self._call("POST", f"/api/oraclyx/mono-depth",
                          params=None, json_body=body)

    def post_oraclyx_object_by_object_id_consensus(self, object_id: str, body: Any = None) -> Any:
        """Consensus"""
        return self._call("POST", f"/api/oraclyx/object/{object_id}/consensus",
                          params=None, json_body=body)

    def post_oraclyx_radar_fuse(self, body: Any = None) -> Any:
        """Radar Fuse"""
        return self._call("POST", f"/api/oraclyx/radar/fuse",
                          params=None, json_body=body)

    def post_oraclyx_tracks4d_stitch(self, body: Any = None) -> Any:
        """Tracks4D Stitch"""
        return self._call("POST", f"/api/oraclyx/tracks4d/stitch",
                          params=None, json_body=body)

    def post_oraclyx_uncertainty(self, body: Any = None) -> Any:
        """Uncertainty"""
        return self._call("POST", f"/api/oraclyx/uncertainty",
                          params=None, json_body=body)

    def get_platforms(self) -> Any:
        """List Platforms"""
        return self._call("GET", f"/api/platforms",
                          params=None, json_body=None)

    def get_predictions_by_prediction_id_crop(self, prediction_id: str, pad: float | None = 0.15) -> Any:
        """Prediction Crop"""
        return self._call("GET", f"/api/predictions/{prediction_id}/crop",
                          params={"pad": pad}, json_body=None)

    def get_projects_by_project_id_assets(self, project_id: str, state: Any | None = None, media_type: Any | None = None, limit: int | None = 200, offset: int | None = 0) -> Any:
        """List Assets"""
        return self._call("GET", f"/api/projects/{project_id}/assets",
                          params={"state": state, "media_type": media_type, "limit": limit, "offset": offset}, json_body=None)

    def post_projects_by_project_id_label_config(self, project_id: str, body: Any = None) -> Any:
        """Set Label Config"""
        return self._call("POST", f"/api/projects/{project_id}/label-config",
                          params=None, json_body=body)

    def get_projects_by_project_id_stats(self, project_id: str) -> Any:
        """Project Stats"""
        return self._call("GET", f"/api/projects/{project_id}/stats",
                          params=None, json_body=None)

    def post_qa_vlm(self, session_id: str, limit: int | None = 40) -> Any:
        """Qa Vlm"""
        return self._call("POST", f"/api/qa/vlm",
                          params={"session_id": session_id, "limit": limit}, json_body=None)

    def get_quality_attr_audit_by_session_id(self, session_id: str) -> Any:
        """Attr Audit"""
        return self._call("GET", f"/api/quality/attr-audit/{session_id}",
                          params=None, json_body=None)

    def post_quality_calibrate_fit(self, body: Any = None) -> Any:
        """Calibrate Fit"""
        return self._call("POST", f"/api/quality/calibrate/fit",
                          params=None, json_body=body)

    def get_quality_gold_sets(self) -> Any:
        """Gold Sets"""
        return self._call("GET", f"/api/quality/gold-sets",
                          params=None, json_body=None)

    def post_quality_gold_seal(self, body: Any = None) -> Any:
        """Seal"""
        return self._call("POST", f"/api/quality/gold/seal",
                          params=None, json_body=body)

    def post_quality_iaa(self, body: Any = None) -> Any:
        """Inter Annotator Agreement"""
        return self._call("POST", f"/api/quality/iaa",
                          params=None, json_body=body)

    def get_quality_sheet(self, gold_id: str) -> Any:
        """Sheet"""
        return self._call("GET", f"/api/quality/sheet",
                          params={"gold_id": gold_id}, json_body=None)

    def get_readyz(self) -> Any:
        """Readyz"""
        return self._call("GET", f"/api/readyz",
                          params=None, json_body=None)

    def get_reasoner_attribution(self, since_hours: Any | None = None) -> Any:
        """Attribution"""
        return self._call("GET", f"/api/reasoner/attribution",
                          params={"since_hours": since_hours}, json_body=None)

    def get_reasoner_coverage(self) -> Any:
        """Coverage"""
        return self._call("GET", f"/api/reasoner/coverage",
                          params=None, json_body=None)

    def post_reasoner_explain(self, body: Any = None) -> Any:
        """Explain"""
        return self._call("POST", f"/api/reasoner/explain",
                          params=None, json_body=body)

    def get_reasoner_outcomes(self, since_hours: Any | None = None) -> Any:
        """Outcomes"""
        return self._call("GET", f"/api/reasoner/outcomes",
                          params={"since_hours": since_hours}, json_body=None)

    def get_reasoner_priors(self) -> Any:
        """Priors"""
        return self._call("GET", f"/api/reasoner/priors",
                          params=None, json_body=None)

    def post_reasoner_rerun_by_session_id(self, session_id: str, limit: int | None = 500, apply: bool | None = False) -> Any:
        """Rerun"""
        return self._call("POST", f"/api/reasoner/rerun/{session_id}",
                          params={"limit": limit, "apply": apply}, json_body=None)

    def get_reasoner_trace_by_object_id(self, object_id: str) -> Any:
        """Trace"""
        return self._call("GET", f"/api/reasoner/trace/{object_id}",
                          params=None, json_body=None)

    def get_reasoner_weights(self, since_hours: Any | None = None) -> Any:
        """Suggested Weights"""
        return self._call("GET", f"/api/reasoner/weights",
                          params={"since_hours": since_hours}, json_body=None)

    def get_recall_candidates(self, status: str | None = 'pending', session_id: Any | None = None, limit: int | None = 200) -> Any:
        """Candidates"""
        return self._call("GET", f"/api/recall/candidates",
                          params={"status": status, "session_id": session_id, "limit": limit}, json_body=None)

    def post_recall_run_by_session_id(self, session_id: str, shortlist_only: bool | None = False) -> Any:
        """Run"""
        return self._call("POST", f"/api/recall/run/{session_id}",
                          params={"shortlist_only": shortlist_only}, json_body=None)

    def post_relabel_ingest(self, body: Any = None) -> Any:
        """Ingest"""
        return self._call("POST", f"/api/relabel/ingest",
                          params=None, json_body=body)

    def get_relabel_runs(self) -> Any:
        """Runs"""
        return self._call("GET", f"/api/relabel/runs",
                          params=None, json_body=None)

    def post_relabel_runs_by_run_id_revert(self, run_id: str) -> Any:
        """Revert"""
        return self._call("POST", f"/api/relabel/runs/{run_id}/revert",
                          params=None, json_body=None)

    def post_relabel_start(self, body: Any = None) -> Any:
        """Start"""
        return self._call("POST", f"/api/relabel/start",
                          params=None, json_body=body)

    def post_relations_by_relationship_id_status(self, relationship_id: str, status: str) -> Any:
        """Relation Status"""
        return self._call("POST", f"/api/relations/{relationship_id}/status",
                          params={"status": status}, json_body=None)

    def delete_relationships_by_relationship_id(self, relationship_id: str) -> Any:
        """Delete Relationship"""
        return self._call("DELETE", f"/api/relationships/{relationship_id}",
                          params=None, json_body=None)

    def get_release_gold_by_gold_id_lineage(self, gold_id: str) -> Any:
        """Lineage"""
        return self._call("GET", f"/api/release/gold/{gold_id}/lineage",
                          params=None, json_body=None)

    def get_release_registry(self, limit: int | None = 100) -> Any:
        """Registry"""
        return self._call("GET", f"/api/release/registry",
                          params={"limit": limit}, json_body=None)

    def get_release_by_commit_id_verify(self, commit_id: str) -> Any:
        """Verify"""
        return self._call("GET", f"/api/release/{commit_id}/verify",
                          params=None, json_body=None)

    def get_sanyx_board(self, limit: int | None = 100) -> Any:
        """Board"""
        return self._call("GET", f"/api/sanyx/board",
                          params={"limit": limit}, json_body=None)

    def get_sanyx_gate_by_session_id(self, session_id: str) -> Any:
        """Gate"""
        return self._call("GET", f"/api/sanyx/gate/{session_id}",
                          params=None, json_body=None)

    def get_sanyx_rig_by_vehicle_id_trends(self, vehicle_id: str) -> Any:
        """Rig Trends View"""
        return self._call("GET", f"/api/sanyx/rig/{vehicle_id}/trends",
                          params=None, json_body=None)

    def get_sanyx_session_by_session_id(self, session_id: str) -> Any:
        """Session Report"""
        return self._call("GET", f"/api/sanyx/session/{session_id}",
                          params=None, json_body=None)

    def post_sanyx_session_by_session_id_override(self, session_id: str, body: Any = None) -> Any:
        """Override"""
        return self._call("POST", f"/api/sanyx/session/{session_id}/override",
                          params=None, json_body=body)

    def post_sanyx_session_by_session_id_run(self, session_id: str) -> Any:
        """Run"""
        return self._call("POST", f"/api/sanyx/session/{session_id}/run",
                          params=None, json_body=None)

    def get_scenarios(self, session_id: Any | None = None, type: Any | None = None, city: Any | None = None, limit: int | None = 100) -> Any:
        """List Scenarios"""
        return self._call("GET", f"/api/scenarios",
                          params={"session_id": session_id, "type": type, "city": city, "limit": limit}, json_body=None)

    def get_scenarios_search(self, q: str, city: Any | None = None, session_id: Any | None = None, limit: int | None = 100, semantic: bool | None = False) -> Any:
        """Scenarios Search"""
        return self._call("GET", f"/api/scenarios/search",
                          params={"q": q, "city": city, "session_id": session_id, "limit": limit, "semantic": semantic}, json_body=None)

    def get_scenarios_by_scenario_id(self, scenario_id: str) -> Any:
        """Scenario Detail"""
        return self._call("GET", f"/api/scenarios/{scenario_id}",
                          params=None, json_body=None)

    def get_scene_graph_vocab(self) -> Any:
        """Scene Graph Vocab"""
        return self._call("GET", f"/api/scene-graph/vocab",
                          params=None, json_body=None)

    def post_scene_classify(self, session_id: Any | None = None, limit: Any | None = None) -> Any:
        """Scene Classify"""
        return self._call("POST", f"/api/scene/classify",
                          params={"session_id": session_id, "limit": limit}, json_body=None)

    def get_search_objects(self, q: str, session_id: Any | None = None, limit: int | None = 24) -> Any:
        """Search Objects"""
        return self._call("GET", f"/api/search/objects",
                          params={"q": q, "session_id": session_id, "limit": limit}, json_body=None)

    def get_search_semantic(self, q: str, k: int | None = 24) -> Any:
        """Search Semantic"""
        return self._call("GET", f"/api/search/semantic",
                          params={"q": q, "k": k}, json_body=None)

    def post_search_similar(self, body: Any = None) -> Any:
        """Search Similar"""
        return self._call("POST", f"/api/search/similar",
                          params=None, json_body=body)

    def get_sec_identities(self, min_cameras: int | None = 1, limit: int | None = 100) -> Any:
        """List Identities"""
        return self._call("GET", f"/api/sec/identities",
                          params={"min_cameras": min_cameras, "limit": limit}, json_body=None)

    def delete_sec_identities_by_identity_id(self, identity_id: str) -> Any:
        """Forget Identity"""
        return self._call("DELETE", f"/api/sec/identities/{identity_id}",
                          params=None, json_body=None)

    def get_sec_identities_by_identity_id(self, identity_id: str) -> Any:
        """Identity Detail"""
        return self._call("GET", f"/api/sec/identities/{identity_id}",
                          params=None, json_body=None)

    def get_sec_incidents(self, camera_id: Any | None = None, kind: Any | None = None, status: Any | None = None, severity: Any | None = None, since_hours: Any | None = None, limit: int | None = 100, offset: int | None = 0) -> Any:
        """List Incidents"""
        return self._call("GET", f"/api/sec/incidents",
                          params={"camera_id": camera_id, "kind": kind, "status": status, "severity": severity, "since_hours": since_hours, "limit": limit, "offset": offset}, json_body=None)

    def post_sec_incidents_by_incident_id_acknowledge(self, incident_id: str, close: bool | None = False) -> Any:
        """Acknowledge"""
        return self._call("POST", f"/api/sec/incidents/{incident_id}/acknowledge",
                          params={"close": close}, json_body=None)

    def post_sec_rtsp_ingest(self, body: Any = None) -> Any:
        """Rtsp Ingest"""
        return self._call("POST", f"/api/sec/rtsp/ingest",
                          params=None, json_body=body)

    def post_sec_sessions_by_session_id_evaluate(self, session_id: str) -> Any:
        """Evaluate Session"""
        return self._call("POST", f"/api/sec/sessions/{session_id}/evaluate",
                          params=None, json_body=None)

    def post_sec_sessions_by_session_id_link_identities(self, session_id: str) -> Any:
        """Link Identities"""
        return self._call("POST", f"/api/sec/sessions/{session_id}/link-identities",
                          params=None, json_body=None)

    def post_sec_sessions_by_session_id_stitch_plates(self, session_id: str) -> Any:
        """Stitch Plates"""
        return self._call("POST", f"/api/sec/sessions/{session_id}/stitch-plates",
                          params=None, json_body=None)

    def get_sec_zones(self, camera_id: Any | None = None, active_only: bool | None = True) -> Any:
        """List Zones"""
        return self._call("GET", f"/api/sec/zones",
                          params={"camera_id": camera_id, "active_only": active_only}, json_body=None)

    def post_sec_zones(self, body: Any = None) -> Any:
        """Create Zone"""
        return self._call("POST", f"/api/sec/zones",
                          params=None, json_body=body)

    def delete_sec_zones_by_zone_id(self, zone_id: str) -> Any:
        """Delete Zone"""
        return self._call("DELETE", f"/api/sec/zones/{zone_id}",
                          params=None, json_body=None)

    def get_security_pack(self, pack_id: str | None = 'sec') -> Any:
        """Security Pack"""
        return self._call("GET", f"/api/security/pack",
                          params={"pack_id": pack_id}, json_body=None)

    def get_security_reads(self, session_id: Any | None = None, camera_id: Any | None = None, plate: Any | None = None, state_code: Any | None = None, hits_only: bool | None = False, limit: int | None = 100, offset: int | None = 0) -> Any:
        """Get Reads"""
        return self._call("GET", f"/api/security/reads",
                          params={"session_id": session_id, "camera_id": camera_id, "plate": plate, "state_code": state_code, "hits_only": hits_only, "limit": limit, "offset": offset}, json_body=None)

    def post_security_recognize(self, body: Any = None) -> Any:
        """Recognize"""
        return self._call("POST", f"/api/security/recognize",
                          params=None, json_body=body)

    def get_security_sessions(self, limit: int | None = 100, offset: int | None = 0) -> Any:
        """Security Sessions"""
        return self._call("GET", f"/api/security/sessions",
                          params={"limit": limit, "offset": offset}, json_body=None)

    def get_security_stats(self) -> Any:
        """Get Stats"""
        return self._call("GET", f"/api/security/stats",
                          params=None, json_body=None)

    def get_security_watchlist(self, active_only: bool | None = True, limit: int | None = 500) -> Any:
        """Get Watchlist"""
        return self._call("GET", f"/api/security/watchlist",
                          params={"active_only": active_only, "limit": limit}, json_body=None)

    def post_security_watchlist(self, body: Any = None) -> Any:
        """Add Watchlist"""
        return self._call("POST", f"/api/security/watchlist",
                          params=None, json_body=body)

    def delete_security_watchlist_by_entry_id(self, entry_id: str) -> Any:
        """Delete Watchlist"""
        return self._call("DELETE", f"/api/security/watchlist/{entry_id}",
                          params=None, json_body=None)

    def post_segment(self, body: Any = None) -> Any:
        """Segment"""
        return self._call("POST", f"/api/segment",
                          params=None, json_body=body)

    def get_sessions(self, limit: int | None = 200) -> Any:
        """Sessions"""
        return self._call("GET", f"/api/sessions",
                          params={"limit": limit}, json_body=None)

    def get_sessions_page(self, limit: int | None = 50, offset: int | None = 0, vehicle_id: Any | None = None) -> Any:
        """Sessions Page"""
        return self._call("GET", f"/api/sessions/page",
                          params={"limit": limit, "offset": offset, "vehicle_id": vehicle_id}, json_body=None)

    def get_sessions_by_session_id_driving_events(self, session_id: str, kind: Any | None = None, state: Any | None = None, severity: Any | None = None, track_id: Any | None = None, limit: int | None = 2000) -> Any:
        """List Driving Events"""
        return self._call("GET", f"/api/sessions/{session_id}/driving-events",
                          params={"kind": kind, "state": state, "severity": severity, "track_id": track_id, "limit": limit}, json_body=None)

    def post_sessions_by_session_id_driving_events(self, session_id: str, body: Any = None) -> Any:
        """Create"""
        return self._call("POST", f"/api/sessions/{session_id}/driving-events",
                          params=None, json_body=body)

    def post_sessions_by_session_id_driving_events_derive(self, session_id: str, prune_stale: bool | None = True) -> Any:
        """Derive"""
        return self._call("POST", f"/api/sessions/{session_id}/driving-events/derive",
                          params={"prune_stale": prune_stale}, json_body=None)

    def post_sessions_by_session_id_driving_events_preview(self, session_id: str) -> Any:
        """Preview"""
        return self._call("POST", f"/api/sessions/{session_id}/driving-events/preview",
                          params=None, json_body=None)

    def get_sessions_by_session_id_driving_events_summary(self, session_id: str) -> Any:
        """Summary"""
        return self._call("GET", f"/api/sessions/{session_id}/driving-events/summary",
                          params=None, json_body=None)

    def get_sessions_by_session_id_egostate(self, session_id: str) -> Any:
        """Ego State"""
        return self._call("GET", f"/api/sessions/{session_id}/egostate",
                          params=None, json_body=None)

    def get_sessions_by_session_id_events(self, session_id: str, modality: Any | None = None) -> Any:
        """List Timeline Events"""
        return self._call("GET", f"/api/sessions/{session_id}/events",
                          params={"modality": modality}, json_body=None)

    def post_sessions_by_session_id_events(self, session_id: str, body: Any = None) -> Any:
        """Create Timeline Event"""
        return self._call("POST", f"/api/sessions/{session_id}/events",
                          params=None, json_body=body)

    def post_sessions_by_session_id_events_auto(self, session_id: str) -> Any:
        """Persist Auto Events"""
        return self._call("POST", f"/api/sessions/{session_id}/events/auto",
                          params=None, json_body=None)

    def post_sessions_by_session_id_events_correlate(self, session_id: str, ts_ns: int, window_ns: int | None = 250000000) -> Any:
        """Correlate Timeline Event"""
        return self._call("POST", f"/api/sessions/{session_id}/events/correlate",
                          params={"ts_ns": ts_ns, "window_ns": window_ns}, json_body=None)

    def post_sessions_by_session_id_events_scene(self, session_id: str) -> Any:
        """Persist Scene Events Ep"""
        return self._call("POST", f"/api/sessions/{session_id}/events/scene",
                          params=None, json_body=None)

    def get_sessions_by_session_id_first_frame(self, session_id: str) -> Any:
        """First Frame"""
        return self._call("GET", f"/api/sessions/{session_id}/first-frame",
                          params=None, json_body=None)

    def get_sessions_by_session_id_inertial_events(self, session_id: str) -> Any:
        """Inertial Events"""
        return self._call("GET", f"/api/sessions/{session_id}/inertial_events",
                          params=None, json_body=None)

    def post_sessions_by_session_id_lanes_classify(self, session_id: str, apply: bool | None = True, reclassify: bool | None = False) -> Any:
        """Classify Lanes"""
        return self._call("POST", f"/api/sessions/{session_id}/lanes/classify",
                          params={"apply": apply, "reclassify": reclassify}, json_body=None)

    def post_sessions_by_session_id_lanes_link(self, session_id: str, apply: bool | None = True) -> Any:
        """Link Lanes"""
        return self._call("POST", f"/api/sessions/{session_id}/lanes/link",
                          params={"apply": apply}, json_body=None)

    def get_sessions_by_session_id_qa_consistency(self, session_id: str) -> Any:
        """Qa Consistency"""
        return self._call("GET", f"/api/sessions/{session_id}/qa/consistency",
                          params=None, json_body=None)

    def get_sessions_by_session_id_static_dynamic(self, session_id: str) -> Any:
        """Session Static Dynamic"""
        return self._call("GET", f"/api/sessions/{session_id}/static-dynamic",
                          params=None, json_body=None)

    def get_sessions_by_session_id_stats(self, session_id: str) -> Any:
        """Session Stats"""
        return self._call("GET", f"/api/sessions/{session_id}/stats",
                          params=None, json_body=None)

    def get_sessions_by_session_id_timeline(self, session_id: str) -> Any:
        """Timeline"""
        return self._call("GET", f"/api/sessions/{session_id}/timeline",
                          params=None, json_body=None)

    def get_sievyx_composition(self, session_id: Any | None = None, top_n: int | None = 500) -> Any:
        """Composition"""
        return self._call("GET", f"/api/sievyx/composition",
                          params={"session_id": session_id, "top_n": top_n}, json_body=None)

    def get_sievyx_discover(self, limit: int | None = 400, min_size: int | None = 2) -> Any:
        """Discover"""
        return self._call("GET", f"/api/sievyx/discover",
                          params={"limit": limit, "min_size": min_size}, json_body=None)

    def post_sievyx_maneuver(self, body: Any = None) -> Any:
        """Maneuver"""
        return self._call("POST", f"/api/sievyx/maneuver",
                          params=None, json_body=body)

    def post_sievyx_odd_gaps(self, body: Any = None) -> Any:
        """Odd Gaps"""
        return self._call("POST", f"/api/sievyx/odd/gaps",
                          params=None, json_body=body)

    def get_sievyx_priority(self, session_id: Any | None = None, limit: int | None = 100) -> Any:
        """Priority"""
        return self._call("GET", f"/api/sievyx/priority",
                          params={"session_id": session_id, "limit": limit}, json_body=None)

    def post_signs_recognize(self, session_id: str, limit: Any | None = None) -> Any:
        """Recognize"""
        return self._call("POST", f"/api/signs/recognize",
                          params={"session_id": session_id, "limit": limit}, json_body=None)

    def get_signs_taxonomy(self) -> Any:
        """Taxonomy"""
        return self._call("GET", f"/api/signs/taxonomy",
                          params=None, json_body=None)

    def post_superpixels_by_frame_id(self, frame_id: str, n: int | None = 300) -> Any:
        """Superpixels"""
        return self._call("POST", f"/api/superpixels/{frame_id}",
                          params={"n": n}, json_body=None)

    def post_tracklets_objects_by_object_id_keyframe(self, object_id: str, body: Any = None) -> Any:
        """Set Keyframe"""
        return self._call("POST", f"/api/tracklets/objects/{object_id}/keyframe",
                          params=None, json_body=body)

    def post_tracklets_objects_by_object_id_propagate(self, object_id: str, direction: str | None = 'both', frames: int | None = 12, refine: bool | None = True) -> Any:
        """Propagate"""
        return self._call("POST", f"/api/tracklets/objects/{object_id}/propagate",
                          params={"direction": direction, "frames": frames, "refine": refine}, json_body=None)

    def get_tracklets_stats_summary(self, session_id: Any | None = None) -> Any:
        """Tracklet Stats"""
        return self._call("GET", f"/api/tracklets/stats/summary",
                          params={"session_id": session_id}, json_body=None)

    def get_tracklets_by_track_id(self, track_id: str) -> Any:
        """Load Tracklet"""
        return self._call("GET", f"/api/tracklets/{track_id}",
                          params=None, json_body=None)

    def post_tracklets_by_track_id_attributes(self, track_id: str, body: Any = None) -> Any:
        """Set Track Attributes"""
        return self._call("POST", f"/api/tracklets/{track_id}/attributes",
                          params=None, json_body=body)

    def post_tracklets_by_track_id_derive(self, track_id: str, method: str | None = 'linear', overwrite_human: bool | None = False) -> Any:
        """Derive"""
        return self._call("POST", f"/api/tracklets/{track_id}/derive",
                          params={"method": method, "overwrite_human": overwrite_human}, json_body=None)

    def get_tracklets_by_track_id_suggest_keyframes(self, track_id: str, budget: int | None = 8) -> Any:
        """Suggest Keyframes"""
        return self._call("GET", f"/api/tracklets/{track_id}/suggest-keyframes",
                          params={"budget": budget}, json_body=None)

    def post_tracks_retrack(self, session_id: str) -> Any:
        """Retrack"""
        return self._call("POST", f"/api/tracks/retrack",
                          params={"session_id": session_id}, json_body=None)

    def delete_tracks_by_track_id(self, track_id: str) -> Any:
        """Delete Track"""
        return self._call("DELETE", f"/api/tracks/{track_id}",
                          params=None, json_body=None)

    def get_tracks_by_track_id(self, track_id: str) -> Any:
        """Get Track"""
        return self._call("GET", f"/api/tracks/{track_id}",
                          params=None, json_body=None)

    def get_tracks_by_track_id_attribute_timeline(self, track_id: str, key: str) -> Any:
        """Attribute Timeline"""
        return self._call("GET", f"/api/tracks/{track_id}/attribute-timeline",
                          params={"key": key}, json_body=None)

    def get_tracks_by_track_id_driving_events(self, track_id: str) -> Any:
        """Track Events"""
        return self._call("GET", f"/api/tracks/{track_id}/driving-events",
                          params=None, json_body=None)

    def post_tracks_by_track_id_intent_propose(self, track_id: str) -> Any:
        """Intent Propose"""
        return self._call("POST", f"/api/tracks/{track_id}/intent/propose",
                          params=None, json_body=None)

    def post_tracks_by_track_id_intent_set(self, track_id: str, body: Any = None) -> Any:
        """Intent Set"""
        return self._call("POST", f"/api/tracks/{track_id}/intent/set",
                          params=None, json_body=body)

    def post_tracks_by_track_id_intent_vlm(self, track_id: str) -> Any:
        """Intent Vlm"""
        return self._call("POST", f"/api/tracks/{track_id}/intent/vlm",
                          params=None, json_body=None)

    def post_tracks_by_track_id_interpolate(self, track_id: str) -> Any:
        """Interpolate"""
        return self._call("POST", f"/api/tracks/{track_id}/interpolate",
                          params=None, json_body=None)

    def post_tracks_by_track_id_interpolate_keyframed(self, track_id: str, method: str | None = 'linear') -> Any:
        """Interpolate Keyframed"""
        return self._call("POST", f"/api/tracks/{track_id}/interpolate-keyframed",
                          params={"method": method}, json_body=None)

    def post_tracks_by_track_id_merge(self, track_id: str, body: Any = None) -> Any:
        """Merge Track Ep"""
        return self._call("POST", f"/api/tracks/{track_id}/merge",
                          params=None, json_body=body)

    def post_tracks_by_track_id_relabel(self, track_id: str, body: Any = None) -> Any:
        """Relabel Track"""
        return self._call("POST", f"/api/tracks/{track_id}/relabel",
                          params=None, json_body=body)

    def get_tracks_by_track_id_seg4d_consistency(self, track_id: str, window: int | None = 2) -> Any:
        """Seg4D Consistency"""
        return self._call("GET", f"/api/tracks/{track_id}/seg4d-consistency",
                          params={"window": window}, json_body=None)

    def post_tracks_by_track_id_smooth(self, track_id: str, window: int | None = 5) -> Any:
        """Smooth Track Path"""
        return self._call("POST", f"/api/tracks/{track_id}/smooth",
                          params={"window": window}, json_body=None)

    def post_tracks_by_track_id_split(self, track_id: str, body: Any = None) -> Any:
        """Split Track Ep"""
        return self._call("POST", f"/api/tracks/{track_id}/split",
                          params=None, json_body=body)

    def get_training(self, limit: int | None = 50) -> Any:
        """List Jobs"""
        return self._call("GET", f"/api/training",
                          params={"limit": limit}, json_body=None)

    def get_training_registry(self) -> Any:
        """Registry"""
        return self._call("GET", f"/api/training/registry",
                          params=None, json_body=None)

    def get_training_runs(self, limit: int | None = 50) -> Any:
        """Runs"""
        return self._call("GET", f"/api/training/runs",
                          params={"limit": limit}, json_body=None)

    def get_training_runs_by_run_id_curve(self, run_id: str) -> Any:
        """Run Curve"""
        return self._call("GET", f"/api/training/runs/{run_id}/curve",
                          params=None, json_body=None)

    def post_training_start(self, body: Any = None) -> Any:
        """Start"""
        return self._call("POST", f"/api/training/start",
                          params=None, json_body=body)

    def post_training_sweep(self, body: Any = None) -> Any:
        """Start Sweep Endpoint"""
        return self._call("POST", f"/api/training/sweep",
                          params=None, json_body=body)

    def get_training_sweep_by_name(self, name: str, metric: str | None = 'map50') -> Any:
        """Sweep Status"""
        return self._call("GET", f"/api/training/sweep/{name}",
                          params={"metric": metric}, json_body=None)

    def get_training_tasks(self) -> Any:
        """Tasks"""
        return self._call("GET", f"/api/training/tasks",
                          params=None, json_body=None)

    def get_training_by_job_id(self, job_id: str) -> Any:
        """Status"""
        return self._call("GET", f"/api/training/{job_id}",
                          params=None, json_body=None)

    def post_training_by_job_id_cancel(self, job_id: str) -> Any:
        """Cancel"""
        return self._call("POST", f"/api/training/{job_id}/cancel",
                          params=None, json_body=None)

    def get_triage(self, states: str | None = 'review,annotate', session_id: Any | None = None, klass: Any | None = None, city: Any | None = None, flywheel: Any | None = None, limit: int | None = 200) -> Any:
        """Triage"""
        return self._call("GET", f"/api/triage",
                          params={"states": states, "session_id": session_id, "klass": klass, "city": city, "flywheel": flywheel, "limit": limit}, json_body=None)

    def post_upload_abort(self, body: Any = None) -> Any:
        """Abort"""
        return self._call("POST", f"/api/upload/abort",
                          params=None, json_body=body)

    def post_upload_complete(self, body: Any = None) -> Any:
        """Complete"""
        return self._call("POST", f"/api/upload/complete",
                          params=None, json_body=body)

    def post_upload_init(self, body: Any = None) -> Any:
        """Init"""
        return self._call("POST", f"/api/upload/init",
                          params=None, json_body=body)

    def post_upload_presign_put(self, body: Any = None) -> Any:
        """Presign Put"""
        return self._call("POST", f"/api/upload/presign-put",
                          params=None, json_body=body)

    def post_upload_sign(self, body: Any = None) -> Any:
        """Sign"""
        return self._call("POST", f"/api/upload/sign",
                          params=None, json_body=body)

    def get_users(self, limit: int | None = 200, offset: int | None = 0) -> Any:
        """List Users"""
        return self._call("GET", f"/api/users",
                          params={"limit": limit, "offset": offset}, json_body=None)

    def post_users(self, body: Any = None) -> Any:
        """Create User"""
        return self._call("POST", f"/api/users",
                          params=None, json_body=body)

    def get_users_me(self) -> Any:
        """Whoami"""
        return self._call("GET", f"/api/users/me",
                          params=None, json_body=None)

    def post_users_by_user_id_revoke_tokens(self, user_id: str) -> Any:
        """Revoke Tokens"""
        return self._call("POST", f"/api/users/{user_id}/revoke-tokens",
                          params=None, json_body=None)

    def post_users_by_user_id_token(self, user_id: str) -> Any:
        """Reissue Token"""
        return self._call("POST", f"/api/users/{user_id}/token",
                          params=None, json_body=None)

    def post_verdyx_evaluate(self, body: Any = None) -> Any:
        """Evaluate"""
        return self._call("POST", f"/api/verdyx/evaluate",
                          params=None, json_body=body)

    def get_verdyx_matrix(self, champion: str, challenger: str, slice_metric: str | None = 'map') -> Any:
        """Slice Matrix View"""
        return self._call("GET", f"/api/verdyx/matrix",
                          params={"champion": champion, "challenger": challenger, "slice_metric": slice_metric}, json_body=None)

    def get_verdyx_model_by_model_version_evals(self, model_version: str, limit: int | None = 20) -> Any:
        """Model Evals"""
        return self._call("GET", f"/api/verdyx/model/{model_version}/evals",
                          params={"limit": limit}, json_body=None)

    def get_verdyx_pairs(self) -> Any:
        """Pairs"""
        return self._call("GET", f"/api/verdyx/pairs",
                          params=None, json_body=None)

    def post_verdyx_safety_recall(self, body: Any = None) -> Any:
        """Safety Recall"""
        return self._call("POST", f"/api/verdyx/safety/recall",
                          params=None, json_body=body)

    def post_verdyx_shadow_triage(self, body: Any = None) -> Any:
        """Shadow Triage"""
        return self._call("POST", f"/api/verdyx/shadow/triage",
                          params=None, json_body=body)

    def post_verdyx_stats_bootstrap(self, body: Any = None) -> Any:
        """Stats Bootstrap"""
        return self._call("POST", f"/api/verdyx/stats/bootstrap",
                          params=None, json_body=body)

    def post_verdyx_stats_significance(self, body: Any = None) -> Any:
        """Stats Significance"""
        return self._call("POST", f"/api/verdyx/stats/significance",
                          params=None, json_body=body)

    def get_vlm_dataset_export(self, session_id: Any | None = None) -> Any:
        """Vlm Dataset Export"""
        return self._call("GET", f"/api/vlm-dataset/export",
                          params={"session_id": session_id}, json_body=None)

    def post_vlm_targets_by_target_id_status(self, target_id: str, status: str) -> Any:
        """Vlm Target Status"""
        return self._call("POST", f"/api/vlm-targets/{target_id}/status",
                          params={"status": status}, json_body=None)

    def get_metrics_2(self) -> Any:
        """Metrics Endpoint"""
        return self._call("GET", f"/metrics",
                          params=None, json_body=None)



def _clean(params: dict | None) -> dict | None:
    """Drop unset query parameters.

    Sending them as the string "None" is the classic generated-client bug: the server sees a value where
    the caller meant absence, and a filter nobody asked for silently applies.
    """
    if not params:
        return None
    return {k: v for k, v in params.items() if v is not None}


# 594 routes generated from the server schema.
