"""A small Python client for the LabeloxAV REST API.

Deliberately thin: it maps endpoints to methods and does not reimplement any server logic, so it cannot drift
into a second, subtly different definition of what a project or an annotation is. Everything it returns is the
server's own JSON.

    from sdk.labelox_client import Labelox
    lbx = Labelox("http://localhost:8000", token="lbx1....")
    lbx.create_assets(project_id, [{"media_type": "text", "text": "hello"}])

Auth: pass the signed Bearer token issued by POST /api/users. On a dev server with auth disabled you can pass
`user_id=` instead, which sends the legacy identity header the server only honours in that mode.
"""

from __future__ import annotations

from typing import Any

import httpx

DEFAULT_TIMEOUT = 30.0


class LabeloxError(RuntimeError):
    """A non-2xx response. Carries the status and the server's message."""

    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class Labelox:
    def __init__(self, base_url: str = "http://localhost:8000", *, token: str | None = None,
                 user_id: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.base = base_url.rstrip("/")
        self._headers: dict[str, str] = {}
        if token:
            self._headers["Authorization"] = f"Bearer {token}"
        elif user_id:
            # Only honoured by a server running with auth disabled; kept for local dev convenience.
            self._headers["X-Lbx-User-Id"] = user_id
        self._client = httpx.Client(timeout=timeout)

    # ---- plumbing -----------------------------------------------------------------------------------
    def _req(self, method: str, path: str, **kw) -> Any:
        r = self._client.request(method, f"{self.base}{path}", headers=self._headers, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except Exception:  # noqa: BLE001
                detail = r.text
            raise LabeloxError(r.status_code, detail)
        return r.json() if r.content else None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Labelox:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- health / meta ------------------------------------------------------------------------------
    def health(self) -> dict:
        return self._req("GET", "/api/health")

    def metrics(self) -> dict:
        return self._req("GET", "/api/metrics")

    # ---- projects and jobs --------------------------------------------------------------------------
    def projects(self) -> list[dict]:
        return self._req("GET", "/api/labelops/projects")["projects"]

    def create_project(self, name: str, **kw) -> dict:
        return self._req("POST", "/api/labelops/projects", json={"name": name, **kw})

    def create_task(self, project_id: str, name: str, **kw) -> dict:
        return self._req("POST", "/api/labelops/tasks",
                         json={"project_id": project_id, "name": name, **kw})

    def jobs(self, **q) -> list[dict]:
        return self._req("GET", "/api/labelops/jobs", params=q)["jobs"]

    def assign(self, job_id: str, assignee_id: str | None, expected_version: int | None = None) -> dict:
        return self._req("POST", f"/api/labelops/jobs/{job_id}/assign",
                         json={"assignee_id": assignee_id, "expected_version": expected_version})

    def submit(self, job_id: str, expected_version: int | None = None) -> dict:
        return self._req("POST", f"/api/labelops/jobs/{job_id}/submit",
                         json={"expected_version": expected_version})

    def scorecards(self, project_id: str | None = None) -> list[dict]:
        p = {"project_id": project_id} if project_id else {}
        return self._req("GET", "/api/labelops/scorecards", params=p)["scorecards"]

    # ---- assets and annotations ---------------------------------------------------------------------
    def create_assets(self, project_id: str, items: list[dict]) -> dict:
        return self._req("POST", "/api/assets", json={"project_id": project_id, "items": items})

    def assets(self, project_id: str, **q) -> dict:
        return self._req("GET", f"/api/projects/{project_id}/assets", params=q)

    def asset(self, asset_id: str) -> dict:
        return self._req("GET", f"/api/assets/{asset_id}")

    def annotate(self, asset_id: str, kind: str, **kw) -> dict:
        return self._req("POST", f"/api/assets/{asset_id}/annotations", json={"kind": kind, **kw})

    def set_label_config(self, project_id: str, config: dict) -> dict:
        return self._req("POST", f"/api/projects/{project_id}/label-config", json=config)

    # ---- explore ------------------------------------------------------------------------------------
    def facets(self, predicate: dict | None = None) -> dict:
        return self._req("POST", "/api/explore/facets", json=predicate or {})

    def select(self, predicate: dict | None = None, level: str = "object", limit: int = 5000) -> dict:
        return self._req("POST", "/api/explore/select", json=predicate or {},
                         params={"level": level, "limit": limit})

    def tag(self, predicate: dict, add: list[str] | None = None, remove: list[str] | None = None,
            level: str = "object") -> dict:
        return self._req("POST", "/api/explore/tag",
                         json={"level": level, "predicate": predicate,
                               "add": add or [], "remove": remove or []})

    def fit_projection(self, **kw) -> dict:
        return self._req("POST", "/api/explore/projection", json=kw)

    def views(self) -> list[dict]:
        return self._req("GET", "/api/explore/views")["views"]

    # ---- import / export ----------------------------------------------------------------------------
    def start_import(self, fmt: str, source_uri: str, **kw) -> dict:
        return self._req("POST", "/api/imports/start",
                         json={"format": fmt, "source_uri": source_uri, **kw})

    def export(self, name: str, formats: list[str] | None = None, **kw) -> dict:
        return self._req("POST", "/api/export",
                         json={"name": name, "formats": formats or ["coco", "parquet"], **kw})

    # ---- integrations -------------------------------------------------------------------------------
    def create_webhook(self, url: str, events: list[str] | None = None, **kw) -> dict:
        return self._req("POST", "/api/integrations/webhooks",
                         json={"url": url, "events": events or [], **kw})

    def webhooks(self) -> list[dict]:
        return self._req("GET", "/api/integrations/webhooks")["webhooks"]

    def register_source(self, name: str, provider: str, bucket: str, **kw) -> dict:
        return self._req("POST", "/api/integrations/sources",
                         json={"name": name, "provider": provider, "bucket": bucket, **kw})
