"""Programmatic RunPod orchestrator. Where cloud/runpod_api.py is a CLI (prints JSON, exits on error) for
the bash runbook, this returns values and raises RunpodError, so the warm-session manager can drive it.

The GraphQL queries mirror cloud/runpod_api.py (the RunPod API drifts; verify on first run). Two teardown
verbs matter and differ in billing: terminate() fully removes the pod and stops ALL billing; pause() is the
RunPod stop, which keeps the pod for a fast reconnect but still bills for the network volume.

The Orchestrator Protocol lets the manager be tested against a fake pod with no network and no billing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

GQL = "https://api.runpod.io/graphql"


class RunpodError(RuntimeError):
    """Any failure talking to RunPod (auth, transport, or a GraphQL error)."""


@dataclass(frozen=True)
class PodInfo:
    id: str
    status: str               # RunPod desiredStatus: RUNNING | EXITED | TERMINATED | ...
    name: str | None = None
    gpu_type: str | None = None
    uptime_s: int = 0
    ssh_ip: str | None = None
    ssh_port: int | None = None

    @property
    def is_running(self) -> bool:
        return self.status == "RUNNING"

    @property
    def is_gone(self) -> bool:
        # terminated/removed pods stop billing; a pod the API no longer returns is also gone
        return self.status in ("TERMINATED", "GONE")


class Orchestrator(Protocol):
    """The pod operations the warm-session manager needs. Implemented for real by RunpodOrchestrator and
    for tests by a fake, so cost-safety can be verified without a billable pod."""

    def gpu_types(self) -> list[dict]: ...
    def provision(self, gpu_type_id: str, image: str, volume_id: str | None) -> PodInfo: ...
    def status(self, pod_id: str) -> PodInfo: ...
    def terminate(self, pod_id: str) -> None: ...
    def pause(self, pod_id: str) -> None: ...
    def list_pods(self) -> list[PodInfo]: ...


def _pod_from_node(node: dict) -> PodInfo:
    rt = node.get("runtime") or {}
    ssh_ip = ssh_port = None
    for p in (rt.get("ports") or []):
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            ssh_ip, ssh_port = p.get("ip"), p.get("publicPort")
    machine = node.get("machine") or {}
    return PodInfo(
        id=node.get("id"),
        status=node.get("desiredStatus") or "UNKNOWN",
        name=node.get("name"),
        gpu_type=machine.get("gpuDisplayName") or node.get("gpuTypeId"),
        uptime_s=int(rt.get("uptimeInSeconds") or 0),
        ssh_ip=ssh_ip,
        ssh_port=ssh_port,
    )


class RunpodOrchestrator:
    """Real RunPod orchestrator over the GraphQL API. Auth via RUNPOD_API_KEY (env, as the runbook uses)."""

    POD_NAME = "labeloxav-warm"

    def __init__(self, api_key: str | None = None, *, timeout: float = 60.0) -> None:
        self._key = api_key if api_key is not None else os.environ.get("RUNPOD_API_KEY")
        self._timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self._key)

    def _require_key(self) -> str:
        if not self._key:
            raise RunpodError("RUNPOD_API_KEY is not set")
        return self._key

    def _gql(self, query: str, variables: dict | None = None) -> dict:
        try:
            r = httpx.post(
                f"{GQL}?api_key={self._require_key()}",
                json={"query": query, "variables": variables or {}},
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as exc:
            raise RunpodError(f"runpod request failed: {exc}") from exc
        if data.get("errors"):
            raise RunpodError(f"runpod graphql errors: {data['errors']}")
        return data["data"]

    def gpu_types(self) -> list[dict]:
        q = ("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud"
             " securePrice communityPrice } }")
        return self._gql(q)["gpuTypes"]

    def provision(self, gpu_type_id: str, image: str, volume_id: str | None) -> PodInfo:
        inp: dict = {
            "cloudType": "ALL", "gpuCount": 1, "gpuTypeId": gpu_type_id, "imageName": image,
            "name": self.POD_NAME, "volumeMountPath": "/workspace", "containerDiskInGb": 40,
            "ports": "22/tcp", "startSsh": True,
        }
        if volume_id:
            inp["networkVolumeId"] = volume_id
        q = ("mutation($input: PodFindAndDeployOnDemandInput!) {"
             " podFindAndDeployOnDemand(input: $input) { id imageName machineId desiredStatus } }")
        node = self._gql(q, {"input": inp})["podFindAndDeployOnDemand"]
        if not node or not node.get("id"):
            raise RunpodError("runpod did not return a pod id (no capacity, or input rejected)")
        # a freshly deployed pod has no runtime yet; report it as provisioning
        return PodInfo(id=node["id"], status=node.get("desiredStatus") or "RUNNING", name=self.POD_NAME)

    def status(self, pod_id: str) -> PodInfo:
        q = ("query($input: PodFilter!) { pod(input: $input) { id name desiredStatus gpuTypeId"
             " machine { gpuDisplayName }"
             " runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic type } } } }")
        node = self._gql(q, {"input": {"podId": pod_id}})["pod"]
        if not node:
            # the API no longer knows this pod: treat as gone (not billing)
            return PodInfo(id=pod_id, status="GONE")
        return _pod_from_node(node)

    def terminate(self, pod_id: str) -> None:
        # full removal, stops ALL billing (volume billing is separate and unaffected by this)
        self._gql("mutation($input: PodTerminateInput!) { podTerminate(input: $input) }",
                  {"input": {"podId": pod_id}})

    def pause(self, pod_id: str) -> None:
        # RunPod stop: keeps the pod for fast reconnect but still bills for volume storage
        self._gql("mutation($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }",
                  {"input": {"podId": pod_id}})

    def list_pods(self) -> list[PodInfo]:
        q = ("query { myself { pods { id name desiredStatus gpuTypeId machine { gpuDisplayName }"
             " runtime { uptimeInSeconds } } } }")
        pods = (self._gql(q).get("myself") or {}).get("pods") or []
        return [_pod_from_node(p) for p in pods]
