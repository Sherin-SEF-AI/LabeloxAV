"""Thin RunPod API client (JSON in, JSON out, never table scraping) for the provisioning runbook.

runpodctl's command surface is narrow (it cannot list GPU types, search templates, or create network
volumes), so those operations go through the RunPod GraphQL/REST API here. Each subcommand prints a
single JSON object to stdout so the bash runbook can parse it with one python call. Auth via
RUNPOD_API_KEY. No external deps beyond the project venv (httpx).

Subcommands:
  gpu-types                       list available GPU types (id, name, memoryInGb, prices)
  create-volume NAME SIZE DC      create a network volume, print {id,...}
  create-pod GPU_ID IMAGE [VOLID] deploy an on-demand pod, print {id,...}
  pod-status POD_ID               print {id,desiredStatus,ssh:{ip,port}|null,uptime}
  stop-pod POD_ID                 stop (not delete) the pod, print {id,desiredStatus}

NOTE: the RunPod API drifts. These queries reflect the API at build time; verify on first run. Network
volume creation needs a data center id (DC); if it is not known, create `labeloxav-vol` once in the
RunPod console and pass its id to create-pod via the runbook (LBX_RUNPOD_VOLUME_ID).
"""

from __future__ import annotations

import json
import os
import sys

import httpx

GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"


def _key() -> str:
    k = os.environ.get("RUNPOD_API_KEY")
    if not k:
        _fail("RUNPOD_API_KEY is not set")
    return k


def _fail(msg: str) -> None:
    print(json.dumps({"error": msg}))
    sys.exit(2)


def _gql(query: str, variables: dict | None = None) -> dict:
    try:
        r = httpx.post(f"{GQL}?api_key={_key()}", json={"query": query, "variables": variables or {}}, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:  # noqa: BLE001
        _fail(f"graphql request failed: {exc}")
    if data.get("errors"):
        _fail(f"graphql errors: {json.dumps(data['errors'])}")
    return data["data"]


def gpu_types() -> None:
    q = "query { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice } }"
    print(json.dumps(_gql(q)["gpuTypes"]))


def create_volume(name: str, size: str, dc: str) -> None:
    try:
        r = httpx.post(
            f"{REST}/networkvolumes",
            headers={"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"},
            json={"name": name, "size": int(size), "dataCenterId": dc}, timeout=60,
        )
        r.raise_for_status()
        print(json.dumps(r.json()))
    except Exception as exc:  # noqa: BLE001
        _fail(f"create volume failed: {exc}")


def create_pod(gpu_id: str, image: str, volume_id: str | None) -> None:
    inp: dict = {
        "cloudType": "ALL", "gpuCount": 1, "gpuTypeId": gpu_id, "imageName": image,
        "name": "labeloxav-pod", "volumeMountPath": "/workspace", "containerDiskInGb": 40,
        "ports": "22/tcp", "startSsh": True,
    }
    if volume_id:
        inp["networkVolumeId"] = volume_id
    q = ("mutation($input: PodFindAndDeployOnDemandInput!) {"
         " podFindAndDeployOnDemand(input: $input) { id imageName machineId } }")
    print(json.dumps(_gql(q, {"input": inp})["podFindAndDeployOnDemand"]))


def pod_status(pod_id: str) -> None:
    q = ("query($input: PodFilter!) { pod(input: $input) { id desiredStatus"
         " runtime { uptimeInSeconds ports { ip publicPort privatePort isIpPublic type } } } }")
    pod = _gql(q, {"input": {"podId": pod_id}})["pod"]
    ssh = None
    rt = (pod or {}).get("runtime") or {}
    for p in (rt.get("ports") or []):
        if p.get("privatePort") == 22 and p.get("isIpPublic"):
            ssh = {"ip": p.get("ip"), "port": p.get("publicPort")}
    print(json.dumps({"id": pod.get("id"), "desiredStatus": pod.get("desiredStatus"),
                      "ssh": ssh, "uptime": rt.get("uptimeInSeconds")}))


def stop_pod(pod_id: str) -> None:
    q = "mutation($input: PodStopInput!) { podStop(input: $input) { id desiredStatus } }"
    print(json.dumps(_gql(q, {"input": {"podId": pod_id}})["podStop"]))


def main(argv: list[str]) -> None:
    if not argv:
        _fail("usage: runpod_api.py <subcommand> [args]")
    cmd, rest = argv[0], argv[1:]
    if cmd == "gpu-types":
        gpu_types()
    elif cmd == "create-volume":
        create_volume(rest[0], rest[1], rest[2])
    elif cmd == "create-pod":
        create_pod(rest[0], rest[1], rest[2] if len(rest) > 2 else None)
    elif cmd == "pod-status":
        pod_status(rest[0])
    elif cmd == "stop-pod":
        stop_pod(rest[0])
    else:
        _fail(f"unknown subcommand: {cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])
