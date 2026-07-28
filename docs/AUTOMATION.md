# Automating LabeloxAV

Three ways to drive the engine from outside it: the generated client, webhooks, and campaigns. Most useful
pipelines are a combination, so this covers the seams between them rather than each in isolation.

---

## The generated client

`sdk/generated_client.py` is produced from the API's own OpenAPI schema and covers every route. Regenerate
it whenever the API changes:

```bash
python -m scripts.generate_sdk --out sdk/generated_client.py
python -m scripts.generate_sdk --check     # CI: fails if it has drifted
```

The direction of truth matters here. The server defines the surface and the client is derived, so a route
that changes shape produces a client that changes shape rather than one that keeps calling the old one.
The hand-written `sdk/labelox_client.py` remains for ergonomic helpers that compose several calls; the
generated module is the complete surface underneath it.

```python
from sdk.generated_client import LabeloxClient

with LabeloxClient(base_url="http://localhost:8000", token=TOKEN) as lbx:
    queue = lbx.get_triage(limit=50)
    lbx.post_objects_by_object_id_review(queue[0]["object_id"],
                                         {"action": "confirm", "state": "accepted"})
```

The token is required at construction rather than optional. Reads are deny-by-default on the server, so a
client built without one fails on its first call with a 401 that looks like an outage; refusing up front
says what is actually wrong.

Mint one server side:

```bash
python -m scripts.mint_token --name pipeline --role reviewer --create
```

Give a pipeline the narrowest role that works. `annotator` can post telemetry and review objects;
`reviewer` is needed to seal datasets, run campaigns, and read incidents; `admin` is needed for governance
and the PII access log. A pipeline holding an admin token because it was easier is the most common way a
credential ends up over-scoped.

---

## Webhooks

Register a receiver, then react to what the engine does. Every delivery is signed and retried.

```python
lbx.post_integrations_webhooks({
    "url": "https://your-pipeline.example.com/labelox",
    "events": ["training.completed", "govern.promotion_blocked", "export.completed"],
})
```

The signing secret is returned exactly once, at creation. Verify every delivery:

```python
import hashlib, hmac

def verify(raw_body: bytes, headers: dict, secret: str) -> bool:
    ts = headers["X-Labelox-Timestamp"]
    want = hmac.new(secret.encode(), f"{ts}.".encode() + raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={want}", headers["X-Labelox-Signature"])
```

The timestamp is inside the signed material, so a captured delivery cannot be replayed forever. Reject
anything older than a few minutes.

### Recipe: retrain when a promotion is blocked

The gate refusing a model is the signal that a class needs more labels. This turns that refusal into work
rather than a line in a log.

```python
@app.post("/labelox")
def on_event(event: dict):
    if event["type"] != "govern.promotion_blocked":
        return

    blocked_class = event["data"]["worst_class"]
    lbx.post_campaigns({
        "name": f"auto-{blocked_class}-{event['data']['run_id'][:8]}",
        "class_name": blocked_class,
        "target_value": 0.6,
        "label_budget": 1000,
        # mine and judge run unattended; label always waits for humans and promote is left to a person.
        "autopilot_stages": ["mine", "judge"],
    })
```

### Recipe: export on every promotion

```python
if event["type"] == "govern.model_promoted":
    lbx.post_datasets_export({
        "name": f"release-{event['data']['model_version']}",
        "formats": ["coco", "parquet", "masks"],
        "states": ["accepted", "approved"],
    })
```

Then poll `get_exports_by_job_id_progress` or watch the `/api/events/jobs` stream. A large export is
chunked and checkpointed, so a failure at ninety percent resumes rather than restarting:

```python
progress = lbx.get_exports_by_job_id_progress(job_id)
if progress["status"] == "error" and progress["resumable"]:
    lbx.post_exports_by_job_id_resume(job_id)
```

---

## Campaigns

A campaign is the improvement loop with a budget and a stopping condition. Every stage existed before;
what was missing was the orchestration between them, so a class stalled whenever nobody was watching.

```python
c = lbx.post_campaigns({
    "name": "cattle-recall",
    "class_name": "cattle",
    "target_metric": "recall",
    "target_value": 0.6,
    "label_budget": 2000,
    "patience": 2,
})

# Advance one step. Returns what it did, or what it is waiting for.
out = lbx.post_campaigns_by_campaign_id_tick(c["campaign_id"])
```

One step per call, deliberately. A long-lived loop cannot survive a restart, be inspected halfway, or be
stopped except by killing something, and all three matter for a process that spends a GPU and commissions
review work.

Three constraints are worth understanding before you automate the tick:

- **The budget is in labels.** The batches a campaign builds are human hours; no wall-clock limit
  constrains that.
- **It stops when it stops improving.** `patience` counts consecutive iterations that did not move the
  metric. A campaign that could only stop by succeeding could not stop.
- **Every stage waits for approval by default.** Add stages to `autopilot_stages` one at a time. `label`
  always waits regardless: there is no version of this system where a machine supplies the human review,
  and iteration 5 trained on unreviewed machine labels and drove pedestrian recall from 0.73 to 0.004.

A cron that ticks every campaign every ten minutes is a reasonable driver:

```python
for c in lbx.get_campaigns(status="running")["campaigns"]:
    lbx.post_campaigns_by_campaign_id_tick(c["campaign_id"])
```

---

## Edge devices

A deployed device registers on boot and posts a reporting window rather than every inference: p50, p95 and
the thermal ceiling reached are properties of a window, and a device posting each inference would spend its
uplink on telemetry.

```python
lbx.post_edge_devices({
    "device_id": "orin-042", "hardware": "jetson_orin_nx",
    "runtime": "tensorrt", "artifact_id": DEPLOYMENT_ID, "fleet": "blr-pilot",
})

lbx.post_edge_telemetry({
    "device_id": "orin-042",
    "window_start_ns": t0, "window_end_ns": t1, "n_inferences": 1500,
    "latency_p50_ms": 18.2, "latency_p95_ms": 41.7,
    "temp_c_max": 78.0, "throttled_fraction": 0.04,
    "conf_histogram": [12, 48, 190, 640, 610],   # accuracy drift without labels
    "dropped_frames": 3,
})
```

The confidence histogram is how field accuracy drift is detected at all. The field has no ground truth, so
a distribution that has moved away from the one measured at gate time is the available signal.

Read it back against the bench:

```python
gate = lbx.get_edge_artifacts_by_artifact_id_gate(DEPLOYMENT_ID)
if gate["verdict"] == "field_regression":
    alert(gate["detail"])
```

This is advisory and will not demote anything. Telemetry arrives from devices, which sit outside the trust
boundary: a single misconfigured unit reporting nonsense must not be able to demote a champion. It also
refuses to draw a fleet conclusion from fewer than three live devices.

---

## What to automate, and what not to

Worth automating: mining, judging, exporting, telemetry, retraining, dataset sealing, and reacting to a
blocked gate.

Worth leaving to a person: accepting labels, promoting a model, acknowledging a security incident, and
erasing data. Each of those is a judgement the system deliberately cannot make, and every one of them is
gated on the server rather than by convention here, so automating around this document will not get you
past them.
