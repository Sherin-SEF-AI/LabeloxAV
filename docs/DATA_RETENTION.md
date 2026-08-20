# Data retention and subject rights

The machinery for this exists and is tested. What did not exist is a document telling an operator what the
windows are, who may erase, and how to answer a subject-rights request — which is the half a regulator
asks for. Under the DPDPA a data fiduciary has to be able to say what it holds, for how long, and act on a
request; code that can do it and nobody who knows how is not compliance.

This describes the AV pack. The legal regime is a pack property (`packs/av/pack.py`), so a second domain
states its own.

---

## What personal data this system holds

Two kinds, and they live in different places.

**Pixels.** Faces and registration plates are in the frames themselves. They are blurred *before* the blob
is written to the object store — `PiiAnonymizer` runs inside both ingest paths and the unredacted original
is never stored. That is deliberate and it is also irreversible: an over-blur cannot be undone, which is
why the re-check thresholds are conservative.

**Records about the pixels.** `pii_audit` says what the redactor found per frame. `pii_access_log` says who
viewed a frame that contained personal data. Neither stores a copy of what it describes.

Speech is a third category with a real gate and a stubbed detector: `evaluate_dpdpa` refuses an export
carrying an unredacted personal speech segment, and the detector that would find one is a runtime seam.
Treat audio as unhandled rather than handled.

---

## Retention

`retention_until` on a session is the expiry. It is set by policy at ingest, and the sweep enforces it.

```bash
# What would be removed, and nothing else. Both commands default to a dry run.
curl -sX POST /api/govern/retention/sweep -H "$AUTH" | jq
# Do it.
curl -sX POST "/api/govern/retention/sweep?dry_run=false" -H "$AUTH" | jq
```

A sweep removes frames, annotations, audits and blobs together, and returns a tamper-evident certificate.
Keep the certificates: they are the evidence the retention policy was applied, and they are the only
artefact that survives the data.

**There is no default window.** A session with no `retention_until` is kept indefinitely, which is a
decision an operator has to make deliberately rather than inherit. Set one at ingest for anything with a
contractual or regulatory limit.

---

## Answering a subject request

### "What do you hold about me?"

There is no identity index, by design: this corpus holds faces that were blurred on the way in and never
associated with a name. In practice a subject request arrives with a time and a place — a drive, a road, a
date — and the answer is scoped by session and frame, not by person.

`pii_access_log` answers the other half: who inside the organisation viewed frames containing personal
data, and whether what they saw was redacted.

### "Erase it"

```bash
curl -sX POST /api/govern/retention/erase \
  -H "$AUTH" -H 'content-type: application/json' \
  -d '{"session_id": "...", "frame_ids": ["..."], "dry_run": true}' | jq
```

Single-subject erasure removes the frames, their annotations, their audit rows and their blobs, and returns
a certificate. Run it dry first and read the counts: erasure is not reversible and there is no recycle bin
(the corpus has no soft delete — see `docs/REMEDIATION_STATUS.md`).

**Erasure propagates unevenly, and you have to know where.** A frame that has already been exported into a
dataset commit, or that trained a model, is gone from the corpus and not from those. Exports are
content-addressed and immutable by design. If a subject request has to reach a delivered dataset, that is a
conversation with the recipient, not a database operation — record it as such.

---

## Before a dataset leaves the system

The DPDPA pre-sale gate runs inside `export_dataset` and refuses rather than warns. It blocks on:

- a frame in the slice with no anonymization audit at all — fail-closed, an unaudited frame is treated as
  unredacted;
- a personal speech segment that is not masked;
- **(advisory today)** an annotated person or vehicle with no redaction region covering it.

The third is `LBX_PII__COVERAGE_GATE`. It ships `advisory` because 82.4% of frames holding an annotated
person have zero faces redacted, so enforcing it immediately would refuse most of the corpus. The sequence
to close that out is in `docs/RUNBOOK.md`. Until it is `enforcing`, a passing export means *every frame was
looked at*, not *every face was blurred* — do not represent it as the latter.

---

## Who may do what

| Action | Role |
| --- | --- |
| View a frame containing personal data | any authenticated user; the view is logged |
| Read `pii_access_log` | admin |
| Run a retention sweep or an erasure | admin |
| Export a dataset (subject to the gate) | reviewer |

---

## What is not covered

Stated plainly, because a retention document that implies more than it does is worse than none.

- **No automatic retention default.** A session with no `retention_until` is kept forever.
- **Speech detection is a seam.** The gate refuses unredacted personal speech; nothing currently detects it.
- **No erasure of delivered exports.** Content-addressed and immutable by design.
- **No soft delete.** Erasure is immediate and total.
- **Backups are outside this.** `scripts/backup.sh` copies the corpus as it stands; an erasure after a
  backup does not reach into the backup. Rotate them on a schedule that matches your retention window, or
  a subject's data survives in a `.sql.gz` after it has left the database.
