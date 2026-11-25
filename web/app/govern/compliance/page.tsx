"use client";

// M18 governance: enforced compliance gates. A release redaction proof (every frame must have passed the PII
// gate, else the proof fails and names the uncovered frames), the export consent gate (fails closed), the
// cloud cost ceiling (a job is refused before dispatch), and the governance lineage trace. Operational
// Materialism: a gate is green only when it admits, red when it blocks; nothing decorative.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Verdict, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { getJSON, runJSON, parseInput } from "@/lib/engine";
import { Segmented } from "@/components/ui/kit";
import { ago, humanize, pct, plural, shortId } from "@/lib/format";

type Proof = { proof_id: string; verdict: string; manifest: { coverage: number; n_frames: number; n_covered: number };
  uncovered: string[] };
type Consent = { allowed: boolean; consent_status: string; reason: string | null };
type Cost = { allowed: boolean; reason: string | null; est_cost_usd: number; remaining_after: number };
type Lineage = { subject: string;
  audit_decisions: { actor: string; decision: string; created_at: string | null }[];
  redaction_proofs: { proof_id: string; verdict: string; coverage: number; n_frames: number }[] };

export default function CompliancePage() {
  const [release, setRelease] = useState("");
  const [frameIds, setFrameIds] = useState("");
  const proof = useRun<Proof>();

  const [status, setStatus] = useState("granted");
  const consent = useRun<Consent>();

  const [gpuHours, setGpuHours] = useState(2);
  const [hourly, setHourly] = useState(1.89);
  const [spent, setSpent] = useState(0);
  const cost = useRun<Cost>();

  const [subject, setSubject] = useState("");
  const lineage = useRun<Lineage>();

  return (
    <PageShell active="Govern" title="Governance compliance"
      subtitle="Check the gates every release, export and cloud job must pass: redaction, consent, cost and an audit trail.">
      <div className="page-wrap grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* redaction proof */}
        <Panel title="Release redaction proof" hint="Every frame must have faces and plates redacted">
          <label className="flex flex-col gap-1.5 text-[12.5px] mb-3">
            <span className="text-ink-2">Release</span>
            <input value={release} placeholder="Release id" onChange={(e) => setRelease(e.target.value)}
              className="field text-[12.5px]" />
          </label>
          <label className="block text-[12.5px] text-ink-2 mb-1.5" htmlFor="proof-frames">Frames in the release</label>
          <textarea id="proof-frames" value={frameIds} placeholder={'Frame ids as a JSON list, e.g. ["0b6f...", "4c21..."]'} onChange={(e) => setFrameIds(e.target.value)} rows={3}
            className="field w-full font-mono text-[11.5px]" />
          <div className="mt-3"><RunButton busy={proof.busy}
            onClick={() => proof.run(() => runJSON<Proof>("/api/govern/redaction/proof",
              { release_commit: release, frame_ids: parseInput(frameIds, "frame ids"), method_version: "pii-v3" }))}
            label="Build proof" /></div>
          <ErrLine err={proof.err} />
          {proof.out && (
            <div className="mt-3">
              <Verdict ok={proof.out.verdict === "pass"} yes="Proof passed" no="Proof failed" />
              <KV k="Coverage" v={`${(proof.out.manifest.coverage * 100).toFixed(1)}%`}
                tone={proof.out.verdict === "pass" ? "pass" : "block"} />
              <KV k="Frames redacted" v={`${proof.out.manifest.n_covered} / ${proof.out.manifest.n_frames}`} tone="ink-3" />
              {proof.out.uncovered.length > 0 &&
                <div className="text-[12px] text-block mt-1">Not redacted: <span className="font-mono">{proof.out.uncovered.slice(0, 8).map((f) => shortId(f)).join(", ")}</span>{proof.out.uncovered.length > 8 ? ` and ${proof.out.uncovered.length - 8} more` : ""}</div>}
            </div>
          )}
        </Panel>

        {/* consent gate */}
        <Panel title="Export consent gate" hint="Blocks export unless consent is granted">
          <div className="flex items-center gap-3 mb-3">
            <span className="text-[12.5px] text-ink-2">Consent status</span>
            <Segmented value={status} onChange={setStatus}
              options={["granted", "denied", "unknown"].map((s) => ({ value: s, label: humanize(s) }))} />
          </div>
          <RunButton busy={consent.busy}
            onClick={() => consent.run(() => getJSON<Consent>(`/api/govern/consent/${status}/gate`))} label="Check" />
          <ErrLine err={consent.err} />
          {consent.out && (
            <div className="mt-3">
              <Verdict ok={consent.out.allowed} yes="Export allowed" no="Export blocked" />
              {consent.out.reason && <div className="text-[12px] text-ink-3 mt-1">{consent.out.reason}</div>}
            </div>
          )}
        </Panel>

        {/* cost gate */}
        <Panel title="Cloud cost ceiling" hint="A job over budget is refused before it starts">
          <div className="space-y-2 mb-3">
            <NumField label="GPU hours" value={gpuHours} onChange={setGpuHours} />
            <NumField label="Price per hour (USD)" value={hourly} onChange={setHourly} />
            <NumField label="Already spent this period (USD)" value={spent} onChange={setSpent} w="w-28" />
          </div>
          <RunButton busy={cost.busy}
            onClick={() => cost.run(() => runJSON<Cost>("/api/govern/cost/gate",
              { gpu_hours: gpuHours, hourly_usd: hourly, spent_usd: spent }))} label="Check job" />
          <ErrLine err={cost.err} />
          {cost.out && (
            <div className="mt-3">
              <Verdict ok={cost.out.allowed} yes="Job admitted" no="Job refused" />
              <KV k="Estimated cost" v={`$${cost.out.est_cost_usd.toFixed(2)}`} tone="ink" />
              <KV k="Budget left after" v={`$${cost.out.remaining_after.toFixed(2)}`} tone="ink-3" />
              {cost.out.reason && <div className="text-[12px] text-block mt-1">{cost.out.reason}</div>}
            </div>
          )}
        </Panel>

        {/* lineage */}
        <Panel title="Governance lineage" hint="Decisions and proofs for one release, model or session">
          <label className="flex flex-col gap-1.5 text-[12.5px] mb-3">
            <span className="text-ink-2">Look up</span>
            <input value={subject} placeholder="Release id, model version or session id" onChange={(e) => setSubject(e.target.value)}
              className="field text-[12.5px]" />
          </label>
          <RunButton busy={lineage.busy}
            onClick={() => lineage.run(() => getJSON<Lineage>(`/api/govern/lineage/${encodeURIComponent(subject)}`))} label="Trace" />
          <ErrLine err={lineage.err} />
          {lineage.out && (
            <div className="mt-3 space-y-1">
              <div className="label-caps">Decisions ({lineage.out.audit_decisions.length})</div>
              {lineage.out.audit_decisions.slice(0, 6).map((a, i) => (
                <div key={i} className="flex justify-between text-[12px]">
                  <span className="text-ink-2">{humanize(a.decision)} <span className="text-ink-3">by {a.actor}</span></span>
                  <span className="text-ink-3">{ago(a.created_at)}</span>
                </div>
              ))}
              <div className="label-caps pt-2">Redaction proofs ({lineage.out.redaction_proofs.length})</div>
              {lineage.out.redaction_proofs.slice(0, 4).map((p) => (
                <div key={p.proof_id} className="flex justify-between text-[12px]">
                  <span className={p.verdict === "pass" ? "text-pass" : "text-block"}>{p.verdict === "pass" ? "Passed" : humanize(p.verdict)}</span>
                  <span className="text-ink-3">{pct(p.coverage)} redacted · {plural(p.n_frames, "frame")}</span>
                </div>
              ))}
              {!lineage.out.audit_decisions.length && !lineage.out.redaction_proofs.length &&
                <div className="text-[12px] text-ink-3">No governance events for this subject.</div>}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
