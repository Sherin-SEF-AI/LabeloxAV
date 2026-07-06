"use client";

// M18 governance: enforced compliance gates. A release redaction proof (every frame must have passed the PII
// gate, else the proof fails and names the uncovered frames), the export consent gate (fails closed), the
// cloud cost ceiling (a job is refused before dispatch), and the governance lineage trace. Operational
// Materialism: a gate is green only when it admits, red when it blocks; nothing decorative.

import { useState } from "react";
import PageShell from "@/components/shell/PageShell";
import { Panel, KV, Verdict, RunButton, useRun, ErrLine, NumField } from "@/components/engine/prim";
import { getJSON, runJSON } from "@/lib/engine";

type Proof = { proof_id: string; verdict: string; manifest: { coverage: number; n_frames: number; n_covered: number };
  uncovered: string[] };
type Consent = { allowed: boolean; consent_status: string; reason: string | null };
type Cost = { allowed: boolean; reason: string | null; est_cost_usd: number; remaining_after: number };
type Lineage = { subject: string;
  audit_decisions: { actor: string; decision: string; created_at: string | null }[];
  redaction_proofs: { proof_id: string; verdict: string; coverage: number; n_frames: number }[] };

export default function CompliancePage() {
  const [release, setRelease] = useState("rel-demo-1");
  const [frameIds, setFrameIds] = useState('["f1","f2","f3"]');
  const proof = useRun<Proof>();

  const [status, setStatus] = useState("granted");
  const consent = useRun<Consent>();

  const [gpuHours, setGpuHours] = useState(2);
  const [hourly, setHourly] = useState(1.89);
  const [spent, setSpent] = useState(0);
  const cost = useRun<Cost>();

  const [subject, setSubject] = useState("rel-demo-1");
  const lineage = useRun<Lineage>();

  return (
    <PageShell active="Govern" title="Governance compliance"
      subtitle="redaction proof, consent, cost ceilings, and lineage, enforced at the spine">
      <div className="p-4 grid grid-cols-1 lg:grid-cols-2 gap-4 max-w-6xl">
        {/* redaction proof */}
        <Panel title="release redaction proof" hint="PII coverage must be complete">
          <label className="flex items-center gap-2 font-mono text-[11px] mb-2">
            <span className="text-ink-3">release</span>
            <input value={release} onChange={(e) => setRelease(e.target.value)}
              className="flex-1 bg-bg-2 border border-line px-1.5 py-0.5 text-ink outline-none focus:border-accent" />
          </label>
          <div className="font-mono text-[10px] text-ink-3 mb-1">frame ids in the release</div>
          <textarea value={frameIds} onChange={(e) => setFrameIds(e.target.value)} rows={3}
            className="w-full bg-bg-2 border border-line px-2 py-1 font-mono text-[10px] text-ink outline-none focus:border-accent" />
          <div className="mt-2"><RunButton busy={proof.busy}
            onClick={() => proof.run(() => runJSON<Proof>("/api/govern/redaction/proof",
              { release_commit: release, frame_ids: JSON.parse(frameIds), method_version: "pii-v3" }))}
            label="build proof" /></div>
          <ErrLine err={proof.err} />
          {proof.out && (
            <div className="mt-3">
              <Verdict ok={proof.out.verdict === "pass"} yes="proof pass" no="proof fail" />
              <KV k="coverage" v={`${(proof.out.manifest.coverage * 100).toFixed(1)}%`}
                tone={proof.out.verdict === "pass" ? "pass" : "block"} />
              <KV k="frames" v={`${proof.out.manifest.n_covered} / ${proof.out.manifest.n_frames}`} tone="ink-3" />
              {proof.out.uncovered.length > 0 &&
                <div className="font-mono text-[10px] text-block mt-1">uncovered: {proof.out.uncovered.slice(0, 8).join(", ")}</div>}
            </div>
          )}
        </Panel>

        {/* consent gate */}
        <Panel title="export consent gate" hint="fails closed">
          <div className="flex gap-2 mb-2">
            {["granted", "denied", "unknown"].map((s) => (
              <button key={s} onClick={() => setStatus(s)}
                className={`font-mono text-[11px] px-2 py-0.5 border ${status === s
                  ? "border-accent text-accent" : "border-line text-ink-3"}`}>{s}</button>
            ))}
          </div>
          <RunButton busy={consent.busy}
            onClick={() => consent.run(() => getJSON<Consent>(`/api/govern/consent/${status}/gate`))} label="check" />
          <ErrLine err={consent.err} />
          {consent.out && (
            <div className="mt-3">
              <Verdict ok={consent.out.allowed} yes="export allowed" no="export blocked" />
              {consent.out.reason && <div className="font-mono text-[10px] text-ink-3 mt-1">{consent.out.reason}</div>}
            </div>
          )}
        </Panel>

        {/* cost gate */}
        <Panel title="cloud cost ceiling" hint="refused before dispatch">
          <div className="space-y-2 mb-2">
            <NumField label="gpu hours" value={gpuHours} onChange={setGpuHours} />
            <NumField label="$/hour" value={hourly} onChange={setHourly} />
            <NumField label="spent this window $" value={spent} onChange={setSpent} w="w-28" />
          </div>
          <RunButton busy={cost.busy}
            onClick={() => cost.run(() => runJSON<Cost>("/api/govern/cost/gate",
              { gpu_hours: gpuHours, hourly_usd: hourly, spent_usd: spent }))} label="gate job" />
          <ErrLine err={cost.err} />
          {cost.out && (
            <div className="mt-3">
              <Verdict ok={cost.out.allowed} yes="admitted" no="refused" />
              <KV k="estimate" v={`$${cost.out.est_cost_usd.toFixed(2)}`} tone="ink" />
              <KV k="remaining after" v={`$${cost.out.remaining_after.toFixed(2)}`} tone="ink-3" />
              {cost.out.reason && <div className="font-mono text-[10px] text-block mt-1">{cost.out.reason}</div>}
            </div>
          )}
        </Panel>

        {/* lineage */}
        <Panel title="governance lineage" hint="audit + proofs for a subject">
          <label className="flex items-center gap-2 font-mono text-[11px] mb-2">
            <span className="text-ink-3">subject</span>
            <input value={subject} onChange={(e) => setSubject(e.target.value)}
              className="flex-1 bg-bg-2 border border-line px-1.5 py-0.5 text-ink outline-none focus:border-accent" />
          </label>
          <RunButton busy={lineage.busy}
            onClick={() => lineage.run(() => getJSON<Lineage>(`/api/govern/lineage/${encodeURIComponent(subject)}`))} label="trace" />
          <ErrLine err={lineage.err} />
          {lineage.out && (
            <div className="mt-3 space-y-1">
              <div className="font-mono text-[10px] uppercase text-ink-3">audit decisions ({lineage.out.audit_decisions.length})</div>
              {lineage.out.audit_decisions.slice(0, 6).map((a, i) => (
                <div key={i} className="flex justify-between font-mono text-[10px]">
                  <span className="text-ink-2">{a.actor} · {a.decision}</span>
                  <span className="text-ink-3">{a.created_at?.slice(0, 10)}</span>
                </div>
              ))}
              <div className="font-mono text-[10px] uppercase text-ink-3 pt-1">redaction proofs ({lineage.out.redaction_proofs.length})</div>
              {lineage.out.redaction_proofs.slice(0, 4).map((p) => (
                <div key={p.proof_id} className="flex justify-between font-mono text-[10px]">
                  <span className={p.verdict === "pass" ? "text-pass" : "text-block"}>{p.verdict}</span>
                  <span className="text-ink-3">{(p.coverage * 100).toFixed(0)}% · {p.n_frames} frames</span>
                </div>
              ))}
              {!lineage.out.audit_decisions.length && !lineage.out.redaction_proofs.length &&
                <div className="font-mono text-[11px] text-ink-3">no governance events for this subject</div>}
            </div>
          )}
        </Panel>
      </div>
    </PageShell>
  );
}
