"use client";

// Why a check thinks one label is wrong (tool #12): the sentence, the frame with the boxes the check compared,
// the crops it leaned on, the near-identical twin frame where there is one, and the vision model's opinion.
// Shown beside the error queue (/quality/errors) and inside the editor's ErrorOverlay card.

import Link from "next/link";

import EvidenceImage, { type DrawBox } from "@/components/quality/EvidenceImage";
import VlmReason from "@/components/quality/VlmReason";
import { humanize } from "@/lib/format";
import { roleTone } from "@/lib/tools/quality";
import type { ErrorWhy } from "@/lib/tools/qualityApi";

export function evidenceBoxes(why: ErrorWhy): DrawBox[] {
  return why.evidence.map((e) => ({
    bbox: e.bbox, tone: roleTone(e.role), dashed: e.role !== "subject",
    label: e.role === "subject" ? humanize(e.label) : `${e.role}: ${humanize(e.label)}`,
  }));
}

export default function ErrorWhyPanel({ why, showFrame = true, showVlm = true, imageMaxW = 440 }: {
  why: ErrorWhy;
  showFrame?: boolean;
  showVlm?: boolean;
  imageMaxW?: number;
}) {
  const w = why.frame.width ?? 0;
  const h = why.frame.height ?? 0;
  return (
    <div className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="chip chip-on" title={why.kind}>{why.detector}</span>
        {why.rule && <span className="text-[11.5px] text-ink-3">{humanize(why.rule)}</span>}
        <span className="text-[11.5px] text-ink-3 tabular-nums">suspicion {why.score.toFixed(2)}</span>
        {why.proposed && <span className="text-[12px] text-accent-2">suggests {humanize(why.proposed.class_name)}</span>}
      </div>
      <p className="text-[13px] text-ink leading-snug">{why.reason}</p>
      {why.facts.length > 0 && (
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-[12px]">
          {why.facts.map((f) => (
            <div key={f.label} className="contents">
              <dt className="text-ink-3">{f.label}</dt>
              <dd className="text-ink-2 tabular-nums">{f.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {showFrame && w > 0 && h > 0 && (
        <div>
          <div className="label-caps mb-1">On the frame</div>
          <EvidenceImage src={why.frame.image_url} width={w} height={h} boxes={evidenceBoxes(why)}
            maxW={imageMaxW} maxH={260} alt={`Frame with the flagged ${humanize(why.subject.class_name)}`} />
          <Link href={`/frame/${why.frame.frame_id}`} className="link text-[12px]">Open the frame</Link>
        </div>
      )}
      {why.twin_frame && why.twin_frame.width && why.twin_frame.height && (
        <div>
          <div className="label-caps mb-1">The near-identical frame ({humanize(why.subject.class_name)} absent)</div>
          <EvidenceImage src={why.twin_frame.image_url} width={why.twin_frame.width} height={why.twin_frame.height}
            focus={false} maxW={imageMaxW} maxH={200}
            boxes={why.twin_frame.objects.map((o) => ({ bbox: o.bbox, tone: "info" as const }))}
            alt="The near-identical frame with its labels" />
        </div>
      )}
      {why.neighbours.length > 0 && (
        <div>
          <div className="label-caps mb-1">What the check compared it with</div>
          <div className="flex flex-wrap gap-1.5">
            {why.neighbours.map((n) => (
              <Link key={n.object_id} href={`/frame/${n.frame_id}`} className="block w-[76px] group"
                title={`${humanize(n.class_name)} · ${n.role}${n.similarity != null ? ` · ${n.similarity.toFixed(2)} alike` : ""}`}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={n.crop_url} alt={humanize(n.class_name)} loading="lazy" decoding="async"
                  className="w-[76px] h-[76px] object-cover rounded border hairline group-hover:border-accent" />
                <span className="block truncate text-[11px] text-ink-2">{humanize(n.class_name)}</span>
              </Link>
            ))}
          </div>
        </div>
      )}
      {showVlm && <VlmReason objectId={why.object_id} />}
    </div>
  );
}
