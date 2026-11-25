"use client";

import PageShell from "@/components/shell/PageShell";
import LoadState from "@/components/shell/LoadState";
import { apiGet } from "@/lib/api";
import { useQuery } from "@/lib/useQuery";

// What is running, where its source is, and which licences reach it. The AGPL components (Ultralytics,
// cleanlab, MinIO) require that people using a modified version over a network can get its source, so the
// link here is part of the product, not decoration. docs/LICENSING.md has the reasoning in full.

type About = {
  version: string | null; commit: string | null; source_url: string; source_of_this_version: string | null;
  modified: boolean; own_licence: string; combined_licence: string; licensing_doc: string;
  copyleft: { name: string; licence: string; where: string; why: string }[];
};

export default function AboutPage() {
  const aboutQ = useQuery(["/api/system/about"], () => apiGet<About>("/api/system/about"));
  const about: About | null = aboutQ.data ?? null;

  return (
    <PageShell active="About" title="About" subtitle="Version, source code and licences">
      <div className="page-wrap-narrow flex flex-col gap-4">
        <LoadState {...aboutQ.loadState} />
        {about && (
          <>
            <section className="panel p-5 grid gap-x-6 gap-y-3 sm:grid-cols-[max-content_1fr] text-[13px]">
              <span className="text-ink-3">Version</span>
              <span className="text-ink tabular-nums">{about.version ?? "Unknown"}</span>
              <span className="text-ink-3">Commit</span>
              <span className="text-ink font-mono text-[12px] break-all">{about.commit ?? <span className="text-ink-3 font-sans" title="No git metadata, and LBX_CODE_SHA is not set">Unknown</span>}</span>
              <span className="text-ink-3">Source</span>
              <span className="flex flex-col gap-1 min-w-0">
                {about.source_of_this_version
                  ? <a className="link break-all" href={about.source_of_this_version} target="_blank" rel="noreferrer">Source of exactly this version</a>
                  : null}
                <a className="link break-all" href={about.source_url} target="_blank" rel="noreferrer">{about.source_url}</a>
                {about.modified && <span className="text-ink-3 text-[12px]">This deployment runs modified code; the link above is where its administrators publish it.</span>}
              </span>
            </section>

            <section className="panel p-5 flex flex-col gap-3 text-[13px]">
              <h2 className="section-title">Licences</h2>
              <p className="text-ink-2 leading-relaxed">
                LabeloxAV&apos;s own code is {about.own_licence}. The application as installed also runs the components below,
                and two of them (Ultralytics and cleanlab) are libraries in the same process, so the combination is covered by the {about.combined_licence}.
                For a team running it for itself that changes nothing; if you modify it and let people outside the team use it, they must be able to get your source.
              </p>
              <div className="overflow-x-auto rounded-md border hairline">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Component</th>
                      <th>Licence</th>
                      <th>Runs in</th>
                      <th>For</th>
                    </tr>
                  </thead>
                  <tbody>
                    {about.copyleft.map((c) => (
                      <tr key={c.name} className="align-top">
                        <td className="text-ink">{c.name}</td>
                        <td className="whitespace-nowrap">{c.licence}</td>
                        <td>{c.where}</td>
                        <td className="text-ink-3">{c.why}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <a className="link self-start" href={about.licensing_doc} target="_blank" rel="noreferrer">
                What this means, situation by situation
              </a>
            </section>
          </>
        )}
      </div>
    </PageShell>
  );
}
