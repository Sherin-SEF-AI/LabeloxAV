# Licensing

LabeloxAV's own source code is licensed under the **Apache License 2.0** (see `LICENSE`). The application
you install, though, is that code combined with third-party components, and a few of those carry
**copyleft** licences that apply to what you run. This page says which ones, what that means for a team
running LabeloxAV on its own machine, and what changes if you offer it to people outside your team.

This is a plain-language summary to help you decide, not legal advice. If your use is commercial or
public-facing, have someone qualified read the licences themselves.

## What carries a copyleft licence

| Component | Licence | Where it runs | Why it is there |
|---|---|---|---|
| Ultralytics (`ultralytics`, `ultralytics-thop`) | AGPL-3.0 | the app image (API and GPU worker) | the YOLO detector behind auto-labelling, training and evaluation |
| cleanlab | AGPL-3.0 | the app image | confident learning, which finds likely label errors (Label errors page) |
| MinIO server | AGPL-3.0 | its own container, built unmodified from the upstream source at a pinned release | object storage for frames, masks and exports |
| PostGIS | GPL-2.0 | the database container | spatial queries |
| psycopg / psycopg2 | LGPL-3.0 / LGPL | the app image | the PostgreSQL driver, used as a library, unmodified |

Two bundled services use source-available licences that restrict offering *them* as a hosted service, not
using them inside your own deployment: **Redis 7.4+** (RSALv2 / SSPLv1) and **Redpanda** (BSL 1.1).

Everything else in the default install is under permissive licences (Apache-2.0, MIT, BSD, PostgreSQL,
MPL-2.0 for the optional Lichtblick viewer).

## What that means for the app as installed

Ultralytics and cleanlab are Python libraries imported into the same process as LabeloxAV's own code. The
combination that runs in the `labeloxav/app` image is therefore covered by the **AGPL-3.0**, even though
LabeloxAV's source files on their own remain Apache-2.0. We keep Ultralytics deliberately: its detector,
training loop and export tooling are what make auto-labelling work on one GPU, and we would rather state
the licence plainly than swap in a weaker model to avoid it.

The AGPL adds one obligation to the GPL's: if you **modify** the program and let people **interact with it
over a network**, you must offer those people the complete corresponding source of the version they are
using (AGPL-3.0, section 13).

## Common situations

**Your team runs it, unmodified, for itself.** This is what LabeloxAV is built for. You may use it freely.
The source of the exact version you run is public, and the app links to it from **Settings > About**.

**Your team modifies it and only your team uses it.** Under the AGPL the people interacting with it are
your team members, and they are entitled to the source of your modified version. In practice: keep your
changes in a repository they can read, and point the app's source link at it (`LBX_ABOUT__SOURCE_URL`, and `LBX_ABOUT__MODIFIED=true`).

**You let people outside your team use it** (a client, annotators from a vendor, a public demo). Anyone who
can sign in must be able to get the complete source of the version you run, including your changes. Set
`LBX_ABOUT__SOURCE_URL` to where that source is published (and `LBX_ABOUT__MODIFIED=true`) before giving
anyone access.

**You redistribute it** (ship the images or a modified fork to someone else). The recipient gets the AGPL
terms and the full source, as with any AGPL program.

## Models and datasets

- **Weights you train** with LabeloxAV are trained by Ultralytics code. Ultralytics' position is that
  models trained with its software are covered by the AGPL unless you hold an Ultralytics Enterprise
  licence. Whether that holds for weights in a given jurisdiction is not settled; if you plan to ship a
  trained detector inside a product, read Ultralytics' licensing terms and decide before you do.
- **Pretrained weights** downloaded by the installer (YOLO11, YOLO-World) are Ultralytics releases under
  AGPL-3.0. SAM weights are released by Meta under Apache-2.0. The Mask2Former checkpoint used for drivable
  space follows the licence on its model card.
- **Sample drives** offered on first run are Wikimedia Commons clips under CC BY-SA, pinned by checksum;
  the attribution is shown wherever they are listed.
- **Your own data and labels are yours.** No licence here reaches the images you upload or the labels your
  team makes, and nothing is sent outside your machine unless you configure a cloud provider.

## If you need to avoid the AGPL

The obligations come from two libraries: Ultralytics and cleanlab. Ultralytics sells an Enterprise licence;
cleanlab's commercial terms are available from Cleanlab Inc. With both licensed commercially, the
remaining components are permissive or run as separate, unmodified services (MinIO, PostGIS) whose own
licences apply only to those services.
