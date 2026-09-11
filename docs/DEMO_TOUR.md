# The recorded tour

A narrated walkthrough of every page in the product, recorded against the running system with a live
database behind it. Twelve chapters, seventy five scenes, 20 minutes and 28 seconds at 1920 by 1080,
50 MB.

It exists because the two earlier films are short. They show the system working; they do not explain
what each surface is for, and they do not visit most of it. This one visits seventy two of the
seventy three routes in the web application. The one it skips is the sign-in page.

## What is real in it

Everything on screen. The browser talks to the API on port 8000 against the `labeloxav` database, so
every count, interval and empty panel in the recording is what the system actually held at the moment
it was filmed.

Every number spoken in the narration is read from that same database at record time and substituted
into the script, so the voice cannot drift from the picture. A figure that cannot be read is spoken as
unreadable rather than as zero, and the reason is printed in the run log.

The uncomfortable numbers are spoken too. The tour says out loud that 788 of 598,179 objects carry a
human verdict, that 30,863 of 30,865 reviews have no timing, and that the multi-camera benefit is
measured over six sessions and is therefore small. A tour that quotes only the flattering figures is a
sales reel.

## The data it was recorded on

Three recordings were ingested for it, and the corpus is measurably better for them.

| | before | after |
| --- | --- | --- |
| measured ego poses | 0 | 154 |
| real LiDAR points | 0 | 18,000,000 |
| point clouds from a laser | 0 | 154 |

The two dashcam clips are the operator's own footage from Bengaluru. The third recording is KITTI drive
`2011_09_26_drive_0005`, 154 frames from Karlsruhe with a sixty four beam Velodyne on the roof, real
GPS and inertial measurements, and the rig's own calibration: focal length 721.54 pixels over a 1242 by
375 image. It was imported by `scripts/import_kitti_raw.py`.

Before that import every point cloud in the corpus was estimated from single camera depth, 453,983
points across 39 clouds, and no ego pose anywhere was measured. The three GNSS fixes the corpus held
across 41,752 frames were the honest reason most three dimensional work in it was estimated.

What was then built on those recordings, all measured from the runs:

| stage | result |
| --- | --- |
| auto-labelling | 3,977 objects on the KITTI drive, 2,443 across the two dashcam clips |
| tracking | 123 tracks and 48 scenarios on the KITTI drive |
| ego pose | 154 poses, all 154 measured, from GPS and inertial |
| cuboids from 2D | 2,089 across 154 frames, no frame failed |
| tracks lifted to 3D | 40 |
| occupancy | 154 grids, 739,309 occupied voxels |

The occupancy grids carry a caveat the builder prints itself and the tour does not hide: none of the
occupied space carries a velocity from a track yet, so the grids are close to static.

## Running it

```bash
.venv/bin/python scripts/demo_tour/postprocess.py --routes demo-tour-2026   # prepare the data
.venv/bin/python scripts/demo_tour/driver.py --check-pages                  # load every page, film nothing
.venv/bin/python scripts/demo_tour/driver.py --out site/demo/labeloxav-tour.mp4
```

`--check-pages` first, always. A broken route found there costs one line of output. Found during a
recording it costs the whole take, because the segments after it are already timed against audio that
has been synthesised.

`--only scene_key,other_key` re-records named scenes, and `--assemble-only` rebuilds the film from
segments already on disk.

The pieces:

- `narration.py` is the script. One `Scene` per page, with the words, the route, and how long to hold.
  Page interactions live in a separate `ACTIONS` table so that a selector which moves is a one line
  change rather than an edit inside a paragraph of narration.
- `record.py` synthesises speech with piper, writes both subtitle formats, and assembles.
- `driver.py` drives the browser, resolves the numbers, and records.
- `postprocess.py` runs tracking, embedding, ego pose, the three dimensional lift and occupancy over
  the tour's sessions, so the later chapters narrate features over pages that have something on them.

## Three decisions worth knowing about

**The browser records itself.** This machine runs a Wayland session, where an X11 grab of `:0` sees
only the XWayland root and returns black frames, which is exactly what the first take produced.
Chromium writing its own video captures the page rather than the desktop, so it is pixel exact at
1920 by 1080, runs headless, and nothing else on the desktop can appear in the film.

**Every scene length is decided, not observed.** The narration is synthesised first and measured, the
length is rounded up to a whole frame, and then the picture is trimmed to that number and the audio
padded to it. Picture, voice and captions are all built from one value that each can represent exactly.
Letting the capture decide its own length drifts them apart by a frame per scene.

**Subtitles are burned from ASS, not SRT.** libass assumes a 384 by 288 play resolution for a format
that carries none and scales everything it draws by the ratio to the real frame, so a font size chosen
in pixels comes out nearly four times too big. The ASS file states the real resolution. The SRT is
still written and attached as the selectable soft track, and both are generated from the same entries
so they cannot disagree.

## Three defects the tour found

Preparing the data for it surfaced two bugs, both of the shape that survives a test suite: each took a
branch only reachable when the interesting data existed, and every test exercised the other branch.

`_gnss_rows` read `Frame.gnss`, a PostGIS geography point, as though it were JSON. The query filters on
the column being non-null, so every row it ever returned raised, and the only sessions that appeared to
work were those with no satellite fixes at all. Measured ego pose had therefore never been written by
that module.

`object_on_real_frame` was a bare `exists().where(...)`, whose FROM clause SQLAlchemy fills in by
auto-correlation. Against a query that also joins `Frame` it correlates the table outwards and leaves
the subquery with no FROM, which raises at compile time. `embed_objects` joins `Frame` to read the
image URI, so the one caller in production raised on every run while all six tests passed.

Both fixes shipped with tests checked against the old code.

Filming found a third, in the product rather than the pipeline. The console's Background panel
interpolated each agent run count straight into a template string, and several agents write a list
there, so every drift and lift row read `findings [object Object],[object Object]`. It had been that way
for as long as the panel existed and nothing threw or logged. The scene was re-recorded after the fix,
which is what `--only` and the manifest merge are for.
