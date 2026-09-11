# Demo film

## Full tour (20 min)

Every page in the product, narrated, with subtitles and twelve chapter marks. Seventy five scenes
covering seventy two of the seventy three routes in the web application; the one it skips is the
sign-in page.

Every number spoken in it is read from the live database at record time and substituted into the
script, so the voice cannot disagree with the screen. That includes the unflattering ones: the tour
says out loud that 788 of 598,179 objects carry a human verdict, that 30,863 of 30,865 reviews have no
timing, and that the multi-camera benefit is measured over six sessions and is therefore small.

It was filmed on three recordings ingested for it: two dashcam clips from Bengaluru and KITTI drive
`2011_09_26_drive_0005`, whose import gave the corpus its first measured ego poses (154) and its first
laser points (154 clouds, 18 million points).

<video controls preload="metadata" style="width:100%; border-radius:6px;"
       src="demo/labeloxav-tour.mp4"></video>

[Download the tour](demo/labeloxav-tour.mp4) (50 MB, H.264 1080p) ·
[subtitles](demo/labeloxav-tour.srt)

How it was made, and the two defects making it uncovered: [the recorded tour](DEMO_TOUR.md).

## The earlier films

Everything in these films is the real application, recorded live from a browser driving the running
system: the real corpus (578,436 objects, 41,752 frames, 377 sessions), the operator's own
`DASHCAM-01` fleet footage in the editor, real GPU SAM segmentation, a real drift scan hitting the
governance plane. Nothing is mocked and nothing is staged; the two labels the demo tools created were
deleted right after filming, and the settlement and control-sample queues were never touched.

## Narrated tour (4 min)

Voiceover, the full platform, and every canvas tool operated on camera: box, polygon, polyline,
point-SAM, box-SAM, magic wand, measure, lane-aligned 3D cuboid, pose keypoints, adverse-region flag,
mask brush, eraser, superpixel cells, and the Lanes / Semantic / Events mode rail.

<video controls preload="metadata" style="width:100%; border-radius:6px;"
       src="demo/labeloxav-demo-narrated.mp4"></video>

[Download the narrated film](demo/labeloxav-demo-narrated.mp4) (16 MB, H.264)

## Silent cut (3 min)

The same hands-on tour without narration - captions only.

<video controls preload="metadata" style="width:100%; border-radius:6px;"
       src="demo/labeloxav-demo.mp4"></video>

[Download the silent film](demo/labeloxav-demo.mp4) (10 MB, H.264)

## What the films show, in order

1. The platform launcher, and the risk-ranked review queue with bulk accept / reject / relabel.
2. The frame editor on real Bengaluru dashcam footage: every drawing tool, the AI family on the
   local GPU, measurement, cuboids, pose, adverse regions, mask surgery, and the mode rail.
3. A 90-frame track timeline where one relabel fixes every frame at once.
4. A settlement lot's acceptance sample - the human verdicts that decide whether 9,586 labels settle.
5. The control queue that audits the auto-accept gate itself.
6. Natural-language search, scenario discovery, driving events, corpus analytics, gold-set health,
   the training factory, jobs, sealed dataset delivery, and multi-format import.
7. The governance console (a real drift scan, clicked on camera) and the autonomy console: a live
   daemon heartbeat, the per-class permission ladder, and the evidence behind every rung.

Two real defects were found *by* filming and fixed on camera: a daemon heartbeat that read as dead
during its busiest tick, and a drift-recovery path that silently re-armed autonomous promotion. Both
fixes shipped with regression tests proven by reverting them.
