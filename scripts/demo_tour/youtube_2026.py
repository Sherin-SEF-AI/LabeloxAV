"""The 2026-11 film: every feature of the product, explained over the live system, for a public audience.

The September tour (`narration.py`) is a page-by-page record for people deciding whether to trust the
system. This one is the product film: it opens on the operator's own Bengaluru dashcam drives rather than
Karlsruhe, it drives each page into the state worth seeing while the voice describes it (the `live` steps
run inside the recorded window, with a visible pointer), and the specialist tools are a fast montage
rather than a scene each.

Same rules as the tour. Every number is read from the database at record time and substituted into the
text; nothing on screen is mocked; a sentence never starts with a placeholder; no dashes in narration.

    LBX_TOUR_TTS=kokoro:af_heart LBX_TOUR_OUT=.scratch/demo/yt/tour \\
      LBX_TOUR_WEB=http://localhost:3011 LBX_TOUR_API=http://localhost:8020 \\
      .venv/bin/python scripts/demo_tour/driver.py --script youtube_2026 --out .scratch/demo/yt/raw.mp4
"""

from __future__ import annotations

from dataclasses import dataclass, field

from narration import Scene as _Base


@dataclass
class Scene(_Base):
    # Steps run after recording starts, so the viewer sees them happen while the narration plays.
    live: list = field(default_factory=list)


CHAPTERS = [
    "Introduction",
    "Getting data in",
    "Labelling",
    "Review",
    "Search and curation",
    "3D and LiDAR",
    "Models and measurement",
    "The assistant and automation",
    "Privacy and governance",
    "The labs",
    "Closing",
]

S = Scene

SCENES: list[Scene] = [
    # ---------------------------------------------------------------- Introduction
    S("home", "Introduction", "Home", "/",
      "This is LabeloxAV, a data engine for autonomous driving, built for Indian roads. "
      "It takes raw drives in, labels them with models, routes the uncertain parts to people, trains on "
      "the result, and measures whether the new model is actually better. "
      "Everything in this video is the live system. It holds {sessions} real drives, {frames} frames and "
      "{objects} labelled objects, and every number you hear is read from that database as it is filmed. "
      "Home opens on what is waiting for a person, then the corpus at a glance.",
      settle_s=4.0,
      live=[("move_text", "Waiting for a person"), ("wait", 2500), ("move_text", "Blind audit"),
            ("wait", 2500), ("move_text", "Recent sessions"), ("wait", 1500)]),
    S("platforms", "Introduction", "One spine, many platforms", "/platforms",
      "Under the hood it is one data spine with several platforms on top. "
      "Ingest health, calibration, curation, annotation, pseudo ground truth, evaluation and edge "
      "deployment each have their own surface, and they all read and write the same sessions, frames, "
      "objects and tracks. "
      "So a correction made in the editor changes what the evaluation gate measures, without an export "
      "in between.",
      live=[("move_text", "SANYX"), ("wait", 1500), ("move_text", "ORACLYX"), ("wait", 1500),
            ("move_text", "VERDYX"), ("wait", 1500)]),

    # ---------------------------------------------------------------- Getting data in
    S("upload", "Getting data in", "Upload", "/annotate/new",
      "Data comes in by dropping files here: dashcam video, image folders or MCAP recordings. "
      "Each file becomes its own session, and faces and number plates are blurred on every frame before "
      "anything is stored. "
      "The upload keeps running while you move around the app, and tells you when it is done.",
      live=[("move_text", "drag clips"), ("wait", 2000), ("move_text", "Each file uploads"), ("wait", 1500)]),
    S("import", "Getting data in", "Importing datasets", "/import",
      "Existing datasets import from COCO, YOLO, Parquet, video and MCAP, of any size. "
      "Every import is a job with its own progress and its own record, and the ones that failed stay on "
      "the list with the reason, rather than disappearing.",
      live=[("wait", 1500), ("scroll", 700), ("wait", 1500)]),
    S("migrate", "Getting data in", "Migrating from another tool", "/import/migrate",
      "Most teams already have labels in another tool. "
      "The migration path reads Labelbox, CVAT and similar exports, maps their class names onto this "
      "ontology, and shows what the mapping costs in a dry run before anything is written.",
      live=[("move", "select"), ("wait", 1500), ("move_text", "dry run"), ("wait", 1000)]),
    S("sessions", "Getting data in", "Sessions", "/annotations",
      "Every drive is a session. "
      "Each card shows how many frames and objects it holds, how much of it a person has reviewed, and "
      "where its labels stand, from waiting for review to accepted.",
      live=[("wait", 1500), ("scroll", 500), ("wait", 1500)]),
    S("inspect", "Getting data in", "The session inspector", "/inspect/{kitti_session}",
      "The session inspector plays a recording back on one clock. "
      "Here a KITTI drive from Karlsruhe, with the roof LiDAR on the left and the front camera on the right, "
      "both scrubbed by the same timeline. "
      "It is how you check a recording before you spend anyone's time labelling it.",
      settle_s=5.0,
      live=[("click", 'button:text-is("play")'), ("wait", 6000), ("point", 700, 300)]),
    S("motion", "Getting data in", "Position and speed", "/inertial",
      "Most dashcams burn the time, speed and GPS position into their own video. "
      "LabeloxAV reads that stamp back with on-device text recognition, so {gnss_frames} frames across "
      "{gnss_sessions} drives now carry a position, and from that, a measured path and the vehicle's speed. "
      "This is one of the operator's own drives near Bengaluru.",
      settle_s=3.0,
      live=[("select", "select", "{dash_session}"), ("wait", 4000), ("point", 800, 300), ("wait", 1500)]),
    S("calibration", "Getting data in", "Calibration", "/calibration",
      "Calibration is validated per session before any three dimensional work is trusted. "
      "Each camera's intrinsics and mounting are resolved from a measured rig, a dataset, or an estimate "
      "from road lines, and the trust level is shown next to it.",
      live=[("move_text", "estimate from road lines"), ("wait", 1500)]),
    S("map", "Getting data in", "The map", "/map",
      "Lanes and signs lifted from the camera are placed on the map, each element traceable back to the "
      "frames and the calibration it came from.",
      settle_s=4.0, live=[("point", 500, 500), ("wait", 1200), ("point", 640, 420)]),

    # ---------------------------------------------------------------- Labelling
    S("editor", "Labelling", "The frame editor", "/frame/{dash_frame}",
      "This is the frame editor, on a street in Bengaluru. "
      "There are {dash_objects} objects on this one frame: autorickshaws, motorcycles, riders, delivery bikes, "
      "hoardings, with the drivable road in green. "
      "The models proposed all of it. Every object lists its class and confidence on the right, and the "
      "checks underneath flag boxes that look wrong before a person even looks.",
      settle_s=5.0,
      live=[("point", 420, 520), ("wait", 1200), ("point", 700, 560), ("wait", 1200),
            ("move_text", "OBJECTS"), ("wait", 1500), ("move_text", "CHECKS"), ("wait", 1500)]),
    S("editor_tools", "Labelling", "Tools in the editor", None,
      "Along the top are the tools: boxes, polygons, pose, measurement, three dimensional boxes, and AI "
      "assist, which turns a click into a mask. "
      "Layers switch on and off for masks, lanes, drivable space and cuboids. "
      "The strip at the bottom is the drive itself, frame by frame, and everything saves as you go.",
      settle_s=5.0,
      live=[("move_text", "AI assist"), ("wait", 1500), ("move_text", "Mask edit"), ("wait", 1200),
            ("move_text", "3D box"), ("wait", 1200), ("move_text", "LAYERS"), ("wait", 1500),
            ("point", 600, 1040), ("wait", 1200)]),
    S("lanes", "Labelling", "Lanes", "/annotate/lane/{frame_id}",
      "Lanes have their own view. Each lane line is typed, dashed, solid or double, and the ego lane is "
      "marked, because a lane change is only measurable against the lane you were in.",
      settle_s=4.0),
    S("track", "Labelling", "A track through time", "/track/{dash_track}",
      "Objects are tracked across frames. "
      "This bus is one track, {dash_track_len} frames long, and every crop is the same vehicle. "
      "Accept the whole track with one key, change its class once for all of its frames, or mark what it "
      "is doing: merging, overtaking, cutting in.",
      settle_s=4.0,
      live=[("move_text", "accept all"), ("wait", 1500), ("move_text", "RELABEL ENTIRE TRACK"), ("wait", 1500),
            ("move_text", "cut_in"), ("wait", 1000)]),
    S("timeline", "Labelling", "Editing a whole track", "/annotate/timeline/{dash_track}",
      "The timeline workspace edits that track as one thing. "
      "Keyframes a person drew, frames the tracker filled in, and gaps it interpolated are all marked "
      "differently, and a correction here re-interpolates the frames around it.",
      settle_s=4.0, live=[("point", 300, 70), ("wait", 1500), ("point", 600, 70)]),
    S("attrsweep", "Labelling", "Attribute sweeps", "/annotate/attrsweep",
      "Some questions are faster asked one at a time across many objects. "
      "An attribute sweep takes one attribute, like whether a rider wears a helmet, and asks it with one "
      "key per object, ordered by how much of it is still unanswered.",
      settle_s=6.0),

    # ---------------------------------------------------------------- Review
    S("queue", "Review", "My queue", "/queue",
      "Review starts at My queue. "
      "Objects waiting for a decision are ordered with the likeliest wrong first, and each row says why it "
      "needs you: a box outside the object, a class conflict, low confidence, a rare class. "
      "With {review_pending} objects waiting corpus wide, the order is the whole point.",
      settle_s=5.0,
      live=[("move_text", "Why this order?"), ("wait", 1500), ("point", 520, 720), ("press", "j"),
            ("wait", 600), ("press", "j"), ("wait", 600), ("press", "j"), ("wait", 1500)]),
    S("grid", "Review", "Rapid review and the crop grid", "/review/grid",
      "For speed there are keyboard views. "
      "Rapid review shows one crop at a time with one key per verdict: accept, reject, reclassify or skip. "
      "The crop grid puts dozens of objects on screen together, so the odd one out is obvious, and a whole "
      "range is decided at once. "
      "Faces and plates stay blurred in every one of them.",
      settle_s=7.0, live=[("point", 1180, 520), ("wait", 900), ("press", "ArrowRight"), ("wait", 500),
                          ("press", "ArrowRight"), ("wait", 500), ("press", "ArrowDown"), ("wait", 1500)]),
    S("value_queue", "Review", "The value queue", "/review/queue",
      "Behind those views, a value queue scores every candidate: how uncertain the model was, how rare "
      "the class is, how different it is from what was already labelled, and how likely it is to be an "
      "error.",
      settle_s=7.0),
    S("errors", "Review", "Label errors", "/quality/errors",
      "Label errors runs several detectors over the corpus: confident learning, track inconsistency, "
      "and a vision language model reading the crop. "
      "Each suspect is confirmed or dismissed with one keystroke, and the page measures which detectors "
      "actually find real errors.",
      settle_s=6.0, live=[("move_text", "all detectors"), ("wait", 1500), ("point", 900, 360), ("wait", 1000)]),
    S("gold", "Review", "Gold sets", "/quality",
      "Accuracy is measured against sealed gold sets, {gold_sets} of them here. "
      "Once sealed, a gold set cannot drift, so a score this month and a score next month answer the same "
      "question.",
      settle_s=4.0),
    S("scorecards", "Review", "Labeller scorecards", "/scorecards",
      "Labeller scorecards measure each person over labels a second person later ruled on. "
      "A rate only counts as proven after enough judged labels, and below that it shows a question mark "
      "rather than a flattering number.",
      settle_s=3.0),

    # ---------------------------------------------------------------- Search and curation
    S("search", "Search and curation", "Search", "/search",
      "Search works in plain language across {embeddings} embedded frames. "
      "Type what you are looking for, and it finds frames that look like it, ranked by similarity, and "
      "spread across different drives so the results are not twenty copies of one moment.",
      settle_s=3.0,
      live=[("type", 'input[placeholder^="natural language"]', "autorickshaw on a busy market street"),
            ("wait", 400), ("click", 'button:text-is("search")'), ("wait", 11000), ("scroll", 400)]),
    S("scenarios", "Search and curation", "Scenarios", "/scenarios",
      "Scenarios are the driving situations mined from tracks: cut ins, wrong side driving, animals on the "
      "road. There are {scenarios} of them so far, each with the actors involved, a criticality score and the city. "
      "They are searchable in plain language too.",
      settle_s=4.0,
      live=[("type", 'input[placeholder^="e.g."]', "motorcycle cutting in"), ("wait", 300),
            ("click", 'button:text-is("search")'), ("wait", 4000)]),
    S("explore", "Search and curation", "Explore", "/explore",
      "Explore lays the whole corpus out by visual similarity. "
      "Each point is an object, and similar things cluster together, so a cluster of mislabelled objects "
      "shows up as a colour in the wrong place. "
      "Lasso a group, tag it, or send it straight to review.",
      settle_s=10.0, live=[("click_text", "class"), ("wait", 3000), ("point", 900, 500), ("wait", 1500)]),
    S("dataset_query", "Search and curation", "Building a dataset", "/datasets/query",
      "Datasets are described rather than hand picked. "
      "A query like night and vulnerable road users is counted against the live corpus before anything is "
      "built, so you know what you will get.",
      settle_s=6.0),
    S("datasets", "Search and curation", "Sealed datasets", "/datasets",
      "Exports are sealed and versioned, in COCO, YOLO, OpenLABEL, nuScenes and Parquet, with a validation "
      "and test split. "
      "A sealed dataset never changes, so a model trained on it can always be traced back to exactly what "
      "it saw.",
      settle_s=3.0, live=[("move_text", "nuscenes"), ("wait", 1500)]),

    # ---------------------------------------------------------------- 3D and LiDAR
    S("lidar", "3D and LiDAR", "Point clouds", "/lidar/annotate",
      "Three dimensional data has its own workspace. "
      "The corpus holds {clouds_all} point clouds and {points_all} million points, from a real sixty four "
      "beam laser on the KITTI drives. "
      "Here one of them, with a bird's eye view underneath.",
      settle_s=3.0,
      live=[("select", "select", "{kitti_cuboids}"), ("click", 'button:text-is("Load")'), ("wait", 3500),
            ("click", 'button:has-text(" pts |")'), ("wait", 5000), ("point", 1400, 300), ("wait", 1500)]),
    S("lidar_linked", "3D and LiDAR", "Camera and LiDAR together", "/lidar/linked",
      "Cuboids drawn in the point cloud project into the camera image through the drive's calibration, "
      "so a three dimensional box can be checked against what the camera saw.",
      settle_s=3.0,
      live=[("select", "select", "{kitti_cuboids}"), ("click", 'button:text-is("Load")'), ("wait", 3500),
            ("click", 'button:has-text(" pts")'), ("wait", 6000)]),

    # ---------------------------------------------------------------- Models and measurement
    S("training", "Models and measurement", "Training", "/training",
      "Training runs on the GPU in this box. "
      "Name a model line, pick a task, the classes and the cities, and queue it. "
      "Every run keeps its history, including the ones that failed and why, because a run that silently "
      "vanished teaches nothing.",
      settle_s=3.0, live=[("move_text", "NEW TRAINING JOB"), ("wait", 1500), ("scroll", 400)]),
    S("registry", "Models and measurement", "The model registry", "/govern",
      "Every trained model lands in the registry with its score on gold. "
      "One is the champion, and the rest are challengers. "
      "A challenger only replaces the champion through a gate, and every automated decision is written to "
      "the audit log underneath, with a kill switch above it all.",
      settle_s=4.0, live=[("move_text", "Champion"), ("wait", 1500), ("move_text", "KILL SWITCH"), ("wait", 1500)]),
    S("slices", "Models and measurement", "Slice evaluation", "/verdyx",
      "A single accuracy number hides the cases that matter. "
      "Slice evaluation compares champion and challenger per slice, like pedestrians at night or "
      "autorickshaws in glare. A regression on a protected slice can block a promotion even when the "
      "average went up.",
      settle_s=3.0, live=[("move_text", "pedestrian_night"), ("wait", 2000)]),
    S("analytics", "Models and measurement", "Analytics", "/analytics",
      "Analytics summarises the corpus: class balance, where labels came from, scenario coverage, review "
      "agreement, and how much of it has been anonymised. "
      "Even where the data was collected is guarded. It is released only as noised map cells, each release "
      "is charged against a privacy budget, and once the budget is spent the page says so, rather than "
      "quietly giving away more.",
      settle_s=10.0,
      live=[("scroll", 1500), ("wait", 1500), ("click", 'button:text-is("release capture density")'),
            ("wait", 3000)]),

    # ---------------------------------------------------------------- The assistant and automation
    S("assistant", "The assistant and automation", "The assistant", "/assistant",
      "The assistant answers questions about the data in plain language. "
      "It runs on a local model, so nothing leaves the machine. "
      "Questions are answered straight from the database. Anything that would change data, like sending "
      "objects back for review or starting a training run, becomes a proposal that a reviewer approves.",
      settle_s=3.0,
      live=[("click_text", "Which drives do we have from Delhi?"), ("wait", 20000)], hold_s=2.0),
    S("agent", "The assistant and automation", "The agent console", "/agent",
      "The agent console runs quality work across the whole corpus. "
      "The overnight auditor samples the day's auto accepted labels within a budget, checks them with a "
      "second model, and queues suspects for a person. "
      "Everything it does is reversible and stamped with where it came from.",
      settle_s=3.0, live=[("move_text", "Overnight Auditor"), ("wait", 2000), ("move_text", "Relabel all frames"),
                          ("wait", 1500)]),
    S("autonomy", "The assistant and automation", "Autonomy", "/autonomy",
      "Autonomy is earned per class, never assumed. "
      "A class only moves up a rung when a measured precision clears its threshold, and large batches of "
      "labels settle by checking a random sample. "
      "Right now no class has climbed above propose only, and the page says so, with the evidence behind "
      "that and how old it is.",
      settle_s=4.0, live=[("move_text", "PER-CLASS LADDER"), ("wait", 1500), ("move_text", "SETTLEMENT LOTS"),
                          ("wait", 1500)]),
    S("jobs", "The assistant and automation", "Jobs", "/jobs",
      "Every long running piece of work is a job: imports, auto labelling, training, map fusion. "
      "Each one shows its progress, and an interrupted job says exactly why it stopped.",
      settle_s=3.0),
    S("console", "The assistant and automation", "The console", "/console",
      "The console shows the machine itself: the GPU, memory, disk, and what the API process is doing, "
      "because on a single box, knowing when it is busy matters.",
      settle_s=3.0),

    # ---------------------------------------------------------------- Privacy and governance
    S("status", "Privacy and governance", "System status", "/settings",
      "Settings opens on a system check. "
      "Database, storage, the GPU, and the privacy models: the face and plate blur, and plate text "
      "verification, which reads the blurred plate back and withholds the frame if any text survives.",
      settle_s=4.0, live=[("move_text", "Privacy blur models"), ("wait", 1500), ("move_text", "Plate text verification"),
                          ("wait", 2000)]),
    S("pii", "Privacy and governance", "Access to personal data", "/govern/pii-access",
      "Every view of personal data is logged: who looked, at what, and whether it was redacted. "
      "There are {pii_access} accesses on record, and the unredacted count is the one that matters.",
      settle_s=3.0),
    S("compliance", "Privacy and governance", "Compliance", "/govern/compliance",
      "Compliance enforces the rules at the data spine: a release cannot ship without redaction proof, an "
      "export fails closed without consent, cloud spending has a ceiling, and lineage traces any subject "
      "through everything it touched.",
      settle_s=3.0),
    S("projects", "Privacy and governance", "Projects", "/settings/projects",
      "Projects group sessions for a client, a study or a partner. "
      "A project can be visible to the whole team or to its members only, and it carries its own class "
      "set: all {classes} Indian road classes, a generic driving set, or a custom list.",
      settle_s=3.0, live=[("move_text", "Who sees its sessions"), ("wait", 1500), ("move_text", "Indian roads"),
                          ("wait", 1200), ("move_text", "Generic driving"), ("wait", 1200)]),
    S("team", "Privacy and governance", "The team", "/settings/team",
      "Teammates are invited by link and choose their own password, with a role: annotator, reviewer or "
      "administrator. "
      "Any member can be signed out everywhere at once.",
      settle_s=3.0, live=[("move", 'input[placeholder="e.g. priya"]'), ("wait", 1200)]),
    S("about", "Privacy and governance", "Licensing", "/settings/about",
      "And the about page states exactly what is running and under which licence, linked to the source of "
      "this exact version.",
      settle_s=3.0),

    # ---------------------------------------------------------------- The labs
    S("labs_pseudogt", "The labs", "Pseudo ground truth", "/oraclyx/pseudogt",
      "Beyond the everyday workflow there is a set of research tools. "
      "Pseudo ground truth fuses several models offline, keeps what they agree on, and routes the "
      "disagreements to people, where a human label is worth most.",
      settle_s=3.0),
    S("labs_longtail", "The labs", "The long tail", "/sievyx/longtail",
      "The long tail tools compare what the fleet has collected with the conditions it needs, and find the "
      "rare clusters worth labelling next.",
      settle_s=3.0),
    S("labs_flywheel", "The labs", "The adaptive flywheel", "/flywheel/adaptive",
      "The adaptive flywheel reads the live corpus and plans the next cycle: which classes are below their "
      "safety floor, what to label, and what to go and collect.",
      settle_s=3.0),
    S("labs_calyx", "The labs", "Calibration drift", "/calyx/recovery",
      "Calibration tools recover a camera's focal length and pitch from the scene itself, and track drift "
      "in a rig over time.",
      settle_s=3.0),
    S("labs_forgyx", "The labs", "Edge deployment", "/forgyx",
      "Edge deployment benchmarks a model on the target hardware, plots latency against accuracy, and "
      "stages a rollout from canary to the full fleet, with a rollback.",
      settle_s=3.0),
    S("labs_sanyx", "The labs", "Ingest health", "/sanyx",
      "And ingest health scores every drive as it arrives, so a broken recording is quarantined before "
      "anyone labels it.",
      settle_s=3.0),

    # ---------------------------------------------------------------- Closing
    S("closing", "Closing", "One loop", "/",
      "That is LabeloxAV. "
      "Drives in, labelled by models, checked by people where it matters, trained on, measured honestly, "
      "and shipped to the vehicle. "
      "One system, on one GPU box, built for the roads it will actually drive on.",
      settle_s=4.0, hold_s=2.0),
]

for _s in SCENES:
    _s.actions = []

__all__ = ["Scene", "SCENES", "CHAPTERS"]
