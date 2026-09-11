"""The tour's script: one entry per scene, with what to show, what to say, and how long to hold.

Kept as data rather than embedded in the driver, so the words can be reviewed and corrected without
touching the automation, and so the subtitle file and the voiceover come from exactly the same text. A
subtitle that disagrees with the audio is worse than no subtitle.

Every number spoken here is read from the live database at record time by `driver.py` and substituted
into the text, so the narration cannot drift from what the screen shows. Where a number would be
embarrassing it is still spoken: a tour that quotes only the flattering figures is a sales reel, and this
is meant to be usable by somebody deciding whether to trust the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scene:
    key: str
    chapter: str
    title: str
    # The page to open, relative to the web root. None means the scene reuses the previous page.
    path: str | None
    narration: str
    # Playwright actions run after load and before the hold, as (action, selector_or_value) pairs.
    actions: list[tuple] = field(default_factory=list)
    hold_s: float = 0.0        # extra seconds after the narration finishes, for the eye to catch up
    settle_s: float = 2.5      # seconds to wait after navigation before speaking


CHAPTERS = [
    "Introduction",
    "Ingest",
    "Auto-labelling",
    "Review",
    "Tracking and multi-camera",
    "3D and LiDAR",
    "Measurement",
    "Autonomy",
    "The data engine",
    "Export and privacy",
    "Edge",
    "Closing",
]

SCENES: list[Scene] = [
    # ---------------------------------------------------------------- Introduction
    Scene("home", "Introduction", "The console", "/",
          "This is LabeloxAV, a data engine for autonomous driving built around Indian roads. "
          "Everything you are about to see runs against a live database with {sessions} driving "
          "sessions, {frames} frames and {objects} labelled objects in it. "
          "No screen in this tour is a mock up. Where a number is small or a page is empty, that is "
          "the real state of the system and the narration will say so."),
    Scene("platforms", "Introduction", "Ten platforms, one spine", "/platforms",
          "The product is ten platforms over a single data spine. "
          "Annotation, review, measurement, governance, edge deployment and the rest each get their own "
          "surface, but they all read and write the same sessions, frames, objects and tracks. "
          "That is why a correction made in the editor changes what the gate measures an hour later."),
    Scene("projects", "Introduction", "Projects and packs", "/projects",
          "Work is scoped by project, and each project belongs to a domain pack. "
          "The autonomous driving pack brings the road ontology, the camera and LiDAR sensor model and "
          "the traffic scenario vocabulary. "
          "A second pack for physical security shares the same spine and none of the road specific code."),

    # ---------------------------------------------------------------- Ingest
    Scene("import", "Ingest", "Bringing data in", "/import",
          "Data arrives here. The importer takes dashcam video, ROS bags, MCAP recordings and raw "
          "sensor drives, and turns each one into a session with frames at a fixed timestamp. "
          "For this tour I ingested three real recordings. Two are dashcam clips from the owner of this "
          "machine, and one is a KITTI drive from Karlsruhe with a sixty four beam LiDAR on the roof."),
    Scene("import_migrate", "Ingest", "Migrating an existing corpus", "/import/migrate",
          "Most teams already have labels somewhere else. "
          "The migration path reads COCO, YOLO, Pascal and CVAT exports, maps their class names onto "
          "this ontology, and records the mapping so that nothing silently changes meaning. "
          "Classes it cannot map are held back rather than guessed at."),
    Scene("inspect", "Ingest", "The session inspector", "/inspect",
          "Every ingested session lands in the inspector. "
          "This is a Foxglove class viewer built into the product, so you can check a recording before "
          "you spend any labelling effort on it. "
          "The health gate at the top is the useful part. It refuses a session with broken timestamps, "
          "missing calibration or a camera that dropped out, instead of letting the problem reach the "
          "labellers."),
    Scene("inspect_session", "Ingest", "Inside one recording", "/inspect/{session_id}",
          "This is the KITTI drive. The timeline shows every topic on one clock, the camera alongside "
          "the LiDAR and the GPS. "
          "The panels are native, not an embedded iframe, so a click on the timeline moves the point "
          "cloud, the image and the ego pose together."),
    Scene("calibration", "Ingest", "Calibration", "/calibration",
          "Calibration is what makes three dimensional work possible. "
          "This session carries real intrinsics from the KITTI rig, a focal length of seven hundred and "
          "twenty one pixels over a twelve forty two by three seventy five image. "
          "Without this the system can still draw boxes, but it cannot tell you how far away anything is."),
    Scene("inertial", "Ingest", "Motion and ego state", "/inertial",
          "Here is the vehicle's own motion. Speed, yaw rate and the trajectory it drove. "
          "The KITTI import brought {poses} measured ego poses from the GPS and inertial unit. "
          "Before this import the entire corpus had three satellite fixes across forty one thousand "
          "frames, which is the honest reason most of the three dimensional work here was estimated "
          "rather than measured."),
    Scene("datasets", "Ingest", "Datasets", "/datasets",
          "A dataset is a frozen selection of frames with a content hash. "
          "Two people training on the same dataset identifier are training on exactly the same images "
          "and labels, which is the only way a comparison between two models means anything."),

    # ---------------------------------------------------------------- Auto-labelling
    Scene("annotate_new", "Auto-labelling", "Starting a labelling run", "/annotate/new",
          "Labelling starts automatically. You choose a session and the classes you care about, and the "
          "auto label runner works through it batch by batch. "
          "Batch by batch is a deliberate constraint, not an implementation detail. "
          "The runner holds a GPU slot, checks free video memory between batches, and stops if a "
          "training job wants the card, so a labelling run cannot take the machine down."),
    Scene("frame_editor", "Auto-labelling", "The frame editor", "/frame/{frame_id}",
          "This is the annotation canvas with real machine predictions on a real frame. "
          "The tools run down the left side. Box, polygon, segment anything, lane, drivable area, "
          "keypoints, amodal extent and depth. "
          "Every tool is registered in one place, so the keyboard shortcut you see in the panel is the "
          "shortcut that actually fires."),
    Scene("frame_describe", "Auto-labelling", "Labelling by description", None,
          "The describe tool is the one worth stopping on. "
          "You type what you want in plain words, for example a cow standing on the median, and an open "
          "vocabulary detector proposes it with a segment anything mask. "
          "This exists because the ontology will never have a word for everything on an Indian road, and "
          "the alternative is that those objects go unlabelled forever."),
    Scene("annotate_lane", "Auto-labelling", "Lanes and drivable space", "/annotate/lane/{frame_id}",
          "Lanes and drivable surface are annotated as geometry rather than boxes. "
          "The drivable mask now covers ninety nine point nine percent of frames in the corpus. "
          "It used to cover under four percent, and the gap was not a model problem. "
          "Three separate consumers were reading the masks in a format nobody had written."),
    Scene("attrsweep", "Auto-labelling", "Attribute sweeps", "/annotate/attrsweep",
          "Attributes are swept across many objects at once instead of one at a time. "
          "Occlusion level, truncation, whether a rider is wearing a helmet, whether a vehicle is parked "
          "or moving. "
          "One pass over a hundred crops of the same class is far faster than a hundred visits to the "
          "editor, and it keeps the judgement consistent."),
    Scene("annotate_asset", "Auto-labelling", "Road assets", "/annotate/asset/{frame_id}",
          "Road furniture gets its own surface. Signs, signals, poles, barriers and markings. "
          "Signs carry an IRC sixty seven code, which is the Indian standard, so a mandatory sign and a "
          "cautionary sign are distinguishable without reading the pictogram."),
    Scene("oraclyx", "Auto-labelling", "The model ensemble", "/oraclyx",
          "Behind the auto labelling sits an ensemble. Several detectors run over the same frame and "
          "their outputs are reconciled. "
          "Where they agree the label is cheap and probably right. Where they disagree is exactly where "
          "a human should be looking, and that disagreement is mined rather than averaged away."),
    Scene("pseudogt", "Auto-labelling", "Consensus pseudo ground truth", "/oraclyx/pseudogt",
          "Agreement between models becomes pseudo ground truth for training. "
          "One lesson from building this is recorded on the page. "
          "Choosing the teacher pair by how many predictions each model made gave nine consensus labels. "
          "Choosing by how many frames the two models shared gave one thousand one hundred and ninety."),

    # ---------------------------------------------------------------- Review
    Scene("review_queue", "Review", "The review queue", "/review/queue",
          "Machine labels are proposals until a person rules on them. "
          "The queue orders work by how much a verdict is worth, combining model uncertainty, class "
          "rarity and whether a gate is currently blocked on that class. "
          "There are {review_pending} objects waiting in review right now, which tells you honestly how "
          "far ahead of the humans the machine is."),
    Scene("review_grid", "Review", "Grid review", "/review/grid",
          "Grid review shows many crops of one class at once. "
          "The eye is very good at spotting the odd one out in a wall of similar images, and very bad at "
          "staying calibrated across a thousand single decisions. "
          "Accept or reject applies to everything selected in one keystroke."),
    Scene("review_rapid", "Review", "Rapid review", "/review/rapid",
          "Rapid review is the keyboard only mode for a long session. "
          "One crop, one key, next crop. "
          "It records how long each verdict took, because the cost of a label in minutes is the number "
          "the whole economics of this system rests on."),
    Scene("annotations", "Review", "Every annotation", "/annotations",
          "This is the full annotation table, filterable by class, state, source, session and reviewer. "
          "State is the important column. A label is in review, auto accepted, settled or accepted, and "
          "accepted means a person ruled on it. "
          "The machine can never write accepted. That is enforced in code and there is a test that fails "
          "the build if any code path tries."),
    Scene("object_detail", "Review", "One object's history", "/object/{object_id}",
          "Any single label opens to its full history. "
          "Which model proposed it, at what confidence, which human touched it and when, what it was "
          "before each change, and which training run consumed it. "
          "You can answer where did this label come from for any of {objects} objects."),
    Scene("quality", "Review", "Quality", "/quality",
          "Quality is measured, not asserted. "
          "Precision comes from stratified samples with Wilson confidence intervals, so a class with "
          "twelve judged crops reports a wide interval rather than a confident number. "
          "Unmeasured is reported as unmeasured. It is never reported as zero and never as fine."),
    Scene("quality_errors", "Review", "Error taxonomy", "/quality/errors",
          "Errors are typed rather than counted. "
          "A missed detection, a duplicate, a class confusion and a loose box are four different "
          "problems with four different fixes, and lumping them into one accuracy number hides which "
          "one you actually have."),
    Scene("labelox_quality", "Review", "Reviewer agreement", "/labelox/quality",
          "This page measures the reviewers. "
          "Control samples with a known answer are mixed into the queue, so agreement between people can "
          "be computed without anybody re-reviewing anybody. "
          "A finding worth admitting is here. Thirty thousand eight hundred and sixty three of thirty "
          "thousand eight hundred and sixty five reviews carry no timing, because the field was only "
          "wired up recently."),

    # ---------------------------------------------------------------- Tracking and multi-camera
    Scene("track_detail", "Tracking and multi-camera", "A track through time", "/track/{track_id}",
          "An object across frames is a track, and a track is the unit that matters for driving. "
          "The corpus holds {tracks} of them. "
          "Eighty four percent of tracks used to blink out and restart as a new identity. "
          "Interpolation across gaps had been written but never run, and turning it on filled one "
          "hundred and thirty seven thousand gaps."),
    Scene("track_timeline", "Tracking and multi-camera", "Editing a whole track", "/annotate/timeline/{track_id}",
          "Corrections apply to the track, not the frame. "
          "Relabelling a motorcycle as a rider used to reach one frame out of ninety three, because the "
          "editor and the review queue took different write paths. "
          "They now share one, and the propagation is visible on this timeline."),
    Scene("multicam", "Tracking and multi-camera", "Multiple cameras", "/annotate/multicam/{session_id}",
          "A vehicle with several cameras sees the same object more than once. "
          "With calibration, a box drawn in one view projects into every other view in the rig, so you "
          "annotate once. "
          "The corpus has six multi camera sessions, so the measured benefit here is small and I will "
          "not inflate it."),
    Scene("calyx", "Tracking and multi-camera", "Ego motion propagation", "/calyx",
          "Between two labelled frames the vehicle moved a known distance. "
          "That motion alone predicts where a static object should appear next, without running any "
          "model, and it is enough to carry a label forward through a stretch of road."),
    Scene("calyx_recovery", "Tracking and multi-camera", "Recovering lost tracks", "/calyx/recovery",
          "When a track breaks behind a bus and reappears, the recovery pass tries to rejoin the two "
          "halves using appearance and predicted position. "
          "A rejoined track is marked as rejoined rather than silently merged, because a wrong merge is "
          "worse than two short tracks."),

    # ---------------------------------------------------------------- 3D and LiDAR
    Scene("lidar", "3D and LiDAR", "Point clouds", "/lidar",
          "This is where the tour gets its real LiDAR. "
          "The KITTI drive brought {clouds} point clouds from a sixty four beam Velodyne, about {points} "
          "million points. "
          "Before this import the corpus had four hundred and fifty three thousand points in total, all "
          "of them estimated from single camera depth rather than measured by a laser."),
    Scene("lidar_annotate", "3D and LiDAR", "Three dimensional boxes", "/lidar/annotate",
          "Cuboids are drawn in the point cloud with the camera image beside them. "
          "The ego trajectory runs underneath, so a box can be placed in the world frame and stays put "
          "while the vehicle moves past it."),
    Scene("lidar_linked", "3D and LiDAR", "Camera and LiDAR together", "/lidar/linked",
          "The linked view keeps the image and the cloud in step. "
          "Click a two dimensional box and the matching three dimensional box highlights, and the other "
          "way round. "
          "This is also how the lifted boxes are checked. A cuboid estimated from camera depth alone is "
          "compared against the laser points that actually fall inside it."),
    Scene("map", "3D and LiDAR", "Where the data was collected", "/map",
          "Sessions on a map, by route and by city. "
          "The locations here are aggregated into cells rather than drawn as raw points, and I will come "
          "back to why in the privacy chapter."),

    # ---------------------------------------------------------------- Measurement
    Scene("verdyx", "Measurement", "Measurement", "/verdyx",
          "Verdyx is where models are measured. "
          "Accuracy is reported per class with an interval, and against a frozen gold set that no "
          "training run is allowed to touch. "
          "There are {models} registered models, each with its lineage recorded. What it was trained "
          "from, which teacher it distilled, and which dataset hash it saw."),
    Scene("verdyx_safety", "Measurement", "Safety slices and counterfactuals", "/verdyx/safety",
          "An average number hides the cases that matter. "
          "Performance is sliced by night, rain, occlusion and by the safety critical classes. "
          "Frames are also perturbed on purpose. Occlusion, dusk, rain, fog and motion blur are applied "
          "to real images and the model is re-scored. "
          "The current champion loses twenty four points of motorcycle recall under occlusion, which is "
          "the kind of thing an average accuracy score will never tell you."),
    Scene("scorecards", "Measurement", "Scorecards", "/scorecards",
          "A scorecard is the summary a person actually reads before a release. "
          "Per class recall and precision, the slices that regressed, the gate result and the reason if "
          "it was blocked. "
          "It is generated from the measurement run, so it cannot drift from the evidence."),
    Scene("training", "Measurement", "Training", "/training",
          "Training runs are scheduled here and they are the only thing allowed to hold the GPU for a "
          "long time. "
          "Everything else in the system checks whether training holds the card before it asks for it, "
          "and backs off if it does. "
          "That single rule is why labelling, embedding, depth and three dimensional lifting can all run "
          "on one desktop without fighting each other."),
    Scene("lineage", "Measurement", "Lineage", "/lineage",
          "Lineage connects a deployed model back to every label it learned from. "
          "If a label is found to be wrong, this answers which models saw it, and therefore which "
          "measurements are now suspect."),
    Scene("events", "Measurement", "Driving events", "/events",
          "Events are the moments worth finding. Hard braking, a cut in, a near miss, a pedestrian "
          "crossing against the signal. "
          "They are detected from ego dynamics and object tracks together. "
          "An honest caveat from building this. The dynamics signal in this corpus is noisier than the "
          "events it is being asked to find, so the detector is tuned to over report and let a human cut."),
    Scene("events_search", "Measurement", "Searching for a situation", "/events/search",
          "Events are searchable by description, so you can ask for every time a rider overtook on the "
          "left while the vehicle was braking, and get frames rather than a report."),

    # ---------------------------------------------------------------- Autonomy
    Scene("autonomy", "Autonomy", "The autonomy console", "/autonomy",
          "This is the part I would show a sceptic first. "
          "The system proposes its own work, but it cannot approve itself. "
          "Every autonomous action is a proposal with evidence, a named rule that fired, and a revert "
          "button that actually undoes the writes."),
    Scene("autonomy_settlement", "Autonomy", "Deciding when a class is done", None,
          "A class is signed off by sequential testing rather than a fixed sample. "
          "A clean class proves itself in far fewer verdicts and a bad one fails in a handful. "
          "Accept is deliberately conjunctive. The sequential test and the confidence interval must both "
          "agree before anything is accepted, while a rejection is allowed to fire early, because "
          "rejecting early is the safe direction to be wrong in."),
    Scene("agent", "Autonomy", "The annotation agent", "/agent",
          "The agent layer plans labelling work, criticises its own output and reconciles conflicts. "
          "It works in chunks, and every chunk is one recorded run with the exact rows it wrote. "
          "Reverting a run is a real database operation, not a flag."),
    Scene("govern", "Autonomy", "Governance and the killswitch", "/govern",
          "Governance sits above all of it. "
          "There is one killswitch that stops every autonomous writer, and every automated path checks "
          "it before doing anything. "
          "Approval thresholds are set per action here, so a team can let the system merge duplicate "
          "tracks on its own while still requiring a human for an ontology change."),
    Scene("jobs", "Autonomy", "Jobs", "/jobs",
          "Every long running piece of work is a job with a state, a progress figure and a log. "
          "Jobs that die are reaped rather than left claiming to run, which sounds obvious and took a "
          "real bug to get right."),
    Scene("activity", "Autonomy", "The activity log", "/activity",
          "One chronological record of everything that changed, by machine or by person. "
          "This is what you read when a number moved and nobody knows why."),
    Scene("reasoner", "Autonomy", "The reasoner", "/reasoner",
          "The reasoner explains a decision in words, with the evidence attached. "
          "Why this class was blocked, why this lot was rejected, why this model did not promote. "
          "The words are generated from the same fields the gate read, so an explanation cannot flatter "
          "a decision that the evidence does not support."),

    # ---------------------------------------------------------------- The data engine
    Scene("curation", "The data engine", "Curation", "/curation",
          "The data engine is the loop that decides what to label next. "
          "More data is not better data. The question is always which thousand frames would move the "
          "number that is currently stuck."),
    Scene("explore", "The data engine", "Exploring by similarity", "/explore",
          "Frames are embedded as vectors, {embeddings} of them, so you can search the corpus by "
          "resemblance rather than by keyword. "
          "Show me more scenes like this one is a query the system can answer directly."),
    Scene("search", "The data engine", "Search", "/search",
          "Text search runs over classes, attributes, sessions and scenario descriptions, and combines "
          "with the vector search, so a natural request narrows to a set of frames you can send straight "
          "to a labelling job."),
    Scene("discovery", "The data engine", "Finding the unknown", "/discovery",
          "Discovery looks for what the ontology has no word for. "
          "Clusters of frames that no class explains well are surfaced as candidates for a new class. "
          "This is how an Indian road corpus grows a vocabulary that a western ontology never had."),
    Scene("sievyx", "The data engine", "Filtering the corpus", "/sievyx",
          "Sievyx filters. Near duplicate frames, frames that are almost all sky, frames the camera "
          "auto exposure ruined. "
          "Removing these before labelling is the cheapest quality improvement available, because a "
          "duplicate frame costs a full label and teaches nothing."),
    Scene("longtail", "The data engine", "The long tail", "/sievyx/longtail",
          "The long tail is the whole problem in India. "
          "Cattle on the carriageway, three people on one motorcycle, an auto rickshaw reversing, a "
          "hand cart at night. "
          "These classes are rare in any corpus and over represented in the situations that matter, and "
          "this page ranks them by how starved each one is."),
    Scene("flywheel", "The data engine", "The adaptive flywheel", "/flywheel/adaptive",
          "This closes the loop. A blocked gate produces a recall deficit per class, the deficit builds "
          "its own review batch, the batch produces labels, the labels retrain the model and the gate "
          "runs again. "
          "That loop cleared a class that had been stuck at zero point one four recall for five "
          "training cycles."),
    Scene("campaigns", "The data engine", "Labelling campaigns", "/campaigns",
          "A campaign is a labelling goal with a budget and a deadline. "
          "The system allocates it against the marginal value of a label per rupee, using the measured "
          "time a verdict actually takes rather than an assumed rate."),
    Scene("scenarios", "The data engine", "Scenarios", "/scenarios",
          "Interesting moments are exported as OpenSCENARIO files, so a situation found in real footage "
          "becomes a test case a simulator can replay. "
          "Where the esmini simulator is installed, each exported scenario is replayed at export time "
          "and the bundle records whether it was actually valid."),
    Scene("analytics", "The data engine", "Analytics", "/analytics",
          "Corpus level analytics. Class balance, geographic coverage, time of day, weather, how much of "
          "each session is labelled. "
          "One number here is uncomfortable and deliberately left visible. Of {objects} objects, "
          "{accepted} carry a human verdict. "
          "The machine has run far ahead of the people, and pretending otherwise would make every "
          "quality number on this tour meaningless."),

    # ---------------------------------------------------------------- Export and privacy
    Scene("datasets_query", "Export and privacy", "Building a dataset by query", "/datasets/query",
          "A dataset is built from a query rather than a folder. "
          "All night time frames containing a rider without a helmet, excluding the gold set, excluding "
          "synthetic frames. "
          "The result is frozen with a hash and a datasheet that states exactly what went in."),
    Scene("synthetic", "Export and privacy", "Synthetic data, kept apart", None,
          "Synthetic frames exist, built by pasting real instances of starved classes into real "
          "backgrounds. "
          "They are quarantined structurally rather than by a filter. A synthetic label carries its own "
          "state and its own source, so every reader that measures anything excludes them without having "
          "to remember to. "
          "No synthetic pixel can reach a validation split or a gold set."),
    Scene("compliance", "Export and privacy", "Compliance", "/govern/compliance",
          "Indian data protection law applies to road footage, because faces and number plates are "
          "personal data. "
          "Faces and plates are detected and redacted, and the redaction is verified by re-running "
          "detection on the redacted image rather than trusting that the blur was applied."),
    Scene("pii_access", "Export and privacy", "Access to personal data", "/govern/pii-access",
          "Access to unredacted imagery is a role, every access is logged with a reason, and the log is "
          "not deletable from the product. "
          "Aggregated releases carry a differential privacy budget. Counts are noised, map cells with "
          "fewer than ten sessions are suppressed entirely, and when the budget for a scope is spent the "
          "endpoint refuses rather than quietly returning a weaker guarantee."),
    Scene("labeloxsec", "Export and privacy", "A second domain", "/labeloxsec",
          "The same spine runs a physical security pack. Cameras, zones, incidents. "
          "It shares ingest, annotation, review, measurement and governance, and shares none of the road "
          "specific code. "
          "An import contract is enforced in the build so that the core can never reach into a pack."),
    Scene("incidents", "Export and privacy", "Incidents", "/labeloxsec/incidents",
          "Incidents in the security pack are assembled from tracks and zones. "
          "There is no transcription anywhere in this pack. No text is read off a frame and no text "
          "content is stored, which is a deliberate scope limit rather than a missing feature."),
    Scene("integrations", "Export and privacy", "Integrations", "/integrations",
          "Exports go out in the formats teams already use. COCO, YOLO, KITTI, nuScenes, OpenLABEL and "
          "the scenario formats. "
          "Every bundle carries a datasheet with the query that built it, the class counts, the "
          "synthetic share and the privacy parameters."),

    # ---------------------------------------------------------------- Edge
    Scene("forgyx", "Edge", "Getting a model onto the vehicle", "/forgyx",
          "Forgyx takes a trained model to the edge. Export, quantise to eight bit, compile for the "
          "target and benchmark it. "
          "A model that is accurate and too slow has not solved the problem, so latency is a gate "
          "condition and not a footnote."),
    Scene("forgyx_deploy", "Edge", "Deployment", "/forgyx/deploy",
          "Deployment is staged. A candidate runs in shadow beside the current model on the vehicle "
          "before it takes over anything. "
          "The distillation loop here trains a small student from a large teacher and repeats until it "
          "fits the latency budget, or stops after four rounds and reports that it could not."),
    Scene("forgyx_field", "Edge", "The fleet in the field", "/forgyx/field",
          "Deployed models report back. Latency, thermal state, frames dropped and where the model was "
          "uncertain. "
          "Uncertain frames come home and re-enter the labelling queue, which is the outer loop of the "
          "whole data engine."),
    Scene("sanyx", "Edge", "Fleet health", "/sanyx",
          "Sanyx watches the hardware. Device temperature, storage, camera health and clock drift across "
          "the fleet."),
    Scene("sanyx_predictive", "Edge", "Predicting failures", "/sanyx/predictive",
          "Predicted failures ahead of time. A camera whose focus is slowly drifting or a drive filling "
          "up. "
          "A vehicle that records unusable footage for a week is more expensive than one that stops, "
          "because nobody notices until the labelling starts."),

    # ---------------------------------------------------------------- Closing
    Scene("ops", "Closing", "Operations", "/ops",
          "Operations shows the system's own health. Queue depths, GPU memory, worker state and the "
          "killswitch. "
          "The resource rules here are the reason this whole tour runs on one desktop. "
          "Every heavy path takes a slot, works in batches, checks free memory between them and yields "
          "to training."),
    Scene("collaborate", "Closing", "Working together", "/collaborate",
          "Several people work the same corpus at once, with assignment, locking and comment threads on "
          "individual objects. "
          "The interface is translated into Hindi, Kannada and Tamil, because the people doing the "
          "labelling are frequently not working in English."),
    Scene("profile", "Closing", "Your own work", "/profile",
          "Each person sees their own throughput, their agreement with control samples and their "
          "history. "
          "Measured, not estimated, and visible to the person it describes."),
    Scene("console_close", "Closing", "What this is and what it is not", "/console",
          "That is the full system. "
          "What is real today is the ingest, the auto labelling, the review surfaces, the tracking, the "
          "measurement with honest intervals, the governance and the export path. "
          "What is thin today is human verdicts, with {accepted} of {objects} objects ruled on, and "
          "measured ego pose, where the {poses} poses from this KITTI import are the first the corpus "
          "has ever had. "
          "Both of those are visible on the pages you just saw, because a data engine that hides its own "
          "gaps is not a data engine. It is a demo."),
]


# What each page has to be driven to do before it is worth filming. Kept apart from the scene text so a
# selector that moves is a one line change in a table, not an edit inside a paragraph of narration, and so
# the spoken script stays readable on its own.
#
# `{session_id}` and the other id placeholders are filled at record time from the same lookup that fills
# the narration, so the page and the words are always about the same session.
ACTIONS: dict[str, list[tuple]] = {
    "lidar": [("fill", 'input[placeholder="session id"]', "{session_id}"),
              ("click_text", "Load"), ("wait", 2500)],
    "lidar_annotate": [("fill", 'input[placeholder="or paste a session id"]', "{session_id}"),
                       ("click_text", "Load"), ("wait", 2500),
                       ("click", 'button:has-text(" pts")'), ("wait", 3500)],
    "lidar_linked": [("fill", 'input[placeholder="or paste a session id"]', "{session_id}"),
                     ("click_text", "Load"), ("wait", 2500),
                     ("click", 'button:has-text(" pts")'), ("wait", 3500),
                     ("click_text", "link"), ("wait", 3000)],
    "search": [("fill", 'input[placeholder^="natural language"]', "motorcycle at night"),
               ("click_text", "search"), ("wait", 4000)],
}

for _s in SCENES:
    _s.actions = ACTIONS.get(_s.key, [])


def check() -> None:
    """Fail loudly on a script that cannot be recorded, before anything is synthesised.

    Cheap to run and worth running: a duplicate key would silently overwrite a segment file, an unknown
    chapter would drop a scene out of the chapter cards, and an em-dash would be read aloud as a pause
    the subtitle does not show.
    """
    keys = [s.key for s in SCENES]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise ValueError(f"duplicate scene keys: {sorted(dupes)}")
    unknown = {s.chapter for s in SCENES} - set(CHAPTERS)
    if unknown:
        raise ValueError(f"scenes in unknown chapters: {sorted(unknown)}")
    stray = set(ACTIONS) - {s.key for s in SCENES}
    if stray:
        raise ValueError(f"actions for scenes that do not exist: {sorted(stray)}")
    for s in SCENES:
        if "—" in s.narration or "–" in s.narration:
            raise ValueError(f"dash in narration for {s.key}")
        if not s.narration.strip():
            raise ValueError(f"empty narration for {s.key}")


__all__ = ["Scene", "SCENES", "CHAPTERS", "ACTIONS", "check"]
