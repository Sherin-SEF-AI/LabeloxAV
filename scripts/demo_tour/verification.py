"""The scene list for the verification recording: what was checked, on the pages that show it.

Separate from `narration.py` because it is a different kind of film. The tour explains what the product
is; this one is a record of a verification pass, and every claim in it is a number produced by a run
earlier the same day rather than a description of a feature.

The numbers that describe the corpus are read from the database at record time, the same way the tour
does it. The numbers that describe a run, like how many routes answered, are quoted from that run and
written here, because a past run cannot be re-read from a live query.
"""

from __future__ import annotations

from narration import Scene

KITTI_0005 = "765c177b-8c46-4869-9188-2e7fd3aa1d61"
COMMIT = "map-fee6980351629588"

CHAPTERS = ["What was checked", "New data", "The HD map", "Every route", "What is not fixed"]

SCENES: list[Scene] = [
    Scene("v_intro", "What was checked", "The pass", "/console",
          "This is a record of a verification pass over LabeloxAV, run against the live system. "
          "Three things were done. More real data was downloaded and ingested, every readable interface "
          "was called to see what answers, and the HD map was taken end to end for the first time. "
          "Every number spoken here came from one of those runs."),

    Scene("v_data", "New data", "Two more KITTI drives", "/inspect",
          "Two KITTI raw drives were downloaded from the public archive and imported, adding four "
          "hundred and eighty five frames that each carry a synchronised Velodyne scan, a GPS fix and "
          "the drive's own calibration. "
          "A third drive stopped part way through its download and was left out rather than imported "
          "half complete."),
    Scene("v_corpus", "New data", "What it changed", "/analytics",
          "The corpus now holds {clouds_all} point clouds and about {points_all} million real laser "
          "points, against eighteen million before this pass. "
          "Measured ego poses went from one hundred and fifty four to {poses}. "
          "Everything else in the corpus is still estimated from single camera depth, and the tooling "
          "says so on every page that reads it."),
    Scene("v_inspect", "New data", "One of the new drives", f"/inspect/{KITTI_0005}",
          "This is a drive in the session inspector, with the camera, the laser and the GPS on one "
          "clock. "
          "The point cloud here is measured by a sixty four beam scanner, not inferred from pixels, "
          "which is what makes the three dimensional work on this session checkable."),

    Scene("v_lanes", "The HD map", "Lanes, at last on frames that know where they are", "/annotate/lane/{frame_id}",
          "Before this pass no session had both lane annotations and a position fix. One hundred and "
          "fifty seven frames of forty two thousand carried a fix, and not one of them had a lane, so "
          "the map could not be built at all. "
          "Lane proposal turned out to run locally rather than only on a rented GPU, so it was run "
          "across a drive that already had a fix on every frame. "
          "One hundred and ninety nine lanes on one hundred and fifty four frames, with forty one "
          "rejected for sitting off the drivable surface."),
    Scene("v_georef", "The HD map", "Placed on the earth", "/map",
          "Those lanes were then georeferenced into world space. One hundred and ninety nine map "
          "elements, every one placed using the drive's own calibration rather than a nominal lens. "
          "That distinction is not cosmetic. With the wrong lens this projection put a lane six metres "
          "ahead at fifteen metres, and one at seventy metres at over a kilometre."),
    Scene("v_topology", "The HD map", "Lanes that connect", "/map",
          "Fusion merged those into fifty three continuous boundaries, and the topology layer paired "
          "twenty nine of them into lanes with three links between them. "
          "The exports now carry twenty nine lanelet relations and fifty eight driving lanes. "
          "Both formats previously carried zero of each, which meant neither was a map anything could "
          "route on."),

    Scene("v_api", "Every route", "Calling everything that reads", "/ops",
          "Every readable interface was then called against the live database. "
          "Three hundred and fifty two of them. Two hundred and twenty answered, fifty three refused "
          "properly, forty four could not be reached because this corpus has no identifier to fill the "
          "address, and one returned a server error. "
          "The four hundred routes that change data were not called, because firing those blindly would "
          "train models, launch paid jobs and delete work."),
    Scene("v_bug", "Every route", "The one that failed", "/collaborate",
          "The failure was the annotator branch listing, which returned a bare server error whenever "
          "the versioning service was not running. "
          "That service is optional here. Assignments, tasks and merge requests are all in the database "
          "and work without it. "
          "It now refuses with the reason and says what is unaffected, which is what every other "
          "optional path in this codebase already did."),
    Scene("v_tests", "Every route", "The suites", "/quality",
          "Three thousand five hundred and six tests pass with nothing failing, seven hundred and three "
          "in the web suite, the typecheck is clean and the import contract holds. "
          "Seventy two of the seventy three pages load without a problem. The one not checked is the "
          "sign in screen."),

    Scene("v_honest", "What is not fixed", "What this does not claim", "/map",
          "Two things in the map are not yet trustworthy and are worth saying plainly. "
          "Fusion produced one boundary eight and a half kilometres long on a drive of a few hundred "
          "metres, which is a merge running away. "
          "And the median paired lane came out two and a half metres wide where a German lane is three "
          "to three and a half, which says the pairing is still matching things that are not the two "
          "sides of a lane. "
          "The layer runs end to end. The geometry underneath it needs work, and calling that finished "
          "would be the easiest thing to get wrong here."),
    Scene("v_close", "What is not fixed", "Where it stands", "/console",
          "So the honest summary. More real data is in, every readable route has been called and one "
          "genuine failure was found and fixed, and the HD map runs from a camera frame through to a "
          "routable export for the first time. "
          "The human verdict count has not moved: {accepted} of {objects} labels carry one. "
          "That remains the thing most worth fixing, and no amount of code changes it."),
]

ACTIONS: dict[str, list[tuple]] = {
    "v_georef": [("wait", 2500)],
    "v_topology": [("wait", 2500)],
    "v_honest": [("wait", 2000)],
}

for _s in SCENES:
    _s.actions = ACTIONS.get(_s.key, [])
