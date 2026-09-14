# A verification pass, with the numbers it produced

Everything below was run against the live system on 2026-09-12, not against fixtures. Where something
did not work it is named, and where a number could not be produced the reason is given instead.

## More data

Three KITTI raw drives were downloaded from the public archive. Two finished and were imported; the
third stopped part way through its 1.77 GB archive and was left out rather than imported half complete.

| drive | frames | Velodyne scans | GPS fixes |
| --- | --- | --- | --- |
| 2011_09_26_drive_0015 | 297 | 297 | 297 |
| 2011_09_26_drive_0027 | 188 | 188 | 188 |

KITTI raw is published by KIT and Toyota Technological Institute under CC BY-NC-SA 3.0 and is attributed
on each session. What that changed in the corpus:

| | before | after |
| --- | --- | --- |
| real LiDAR points | 18 million | 159 million |
| point clouds | 564 | 1,049 |
| measured ego poses | 154 | 639 |
| frames | 42,766 | 43,251 |

## The HD map ran end to end for the first time

Before this pass no session had both lane annotations and a position fix, so the georeferencer could not
run at all. Lane proposal turned out to be local rather than cloud-only, so it could be run across a
drive that already had GPS on every frame.

| stage | result |
| --- | --- |
| lanes proposed on drive 0005 | 199 on 154 frames, 41 rejected as off the drivable surface |
| georeferenced to world space | 199 elements, all with dataset calibration at quality 0.95 |
| fused into continuous boundaries | 53, of which 27 carry cross-frame consensus |
| paired into lanelets | 29, with 3 successor links |
| Lanelet2 export | 29 lanelet relations, 3 succession relations, 53 boundary ways |
| OpenDRIVE export | 48 roads, 58 driving lanes, 1 road link |

Both exports previously contained zero lanelet relations and zero driving lanes, so neither was routable.

Two things in that table are not yet trustworthy and should not be read as a working map. Fusion emits
one boundary 8,439 m long on a drive of a few hundred metres, which is a merge running away. And the
median paired lane is 2.54 m wide where a German lane is 3.0 to 3.5 m, which says the pairing is still
matching things that are not the two sides of a lane. The layer runs; the geometry needs work.

The georeferencing itself was also wrong before this pass, by a factor of two and a half in the near
field and fifteen approaching the horizon, because it used a nominal lens rather than the drive's own
calibration. That is recorded in the engineering log.

## Every readable route was called

`scripts/smoke_api.py` calls every GET route against the live database with an admin token. Writes are
not swept: firing 400 POST, PUT, PATCH and DELETE routes blindly would train models, launch cloud jobs
and delete data.

| outcome | count |
| --- | --- |
| answered | 220 |
| refused with 4xx, which is a working route | 53 |
| skipped, no id in this corpus to fill the path | 44 |
| slower than the 10 second limit | 34 |
| server error | 1 |

The one server error was `GET /api/collaborate/branches`, which raised a bare 500 whenever lakeFS was not
running. lakeFS is optional here: assignments, tasks and merge requests are all in Postgres and work
without it. The route now refuses with a 503 naming the reason and saying what is unaffected, which is
what every other optional-service path in this codebase already did. A test covers it and fails against
the old code.

The 34 slow routes are not failures. Several answered in 8 to 9 seconds when the limit was 30, so they
are slow rather than broken; the slowest answering route was `/api/analytics/source-mix` at 9.6 seconds.
The sweep itself had to be fixed twice before it could report any of this: its path-parameter lookup
scanned 600,000 rows on an unindexed JSON path and hung before sending a single request, and calling 352
routes serially could not finish inside any sensible timeout.

## Everything else

| check | result |
| --- | --- |
| Python test suite | 3,506 passed, 6 skipped, 4 xfailed, 0 failed |
| Web test suite | 703 passed across 62 files |
| TypeScript typecheck | clean |
| Import contract | 1 kept, 0 broken |
| Web pages loaded | 72 of 73 routes, 0 problems |
| Documentation build | strict, clean |

The one page not checked is the sign-in screen.
