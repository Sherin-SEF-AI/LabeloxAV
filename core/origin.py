"""Where a frame's pixels came from, and the one predicate every measuring reader must carry.

`real` is a camera on a road. `synthetic` is a composite the copy-paste generator built from real
donors and a real background; `perturbed` is a real frame with a counterfactual applied (occlusion,
dusk, rain). Only real pixels may teach a model that will be measured, and only real pixels may be
measured against. The column lives on `frame` (nearly every object reader already joins it) and on
`session` (for listing); the two agree by construction because a session's frames all share its origin.

Readers on the allow-list side (source == 'human', a state list, a cycle id) exclude synthetic labels
for free, because synthetic objects sit in their own state and source. Readers that sweep frames by
time or by absence of a derived row are the deny-list side and must say `Frame.origin == REAL`.
"""

from __future__ import annotations

REAL = "real"
SYNTHETIC = "synthetic"
PERTURBED = "perturbed"
ORIGINS = (REAL, SYNTHETIC, PERTURBED)

# The object state and source a generated label carries. A machine-only state, like `settled`, but in
# the other direction: it never means "a person ruled" and no person may write it.
SYNTHETIC_STATE = "synthetic"
SYNTHETIC_SOURCE = "synthetic"
