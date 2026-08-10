"""The on-rig half of FORGYX: deciding what is worth the uplink, on the vehicle.

`services/ingest/extract_smart.py` makes this decision well and makes it server-side, after every frame has
already been driven home and stored. Moving it onto the rig is the largest scale lever available: a device
that uploads the interesting 8% of its day costs a twelfth as much to operate.

Deliberately free of the server stack. numpy, and cv2 only when a frame is actually written, because this
has to run on a board where installing the repo is not an option.
"""
