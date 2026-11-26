"""Sensors beyond the camera and the LiDAR, and what can be rebuilt from them (agent E's tools).

- radar.py: radar sweeps (nuScenes RADAR_* .pcd, MCAP PointCloud2 with velocity fields, radar_msgs/RadarScan),
  stored as `radar_sweep` rows with their arrays as npz in the object store, projected onto camera frames and
  fused into Track3D speed.
- events.py: event-camera streams ((x, y, t, polarity) from .npy, .h5 and MCAP EventArray), accumulated into
  ordinary frames so a person annotates them in the normal editor.
- reconstruct/: a session window rebuilt in 3D, either as trained Gaussian splats (gsplat) or as a colourised
  aggregated LiDAR point map, with novel views rendered for placing a cuboid that then goes to every frame.
"""
