// COCO-17 person/cyclist pose skeleton: 17 named keypoints + the edges that connect them. Keypoint
// values are [x, y, v] in image pixels with v in {0 not-labeled, 1 occluded, 2 visible}.
export const PERSON_17 = {
  name: "person_17",
  points: [
    "nose", "l_eye", "r_eye", "l_ear", "r_ear",
    "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle",
  ],
  edges: [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16],
  ],
} as const;
