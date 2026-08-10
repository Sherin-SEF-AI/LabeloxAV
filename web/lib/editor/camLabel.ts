// The camera's display name.
//
// The HUD rendered `cam {meta.cam_id}` against ids that already carry the prefix, so a front camera read
// "cam cam_front". Normalising here rather than at the call site keeps every surface that names a camera
// agreeing with the rig configuration, which is where these ids come from.
export function camLabel(camId: string | null | undefined): string {
  if (!camId) return "cam";
  const id = String(camId).trim();
  // Ids are conventionally `cam_front`, `cam_rear_left`. Strip the redundant prefix and re-add one word so
  // the label reads "cam front" rather than "cam cam_front" or a bare "cam_front".
  const bare = id.replace(/^cam[_-]?/i, "");
  return bare ? `cam ${bare.replace(/[_-]+/g, " ")}` : id;
}
