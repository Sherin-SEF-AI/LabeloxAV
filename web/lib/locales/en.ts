// The English dictionary, and the key list every other locale is checked against.
//
// Only the strings an annotator reads while working live here. The governance and platform surfaces stay
// English on purpose: they are read by operators who configured the deployment, and a half-translated
// compliance page is worse than an English one because it invites the reader to trust a phrasing nobody
// reviewed for legal meaning.
//
// Data is never translated. Class names, session ids and model versions stay as they are, because
// renaming a class in the interface would make it impossible to talk about with the ontology.

export type Dict = Record<string, string>;

export const EN: Dict = {
  "action.accept": "accept",
  "action.reject": "reject",
  "action.reclassify": "reclassify",
  "action.skip": "skip",
  "action.undo": "undo",
  "action.save": "save",
  "action.cancel": "cancel",
  "action.confirm": "confirm frame",
  "action.next": "next",
  "action.previous": "previous",
  "review.queue": "review queue",
  "review.rapid": "rapid review",
  "review.empty": "nothing in the queue",
  "review.decided": "decided",
  "editor.objects": "objects",
  "editor.lanes": "lanes",
  "editor.pose": "pose",
  "editor.review": "review",
  "editor.class": "class",
  "editor.confidence": "confidence",
  "editor.saved": "saved",
  "editor.unsaved": "unsaved changes",
  "editor.pick_label": "pick a label first",
  "nav.home": "home",
  "nav.activity": "activity",
  "nav.profile": "your account",
  "notify.empty": "nothing yet",
  "notify.mark_all": "mark all read",
  "onboarding.welcome": "Welcome to LabeloxAV",
  "onboarding.skip": "skip the tour",
  "onboarding.next": "next",
  "onboarding.done": "start working",

  // ---- Editor tools. The strip's labels, so the tool a person reaches for is named in their language.
  "tool.select": "select",
  "tool.box": "box",
  "tool.polygon": "polygon",
  "tool.polyline": "polyline",
  "tool.amodal": "whole extent",
  "tool.brush": "brush",
  "tool.eraser": "eraser",
  "tool.measure": "measure",
  "tool.keypoint": "keypoint",
  "tool.describe": "describe",
  "tool.cuboid": "3D box",
  "group.select": "Select",
  "group.draw": "Draw",
  "group.ai": "AI assist",
  "group.mask": "Mask edit",
  "group.region": "Region",
  "group.measure": "Measure",
  "group.pose": "Pose",

  // ---- Describe by phrase. The tool for the things the class list has no word for.
  "describe.title": "describe what you see",
  "describe.placeholder": "a few words, for example: cycle rickshaw",
  "describe.run": "find it",
  "describe.none": "nothing matched that phrase on this frame",
  "describe.recent": "recent phrases",
  // The caveat is part of the tool. An open-vocabulary model finds whatever it is asked for, so a weak
  // proposal is the model complying with the prompt rather than evidence that the object is there.
  "describe.caveat": "these are proposals from your words, not detections of a known class",
  "describe.count": "{n} proposals",

  // ---- Next object. Where to look next inside a frame, and why.
  "next.title": "next object",
  "next.none": "nothing on this frame is waiting",
  "next.continues": "continues a track you corrected",
  "next.value": "value {value}",

  // ---- Tube verdict. What the judge said about a whole track.
  "tube.title": "tube verdict",
  "tube.unjudged": "the tube judge has not looked at this track",
  "tube.judge": "judge this tube",
  "tube.correct": "the class looks right",
  "tube.incorrect": "the class looks wrong",
  "tube.unsure": "the judge could not tell",
  "tube.proposed": "suggests {klass}",
  "tube.crops": "from {n} views of the track",

  // ---- Frame editor chrome an annotator reads constantly.
  "frame.objects_count": "{n} objects",
  "frame.of_session": "frame {index} of {total}",
  "frame.loading": "loading the frame",
  "frame.failed": "the frame could not be loaded",
};
