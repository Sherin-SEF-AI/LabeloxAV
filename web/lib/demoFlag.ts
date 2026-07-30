// Demo-only chrome.
//
// Some surfaces exist to explain the product rather than to operate it. They belong in a walkthrough and
// not in the toolbar an annotator uses all day, where they cost space and read as unfinished to anyone
// evaluating the tool.
//
// Read from the build environment rather than a runtime toggle: this gates presentation, and a production
// build should not ship the markup at all. NEXT_PUBLIC_ is required for the value to reach the browser.
export const IS_DEMO_BUILD = process.env.NEXT_PUBLIC_LBX_DEMO === "1";
