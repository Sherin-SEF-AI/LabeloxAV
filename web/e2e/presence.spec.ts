import { expect, test } from "@playwright/test";

import { API, AUTOSAVE_MS, openFrame, requireFixture, serverObjects, signature, signedIn } from "./editorFixture";

// Two people on one frame.
//
// Before the edit lock, the second person in could edit freely and found out at save time, with a 409 and
// "reload to continue". These walk the whole arrangement with three real identities: the second person is
// view-only and is told who has the frame; the server refuses their editor's save whatever their page
// shows; a reviewer can take the frame over, is asked first, and the person displaced is told; and a lock
// whose holder vanished is handed on.

const banner = "status";

test.describe("presence and the frame edit lock", () => {
  test.setTimeout(120_000);

  test("the second person is view-only, and the server enforces it", async ({ browser }) => {
    const fx = requireFixture();
    const frame = fx.frames[5];
    const admin = await signedIn(browser, fx, "admin");
    const adminPage = await openFrame(admin, frame);
    await expect(adminPage.getByRole("button", { name: "Take over" })).toHaveCount(0);

    const annotator = await signedIn(browser, fx, "annotator");
    const viewer = await openFrame(annotator, frame);
    await expect(viewer.getByRole(banner).filter({ hasText: /view only/i })).toContainText(fx.users.admin.name);
    // An annotator may not take a frame from somebody.
    await expect(viewer.getByRole("button", { name: "Take over" })).toHaveCount(0);

    // The editor learns who else is here on its next heartbeat.
    await expect(adminPage.getByLabel(/Also on this frame/)).toHaveAttribute(
      "aria-label", new RegExp(fx.users.annotator.name), { timeout: 15_000 });

    // The viewer's keys do nothing.
    const before = await serverObjects(fx, frame);
    await viewer.mouse.move(800, 500);
    await viewer.keyboard.press("Control+a");
    for (let i = 0; i < 5; i++) await viewer.keyboard.press("ArrowRight");
    await viewer.keyboard.press("Delete");
    await viewer.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, frame))).toBe(signature(before));

    // And a save from an editor that does not hold the lock is refused by the server, not just the page.
    // A reviewer's token, because deleting an object needs that role and the lock is what is under test.
    const res = await fetch(`${API}/api/objects/${before[0].object_id}`, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${fx.users.reviewer.token}`, "X-Lbx-Tab": "e2e-not-the-holder-0001" },
    });
    expect(res.status).toBe(423);
    expect((await serverObjects(fx, frame)).some((o) => o.object_id === before[0].object_id)).toBe(true);

    // The holder is unaffected.
    await adminPage.mouse.move(800, 500);
    await adminPage.keyboard.press("Control+a");
    await adminPage.keyboard.press("ArrowRight");
    await adminPage.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, frame))).not.toBe(signature(before));

    await admin.close();
    await annotator.close();
  });

  test("a reviewer takes over, the displaced editor is told, and a vanished holder's lock is handed on", async ({ browser }) => {
    const fx = requireFixture();
    const frame = fx.frames[6];
    const admin = await signedIn(browser, fx, "admin");
    const adminPage = await openFrame(admin, frame);

    const reviewer = await signedIn(browser, fx, "reviewer");
    const reviewerPage = await openFrame(reviewer, frame);
    await reviewerPage.getByRole("button", { name: "Take over" }).click();
    // It asks first: taking a frame switches somebody else to view-only mid-task.
    const confirm = reviewerPage.getByRole("alertdialog");
    await expect(confirm).toContainText(fx.users.admin.name);
    await confirm.getByRole("button", { name: "Take over" }).click();
    await expect(reviewerPage.getByRole("button", { name: "Take over" })).toHaveCount(0);

    // The displaced editor finds out on its next heartbeat and can no longer change the frame.
    await expect(adminPage.getByRole(banner).filter({ hasText: /view only/i })).toContainText(
      fx.users.reviewer.name, { timeout: 15_000 });
    const held = await serverObjects(fx, frame);
    await adminPage.mouse.move(800, 500);
    await adminPage.keyboard.press("Control+a");
    await adminPage.keyboard.press("ArrowLeft");
    await adminPage.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, frame))).toBe(signature(held));

    // The reviewer's browser goes away without releasing anything. The lock lapses (15 s) and the tab
    // that was waiting gets the frame back by itself.
    await reviewer.close();
    await adminPage.waitForTimeout(22_000);
    await adminPage.mouse.move(800, 500);
    await adminPage.keyboard.press("Control+a");
    await adminPage.keyboard.press("ArrowDown");
    await adminPage.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, frame))).not.toBe(signature(held));
    await admin.close();
  });
});
