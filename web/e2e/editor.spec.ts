import { expect, test } from "@playwright/test";

import { API, AUTOSAVE_MS, openFrame, requireFixture, serverObjects, signature, signedIn } from "./editorFixture";

// The editor's keyboard editing, checked where it counts: in what the server has afterwards.
//
// Every assertion about an edit reads the objects back from the API. A box that moved on screen and was
// never saved is the failure this suite exists to catch, and the page cannot tell you about it.
//
// Each test that edits has a frame to itself. Opening a frame takes its edit lock, and a browser context
// that is closed does not always get to give it back, so a second test on the same frame would find it
// held by "your other tab" for fifteen seconds and its keys would, correctly, do nothing.

test.describe("keyboard editing in the frame editor", () => {
  test("the canvas is layered, so a drag does not repaint the frame image", async ({ browser }) => {
    const fx = requireFixture();
    const ctx = await signedIn(browser, fx, "admin");
    const page = await openFrame(ctx, fx.frames[0]);
    // Konva draws one <canvas> per layer: static (image), objects, interaction.
    expect(await page.locator("canvas").count()).toBeGreaterThanOrEqual(3);
    await ctx.close();
  });

  test("arrow keys nudge the whole selection, the move is saved, and undo saves it back", async ({ browser }) => {
    const fx = requireFixture();
    const frame = fx.frames[1];
    const ctx = await signedIn(browser, fx, "admin");
    const page = await openFrame(ctx, frame);
    const before = await serverObjects(fx, frame);
    expect(before.length).toBeGreaterThan(1);

    await page.mouse.move(800, 500);
    await page.keyboard.press("Control+a");
    for (let i = 0; i < 3; i++) await page.keyboard.press("ArrowRight");
    await page.keyboard.press("Shift+ArrowDown");
    await page.waitForTimeout(AUTOSAVE_MS);

    const after = await serverObjects(fx, frame);
    for (const o of after) {
      const was = before.find((b) => b.object_id === o.object_id);
      expect(was, "an object kept its identity through the edit").toBeTruthy();
      expect(o.bbox[0] - was!.bbox[0]).toBeCloseTo(3, 1);       // three presses, one pixel each
      expect(o.bbox[1] - was!.bbox[1]).toBeCloseTo(10, 1);      // shift: ten pixels
    }

    for (let i = 0; i < 4; i++) await page.keyboard.press("Control+z");   // four presses, four entries
    await page.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, frame))).toBe(signature(before));
    await ctx.close();
  });

  test("a held arrow key is one undo step, not one per repeat", async ({ browser }) => {
    const fx = requireFixture();
    const frame = fx.frames[2];
    const ctx = await signedIn(browser, fx, "admin");
    const page = await openFrame(ctx, frame);
    const before = await serverObjects(fx, frame);

    await page.mouse.move(800, 500);
    await page.keyboard.press("Control+a");
    await page.keyboard.down("ArrowLeft");
    for (let i = 0; i < 6; i++) await page.keyboard.down("ArrowLeft");    // auto-repeat
    await page.keyboard.up("ArrowLeft");
    await page.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, frame))).not.toBe(signature(before));

    await page.keyboard.press("Control+z");
    await page.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, frame))).toBe(signature(before));
    await ctx.close();
  });

  test("copy on one frame and paste on another, then delete the lot", async ({ browser }) => {
    const fx = requireFixture();
    const [from, to] = [fx.frames[3], fx.frames[4]];
    const ctx = await signedIn(browser, fx, "admin");
    const page = await openFrame(ctx, from);
    const source = await serverObjects(fx, from);
    const target = await serverObjects(fx, to);

    await page.mouse.move(800, 500);
    await page.keyboard.press("Control+a");
    await page.keyboard.press("Control+c");

    // The clipboard has to survive the navigation, which is the whole point of pasting across frames.
    await page.goto(`/frame/${to}`, { waitUntil: "domcontentloaded" });
    await page.waitForSelector("canvas", { timeout: 60_000 });
    await page.waitForTimeout(1500);
    await page.mouse.move(800, 500);
    await page.keyboard.press("Control+v");
    await page.waitForTimeout(AUTOSAVE_MS);

    const pasted = await serverObjects(fx, to);
    expect(pasted.length).toBe(target.length + source.length);
    const fresh = pasted.filter((o) => !target.some((t) => t.object_id === o.object_id));
    expect(fresh.length).toBe(source.length);
    for (const o of fresh) {
      expect(source.some((s) => s.object_id === o.object_id), "a pasted object is a new object").toBe(false);
      expect(o.track_id, "a pasted object does not join the source's track").toBeNull();
    }

    // The paste leaves the new objects selected. Delete removes every one of them, after asking.
    await page.keyboard.press("Delete");
    const confirm = page.getByRole("alertdialog");
    await expect(confirm).toBeVisible();
    await confirm.getByRole("button", { name: /delete|remove/i }).first().click();
    await page.waitForTimeout(AUTOSAVE_MS);
    expect(signature(await serverObjects(fx, to))).toBe(signature(target));
    await ctx.close();
  });

  test("an annotator can fix a label nobody has reviewed, and cannot move one a reviewer accepted", async ({ browser }) => {
    const fx = requireFixture();
    const frame = fx.frames[7];
    const before = await serverObjects(fx, frame);
    expect(before.length).toBeGreaterThan(1);

    // A reviewer accepts one of the frame's labels before the annotator arrives.
    const settled = before[0];
    const accept = await fetch(`${API}/api/objects/${settled.object_id}/review`, {
      method: "POST",
      headers: { Authorization: `Bearer ${fx.users.reviewer.token}`, "Content-Type": "application/json" },
      body: JSON.stringify({ action: "accept" }),
    });
    expect(accept.status).toBe(200);

    const ctx = await signedIn(browser, fx, "annotator");
    const page = await openFrame(ctx, frame);
    await page.mouse.move(800, 500);
    await page.keyboard.press("Control+a");
    await page.keyboard.press("ArrowRight");
    await page.waitForTimeout(AUTOSAVE_MS + 1000);

    const after = await serverObjects(fx, frame, "reviewer");
    for (const o of after) {
      const was = before.find((b) => b.object_id === o.object_id)!;
      if (o.object_id === settled.object_id) {
        // The reviewer's label is where the reviewer left it.
        expect(o.bbox[0]).toBeCloseTo(was.bbox[0], 1);
        expect(o.state).toBe("accepted");
      } else {
        // Everything else moved, and an annotator's save is a submission, never an acceptance.
        expect(o.bbox[0] - was.bbox[0]).toBeCloseTo(1, 1);
        expect(o.state).not.toBe("accepted");
      }
    }
    await ctx.close();
  });

  test("the shortcut overlay opens as a dialog and offers rebinding", async ({ browser }) => {
    const fx = requireFixture();
    const ctx = await signedIn(browser, fx, "admin");
    const page = await openFrame(ctx, fx.frames[0]);
    await page.mouse.move(800, 500);
    await page.keyboard.press("Shift+Slash");
    const overlay = page.getByRole("dialog", { name: /keyboard shortcuts/i });
    await expect(overlay).toBeVisible();
    await expect(overlay.getByText("click a key to change it").first()).toBeVisible();
    await expect(overlay.getByRole("button", { name: /reset to defaults/i })).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(overlay).toBeHidden();
    await ctx.close();
  });

  test("an editing session raises no page errors and no server errors", async ({ browser }) => {
    const fx = requireFixture();
    const ctx = await signedIn(browser, fx, "admin");
    const errors: string[] = [];
    ctx.on("page", (p) => {
      p.on("pageerror", (e) => errors.push(String(e)));
      p.on("response", (r) => { if (r.status() >= 500) errors.push(`${r.status()} ${r.url()}`); });
    });
    const page = await openFrame(ctx, fx.frames[8]);            // "next frame" below lands on the last one
    await page.mouse.move(800, 500);
    await page.keyboard.press("Control+a");
    await page.keyboard.press("ArrowUp");
    await page.keyboard.press("Control+z");
    await page.keyboard.press("BracketRight");                  // next frame
    await page.waitForTimeout(AUTOSAVE_MS);
    expect(errors).toEqual([]);
    await ctx.close();
  });
});
