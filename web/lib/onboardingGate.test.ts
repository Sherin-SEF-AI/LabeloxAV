// Observed on a clean browser profile: navigating to /login renders the tour ("WELCOME TO LABELOXAV, 1 OF
// 4") over the sign-in card, and Playwright cannot click "Sign in as the dev administrator" at all. The
// retry log names the reason, which is the tour's own backdrop:
//
//   <div class="fixed inset-0 z-[100] bg-bg/85 ..."> intercepts pointer events
//
// Onboarding's user check did not stop it because AuthBootstrap had already stored an admin token on mount,
// as it is designed to in local dev. Two guards that look equivalent are not: "is somebody signed in" and
// "is this a screen you reach before signing in" disagree exactly here.

import { describe, expect, it } from "vitest";

import { isPreAuthRoute, shouldOpenOnboarding } from "./onboardingGate";

describe("isPreAuthRoute", () => {
  it("recognises the sign-in screen", () => {
    expect(isPreAuthRoute("/login")).toBe(true);
  });

  it("recognises nested auth routes", () => {
    expect(isPreAuthRoute("/auth/callback")).toBe(true);
    expect(isPreAuthRoute("/login/reset")).toBe(true);
  });

  it("does not match a route that merely starts with the same letters", () => {
    // A prefix test without the boundary would silently suppress chrome on a real page.
    expect(isPreAuthRoute("/loginless")).toBe(false);
    expect(isPreAuthRoute("/authoring")).toBe(false);
  });

  it("treats an unknown pathname as a normal screen", () => {
    expect(isPreAuthRoute(null)).toBe(false);
    expect(isPreAuthRoute(undefined)).toBe(false);
    expect(isPreAuthRoute("/")).toBe(false);
  });
});

describe("shouldOpenOnboarding", () => {
  it("stays shut while the route is still unknown", () => {
    // A full-screen modal opened before we know where we are can land on the login page, which is the bug
    // this file exists for. Not showing a tour costs a tour; showing it there costs the session.
    expect(shouldOpenOnboarding(null, true, false)).toBe(false);
    expect(shouldOpenOnboarding(undefined, true, false)).toBe(false);
    expect(shouldOpenOnboarding("", true, false)).toBe(false);
  });

  it("stays shut on the login page even when a token is already stored", () => {
    // The exact production of the bug: AuthBootstrap signed us in, so hasUser is true on /login.
    expect(shouldOpenOnboarding("/login", true, false)).toBe(false);
  });

  it("opens on a real page for a signed-in first-timer", () => {
    expect(shouldOpenOnboarding("/", true, false)).toBe(true);
  });

  it("does not open for somebody who has seen it", () => {
    expect(shouldOpenOnboarding("/", true, true)).toBe(false);
  });

  it("does not open with no credential at all", () => {
    expect(shouldOpenOnboarding("/", false, false)).toBe(false);
  });
});
