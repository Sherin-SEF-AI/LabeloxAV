// Whether the welcome tour may open on this screen.
//
// Onboarding already meant to stay off the login page, and said so: "Showing it on the login page would put
// a tour in front of somebody who cannot yet reach any of the things it points at." It enforced that by
// checking for a signed-in user, which is not the same question and does not hold here.
//
// AuthBootstrap runs app-wide in local dev and mints a real admin token on mount, login page included. So by
// the time Onboarding looks, a user exists, the tour opens, and it renders at z-100 over the whole viewport
// with `inset-0`, which means it also swallows every click aimed at the sign-in form underneath. A first
// visit lands on a modal about a review queue and a sign-in button that does not respond.
//
// The route is the thing that was actually being asked about, so that is what is checked. Pure, and separate
// from the component, because "which screens are pre-auth" is a fact about the app that other chrome will
// want too.

/** Screens reached before or during sign-in. Nothing modal may cover these. */
export const PRE_AUTH_ROUTES = ["/login", "/logout", "/auth"];

export function isPreAuthRoute(pathname: string | null | undefined): boolean {
  if (!pathname) return false;
  return PRE_AUTH_ROUTES.some((r) => pathname === r || pathname.startsWith(`${r}/`));
}

/**
 * Whether the first-run tour should open.
 *
 * `seen` is the persisted marker, `hasUser` the stored credential. The route check comes first because it is
 * the only one of the three that a local-dev auto-login cannot make true by accident.
 */
export function shouldOpenOnboarding(pathname: string | null | undefined,
                                     hasUser: boolean, seen: boolean): boolean {
  if (isPreAuthRoute(pathname)) return false;
  if (!hasUser) return false;
  return !seen;
}
