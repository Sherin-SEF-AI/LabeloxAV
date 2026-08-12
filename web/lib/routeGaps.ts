// Breadcrumb ancestors that are not pages.
//
// `fromPath` builds the trail by accumulating path segments and linking every one but the last, which
// assumes a page exists at each level. It does not. `/annotate/new` renders "Home / Annotate / New" and
// links Annotate to `/annotate`, where nothing is routed, and `<Link>` prefetches, so every page under one
// of these fires an RSC request that 404s the moment it renders:
//
//   :3000/annotate?_rsc=1jxsw   404 (Not Found)
//
// Nobody clicks these often enough to notice the dead link, but the prefetch happens on every load, so the
// console fills with 404s describing the app's normal navigation. That is the same failure as the operation
// chips: noise that means "everything is fine" trains people to stop reading the console.
//
// Listed as the gaps rather than as the routes. There are 69 page routes and 11 gaps, so the short list is
// the one worth maintaining by hand, and the test beside this file walks web/app and asserts the two agree.
// Adding `app/review/page.tsx` later makes that test fail and say to delete the entry, which is the right
// direction for the reminder to point.
//
// The alternative was to add eleven redirect pages so the links resolve. That invents navigation targets to
// satisfy a breadcrumb, and a redirect to a child is a worse answer than a segment that is simply not a
// link: `/annotate` meaning "whichever annotate page we happened to pick" is not a URL anybody wants to
// land on or share.

/** Path prefixes that appear in a breadcrumb trail but have no page of their own. */
export const NON_PAGE_ANCESTORS: ReadonlySet<string> = new Set([
  "/annotate",
  "/annotate/asset",
  "/annotate/lane",
  "/annotate/multicam",
  "/annotate/timeline",
  "/flywheel",
  "/frame",
  "/labelox",
  "/object",
  "/review",
  "/track",
]);

/** Whether a breadcrumb segment at this accumulated path should be a link. */
export function isNavigable(path: string): boolean {
  return !NON_PAGE_ANCESTORS.has(path);
}
