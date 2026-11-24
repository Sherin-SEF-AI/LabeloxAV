// Warm the browser's cache for the frame on either side of this one.
//
// `[` and `]` are the most pressed keys in the editor and each landed on a cold image: the next frame's
// JPEG was requested only after the route had changed and its metadata had come back. The server already
// sends the image with a private one-hour cache lifetime, so a request made a second early is a request
// the next page does not wait for.
//
// One frame each way and no further. The image endpoint is also the access record for frames holding
// personal data, and reading ten frames ahead would log ten views of frames nobody opened.

/**
 * The image URLs of the previous and next frame, derived from this frame's own.
 *
 * There is no client-side builder for a frame's image URL: the server puts it in the frame's metadata, and
 * the next frame's metadata is exactly what has not been fetched yet. A neighbour's URL is this frame's
 * with the id swapped, which is the server's shape rather than a second copy of it written here. If the
 * id is not in the URL the shape is not one this understands, and nothing is prefetched.
 */
export function neighbourImageUrls(imageUrl: string, frameId: string,
                                   prev: string | null | undefined, next: string | null | undefined): string[] {
  if (!imageUrl || !frameId || !imageUrl.includes(frameId)) return [];
  const out: string[] = [];
  for (const id of [prev, next]) {
    if (!id || id === frameId) continue;
    const url = imageUrl.split(frameId).join(id);
    if (!out.includes(url)) out.push(url);
  }
  return out;
}

type ImageLike = { src: string; crossOrigin: string | null; onload: unknown; onerror: unknown };

/**
 * Start loading each URL and return a function that lets go of them.
 *
 * Letting go detaches and forgets; it does not abort. The usual reason this is called is that the
 * annotator pressed `]`, and the request it would abort is the one the page they are arriving at is about
 * to make. A failed prefetch is not reported anywhere: the frame it was for reports its own failure when
 * it is opened.
 */
export function prefetchImages(urls: string[], make: () => ImageLike = () => new Image()): () => void {
  // Held so the request has an owner for as long as the page that asked is mounted.
  let held: ImageLike[] = [];
  for (const url of urls) {
    const im = make();
    // The same mode the canvas loads with, so the cached response is one the canvas is allowed to reuse.
    im.crossOrigin = "anonymous";
    const done = () => { held = held.filter((x) => x !== im); };
    im.onload = done;
    im.onerror = done;
    held.push(im);
    im.src = url;
  }
  return () => {
    for (const im of held) { im.onload = null; im.onerror = null; }
    held = [];
  };
}
