"use client";

import { useRouter } from "next/navigation";

// One definition of "back" for the whole app: return to wherever the user actually came from, and only fall
// back to a fixed route when there is no history to return to (a deep link or a fresh tab). Pages used to
// hard-code router.push("/") after a save, which threw away the user's place and dumped them on the home
// dashboard; useSmartBack replaces that so finishing with an object returns you to the list you were working.

export function useSmartBack(fallback = "/") {
  const router = useRouter();
  return () => {
    if (typeof window !== "undefined" && window.history.length > 1) router.back();
    else router.push(fallback);
  };
}
