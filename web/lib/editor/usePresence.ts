"use client";

// Who else is on this frame, and whether this tab may edit it, kept current by a heartbeat.
//
// What an answer means is decided in ./presence.ts, which knows nothing about React. This is the part
// that cannot be pure: the timer, the tab going into the background and coming back, the page closing,
// and the frame changing under a component that stays mounted.
//
// Polling, because the heartbeat this tab has to send anyway brings back everything it needs. There is no
// second channel to fall out of step with the first.

import { useCallback, useEffect, useMemo, useReducer, useRef } from "react";

import { presenceApi, type LockHolder, type PresenceState, type PresenceUser } from "@/lib/presenceApi";
import { getUser, setEditorTab } from "@/lib/user";

import {
  initialPresence, isReadOnly, newTabToken, presenceReducer, type PresenceModel, type PresenceStatus,
} from "./presence";

/** The calls the hook makes. `presenceApi` in the app; a fake in tests. */
export type PresenceTransport = Pick<typeof presenceApi,
  "heartbeat" | "acquire" | "release" | "takeOver" | "releaseOnUnload">;

export type UsePresenceOptions = {
  transport?: PresenceTransport;
  /** Whether there is a signed-in user to be present as. Defaults to asking lib/user. */
  hasUser?: () => boolean;
};

export type Presence = {
  status: PresenceStatus;
  /** Everyone else on the frame right now. Never includes the signed-in user. */
  others: PresenceUser[];
  lock: { mine: boolean; holder: LockHolder | null };
  /** Presence cannot be reached, so nothing is gated and nobody is listed. */
  degraded: boolean;
  /** The editor must refuse edits: somebody else holds the frame, or the first answer is not in yet. */
  readOnly: boolean;
  canTakeOver: boolean;
  /** Who took the frame from this tab, while that is still why it is view-only. */
  lostTo: LockHolder | null;
  /** Goes up each time this tab takes a lock it was not already holding. See ./presence.ts. */
  acquisitions: number;
  /** Who held the frame just before this tab last took it, if anyone did. */
  regainedFrom: LockHolder | null;
  /** Take the frame from whoever is editing it. Rejects when the server refuses (403 for an annotator). */
  takeOver: () => Promise<PresenceState>;
  /** The whole model, for whatever renders a banner from it. */
  model: PresenceModel;
};

// One token for the life of the page, shared by every frame this tab visits.
let tabToken: string | null = null;
const thisTab = () => (tabToken ??= newTabToken());

// A failed release has nobody left to report to (the component is gone) and nothing depends on it: the
// lock lapses by its TTL exactly as it does for a tab that crashed.
function lockWillExpireByItself(): void {}

export function usePresence(frameId: string, opts: UsePresenceOptions = {}): Presence {
  const transport = opts.transport ?? presenceApi;
  const hasUserRef = useRef(opts.hasUser);
  hasUserRef.current = opts.hasUser;

  const [stored, dispatch] = useReducer(presenceReducer, frameId, initialPresence);
  // The frame changes a render before the effect below can reset the model. For that one render the
  // stored model is the PREVIOUS frame's, and reading "this tab holds the lock" off it would leave the
  // next frame editable before anybody has been asked.
  const model = useMemo(
    () => (stored.frameId === frameId ? stored : initialPresence(frameId)), [stored, frameId]);

  const beatSecondsRef = useRef(model.heartbeatS);
  beatSecondsRef.current = model.heartbeatS;

  useEffect(() => {
    dispatch({ t: "reset", frameId });
    if (!frameId) return undefined;
    if (!(hasUserRef.current ? hasUserRef.current() : !!getUser())) {
      dispatch({ t: "unavailable", frameId });
      return undefined;
    }

    const tab = thisTab();
    // From here until this frame is left, this tab's saves carry the token and the server checks them
    // against the lock.
    setEditorTab(tab);
    let live = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let inFlight = false;
    let answered = false;

    const beat = async () => {
      if (!live || inFlight) return;
      inFlight = true;
      try {
        // The first call is the explicit "I am opening this to edit"; every later one is a heartbeat
        // that also asks for the lock, which is how a viewer becomes the editor once the holder leaves
        // and how an editor back from a background tab gets a lapsed lock back.
        const state = answered
          ? await transport.heartbeat(frameId, tab, true)
          : await transport.acquire(frameId, tab);
        answered = true;
        if (live) dispatch({ t: "answer", frameId, state });
      } catch {
        // Counted, not swallowed: the reducer turns enough of these in a row into `degraded`, which the
        // editor shows. One dropped request changes nothing on screen, on purpose.
        if (live) dispatch({ t: "failed", frameId });
      } finally {
        inFlight = false;
      }
    };

    // A timeout that re-arms itself after each beat rather than an interval, so a slow answer delays the
    // next beat instead of stacking a second request behind it.
    const arm = () => {
      if (!live || timer !== null) return;
      timer = setTimeout(async () => {
        timer = null;
        await beat();
        if (!document.hidden) arm();
      }, beatSecondsRef.current * 1000);
    };
    const disarm = () => {
      if (timer !== null) clearTimeout(timer);
      timer = null;
    };

    // A hidden tab stops beating. Its lock then lapses by itself, which frees the frame for whoever is
    // waiting without this tab having to decide that its user has gone. Coming back beats at once.
    const onVisibility = () => {
      if (document.hidden) { disarm(); return; }
      disarm();
      void beat().then(arm);
    };
    const onPageHide = () => transport.releaseOnUnload(frameId, tab);
    // Restored from the back/forward cache: the release above already went, so start again.
    const onPageShow = (e: PageTransitionEvent) => { if (e.persisted) onVisibility(); };

    if (!document.hidden) void beat().then(arm);
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("pagehide", onPageHide);
    window.addEventListener("pageshow", onPageShow);

    return () => {
      live = false;
      setEditorTab(null);
      disarm();
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("pagehide", onPageHide);
      window.removeEventListener("pageshow", onPageShow);
      // Leaving the frame (another frame, or away from the editor): give the lock back now rather than
      // making the next person wait out the TTL.
      void transport.release(frameId, tab).catch(lockWillExpireByItself);
    };
  }, [frameId, transport]);

  const takeOver = useCallback(async () => {
    const state = await transport.takeOver(frameId, thisTab());
    dispatch({ t: "answer", frameId, state });
    return state;
  }, [frameId, transport]);

  return useMemo(() => ({
    status: model.status,
    others: model.others,
    lock: model.lock,
    degraded: model.status === "degraded",
    readOnly: isReadOnly(model),
    canTakeOver: model.canTakeOver,
    lostTo: model.lostTo,
    acquisitions: model.acquisitions,
    regainedFrom: model.regainedFrom,
    takeOver,
    model,
  }), [model, takeOver]);
}
