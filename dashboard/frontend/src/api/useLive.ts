/**
 * Live data hooks: one initial fetch, then SSE for updates.
 *
 * Reconnection is deliberately absent. The server's first frame is `retry: <ms>`
 * tied to its own push interval, and native `EventSource` honours that and
 * reconnects on its own. Hand-rolling a reconnect loop on top would double every
 * retry and fight the server's cadence, which is exactly what the `retry:` frame
 * exists to prevent.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamUrls } from "./client";
import type { RunStatus, RunSummary } from "./types";

/** Connection state of a live view, so the UI can say "stale" rather than lie. */
export type LiveState = "connecting" | "live" | "reconnecting" | "error";

interface Live<T> {
  data: T | null;
  liveState: LiveState;
  /** Last error text, kept alongside data so a hiccup does not blank the page. */
  error: string | null;
  /** Wall-clock of the last accepted payload; drives the "as of" readout. */
  updatedAt: number | null;
  refetch: () => void;
}

function useSse<T>(url: string, initial: () => Promise<T>, enabled = true): Live<T> {
  const [data, setData] = useState<T | null>(null);
  const [liveState, setLiveState] = useState<LiveState>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [nonce, setNonce] = useState(0);
  // The initial fetch is a fresh closure every render; a ref keeps the effect
  // from re-subscribing the stream every time the caller re-renders.
  const initialRef = useRef(initial);
  initialRef.current = initial;

  const refetch = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    // Fetch once up front rather than waiting a full push interval: the SSE loop
    // sleeps *after* its first payload, but a cold page should not be blank for
    // two seconds when the data is one GET away.
    initialRef.current()
      .then((value) => {
        if (cancelled) return;
        setData(value);
        setUpdatedAt(Date.now());
        setError(null);
      })
      .catch((exc: unknown) => {
        if (cancelled) return;
        setError(exc instanceof Error ? exc.message : String(exc));
        setLiveState("error");
      });

    const source = new EventSource(url);

    source.addEventListener("open", () => {
      if (!cancelled) setLiveState("live");
    });

    source.addEventListener("update", (event) => {
      if (cancelled) return;
      try {
        setData(JSON.parse((event as MessageEvent<string>).data) as T);
        setUpdatedAt(Date.now());
        setLiveState("live");
        setError(null);
      } catch (exc: unknown) {
        setError(exc instanceof Error ? exc.message : String(exc));
      }
    });

    // The server emits `error` events for a failed render but keeps the stream
    // open, because the files it reads are written by a live process and a
    // transient failure is expected. Surfacing it without dropping `data` is the
    // difference between "one read hiccuped" and "the run is gone".
    source.addEventListener("error", (event) => {
      if (cancelled) return;
      const message = (event as MessageEvent<string>).data;
      if (message) {
        try {
          setError((JSON.parse(message) as { detail?: string }).detail ?? message);
        } catch {
          setError(message);
        }
        return;
      }
      // No payload means the transport dropped. EventSource is already retrying.
      setLiveState(source.readyState === EventSource.CLOSED ? "error" : "reconnecting");
    });

    return () => {
      cancelled = true;
      source.close();
    };
  }, [url, enabled, nonce]);

  return { data, liveState, error, updatedAt, refetch };
}

/** All runs, for the run list. */
export function useRuns(): Live<RunSummary[]> {
  const initial = useCallback(() => api.listRuns(), []);
  return useSse<RunSummary[]>(streamUrls.allRuns(), initial);
}

/** One run's full status. `runId` may be null while routing settles. */
export function useRun(runId: string | null): Live<RunStatus> {
  const initial = useCallback(() => api.run(runId as string), [runId]);
  return useSse<RunStatus>(runId ? streamUrls.run(runId) : "", initial, runId !== null);
}

/**
 * One-shot fetch with the same shape as the live hooks.
 *
 * Templates, series, media and events are all either immutable for a run or
 * expensive enough that pushing them every two seconds would be worse than a
 * manual refresh. Refetch is explicit, and on the deep-dive view it is tied to
 * the run's step advancing rather than to a timer.
 */
export function useFetch<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
): { data: T | null; loading: boolean; error: string | null; reload: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    fetcherRef.current(controller.signal)
      .then((value) => {
        setData(value);
        setError(null);
      })
      .catch((exc: unknown) => {
        if (controller.signal.aborted) return;
        setError(exc instanceof Error ? exc.message : String(exc));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  return { data, loading, error, reload: useCallback(() => setNonce((n) => n + 1), []) };
}
