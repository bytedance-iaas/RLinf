/**
 * Live data hooks: one initial fetch, then SSE for updates.
 *
 * Reconnection is deliberately absent. The server's first frame is `retry: <ms>`
 * tied to its own push interval, and native `EventSource` honours that and
 * reconnects on its own. Hand-rolling a reconnect loop on top would double every
 * retry and fight the server's cadence, which is exactly what the `retry:` frame
 * exists to prevent.
 *
 * Every payload is stored together with the identity it was fetched for, and
 * read back only when that identity still matches. Holding the two in one piece
 * of state is what makes "the previous run's numbers under the new run's URL"
 * unrepresentable rather than merely unlikely: there is no ordering of fetches,
 * stream events or renders that can pair them up. A hook that cleared stale data
 * in an effect would still render once with the mismatch before the effect ran.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { commit as commitKeyed, read as readKeyed, type Keyed } from "../lib/identity";
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

/** The part of a live view that belongs to one identity. */
interface Snapshot<T> {
  data: T | null;
  liveState: LiveState;
  error: string | null;
  updatedAt: number | null;
}

function blank<T>(): Snapshot<T> {
  return { data: null, liveState: "connecting", error: null, updatedAt: null };
}

function useSse<T>(url: string, initial: () => Promise<T>, enabled = true): Live<T> {
  const [snapshot, setSnapshot] = useState<Keyed<Snapshot<T>> | null>(null);
  const [nonce, setNonce] = useState(0);
  // The initial fetch is a fresh closure every render; a ref keeps the effect
  // from re-subscribing the stream every time the caller re-renders.
  const initialRef = useRef(initial);
  initialRef.current = initial;

  const refetch = useCallback(() => setNonce((value) => value + 1), []);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;

    // Writes are stamped with the URL this effect run owns. If what is stored
    // belongs to an earlier URL it is discarded rather than patched, so a field
    // from the previous identity cannot survive into the new one.
    const commit = (patch: Partial<Snapshot<T>>) =>
      setSnapshot((prev) => commitKeyed(prev, url, patch, blank<T>));

    // Fetch once up front rather than waiting a full push interval: the SSE loop
    // sleeps *after* its first payload, but a cold page should not be blank for
    // two seconds when the data is one GET away.
    initialRef.current()
      .then((value) => {
        if (cancelled) return;
        commit({ data: value, updatedAt: Date.now(), error: null });
      })
      .catch((exc: unknown) => {
        if (cancelled) return;
        commit({
          error: exc instanceof Error ? exc.message : String(exc),
          liveState: "error",
        });
      });

    const source = new EventSource(url);

    source.addEventListener("open", () => {
      if (!cancelled) commit({ liveState: "live" });
    });

    source.addEventListener("update", (event) => {
      if (cancelled) return;
      try {
        commit({
          data: JSON.parse((event as MessageEvent<string>).data) as T,
          updatedAt: Date.now(),
          liveState: "live",
          error: null,
        });
      } catch (exc: unknown) {
        commit({ error: exc instanceof Error ? exc.message : String(exc) });
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
          commit({
            error: (JSON.parse(message) as { detail?: string }).detail ?? message,
          });
        } catch {
          commit({ error: message });
        }
        return;
      }
      // No payload means the transport dropped. EventSource is already retrying.
      commit({
        liveState: source.readyState === EventSource.CLOSED ? "error" : "reconnecting",
      });
    });

    return () => {
      cancelled = true;
      source.close();
    };
  }, [url, enabled, nonce]);

  // The read side is the other half of the guarantee. A snapshot left over from
  // a previous URL is never unwrapped, so the very first render after a route
  // change already shows an empty view rather than the old run.
  const view = readKeyed(snapshot, url) ?? blank<T>();
  return useMemo(
    () => ({
      data: view.data,
      liveState: view.liveState,
      error: view.error,
      updatedAt: view.updatedAt,
      refetch,
    }),
    [view.data, view.liveState, view.error, view.updatedAt, refetch],
  );
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
  // The deps are the identity of the request, exactly as for the SSE hook: a
  // series fetched for `env/episode_len` must not be readable once the caller
  // has moved to `env/return`, or the chart renders the old numbers under the
  // new label for as long as the new request takes.
  const key = JSON.stringify(deps);
  const [entry, setEntry] = useState<Keyed<{ data: T | null; error: string | null }> | null>(
    null,
  );
  const [loadingKey, setLoadingKey] = useState<string | null>(key);
  const [nonce, setNonce] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    const controller = new AbortController();
    setLoadingKey(key);
    fetcherRef.current(controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return;
        setEntry({ key, value: { data: value, error: null } });
      })
      .catch((exc: unknown) => {
        if (controller.signal.aborted) return;
        setEntry({
          key,
          value: { data: null, error: exc instanceof Error ? exc.message : String(exc) },
        });
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingKey(null);
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, nonce]);

  const fresh = readKeyed(entry, key);
  return {
    data: fresh?.data ?? null,
    // A request for the current identity is outstanding, or the identity just
    // changed and the effect has not run yet. Both are "not loaded", and the
    // caller must not be told otherwise while it is holding a stale payload.
    loading: loadingKey === key || fresh === null,
    error: fresh?.error ?? null,
    reload: useCallback(() => setNonce((n) => n + 1), []),
  };
}
