/**
 * Pairing a payload with the identity it was fetched for.
 *
 * Every live view in this app answers a question about one thing -- this run,
 * this metric -- and the answer arrives later than the question changes. Storing
 * the two separately is what let a run's numbers be rendered under another run's
 * URL for as long as the new request took: the route had already advanced, the
 * payload had not, and nothing in between knew they disagreed.
 *
 * These two functions are the whole rule. `commit` stamps every write with the
 * identity that produced it and refuses to merge into a snapshot from a
 * different one; `read` unwraps a snapshot only for the identity that is current.
 * Together they make a mismatched pair unrepresentable rather than unlikely --
 * there is no interleaving of responses, stream events or renders that produces
 * one, so no caller has to remember to check.
 *
 * They are pure and exported so `scripts/check-identity.mjs` can hold them to
 * that claim, including against the behaviour they replaced.
 */

export interface Keyed<T> {
  /** The identity this payload answers for: a stream URL, or serialised deps. */
  key: string;
  value: T;
}

/**
 * Apply a patch on behalf of `key`.
 *
 * A snapshot belonging to another identity is discarded, not merged: carrying
 * even one field across -- a timestamp, an error, a connection state -- would
 * describe the new identity with the old one's facts.
 */
export function commit<T extends object>(
  previous: Keyed<T> | null,
  key: string,
  patch: Partial<T>,
  empty: () => T,
): Keyed<T> {
  const base = previous && previous.key === key ? previous.value : empty();
  return { key, value: { ...base, ...patch } };
}

/** Unwrap a snapshot only for the identity that is current; otherwise nothing. */
export function read<T>(snapshot: Keyed<T> | null, key: string): T | null {
  return snapshot && snapshot.key === key ? snapshot.value : null;
}
