/**
 * HTTP client for the dashboard API.
 *
 * Deliberately thin: no query cache, no request dedup library, no generated
 * client. Every response here is either small (a status document) or fetched once
 * per view (a series batch), and the SSE stream already handles the one case that
 * would need invalidation.
 */

import type {
  CheckpointEntry,
  MediaEntry,
  RunEvent,
  RunStatus,
  RunSummary,
  RunTemplate,
  Series,
  ServerHealth,
} from "./types";

/** Same-origin by default; Vite's dev server proxies /api to the Python side. */
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(`HTTP ${status}: ${detail}`);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    // FastAPI puts the useful part in `detail`; falling back to the status text
    // keeps a proxy's HTML error page from being shown as a JSON parse failure.
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export const api = {
  serverHealth: (signal?: AbortSignal) => get<ServerHealth>("/api/health", signal),

  listRuns: (signal?: AbortSignal) => get<RunSummary[]>("/api/runs", signal),

  run: (runId: string, signal?: AbortSignal) =>
    get<RunStatus>(`/api/runs/${encodeURIComponent(runId)}`, signal),

  template: (runId: string, signal?: AbortSignal) =>
    get<RunTemplate>(`/api/runs/${encodeURIComponent(runId)}/template`, signal),

  keys: (runId: string, signal?: AbortSignal) =>
    get<{ keys: string[]; sources: string[] }>(
      `/api/runs/${encodeURIComponent(runId)}/keys`,
      signal,
    ),

  /**
   * Fetch several series in one request.
   *
   * `keys` is a repeated query parameter, not a comma-joined list -- a metric key
   * may legitimately contain a comma, and the server declares `list[str]`.
   */
  series: async (runId: string, keys: string[], signal?: AbortSignal) => {
    if (keys.length === 0) return {};
    const params = new URLSearchParams();
    for (const key of keys) params.append("keys", key);
    return get<Record<string, Series>>(
      `/api/runs/${encodeURIComponent(runId)}/series?${params}`,
      signal,
    );
  },

  events: (runId: string, limit = 200, signal?: AbortSignal) =>
    get<RunEvent[]>(`/api/runs/${encodeURIComponent(runId)}/events?limit=${limit}`, signal),

  checkpoints: (runId: string, signal?: AbortSignal) =>
    get<CheckpointEntry[]>(`/api/runs/${encodeURIComponent(runId)}/checkpoints`, signal),

  media: (
    runId: string,
    options: { split?: string; step?: number; limit?: number } = {},
    signal?: AbortSignal,
  ) => {
    const params = new URLSearchParams();
    if (options.split) params.set("split", options.split);
    if (options.step !== undefined) params.set("step", String(options.step));
    params.set("limit", String(options.limit ?? 500));
    return get<MediaEntry[]>(
      `/api/runs/${encodeURIComponent(runId)}/media?${params}`,
      signal,
    );
  },

  mediaSteps: (runId: string, signal?: AbortSignal) =>
    get<number[]>(`/api/runs/${encodeURIComponent(runId)}/media/steps`, signal),

  /** Absolute URL for a clip, for a `<video src>`. */
  mediaUrl: (entry: MediaEntry) => (entry.url ? `${BASE}${entry.url}` : null),
};

/** SSE endpoint URLs. Consumed by native `EventSource`, which owns reconnection. */
export const streamUrls = {
  allRuns: () => `${BASE}/api/stream/runs`,
  run: (runId: string) => `${BASE}/api/stream/runs/${encodeURIComponent(runId)}`,
};
