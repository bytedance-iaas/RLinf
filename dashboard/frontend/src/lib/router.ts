/**
 * Hash routing, hand-rolled in ~50 lines.
 *
 * A router library would be the largest dependency in this bundle for four routes.
 * Hash rather than history API because the production deployment serves this
 * bundle from a FastAPI mount whose only HTML route is `/`: a deep link under the
 * history API would need a catch-all rewrite on the server, and that server is
 * owned by another process. A hash URL is still copy-pasteable, which is the
 * requirement -- an operator shares a link to a specific run.
 */

import { useCallback, useEffect, useState } from "react";

export type Route =
  | { name: "runs" }
  | { name: "overview"; runId: string }
  | { name: "metrics"; runId: string }
  | { name: "media"; runId: string }
  | { name: "events"; runId: string }
  | { name: "compare"; runIds: string[]; key: string | null };

function parse(hash: string): Route {
  const raw = hash.replace(/^#\/?/, "");
  const [path, query] = raw.split("?");
  const parts = (path ?? "").split("/").filter(Boolean).map(decodeURIComponent);
  const params = new URLSearchParams(query ?? "");

  if (parts[0] === "compare") {
    return {
      name: "compare",
      runIds: params.getAll("run"),
      key: params.get("key"),
    };
  }
  if (parts[0] === "runs" && parts[1]) {
    const view = parts[2] ?? "overview";
    const runId = parts[1];
    if (view === "metrics") return { name: "metrics", runId };
    if (view === "media") return { name: "media", runId };
    if (view === "events") return { name: "events", runId };
    return { name: "overview", runId };
  }
  return { name: "runs" };
}

export function href(route: Route): string {
  switch (route.name) {
    case "runs":
      return "#/";
    case "overview":
      return `#/runs/${encodeURIComponent(route.runId)}`;
    case "metrics":
      return `#/runs/${encodeURIComponent(route.runId)}/metrics`;
    case "media":
      return `#/runs/${encodeURIComponent(route.runId)}/media`;
    case "events":
      return `#/runs/${encodeURIComponent(route.runId)}/events`;
    case "compare": {
      const params = new URLSearchParams();
      for (const runId of route.runIds) params.append("run", runId);
      if (route.key) params.set("key", route.key);
      return `#/compare?${params}`;
    }
  }
}

export function useRoute(): [Route, (route: Route) => void] {
  const [route, setRoute] = useState<Route>(() => parse(window.location.hash));

  useEffect(() => {
    const onChange = () => setRoute(parse(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const navigate = useCallback((next: Route) => {
    window.location.hash = href(next);
  }, []);

  return [route, navigate];
}

/** The run a route is about, or null for the list and compare views. */
export function routeRunId(route: Route): string | null {
  return "runId" in route ? route.runId : null;
}
