import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to the Python dashboard rather than relying on
// CORS. The server only enables CORS when `cors_origins` is configured, and a
// proxy keeps the browser's origin identical to production (where FastAPI serves
// the built bundle), so nothing behaves differently between the two.
// Declared locally rather than by adding `@types/node`. This is the only Node
// global the project touches, and a whole platform typings package pulled in for
// one property is weight in an install that has to work offline.
declare const process: { env: Record<string, string | undefined> };

// Must match `--port`'s default in rlinf_dashboard/__main__.py. A mismatch here
// is not a build error: the dev server starts, the page renders, and every
// request fails at runtime with a proxy error that looks like the backend is
// down.
const API_TARGET = process.env.RLINF_DASHBOARD_API ?? "http://127.0.0.1:8420";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5273,
    proxy: {
      "/api": {
        target: API_TARGET,
        changeOrigin: true,
        // SSE dies behind a proxy that buffers. Vite's proxy streams by default
        // as long as compression is off for the response, which it is for
        // text/event-stream, but the timeout has to be disabled or a quiet
        // stream is severed after the default idle window.
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
  build: {
    // Fingerprinted assets under a fixed prefix, so a FastAPI StaticFiles mount
    // can cache /assets aggressively and index.html not at all.
    assetsDir: "assets",
    sourcemap: true,
    chunkSizeWarningLimit: 900,
  },
});
