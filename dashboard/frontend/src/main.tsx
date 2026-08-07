/**
 * Entry point.
 *
 * Import order matters: the fonts and the generated tokens must land before the
 * application sheet, since `app.css` reads `--font-*` and both theme palettes from
 * `tokens.css`, and Vite preserves import order when it concatenates the bundle's
 * CSS. `tokens.css` is generated from DESIGN.md front matter and is never edited by
 * hand -- see `scripts/gen_tokens.py`.
 *
 * The fonts are self-hosted npm packages rather than a CDN link: this bundle ships
 * into air-gapped clusters, where a Google Fonts request is a two-second delay
 * followed by a fallback font.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "@fontsource-variable/jetbrains-mono";
import "./styles/tokens.css";
import "./styles/app.css";
import { App } from "./App";

const host = document.getElementById("root");
if (!host) {
  // A missing mount node means index.html and this file disagree, which is a build
  // error, not a runtime condition worth degrading gracefully for.
  throw new Error("#root is missing from index.html");
}

createRoot(host).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
