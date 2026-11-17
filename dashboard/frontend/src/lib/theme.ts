/** Theme state shared by the shell and canvas charts.
 *
 * The chosen value lives on ``<html>`` so CSS can switch before React mounts.
 * A tiny external store keeps every chart subscribed without threading a visual
 * preference through the run-data component tree.
 */

import { useSyncExternalStore } from "react";

export type Theme = "dark" | "light";

export const THEME_STORAGE_KEY = "rlinf-dashboard-theme";
const THEME_EVENT = "rlinf-dashboard-theme-change";

function currentTheme(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  document.documentElement.style.colorScheme = theme;
}

function subscribe(listener: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key !== THEME_STORAGE_KEY) return;
    applyTheme(event.newValue === "light" ? "light" : "dark");
    listener();
  };
  window.addEventListener(THEME_EVENT, listener);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(THEME_EVENT, listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** Read the active theme and re-render when it changes. */
export function useTheme(): Theme {
  return useSyncExternalStore(subscribe, currentTheme, () => "dark");
}

/** Apply and remember a theme selected by the operator. */
export function setTheme(theme: Theme): void {
  applyTheme(theme);
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    // Storage can be unavailable in hardened/private browser contexts. The
    // current tab still switches correctly; only persistence is lost.
  }
  window.dispatchEvent(new Event(THEME_EVENT));
}
