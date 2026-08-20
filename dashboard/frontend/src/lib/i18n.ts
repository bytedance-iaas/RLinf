/**
 * UI language state and message lookup.
 *
 * The chosen language lives on ``<html lang>`` for the same reason the theme
 * lives on ``<html data-theme>``: the value has to be right before React mounts.
 * It is not only a React concern -- `:lang()` selectors, the browser's font
 * fallback for CJK, and any assistive technology reading the page all take it
 * from that attribute, so keeping a second copy in a module variable would give
 * two answers to one question.
 *
 * There is one subscription, in `App`. Everything below it re-renders when the
 * root does, so views and helpers call the plain `t()` -- which reads the
 * attribute -- rather than each holding their own subscription. That also lets
 * non-component helpers (`format.ts`, `signals.ts`) translate without becoming
 * hooks.
 *
 * Only the *chrome* is translated here: labels, prose, empty states, the words
 * this bundle writes itself. Metric keys, run ids, paths, event payloads and the
 * server's health sentences are data, and are rendered exactly as received --
 * translating a TensorBoard tag would break the one thing it is for, which is
 * being grep-able against the training side.
 */

import { createElement, Fragment, useSyncExternalStore, type ReactNode } from "react";
import { en } from "../locales/en";
import { zh } from "../locales/zh";

export type Lang = "en" | "zh";

/** Every message key. `en` is the source of truth; `zh` is typed against it. */
export type MessageKey = keyof typeof en;

/** Values a message can interpolate. Numbers are formatted by the call site. */
export type Vars = Record<string, string | number>;

export const LANG_STORAGE_KEY = "rlinf-dashboard-lang";
const LANG_EVENT = "rlinf-dashboard-lang-change";

const CATALOGS: Record<Lang, Record<MessageKey, string>> = { en, zh };

/** The `lang` attribute value written for each language. */
const HTML_LANG: Record<Lang, string> = { en: "en", zh: "zh-CN" };

/**
 * A stored or advertised language tag as one of ours.
 *
 * Prefix matching rather than equality: a browser may advertise `zh-Hans-CN`,
 * `zh-TW` or `en-GB`, and all of them have an answer here.
 */
export function normalizeLang(value: string | null | undefined): Lang | null {
  if (!value) return null;
  const tag = value.toLowerCase();
  if (tag.startsWith("zh")) return "zh";
  if (tag.startsWith("en")) return "en";
  return null;
}

/**
 * The language to start in when nothing has been chosen yet.
 *
 * The browser's own preference order, first match wins, English otherwise. A
 * console opened from a Chinese-locale machine should not need a click before it
 * is readable, and one opened anywhere else must not turn Chinese for it.
 */
export function preferredLang(navigatorLanguages?: readonly string[]): Lang {
  const advertised =
    navigatorLanguages ??
    (typeof navigator === "undefined"
      ? []
      : (navigator.languages ?? [navigator.language]));
  for (const tag of advertised) {
    const lang = normalizeLang(tag);
    if (lang) return lang;
  }
  return "en";
}

function currentLang(): Lang {
  if (typeof document === "undefined") return "en";
  return normalizeLang(document.documentElement.lang) ?? "en";
}

/**
 * The active language, for the few places that need the language itself rather
 * than a message: casing rules and `Intl` locale tags are properties of the
 * language, not strings that can be translated.
 */
export function activeLang(): Lang {
  return currentLang();
}

/** The active language as a BCP 47 tag, for `toLocaleString` and friends. */
export function localeTag(): string {
  return HTML_LANG[currentLang()];
}

function applyLang(lang: Lang): void {
  document.documentElement.lang = HTML_LANG[lang];
}

function subscribe(listener: () => void): () => void {
  const onStorage = (event: StorageEvent) => {
    if (event.key !== LANG_STORAGE_KEY) return;
    applyLang(normalizeLang(event.newValue) ?? preferredLang());
    listener();
  };
  window.addEventListener(LANG_EVENT, listener);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(LANG_EVENT, listener);
    window.removeEventListener("storage", onStorage);
  };
}

/** Read the active language and re-render when it changes. */
export function useLang(): Lang {
  return useSyncExternalStore(subscribe, currentLang, () => "en");
}

/** Apply and remember a language selected by the operator. */
export function setLang(lang: Lang): void {
  applyLang(lang);
  try {
    window.localStorage.setItem(LANG_STORAGE_KEY, lang);
  } catch {
    // Storage can be unavailable in hardened/private browser contexts. The
    // current tab still switches correctly; only persistence is lost.
  }
  window.dispatchEvent(new Event(LANG_EVENT));
}

/**
 * Substitute `{name}` placeholders.
 *
 * An unknown placeholder is left standing rather than replaced with an empty
 * string: a visible `{count}` in the UI names the bug, while a silent gap reads
 * as a missing number the server failed to send.
 */
function interpolate(template: string, vars?: Vars): string {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (whole, name: string) =>
    name in vars ? String(vars[name]) : whole,
  );
}

function message(key: MessageKey): string {
  const lang = currentLang();
  // The English string is the fallback for a key a catalogue somehow lacks. The
  // type system makes that unreachable through normal edits; a stale bundle in a
  // long-lived tab is the case it covers, and English copy beats a raw key.
  return CATALOGS[lang][key] ?? en[key];
}

/** Translate a key into the active language. */
export function t(key: MessageKey, vars?: Vars): string {
  return interpolate(message(key), vars);
}

/**
 * Translate a key whose message embeds elements -- a `<Code>`, an `<em>`.
 *
 * The alternative is splitting the sentence into a prefix and a suffix key,
 * which fixes the element's position in the sentence to wherever English puts
 * it. Chinese does not put it there, so the whole sentence stays one message and
 * the elements are named placeholders inside it.
 */
export function tNode(key: MessageKey, vars: Record<string, ReactNode>): ReactNode {
  const template = message(key);
  const parts: ReactNode[] = [];
  const pattern = /\{(\w+)\}/g;
  let last = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(template)) !== null) {
    if (match.index > last) parts.push(template.slice(last, match.index));
    const name = match[1] as string;
    parts.push(name in vars ? vars[name] : match[0]);
    last = match.index + match[0].length;
  }
  if (last < template.length) parts.push(template.slice(last));

  return parts.map((part, index) => createElement(Fragment, { key: index }, part));
}

/**
 * A `Health` or `RunState` value as a word.
 *
 * Both vocabularies share one key space because they never collide on a value
 * except `unknown`, where they mean the same thing anyway. A value with no
 * message -- a state a newer runner invented -- is shown raw rather than hidden.
 */
export function statusLabel(value: string): string {
  const key = `status.${value}` as MessageKey;
  return key in en ? t(key) : value;
}

/**
 * Subscribe the caller to language changes.
 *
 * Used once, at the root. It returns `t` so a component that needs both the
 * subscription and the function does not have to import two things and
 * remember which one re-renders it.
 */
export function useT(): typeof t {
  useLang();
  return t;
}
