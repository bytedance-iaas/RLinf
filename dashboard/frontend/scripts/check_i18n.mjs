// Assert that the two message catalogues describe the same UI.
//
// `tsc` already rejects a key that exists in one catalogue and not the other --
// `zh` is typed as `Record<keyof typeof en, string>`. What it cannot see is
// everything that makes a *present* key wrong:
//
//   * an empty string, which renders as a missing label rather than an error;
//   * `{placeholders}` that drifted, so one language interpolates a number the
//     other silently drops -- the count disappears and the sentence still reads;
//   * a Chinese entry that is still the English text, i.e. a forgotten line;
//   * a key no view references any more, or a `t("...")` naming a key that does
//     not exist, which `t` resolves to English rather than to a crash.
//
// None of those fail a build or a test, and all of them are visible to whoever
// opens the page. Hence a check.
//
// Run with node >= 22:  node --experimental-strip-types scripts/check_i18n.mjs
// The catalogues are imported from source, so this asserts against what ships.

import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { en } from "../src/locales/en.ts";
import { zh } from "../src/locales/zh.ts";

const SRC = fileURLToPath(new URL("../src", import.meta.url));

/**
 * Keys whose two catalogues are allowed to be identical.
 *
 * A brand name is not translated. Everything else being identical means the
 * Chinese line was never written.
 */
const SAME_BY_DESIGN = new Set(["app.brandAlt"]);

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name}${detail ? ` -- ${detail}` : ""}`);
  }
}

/** Every `.ts` / `.tsx` file under `src/`, minus the catalogues themselves. */
function sources(dir = SRC) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      if (entry === "locales") continue;
      out.push(...sources(path));
    } else if (path.endsWith(".ts") || path.endsWith(".tsx")) {
      out.push(path);
    }
  }
  return out;
}

const files = sources();
const text = files.map((path) => readFileSync(path, "utf8")).join("\n");

const enKeys = Object.keys(en);
const zhKeys = Object.keys(zh);

console.log(`catalogues (${enKeys.length} keys, ${files.length} source files)`);

const missingInZh = enKeys.filter((key) => !(key in zh));
const extraInZh = zhKeys.filter((key) => !(key in en));
check("zh covers every en key", missingInZh.length === 0, missingInZh.join(", "));
check("zh invents no key", extraInZh.length === 0, extraInZh.join(", "));

console.log("every message says something");
const blank = [];
for (const [name, catalogue] of [
  ["en", en],
  ["zh", zh],
]) {
  for (const [key, value] of Object.entries(catalogue)) {
    if (typeof value !== "string" || value.trim() === "") blank.push(`${name}:${key}`);
  }
}
check("no empty or blank message", blank.length === 0, blank.join(", "));

console.log("placeholders agree across languages");
const placeholders = (value) =>
  [...String(value).matchAll(/\{(\w+)\}/g)].map((match) => match[1]).sort();
const drifted = enKeys.filter((key) => {
  const a = placeholders(en[key]).join(",");
  const b = placeholders(zh[key] ?? "").join(",");
  return a !== b;
});
check(
  "no key interpolates different names",
  drifted.length === 0,
  drifted.map((key) => `${key}: en{${placeholders(en[key])}} zh{${placeholders(zh[key])}}`).join("; "),
);

console.log("the Chinese catalogue is actually translated");
const untranslated = enKeys.filter(
  (key) => !SAME_BY_DESIGN.has(key) && zh[key] === en[key] && /[A-Za-z]{3}/.test(en[key]),
);
check("no zh entry is still the English string", untranslated.length === 0, untranslated.join(", "));

// Keys reached through a computed name: `t(`status.${value}`)` and friends. The
// prefix is what the source can be checked for; the suffix is a runtime value.
const dynamicPrefixes = [
  ...text.matchAll(/[`"']([\w.]+\.)\$\{/g),
  ...text.matchAll(/`([\w.]+\.)\$\{/g),
].map((match) => match[1]);

console.log(`keys reached dynamically: ${[...new Set(dynamicPrefixes)].join(", ") || "none"}`);

const referenced = (key) =>
  text.includes(`"${key}"`) ||
  text.includes(`'${key}'`) ||
  dynamicPrefixes.some((prefix) => key.startsWith(prefix));

console.log("every key is used, and every used key exists");
const unused = enKeys.filter((key) => !referenced(key));
check("no key is dead copy", unused.length === 0, unused.join(", "));

// `t("...")` / `tNode("...")` call sites, which must name a key that exists.
const called = [...text.matchAll(/\bt(?:Node)?\(\s*"([\w.]+)"/g)].map((match) => match[1]);
const unknown = [...new Set(called)].filter((key) => !(key in en));
check("no call site names a missing key", unknown.length === 0, unknown.join(", "));

// A dynamic prefix that matches nothing is a rename that left a `t()` behind.
console.log("every dynamic prefix resolves to real keys");
const emptyPrefixes = [...new Set(dynamicPrefixes)].filter(
  (prefix) => !enKeys.some((key) => key.startsWith(prefix)),
);
check("no dynamic prefix is empty", emptyPrefixes.length === 0, emptyPrefixes.join(", "));

if (failures > 0) {
  console.log(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log("\nall i18n checks passed");
