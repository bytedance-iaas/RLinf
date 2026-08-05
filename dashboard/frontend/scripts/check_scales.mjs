// Assert that the x extent a chart pins can be walked by uPlot's axis-split loop.
//
// This is the one frontend check that is not the compiler or the bundler, and it
// exists because both were green while the Metrics tab killed the browser. A
// one-step run pinned x to `min === max`; uPlot's own rejection of a zero-width
// range sits behind `dataLen > 1`, so a single point slipped past it, `findIncr`
// returned `1e-16`, and `numAxisSplits` ran
//
//     for (let val = min; val <= max; val = val + incr) splits.push(val)
//
// forever, because `1 + 1e-16` is `1` in float64. Chrome killed the renderer
// ("Error code: 5") and Safari stopped responding. A type error it is not, and no
// bundle check can see it -- so it is asserted here.
//
// Run with node >= 22:  node --experimental-strip-types scripts/check_scales.mjs
// The import is a .ts file read directly, so this asserts against the shipped
// source rather than a copy of its logic.

import { xExtent } from "../src/lib/series.ts";

// uPlot's x-axis default `space` over a plausible plot width (`xAxisOpts.space`
// is 50). `findIncr` picks the smallest tabulated increment with
// `dim * incr / delta >= minSpace`, so no increment it can choose is below
// `delta * minSpace / dim`. Using the ratio -- rather than uPlot's increment
// table -- keeps this from being a copy of internals that can drift: a narrower
// plot only makes the increment larger, which is the safe direction.
const MIN_INCR_RATIO = 50 / 800;

/** Walk uPlot's split loop and report whether it terminates. */
function walk(min, max, cap = 100_000) {
  const incr = ((max - min) * MIN_INCR_RATIO) || 0;
  if (incr === 0) return { advances: false, splits: 0 };
  let splits = 0;
  for (let val = min; val <= max; val = val + incr) {
    // The actual hang: the accumulator stops moving while still <= max.
    if (val + incr === val) return { advances: false, splits };
    if (++splits > cap) return { advances: false, splits, runaway: true };
  }
  return { advances: true, splits };
}

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name}${detail ? ` -- ${detail}` : ""}`);
  }
}

console.log("xExtent: a pinned extent is always walkable");

// Step 0 is the fixture tree's floor and step 1 is what a real runner logs first.
// Both are one-point runs; only the second one hung, because `0 + 1e-16` is not
// `0` while `1 + 1e-16` is. Testing only the fixture value is how this shipped.
for (const step of [0, 1, 2, 3, 7, 50, 12345, 2 ** 31, -1, -7, 0.5]) {
  const extent = xExtent([step]);
  const result = walk(extent.min, extent.max);
  check(
    `one-step run at ${step} -> [${extent.min}, ${extent.max}]`,
    result.advances && result.splits > 0,
    `splits=${result.splits} runaway=${result.runaway === true}`,
  );
}

console.log("xExtent: a growing run is still pinned exactly");

// The padding must not touch a real extent: the whole point of pinning x is that
// a live chart's right edge tracks the newest point, and padding a multi-step run
// would leave a visible gap that grows as the run does.
for (const xs of [
  [0, 1],
  [1, 2],
  [1, 2, 3, 4, 5],
  [0, 5, 10, 15],
]) {
  const extent = xExtent(xs);
  const first = xs[0];
  const last = xs[xs.length - 1];
  const result = walk(extent.min, extent.max);
  check(
    `[${xs.join(", ")}] -> [${extent.min}, ${extent.max}]`,
    extent.min === first && extent.max === last && result.advances,
    `expected [${first}, ${last}], advances=${result.advances}`,
  );
}

console.log("xExtent: nothing to pin");
check("empty data -> null", xExtent([]) === null);

// Negative control. Without this the checks above pass against a `walk` that
// cannot fail, and the guard is decoration: this asserts the unpadded extent --
// exactly what the code did before -- still does not terminate, so the loop being
// walked is the one that took the tab down.
console.log("negative control: the unfixed pin still hangs");
const unfixed = walk(1, 1);
check(
  "pinning [xs[0], xs[last]] on a one-step run at 1 does not advance",
  unfixed.advances === false,
  "the split walk terminated, so this check no longer proves anything",
);

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log("\nall scale checks passed");
