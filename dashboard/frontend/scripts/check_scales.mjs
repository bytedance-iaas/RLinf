// Assert that the x extent a chart pins can be walked by uPlot's axis-split loop.
//
// A one-step run can pin x to `min === max`; uPlot's zero-width rejection sits
// behind `dataLen > 1`, so a single point can reach `numAxisSplits` with an
// increment too small to advance in float64:
//
//     for (let val = min; val <= max; val = val + incr) splits.push(val)
//
// This numerical invariant is not visible to a compiler or bundle check.
//
// Run with node >= 22:  node --experimental-strip-types scripts/check_scales.mjs
// The import is a .ts file read directly, so this asserts against the shipped
// source rather than a copy of its logic.

import { xExtent, yGrowth } from "../src/lib/series.ts";

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

// Exercise zero, positive, negative, fractional, and large one-point domains.
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

// ---------------------------------------------------------------- y growth
//
// Same class of defect as the one above and equally invisible to the compiler:
// `setData(data, false)` never recomputes the y scale, so a value that arrives
// after the chart was built and exceeds its range is drawn outside the plot area.
// The curve then looks flat exactly when a loss exploded. What is asserted here
// is the property, not the pixels: any incoming value ends up inside the range.

console.log("yGrowth: a new extreme always ends up inside the range");

const contains = (range, value) => value >= range.min && value <= range.max;

for (const [current, spike] of [
  [{ min: 0, max: 1 }, 5000],       // loss explosion
  [{ min: 0, max: 1 }, -5000],      // reward collapse
  [{ min: -1, max: 1 }, 1.0001],    // barely over
  [{ min: 100, max: 200 }, 1e12],   // orders of magnitude out
  [{ min: 0, max: 1 }, 1e-9],       // inside; must not move
]) {
  const columns = [[0.5, 0.5, spike]];
  const grown = yGrowth(columns, current);
  const range = grown ?? current;
  check(
    `value ${spike} against [${current.min}, ${current.max}] -> [${range.min}, ${range.max}]`,
    contains(range, spike),
    "the value would be drawn outside the plot area and read as a flat curve",
  );
}

console.log("yGrowth: the axis only ever grows");
{
  const current = { min: -10, max: 10 };
  // Data well inside the current range must not pull the axis in: a range that
  // tracked the data downward too would rescale on nearly every push, which is
  // the walking-gridlines problem `resetScales: false` exists to prevent.
  check("data inside the range returns null", yGrowth([[0, 1, 2]], current) === null);
  const grown = yGrowth([[0, 1, 50]], current);
  check(
    "growing upward keeps the old lower bound",
    grown !== null && grown.min === current.min && grown.max > 50,
    JSON.stringify(grown),
  );
}

console.log("yGrowth: growing lands where a fresh render would");
{
  // Chart.tsx's initial y `range` callback, restated. If these two ever disagree,
  // reloading the page shifts every axis that had grown during the session, and
  // the reader cannot tell which of the two views was the honest one.
  const initialRange = (min, max) => {
    const pad = min === max ? (Math.abs(min) > 0 ? Math.abs(min) * 0.15 : 1) : (max - min) * 0.08;
    return { min: min - pad, max: max + pad };
  };
  for (const values of [
    [0.5, 0.5, 1e12],
    [-3, 7, 200],
    [0.001, 0.002, 0.5],
  ]) {
    const low = Math.min(...values);
    const high = Math.max(...values);
    // A current range strictly inside the data forces both bounds to move.
    const current = { min: low + (high - low) * 0.4, max: low + (high - low) * 0.6 };
    const grown = yGrowth([values], current);
    const fresh = initialRange(low, high);
    check(
      `[${values.join(", ")}] grows to the fresh bounds`,
      grown !== null && grown.min === fresh.min && grown.max === fresh.max,
      `grown=${JSON.stringify(grown)} fresh=${JSON.stringify(fresh)}`,
    );
  }
}

console.log("yGrowth: a stacked chart keeps zero on the axis");
{
  // Stacking asserts the bands sum to a total, and a baseline that is not zero
  // misstates every band's share of it. The columns arriving here are already
  // cumulative, so the maximum is the top of the stack.
  const grown = yGrowth([[3, 4, 900]], { min: 0, max: 10 }, { stacked: true });
  check(
    "growth admits the new top without lifting the baseline",
    grown !== null && grown.min === 0 && contains(grown, 900),
    JSON.stringify(grown),
  );
}

console.log("yGrowth: values with no place on an axis are ignored");
{
  const current = { min: 0, max: 1 };
  check("all-null column returns null", yGrowth([[null, null]], current) === null);
  check(
    "NaN and Infinity do not move the axis",
    yGrowth([[NaN, Infinity, -Infinity, 0.5]], current) === null,
  );
  // A log scale cannot render <= 0; admitting one would produce a range uPlot
  // refuses to draw, which is a blank panel rather than a clipped one.
  const logGrown = yGrowth([[-5, 0, 900]], current, { positiveOnly: true });
  check(
    "log scale ignores non-positive values but still admits the spike",
    logGrown !== null && logGrown.min === current.min && contains(logGrown, 900),
    JSON.stringify(logGrown),
  );
}

// Negative controls. Without these the checks above pass against helpers that
// cannot fail, and the guards are decoration.
console.log("negative control: the unfixed pin still hangs");
const unfixed = walk(1, 1);
check(
  "pinning [xs[0], xs[last]] on a one-step run at 1 does not advance",
  unfixed.advances === false,
  "the split walk terminated, so this check no longer proves anything",
);

console.log("negative control: leaving the scale alone still hides the spike");
check(
  "a spike of 5000 is outside the untouched range [0, 1]",
  contains({ min: 0, max: 1 }, 5000) === false,
  "`contains` accepts anything, so the yGrowth checks prove nothing",
);

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log("\nall scale checks passed");
