// Assert that a payload can never be read under an identity it does not belong to.
//
// Every live view answers a question about one thing -- this run, this metric --
// and the answer arrives later than the question changes. The failure this
// guards against is not a crash: the page renders normally, with the previous
// run's numbers under the new run's URL, for as long as the new request takes.
// Nothing in the type system or the bundle notices, and a reviewer only catches
// it by watching the screen during the two seconds it lasts.
//
// Run with node >= 22:  node --experimental-strip-types scripts/check_identity.mjs
// The import is a .ts file read directly, so this asserts against the shipped
// source rather than a copy of its logic.

import { commit, read } from "../src/lib/identity.ts";

const blank = () => ({ data: null, error: null, liveState: "connecting" });

let failures = 0;
function check(name, condition, detail = "") {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name}${detail ? ` -- ${detail}` : ""}`);
  }
}

const RUN_A = "/api/stream/runs/libero-baseline";
const RUN_B = "/api/stream/runs/maniskill-baseline";

console.log("a payload is readable only under its own identity");
let snap = commit(null, RUN_A, { data: { step: 41 } }, blank);
check("run A's payload reads back for run A", read(snap, RUN_A)?.data?.step === 41);
check(
  "the same payload is nothing for run B",
  read(snap, RUN_B) === null,
  "run A's data is readable under run B's route -- the P0 this file exists for",
);

console.log("the route changing empties the view before anything is fetched");
// This is the exact sequence a reviewer saw: run A loaded, the user clicks run
// B, and B's first response has not arrived. Every field must already be gone,
// not just `data`: a timestamp or error carried across describes run B with run
// A's facts.
const viewDuringSwitch = read(snap, RUN_B);
check("no data survives the switch", viewDuringSwitch === null);

console.log("a late response for the old route cannot land on the new one");
// A slow request for A resolves after the user has moved to B. `commit` stamps
// it with A, so B's view is unaffected and the stale answer is unreadable.
snap = commit(snap, RUN_B, { data: { step: 7 } }, blank);
const lateForA = commit(snap, RUN_A, { data: { step: 999 } }, blank);
check(
  "B's view does not show A's late payload",
  read(lateForA, RUN_B) === null,
  "a response for the previous route was merged into the current one",
);

console.log("fields never mix across identities");
snap = commit(null, RUN_A, { data: { step: 41 }, error: "one read hiccuped" }, blank);
const afterSwitch = commit(snap, RUN_B, { data: { step: 7 } }, blank);
check(
  "run A's error does not appear on run B",
  read(afterSwitch, RUN_B)?.error === null,
  "a field from the previous identity survived into the new one",
);

console.log("patches within one identity still merge");
// The guarantee must not be bought by throwing away real updates: a stream event
// that carries only `data` has to leave the connection state alone.
let live = commit(null, RUN_A, { data: { step: 1 }, liveState: "live" }, blank);
live = commit(live, RUN_A, { data: { step: 2 } }, blank);
check("a later patch keeps earlier fields", read(live, RUN_A)?.liveState === "live");
check("a later patch applies its own fields", read(live, RUN_A)?.data?.step === 2);

// Negative controls. Without these the checks above pass against helpers that
// cannot fail, and the guards are decoration.
console.log("negative control: the unfixed read still shows the wrong run");
// What the hooks did before: one `data` state, no identity beside it, cleared
// only by an effect that has not run yet on the render in question.
const unkeyedRead = (entry) => entry?.value ?? null;
const stale = commit(null, RUN_A, { data: { step: 41 } }, blank);
check(
  "ignoring the key surfaces run A's payload while on run B",
  unkeyedRead(stale)?.data?.step === 41,
  "the negative control no longer reproduces the bug, so `read` proves nothing",
);

console.log("negative control: the unfixed merge still carries fields across");
const unkeyedCommit = (previous, key, patch) => ({
  key,
  value: { ...(previous?.value ?? blank()), ...patch },
});
const carried = unkeyedCommit(stale, RUN_B, { data: { step: 7 } });
check(
  "merging without an identity check keeps A's fields on B",
  unkeyedCommit(
    unkeyedCommit(null, RUN_A, { data: { step: 41 }, error: "one read hiccuped" }),
    RUN_B,
    { data: { step: 7 } },
  ).value.error === "one read hiccuped",
  "the negative control no longer reproduces the bug, so `commit` proves nothing",
);
check("and the negative control does produce a value", carried.value.data.step === 7);

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log("\nall identity checks passed");
