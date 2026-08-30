# RLinf Dashboard — frontend

The browser half of the dashboard: an Arco-inspired operator console with dark
and light themes that answers "should I be worried about this run" before it
answers anything else.

It talks to the Python service over HTTP only. There is no shared code,
no build-time coupling, and nothing here imports from the Python side — the
filesystem layout under `<log_path>/_rlinf/runs/<run_id>/` and the JSON API are the
entire contract. See [`../README.md`](../README.md) for why.

## Stack, and why it is this small

| Choice | Reason |
| --- | --- |
| Vite + TypeScript + React | The only build step is `tsc` and a bundle; no framework server. |
| [uPlot](https://github.com/leeoniya/uPlot) for charts | ~45 kB, canvas, and it can be handed a new data array without rescaling axes. A React charting library re-renders a DOM tree per point and shifts layout on every SSE push, which this app is not allowed to do. |
| Plain CSS over generated custom properties | No Tailwind, no CSS-in-JS, no component library. |
| Hash routing, hand-rolled (`src/lib/router.ts`) | ~50 lines against a router library that would be the biggest dependency in the bundle for four routes. Also survives being served from a FastAPI mount whose only HTML route is `/`. |

Five runtime dependencies total, two of which are fonts. This ships into air-gapped
clusters, so every dependency is weight in an install that has to work with no
network.

## Layout

```
DESIGN.md              normative design system; the source of every colour and dimension
src/styles/tokens.css  GENERATED from DESIGN.md front matter -- never hand-edit
src/styles/app.css     everything else, consuming only tokens
src/api/               types.ts (mirrors the server's Pydantic models), client.ts, useLive.ts (SSE)
src/lib/               router, i18n, number/date formatting, series alignment, metric-side signals
src/locales/           en.ts (source of truth) and zh.ts (typed against it)
src/components/        Chart.tsx (uPlot wrapper), primitives.tsx, SmoothingControl.tsx
src/views/             RunList, Overview, Metrics, Media, Events, Compare
scripts/gen_tokens.py       DESIGN.md -> tokens.css
scripts/lint_design_md.py   validate DESIGN.md against the design.md spec
scripts/check_i18n.mjs      catalogue parity, placeholders, dead and missing keys
scripts/make_demo_runs.py   write a fixture run tree with every state this UI must render
```

### `tokens.css` is generated

`DESIGN.md`'s YAML front matter is the single source of truth for colours, type
levels and spacing. `src/styles/tokens.css` is derived from it and committed.

```bash
python scripts/gen_tokens.py          # regenerate
python scripts/gen_tokens.py --check  # exit 1 if stale; this is what CI runs
python scripts/lint_design_md.py      # spec conformance: refs, section order, duplicates
```

To add a token: put it in the front matter, add a line of prose saying why, then
regenerate. Editing `tokens.css` directly makes `--check` fail, by design.

The lint also reports **orphaned tokens** — declared, but reachable from nothing:
no other token references them, the prose does not name them, and no file under
`src/` uses the custom property they generate. That third source is why the check
is worth reading: on prose alone it fired for eleven live tokens (the eight series
colours are consumed from `src/lib/series.ts`, three type levels only from
`app.css`) and one real orphan, and a check that is eleven-twelfths noise gets
ignored. It is a warning, not an error — but the intended response is to delete
the token or give it a role, not to leave it sitting there.

### Two languages, one catalogue

The UI reads in English or Simplified Chinese. The switch is the globe button in
the header, next to the theme toggle; both are stored the same way, for the same
reason.

```
src/locales/en.ts   every string the UI writes itself -- the source of truth
src/locales/zh.ts   Record<keyof typeof en, string>, so tsc rejects a missing key
src/lib/i18n.ts     the store, t(), tNode() for messages with an element inside
```

To add a string: put it in `en.ts`, translate it in `zh.ts`, call
`t("area.thing")`. Nothing else. There is one subscription — `App` — and
everything under it re-renders, so views and even non-component helpers
(`format.ts`, `signals.ts`) call the plain `t()` rather than a hook.

Three rules the catalogues follow, all enforced by `npm run check:i18n`:

- **Placeholders are named** (`{count}`, not `%s`) and must match across
  languages, because word order does not survive the trip.
- **A message with an element inside it stays one message.** `tNode("...", {code:
  <Code>runner.run_id</Code>})` substitutes the node where the sentence wants it;
  splitting into a prefix and a suffix key would pin the element to where English
  puts it.
- **Server-written text is never translated.** Metric keys, run ids, paths, event
  payloads, template chart titles and the health verdict's `reason` all render
  verbatim in both languages — a translated TensorBoard tag cannot be grepped
  against the training side, and a paraphrased verdict cannot be matched against
  the API response.

The language lives on `<html lang>`, resolved by the inline script in
`index.html` before React mounts, exactly like the theme: `lang` decides the
browser's CJK font fallback, so mounting in one language and correcting to the
other would reflow every line of text already painted. Neither shipped webfont
has CJK glyphs, so both stacks name the platform CJK families after them — see
DESIGN.md's Typography section.

Two languages means both catalogues ship in one bundle (about +37 kB raw, +11 kB
gzipped). A split chunk per language would save that on the first paint and cost
a network round trip on the switch, which is the wrong trade for a console whose
whole point is that it works in a cluster with no egress.

## Install and build

```bash
cd frontend
npm install
npm run typecheck     # tsc --noEmit
npm run build         # tsc --noEmit && vite build  -> dist/
npm run check:scales  # the one runtime assertion; see below
npm run check:identity # cross-run payload isolation during route changes
npm run check:i18n    # the two message catalogues describe the same UI
```

### Runtime scale checks

Type checking cannot prove that a numerical chart extent is safe. A one-point
run can otherwise pass a zero-width range to uPlot and make its float64
axis-split loop fail to advance. `scripts/check_scales.mjs` imports the shipped
`src/lib/series.ts` implementation and asserts that every `xExtent` is walkable,
that live y ranges grow to include new extremes, and that zoom-stable behavior
remains intact.

The Python server mounts `dist/` on its own origin when it exists (a missing `dist/`
is fine — the API is useful on its own). So the production check needs no dev server
and no proxy at all:

```bash
npm run build
python -m rlinf_dashboard /tmp/rlinf-demo --port 8871
# open http://127.0.0.1:8871/ -- the hash routes below all work unchanged
```

Worth doing at least once per change, because it is the only configuration that
exercises the real origin: same-origin `/api`, no Vite middleware, and hash routing
against a mount whose only HTML route is `/`.

## Development

The dev server proxies `/api` to the Python server rather than relying on CORS, so
the browser's origin is identical to production.

```bash
# terminal 1 -- the API, pointed at the scan root
python -m rlinf_dashboard /tmp/rlinf-demo

# terminal 2 -- Vite, proxying to it
npm run dev
# -> http://localhost:5273
```

Both defaults are `8420`, so neither command needs a port. On a non-default port,
set both: `python -m rlinf_dashboard <roots> --port 8861` and
`RLINF_DASHBOARD_API=http://127.0.0.1:8861 npm run dev`. They have to agree —
a mismatch is not a startup error, the page just renders with every request
failing, which reads as the backend being down.

One more note that costs time otherwise: open `localhost:5273`, not
`127.0.0.1:5273`. Vite 8 binds the hostname it prints, and on a machine with IPv6
the two are not interchangeable.

## Manual verification

The following fixture workflow covers the principal operator states and views.

### 0. Build the fixture tree

One live run cannot be in every state at once. `scripts/make_demo_runs.py` writes
a scan root covering all four health verdicts, `running`/`finished`/`failed`/
`pending`, a run with no `max_steps`, a run with a NaN in its loss, a run whose
step is a minibatch rather than an RL iteration, a GRPO arm with no critic keys,
sharded media indices with recorded, unrecorded and zero-success clips, one run
with per-worker logging on and one rank deliberately slow, and real TensorBoard
event files.

`libero_10_ppo_onestep` is a degenerate chart fixture: one logged point at step
**1**, which exercises float precision around zero-width extents.

```bash
cd frontend
# any venv with tensorboard + pyyaml; the dashboard's own venv works
python scripts/make_demo_runs.py --root /tmp/rlinf-demo --clean
```

Takes ~2 minutes (it writes real event files). Three options matter:

- `--clean` removes the root first. Without it a re-run leaves the previous tree's
  files in place, and a half-written tree makes the server 404 on `/keys`.
- `--sample-clip /path/to/any.mp4` copies a real mp4 into every clip slot. Without
  it the clips are undecodable stubs, which exercises the media view's decode-error
  path but not playback. Use it at least once to check playback, and once without
  to check the error path.
- `--ranks N` (default 4) sets how many ranks per group the per-worker run gets.
  It decides which branch of the rank drill-down is reachable — see step 5.

**The tree decays, and there is a one-second fix.** Health is a function of age;
the generated runs use a five-second heartbeat and therefore expire after the
configured missed-tick budget. Re-stamp instead of rebuilding:

```bash
python scripts/make_demo_runs.py --root /tmp/rlinf-demo --touch   # <1s
```

`--touch` shifts every timestamp by one delta — the interval since the tree was
generated — so the constructed verdicts survive: the wedged run stays 6000s stale,
the frozen-snapshot run keeps its fresh heartbeat *file* over a stale snapshot, and
the no-heartbeat run keeps having no heartbeat. Run it whenever the run list has
gone all-`unreachable`, then reload the page. No server restart needed; only
timestamps move, so the accumulator's cached series stay valid.

### 1. Start the server and the dev server

```bash
python -m rlinf_dashboard /tmp/rlinf-demo --port 8861
RLINF_DASHBOARD_API=http://127.0.0.1:8861 npm run dev
```

Sanity check before opening a browser — this should report a positive
`run_count` and `exists: true` for the root:

```bash
curl -s http://127.0.0.1:8861/api/health
```

Restart the server after regenerating the fixture. The TensorBoard accumulator
caches file offsets per process, so a long-lived server keeps serving the old
values for event files that were replaced underneath it.

### 1b. Both lists page at twenty

Point the server at a scan root with more than twenty runs (copying the fixture
tree beside itself is enough) and open the list.

- Twenty rows, `1–20 of 48` in the control bar, and the pager centred under the
  table. Twenty is `PAGE_SIZE` in `src/components/Pager.tsx` — one constant for
  every list, because a console where two lists disagree about how much a page
  holds makes the reader relearn the pager on each view.
- The last page carries the remainder (`41–48 of 48`, eight rows) with `next`
  and `last` disabled.
- Filtering or searching returns to page 1. Staying on page 3 of a filter that
  now has one page shows an empty table over a list that is not empty, which
  reads as "no runs match" — the one thing the page must not say wrongly.
- A filter that leaves one page removes the pager entirely.
- The page position is vertically centred with the arrow buttons. It sounds
  cosmetic; it was a real defect, because `.control-group` had no `align-items`
  and the default `stretch` left every bare span in a button row sitting at the
  top of its own box.

### 1c. The attention card stops at five

A scan root where thirty runs are unhealthy used to put a page of names above
the table. The card is a summary, so it names five — enough to see whether the
problem is one run or all of them — and offers the rest.

- Five rows, then `… and 31 more`. The five are the worst, since the list is
  ordered `unreachable`, `degraded`, `unknown`, and the card shares that order
  with the dialog rather than having its own.
- Clicking opens a modal with all of them, scrollable, same rows in the same
  order.
- It is a native `<dialog>` opened with `showModal()`, so the platform gives
  what a hand-rolled overlay usually misses. Check all four: **Escape** closes
  it, a click on the **backdrop** closes it, focus is **trapped** inside it
  while open, and on close focus **returns to the `… and N more` button**.
- Reopening after Escape must work. If it does not, the element closed itself
  while React still believed it was open — which is the whole reason `Dialog`
  listens for the native `close` event.
- The page behind is dimmed by `--color-scrim`, not by `--color-shadow-overlay`.
  The latter is the dialog's own shadow; reused as a backdrop it is 32% black
  over an already dark page, and the modal reads as a card someone forgot to
  close.

### 2. Overview — eight cards, legible in five seconds

Open `http://localhost:5273/#/runs/20260801-142200-libero_10_ppo_lr3e6`.

- Exactly **eight** cards: State, Components, Progress, Timing, Latest checkpoint,
  Health, the north-star metric, Anomalies.
- Progress reads `96 / 200` with `RL ITERATION` **beside** the bar. Step semantics
  are never implied.
- Open `.../20260802-090000-libero_10_ppo_openended` instead: it has no `max_steps`,
  so Progress reads `40 / ?` with `NO HORIZON` in place of a percentage and Timing
  shows `ETA —`. A run with no declared horizon has no percentage and no ETA, and
  inventing either would be presenting a guess as a measurement.
- Card 2 is a `COMPONENTS` strip with an `ASYNC` chip for this run (env, rollout
  and actor are all live for the whole loop, so a single scalar phase would be a
  semantic error). `.../libero_10_ppo_baseline` shows a scalar `PHASE` instead.
  Both shapes must render.
- Every live number is tabular: watch the `Timing` card tick and confirm no digit
  changes width.

### 3. Health bar — verbatim, with metric signals kept separate

The bar under the header and the Health card render the server's `health` and
`reason` **exactly as received**. Nothing in `src/` recomputes a verdict.

Open each of these and compare the reason text against
`curl -s http://127.0.0.1:8861/api/runs/<id> | python -m json.tool`:

| Run | Verdict | Reason begins |
| --- | --- | --- |
| `20260802-233000-libero_10_ppo_wedged` | `unreachable` | "No heartbeat for …s (over 5x the …s budget); the driver process is probably gone." |
| `20260802-201000-libero_10_ppo_frozen_snapshot` | `degraded` | "run.json is stale but the heartbeat file was touched …s ago; the process is alive and its snapshot writes are failing." |
| `20260803-041200-libero_10_ppo_hung` | `degraded` | heartbeat fresh, progress stale — a different remedy from the row above |
| `20260804-060000-libero_10_ppo_noheartbeat` | `unknown` | "Snapshot claims to be running but has no heartbeat." |

If every row reads `unreachable`, the tree has aged out — run `--touch` (see step 0)
and reload. That the two `degraded` runs have *different* reasons is the point of
having both: same badge, different thing to go do about it, and the frontend is not
allowed to paraphrase either one into a shared string.

`unknown` is a real verdict, not a loading state. Confirm it has its own colour and
no spinner, shimmer or pulse:

```bash
grep -n "@keyframes\|animation:\|transition" src/styles/app.css
```

Two hits, both on the media play badge's hover, both behind a
`prefers-reduced-motion` opt-out. Nothing that carries information ever moves:
no spinner, no shimmer, no pulse, and no entrance animation on the modal.

The **Anomalies** card holds the three signals the server cannot compute, because
`derive_health` is a pure function of a snapshot and a clock and these need the
series. They carry a `METRIC-SIDE` chip and live in `src/lib/signals.ts`, never
mixed into the health path:

- `.../libero_10_ppo_lr3e6` → "Step time degraded (warning) time/step is 2.6x its
  early baseline (250.9s now versus 96.1s over steps 2-19)." plus "No eval
  improvement in 5 rounds".
- `.../20260803-180000-libero_10_ppo_nan` → a red critical signal, "Non-finite
  metric value … train/actor/policy_loss first went NaN or Inf at step 31 (21 of 52
  points)." The server reports this run `healthy` / "Run is failed; silence is
  expected." — its liveness question has a correct answer for a terminated run —
  and that verdict must still render verbatim beside the red signal. The two are
  answering different questions, which is exactly why they are separate cards.
- `.../libero_10_ppo_baseline` → "none / No step-time regression, eval plateau or
  non-finite value in 12 watched series." A quiet card says what it looked at.

### 4. Metrics — one generic renderer, driven by the template

Open `http://localhost:5273/#/runs/20260801-101500-libero_10_ppo_baseline/metrics`.

Everything on this page comes from `/api/runs/{id}/template`. There is no
per-task-type code in `src/views/Metrics.tsx`; compare the page against:

```bash
curl -s http://127.0.0.1:8861/api/runs/20260801-101500-libero_10_ppo_baseline/template \
  | python -m json.tool | head -40
```

Expect 7 groups / 25 panels, and specifically:

- Header reads `TEMPLATE EMBODIED`, `RL ITERATION AXIS`, `56/56 keys`.
- Groups in template order, each with the template's own prose subtitle:
  Task performance, Policy optimization, Value function, Rollout, Throughput, then
  `▸ Evaluation` **folded** because the template says `collapsed: true`.
- `▸ Unmatched keys` — 6 keys the run logs that no chart claims, folded, captioned
  "Shown so a metric cannot silently disappear." A metric must never vanish
  because nobody wrote a chart for it.
- Per-panel flags derived from the template, not from the key name: exactly two `%`
  (the `format: percent` panels), three `LOG` (`scale: log`) and one `STACKED` (the
  phase-breakdown panel). Change `scale: log` to linear in the YAML and the flag
  moves with it.

Every demo run has history, so one case is not reachable here: a run with a single
recorded step. Point the server at a real one-iteration run tree instead and the
affected panels say `SINGLE POINT — PLOTTED AS A MARKER` (16 of 25, on the tree this
was checked against). A line through one point is invisible; a dot is not.

The claim being tested is that **adding a template is a YAML change**. To check it,
edit a title or add a group in `../rlinf_dashboard/templates/embodied.yaml`
and reload the page. The change appears with no frontend rebuild.

Then open `.../events`. Newest first — an operator arriving after an alert wants the
last thing that happened at the top:

- `32 of 32` events, and the `warn + error` filter narrows to just those. The
  control bar carries the filter and the range and nothing else: the pager lives
  under the rows, where a reader is when they want the next page, and reads
  `Page 1 of 3` between its arrows rather than a `Page` label attached to
  whichever field precedes it. It is absent, not disabled, on a log that fits on
  one page — which a 32-event log does, at twenty rows a page. Append a hundred
  lines to `events.jsonl` to see the pager here. On
  `.../libero_10_ppo_nan` the filter finds the "non-finite loss observed" warning and
  the page carries a red `Exit` note above the log, because for a failed run that is
  the single most useful thing on the page.
- Payloads render as compact `key=value` pairs, never interpreted. The runner is free
  to change the payload shape, and a wrong reading presented confidently is worse
  than a raw one.
- Below the log, the checkpoint table: step, saved-at, size, duration, path, with
  a `best` chip where the writer set one. No prose above it and no resume-command
  card — the `resume_dir` is still in `/api/runs/{id}/checkpoints` for anything
  that wants to build a command from it.

### 5. Expand to ranks — and the card that is holding the job up

Only `libero_10_ppo_lr3e6` has per-worker data; the other ten runs are how a run
without `runner.per_worker_log: true` must behave. Start on one of those:

```
http://localhost:5273/#/runs/20260801-101500-libero_10_ppo_baseline/metrics
```

There is **no** "Expand to ranks" control here at all. Not a disabled one — a
control that can never do anything asks about a feature instead of answering
something about this run. Confirm the server agrees the data is absent:

```bash
curl -s http://127.0.0.1:8861/api/runs/20260801-101500-libero_10_ppo_baseline/keys \
  | python -c 'import json,sys; print(json.load(sys.stdin)["workers"])'   # []
```

Now `.../20260801-142200-libero_10_ppo_lr3e6/metrics`. The toggle is present and
reads `12` — three worker groups × four ranks, and hovering the count lists them.
Tick it. What to look for, in the order it matters:

- **Iteration time** and the other driver-only charts do not change. `time/step` is
  logged by the driver, not by any rank, so there is nothing to expand. It stays a
  single line rather than growing four flat ones at zero.
- **Environment breakdown** (7 keys) draws its 28 rank lines faint and unlabelled,
  with a panel note saying so. Seven metrics × four ranks cannot each get a colour
  from an eight-slot ramp, and two same-coloured legend rows meaning different
  metrics is worse than no legend at all.
- Any single-key `env/*` chart — **Success rate**, **Return** — gives each rank its
  own colour and legend row, and the aggregate entry picks up a faint `mean`. That
  word is load-bearing: cross-rank aggregation is an arithmetic mean while the
  driver's own timers take a max, and the two conventions sit on one page.
- **Phase durations** keeps its aggregates and says `stacked — shown as the
  aggregate`. Stacking asserts the bands sum to something meaningful; stacking four
  ranks of each band would add the same seconds four times.

The point of the whole feature is the last one. `rank_3` of `EnvGroup` was written
4x slow, and the mean hides most of it:

```bash
curl -s "http://127.0.0.1:8861/api/runs/20260801-142200-libero_10_ppo_lr3e6/series?keys=time/env/interact&expand=ranks" \
  | python -c '
import json, sys
d = json.load(sys.stdin)
last = lambda s: [p["value"] for p in s["points"] if p["value"] is not None][-1]
agg = last(d["time/env/interact"])
per = {s["rank"]: last(s) for s in d.values() if s["rank"] is not None}
for r in sorted(per):
    print(f"  rank {r}: {per[r]:8.2f}")
print(f"aggregate {agg:.2f}; slowest is {max(per.values())/min(per.values()):.2f}x "
      f"the fastest rank but only {max(per.values())/agg:.2f}x the aggregate")'
```

Expect roughly `82 / 88 / 93 / 395`, aggregate `165`: **4.8x between cards, 2.4x
against the mean.** Unexpanded, that page says the job got slower and nothing else.

With four ranks every single-metric chart still fits the colour ramp. The other
branch — too many lines to name, so only the extremes and the median get labels —
needs a wider node:

```bash
python scripts/make_demo_runs.py --root /tmp/rlinf-demo --clean --ranks 8
# restart the server: replaced event files keep their old offsets in a live process
```

Now the same charts name three lines out of eight (`lowest`, `highest`, `median`)
and draw the rest faint, with `5 more ranks drawn unlabelled — named lines are the
extremes and the median` on the panel.
`--ranks 0` writes no per-worker tree at all, which is the shape of a run without
`per_worker_log` — including moving the aggregate event files back out of
`tensorboard/all/`.

### 6. Media — a count, never a boolean

Open `.../20260801-101500-libero_10_ppo_baseline/media`.

One mp4 tiles N environments, so an outcome is a **count**:

- Cards read `3/8 succeeded`, `2/8 succeeded`, `0/8 succeeded`.
- Clips with no recorded outcome read `outcome not recorded`. This is the important
  one: **unrecorded is not failure**, and a `null` must never render as a red
  failure marker.
- Header chips keep the two apart: `42/196 ENVS SUCCEEDED` and `4 NOT RECORDED`.
- Split (`all`/`train`/`eval`) and step filters narrow the grid.
- Generated with `--sample-clip`, a clip reaches `readyState 4` at `640x480`
  **once you press play on it**. Without `--sample-clip`, that card reads "This
  clip could not be decoded" — the error path, distinct from "not recorded".

**Nothing is fetched until it is looked at.** Cards mount a player only when they
scroll near the viewport, and mount it with `preload="none"`, so opening the tab
issues no video requests at all. Check that: on a twenty-clip run the DOM has
twenty `.media-card`s, about four `video` elements and the rest
`.media-frame-idle`, and the network panel shows no `/media/file` request until
you press play.

This is a correctness property: eagerly loading many clips can exhaust the
browser's per-origin connections and delay the control-plane API.

### 7. Compare — across runs that do not agree

Select two runs' checkboxes in the run list and press **Compare**, or go straight
to:

```
http://localhost:5273/#/compare?run=20260801-101500-libero_10_ppo_baseline&run=20260802-151500-libero_10_grpo
```

- The metric picker is split into `In all 2 runs` (52 keys) and `In some runs only`
  (4 keys — `train/critic/*`, which the GRPO arm has no value head to produce).
  Picking one of those draws a single line and the table shows `—` for the run that
  does not log it, rather than looking like a failed request.
- Per-run toggles switch a curve off (`data-off="true"`) while leaving its row in
  the table, so turning a line off does not change the page's geometry.
- The smoothing slider is off at 0. Move it to 6 and the panel flag becomes
  `SMOOTHED 49PT`; the raw curve stays visible underneath. Smoothed data is always
  labelled as smoothed.
- Now swap in the minibatch run:

```
http://localhost:5273/#/compare?run=20260801-101500-libero_10_ppo_baseline&run=20260803-063000-libero_10_ppo_minibatch
```

  A **Mixed step semantics on one axis** note must appear: "The selected runs do not
  agree on what a step is (1x RL iteration, 1x Minibatch). The axis is labelled RL
  iteration; runs using a different unit are drawn dashed." Two runs whose steps mean
  different things are never silently put on one axis — the overlay would look
  meaningful and not be.
- The table carries a **Step semantics** column for the same reason.

### 7b. Zoom, and the axis that grows

On any metrics page, drag horizontally across one chart. Every chart in the run
zooms together — they share one x axis, which is the point of the cursor group —
and each grows a **`ZOOMED · RESET`** badge. Clicking it on any one of them
returns all of them.

Zoom is recorded in uPlot's `setSelect` hook and remains pinned across SSE
updates. The badge matters because a chart that stopped following live data can
otherwise look like a run that stopped producing data.

The y axis grows to admit a value that exceeds it and never shrinks. Live pushes
use `setData(data, false)`, which does not recompute scales. `npm run
check:scales` asserts that new extremes stay visible and live growth matches a
freshly rendered chart.

### 8. Geometry must not shift when an SSE update lands

This is the one property that cannot be checked by looking, because the page is
correct exactly when nothing happens. Leave a run overview open for a minute with
the browser's element inspector on a card: values change, positions do not.

The reason it holds: `Chart.tsx` calls uPlot's `setData(data, false)`, so a push
never rescales an axis; charts have fixed heights from tokens; and no view swaps an
element for a placeholder on update.

Reconnection is native `EventSource` only. The server emits `retry:`, so there is
no hand-rolled backoff to test — kill the server, watch the header go stale, start
it again, and the stream comes back on its own.

### 9. Degenerate inputs

- Point the server at a path that does not exist
  (`python -m rlinf_dashboard /tmp/nope --port 8870`). The list names the root and
  says it does not exist, the footer marks it `MISSING`, and the health bar reads
  `unknown` / "No runs discovered yet — nothing to report on." An empty dashboard
  that looks like a working dashboard is the failure mode here.
- Point it at a directory that exists and holds no run
  (`mkdir -p /tmp/rlinf-empty && python -m rlinf_dashboard /tmp/rlinf-empty --port 8872`).
  This must read differently from the case above: the root is fine, it is the
  level that is wrong, and "missing" would send the reader to fix the wrong thing.
  The footer shows `no runs found` beside the path. A root that does hold runs
  shows the path alone: the count is already the `runs` row directly above it,
  and that row follows the live list while this one froze at page load.
- Passing two paths must fail with a sentence, not scan one of them:
  `python -m rlinf_dashboard /tmp/a /tmp/b` exits 2 saying to use the common
  ancestor.
- One root holding two copies of the same tree — the ordinary case of a run copied
  off a cluster beside the original:

  ```bash
  mkdir -p /tmp/rlinf-both/original /tmp/rlinf-both/copy
  cp -R /tmp/rlinf-demo/libero_10_ppo_baseline /tmp/rlinf-both/original/
  cp -R /tmp/rlinf-demo/libero_10_ppo_baseline /tmp/rlinf-both/copy/
  python -m rlinf_dashboard /tmp/rlinf-both --port 8871
  ```

  Both rows are listed — the server does not deduplicate, because that
  would hide the fact that there are two — and each carries its own `run_root` in the
  sub-label, since without it the list reads as a duplicated row and looks like a bug
  in the dashboard. **The browser console must stay empty.** Rows are keyed by
  `run_id` + `run_root`; on a key collision React silently drops or reuses a row, and
  on a click target that is a wrong-run-opened bug, not a cosmetic one.
- `.../20260804-070000-libero_10_ppo_queued` is a short `pending` run whose ETA
  is extrapolated from little data. The Timing card must expose low confidence so
  the estimate is not mistaken for a measurement.

### 10. The language switch

The globe button in the header, next to the theme toggle. What to check, in the
order it goes wrong:

```
http://localhost:5273/#/runs
```

- Clicking it turns the **whole shell** over at once — breadcrumb, tab strip,
  table headers, state and health words, the `2 分钟前` ages, the run-needs-
  attention card, and the browser tab's own title. Nothing is left in the other
  language and nothing waits for the next SSE push.
- The button's label is the language it switches **to**, in that language
  (`中文` / `EN`). Someone who cannot read the current UI still has to be able to
  find the way out of it. That is also why its label survives the narrow-width
  media query that hides the theme toggle's: a sun and a moon explain themselves,
  a globe does not.
- Open a run and confirm what does **not** change: the health bar's reason
  sentence, the template's chart titles and axis label, metric keys, event kinds
  and payloads, paths. Compare against the API to be sure they are identical:

  ```bash
  curl -s http://127.0.0.1:8861/api/runs/<id> | python -c 'import json,sys; print(json.load(sys.stdin)["health"]["reason"])'
  ```

- The Chinese overview must read `每 RL 迭代` and `位于第 95 RL 迭代`, not
  `每rl 迭代`. The English UI lowercases the step-semantics label mid-sentence;
  Chinese must not, or the acronym becomes a typo. `semanticsInline()` in
  `src/lib/format.ts` owns that rule.
- Timestamps follow the UI language, not the browser: `08/19/2026, 15:10:48` in
  English, `2026/08/19 15:10:48` in Chinese. Durations, byte counts and step
  counts do not — those are numbers, and their column alignment is the point.
- Reload. The choice survives, and it survives as `<html lang>` **before** React
  mounts — check the DOM in the elements panel, or:

  ```js
  localStorage.setItem("rlinf-dashboard-lang", "zh-Hans-CN"); location.reload();
  // -> document.documentElement.lang === "zh-CN"; a regional tag still resolves
  localStorage.removeItem("rlinf-dashboard-lang"); location.reload();
  // -> back to whatever navigator.languages asks for
  ```

- Two tabs stay in step: switching in one flips the other within a paint, over the
  same `storage` event the theme uses.

Then run `npm run check:i18n`, which is what actually keeps this from rotting:
a key with no call site, a call site with no key, a placeholder that exists in one
language only, or a Chinese line that is still English all fail it.

### 11. The scan root, changed from the page

The `scan root` row in the Server card carries the control. Point the server at
the fixture tree, then:

- The row reads the path with a `change` button beside it, and no count — the
  count is the `runs` row above.
- `change` swaps the path for an input, prefilled and focused. **Escape leaves
  it**, and so does `cancel`; a field you can only leave by submitting is how a
  wrong path gets submitted.
- Type a path that does not exist and save. The request is refused, the server's
  own sentence appears under the field (`No such directory on the server: …`,
  not an `HTTP 400:` prefix), **the run table keeps its rows**, and the form
  stays open with what you typed. A typo must not cost the root that was
  working:

  ```bash
  curl -s -X PUT -H 'Content-Type: application/json' \
    -d '{"path":"/tmp/nope"}' http://127.0.0.1:8861/api/scan-root
  # {"detail":"No such directory on the server: /tmp/nope"}
  ```

- Type a real directory holding a different run tree and save. The row shows the
  new path, a `reset` appears beside `change`, and within one SSE push the table
  and the `runs` count are the other root's. `reset` returns to the configured
  default and the `reset` button disappears with it.
- The change is **global and unpersisted**: a second browser sees it on its next
  push, and restarting the server returns to `RLINF_DASHBOARD_SCAN_ROOT`.
- With `RLINF_DASHBOARD_SCAN_ROOT_EDITABLE=false` the row renders the path and
  **no buttons at all** — absent, not disabled, for the same reason the metrics
  view hides the rank drill-down on a run that has no per-worker data.

### 12. A run opened before its first step

The layout is bound server-side against the keys the run has logged, so a run
opened during startup binds against nothing. That state is correct and says so;
what matters is that the page leaves it on its own.

```bash
# a run with a manifest, a snapshot at step 0, and an empty tensorboard/
python -m rlinf_dashboard /tmp/rlinf-lazy --port 8861
```

- The metrics tab reads `0/0 KEYS`, no panels, and **North-star metric not
  logged**. All three are the truth about a run that has logged nothing yet.
- Now let the first step land: write an event file into `tensorboard/` and
  advance `progress.step` in `run.json`.
- Within one SSE push the tab must fill in by itself — key count, groups and
  charts — with **no page reload**. Confirm it really was not reloaded:
  `performance.getEntriesByType("navigation")[0].type` is still `navigate`.
- Then watch the cost. Advance several more steps without adding keys and count
  the requests:

  ```js
  performance.getEntriesByType("resource").filter(e => e.name.includes("/template")).length
  ```

  It must stay put. The layout is fetched per `(run, key set)`, so steps that add
  no key add no request — one on open, one when the keys first appear, none
  after. A count that climbs with the step count means the fetch has been keyed
  on `dataVersion` instead, which refetches the layout for every run forever.

## Conventions worth knowing before editing

- **The server's verdict is never recomputed.** `health` and `reason` are rendered
  verbatim. Client-side observations live in `src/lib/signals.ts` and are labelled
  `METRIC-SIDE` in the UI.
- **`null` in a series means non-finite, not missing.** Pydantic serialises
  `float('nan')` and `float('inf')` to JSON `null`, so a `null` value at a real
  step is a NaN. `src/lib/series.ts` keeps that distinct from a gap.
- **The step axis is always labelled with the run's own `step_semantics`.** Two
  runs with different semantics never share an axis without a warning.
- **No animated numbers, no shadows, no reordering lists.** All three are hard
  constraints in DESIGN.md's "Do's and Don'ts", and the last one matters most here:
  the run list's rows are click targets, so it sorts by start time — an immutable
  property — and expresses "needs attention" with a marker rather than by moving
  the row under the pointer.
- **No user-visible string is written in a component.** It goes in
  `src/locales/en.ts` with a Chinese line beside it in `zh.ts`, and the component
  calls `t()`. `npm run check:i18n` fails on a key that nothing renders and on a
  `t()` naming a key that does not exist.
- Comments explain **why**, in English, Google style. The what is in the code.
