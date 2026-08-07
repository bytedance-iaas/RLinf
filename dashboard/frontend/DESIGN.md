---
version: alpha
name: RLinf Control Plane
description: >
  Arco-inspired operator console for RLinf training runs, with first-class dark
  and light themes. Built for the question "is my job OK?" answered in under
  five seconds, on a screen that may be left open for days.
colors:
  primary: "#3C7EFF"
  primary-solid: "#165DFF"
  primary-solid-hover: "#0E42D2"
  secondary: "#8D4EDA"
  tertiary: "#6AA1FF"

  bg: "#17171A"
  surface: "#232324"
  surface-raised: "#2A2A2B"
  surface-sunken: "#1D1D1F"
  border: "#333335"
  border-strong: "#737379"

  text: "#F2F3F5"
  text-muted: "#B3B6BD"
  text-faint: "#86909C"

  healthy: "#27C346"
  healthy-text: "#65D77E"
  degraded: "#FF9626"
  degraded-text: "#FFB65C"
  unreachable: "#F76965"
  unreachable-text: "#FF9692"
  unknown: "#86909C"
  unknown-text: "#C9CDD4"

  shadow-overlay: "rgba(0, 0, 0, 0.32)"

  series-1: "#3C7EFF"
  series-2: "#8D4EDA"
  series-3: "#27C346"
  series-4: "#FF9626"
  series-5: "#F76965"
  series-6: "#33D1C9"
  series-7: "#FADC19"
  series-8: "#F54BAE"

themes:
  light:
    colors:
      primary: "#165DFF"
      primary-solid: "#165DFF"
      primary-solid-hover: "#0E42D2"
      secondary: "#722ED1"
      tertiary: "#165DFF"

      bg: "#F7F8FA"
      surface: "#FFFFFF"
      surface-raised: "#F2F3F5"
      surface-sunken: "#F7F8FA"
      border: "#E5E6EB"
      border-strong: "#86909C"

      text: "#1D2129"
      text-muted: "#4E5969"
      text-faint: "#5F6B7A"

      healthy: "#00B42A"
      healthy-text: "#007A1B"
      degraded: "#FF7D00"
      degraded-text: "#B54700"
      unreachable: "#F53F3F"
      unreachable-text: "#B71F2A"
      unknown: "#86909C"
      unknown-text: "#4E5969"

      shadow-overlay: "rgba(0, 0, 0, 0.12)"

      series-1: "#165DFF"
      series-2: "#722ED1"
      series-3: "#008A1E"
      series-4: "#B54700"
      series-5: "#B71F2A"
      series-6: "#078A85"
      series-7: "#946200"
      series-8: "#A61E9C"

typography:
  display:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: 650
    lineHeight: 1.15
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: 600
    lineHeight: 1.35
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.5
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.45
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: 550
    lineHeight: 1.3
    letterSpacing: 0.04em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: 550
    lineHeight: 1.25
    letterSpacing: 0.06em
  numeric-lg:
    fontFamily: "JetBrains Mono"
    fontSize: 28px
    fontWeight: 500
    lineHeight: 1.1
    fontFeature: "'tnum' 1, 'zero' 1"
  numeric-md:
    fontFamily: "JetBrains Mono"
    fontSize: 14px
    fontWeight: 450
    lineHeight: 1.4
    fontFeature: "'tnum' 1, 'zero' 1"
  numeric-sm:
    fontFamily: "JetBrains Mono"
    fontSize: 12px
    fontWeight: 450
    lineHeight: 1.35
    fontFeature: "'tnum' 1, 'zero' 1"

rounded:
  none: 0px
  sm: 2px
  md: 4px
  lg: 8px
  full: 999px

spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 24px
  2xl: 32px
  3xl: 48px
  card-gap: 12px
  overview-columns: 4
  chart-height: 180px
  chart-height-tall: 260px
  chart-height-spark: 44px

components:
  card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.text}"
    rounded: "{rounded.lg}"
    padding: 16px
  card-label:
    typography: "{typography.label-sm}"
    textColor: "{colors.text-faint}"
  card-value:
    typography: "{typography.numeric-lg}"
    textColor: "{colors.text}"
  card-hint:
    typography: "{typography.body-sm}"
    textColor: "{colors.text-muted}"

  badge-healthy:
    backgroundColor: "color-mix(in srgb, var(--color-healthy) 14%, transparent)"
    textColor: "{colors.healthy-text}"
    typography: "{typography.label-sm}"
    rounded: "{rounded.full}"
    padding: 4px
  badge-degraded:
    backgroundColor: "color-mix(in srgb, var(--color-degraded) 14%, transparent)"
    textColor: "{colors.degraded-text}"
    typography: "{typography.label-sm}"
    rounded: "{rounded.full}"
    padding: 4px
  badge-unreachable:
    backgroundColor: "color-mix(in srgb, var(--color-unreachable) 14%, transparent)"
    textColor: "{colors.unreachable-text}"
    typography: "{typography.label-sm}"
    rounded: "{rounded.full}"
    padding: 4px
  badge-unknown:
    backgroundColor: "color-mix(in srgb, var(--color-unknown) 14%, transparent)"
    textColor: "{colors.unknown-text}"
    typography: "{typography.label-sm}"
    rounded: "{rounded.full}"
    padding: 4px

  health-bar:
    height: 4px
    rounded: "{rounded.none}"
  progress-track:
    backgroundColor: "{colors.surface-sunken}"
    height: 6px
    rounded: "{rounded.full}"
  progress-fill:
    backgroundColor: "{colors.primary}"
    rounded: "{rounded.full}"

  chart-panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.lg}"
    padding: 12px
  chart-grid-line:
    backgroundColor: "{colors.border}"
  chart-axis-label:
    typography: "{typography.numeric-sm}"
    textColor: "{colors.text-faint}"

  button-ghost:
    backgroundColor: transparent
    textColor: "{colors.text-muted}"
    typography: "{typography.label-md}"
    rounded: "{rounded.md}"
    padding: 8px
  button-ghost-hover:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.text}"
  button-primary:
    backgroundColor: "{colors.primary-solid}"
    textColor: "#FFFFFF"
    typography: "{typography.label-md}"
    rounded: "{rounded.md}"
    padding: 8px
  button-primary-hover:
    backgroundColor: "{colors.primary-solid-hover}"

  table-row:
    backgroundColor: transparent
    textColor: "{colors.text}"
    typography: "{typography.body-sm}"
  table-row-hover:
    backgroundColor: "{colors.surface-raised}"
  table-header:
    typography: "{typography.label-sm}"
    textColor: "{colors.text-faint}"

  code-inline:
    backgroundColor: "{colors.surface-sunken}"
    textColor: "{colors.tertiary}"
    typography: "{typography.numeric-sm}"
    rounded: "{rounded.sm}"
    padding: 4px
---

# RLinf Control Plane

## Overview

This is an operator console, not a dashboard product. The person opening it has
a training job on a cluster and one question: *should I be worried?* Everything
here is arranged to answer that before they finish reading the page.

Three facts about the audience shape every decision below:

**They may be looking at it in a dark room at 2am, or beside a bright notebook
during a daytime incident review.** Dark remains the safe default for long
training sessions, while light is a first-class theme rather than an inverted
afterthought. The choice is remembered and applied before React starts so neither
mode flashes the other on load. Long-session comfort also means no animation that
repeats: anything you learn to ignore is dead pixels.

**The numbers matter more than the chrome.** A step count, a success rate, an
ETA — these get read, compared against a number remembered from an hour ago, and
acted on. They are set in a monospaced face with tabular figures so digits sit
in fixed columns and a value ticking from `9` to `10` does not shift the layout.
Prose labels use Inter; anything a person might diff against a remembered value
uses JetBrains Mono.

**A wrong reassurance is worse than no information.** RLinf runs die in ways
that look healthy: the driver process stays up and its heartbeat thread keeps
ticking while the training thread is blocked forever in an NCCL collective. The
server already distinguishes these (`healthy` / `degraded` / `unreachable` /
`unknown`), and the UI's job is to render that verdict faithfully and never
invent a fourth answer. `unknown` gets its own grey treatment rather than
collapsing into green — "we could not tell" is real information and must not be
laundered into "fine".

The overall feel combines an instrument panel with Arco's enterprise-console
values: clear hierarchy, consistent rules, a 4px rhythm, and restrained chrome.
It is dense, quiet and legible at a glance, with colour reserved for action,
selection, data series and status. A screen where six things shout at once has
told you nothing about which one to look at.

## Colors

**Brand and action.** Arco blue is the action and selection colour: `#165DFF` in
light and `#3C7EFF` in dark. The header uses RLinf's official purple-blue gradient
wordmark; purple also remains the second data series, so the product retains its
identity without turning every active control purple. Neither colour means
anything about health.
`primary-solid` is the darker blue reserved for filled buttons, where white label
text needs more contrast than the selection swatch provides. `tertiary` is the
theme-adjusted link and inline-code blue.

**Surfaces** follow Arco's neutral ladder. Light uses `#F7F8FA` canvas, white
containers and `#E5E6EB` boundaries; dark uses `#17171A` canvas and stepped
`#232324` / `#2A2A2B` containers. `surface-sunken` holds code, progress troughs
and chart wells. Neither palette uses pure black against pure white for the main
reading surface.

**Status is the only semantic axis, and it has exactly four values** because the
server's `health` field has exactly four:

| Token | Meaning |
|---|---|
| `healthy` (green) | heartbeat, progress and metrics all current |
| `degraded` (amber) | alive but something is wrong — hung thread, dead metric path, degrading step time |
| `unreachable` (red) | no heartbeat; the driver process is probably gone |
| `unknown` (grey) | no readable snapshot — we genuinely do not know |

The semantic backgrounds follow Arco's success, warning and danger ramps. Each
has a separate `*-text` token, darker in light mode and lighter in dark mode, so
12px badge labels clear 4.5:1 without distorting the full-strength status strip.
Status is never carried by colour alone — every badge pairs its colour with a
word, and the health card carries the server's reason.

**Series colours** are an ordered eight-slot ramp, assigned by position, and
they mean nothing except "a different line from the one next to it". Slots 1 and
2 are the brand hues so a single-series chart looks like it belongs to RLinf.
Neighbouring slots alternate hue family rather than shading, because two
adjacent lines are told apart by hue far more reliably than by lightness.
Light-theme series use deeper steps from the same ramps so even yellow and cyan
lines clear 3:1 on a white chart. Series colours overlap the status palette by
design — a *line* is never a verdict, so there is no ambiguity to resolve.

## Typography

Two families, and the split between them is functional:

- **Inter** for everything a person reads as language: headings, labels, prose,
  explanations of a verdict.
- **JetBrains Mono** for everything a person reads as a value: step counts,
  metric values, durations, IDs, paths, config keys. Always with `tnum` (tabular
  figures) and `zero` (slashed zero) enabled — a live-updating number that
  reflows its own width is a distraction, and `0` versus `O` matters in a run ID.

Both are variable fonts self-hosted as woff2 subsets. No webfont CDN: this
console runs on clusters that may have no egress, and a page whose numbers
render in a fallback face while waiting on a font request is worse than one that
never asked.

Ten roles share seven deliberately distinct sizes (`32 / 28 / 20 / 15 / 14 / 13
/ 12`). Weight does more work than adding near-duplicate sizes: `label-sm` at
12px and weight 550 in `text-faint` is the standard structural label, and it
recedes without shrinking below Arco's minimum readable size.
Card and table labels use sentence case; compact status tags keep uppercase
tracking so state is visually distinct from navigation.

`numeric-lg` at 28px is the hero number on a card. It is smaller than `display`
on purpose — the card's *label* tells you what you are looking at, and a number
that outsizes the page title fights the page.

## Layout

**A four-column grid on desktop, and the Overview page's eight cards fill it as
two rows of four.** That pairing is why the count is eight: state, phase,
progress, timing, checkpoint, health, north-star metric, anomalies. Below
1200px it becomes two columns, below 720px one; the card order is the reading
order, so reflow degrades gracefully without media-query-specific rules.

**Spacing is a 4px scale**, with 12px (`card-gap`) between dense overview cards,
16px inside cards, and a 24px page gutter. This is the rhythm Arco uses to keep
enterprise surfaces compact without making controls feel accidental.

**Charts are 180px tall by default** — enough to read a trend, short enough that
four rows fit a laptop screen without scrolling. `chart-height-tall` (260px) is
for the one chart a page wants to emphasise, typically the north-star metric.
`chart-height-spark` (44px) is the sparkline that sits inside an overview card,
and it is a token rather than a number at the call site for a concrete reason: it
was a number once, written as an inline height on a wrapper `div`, and the chart
inside kept its own 180px and drew straight through the bottom of the card. A
chart's height belongs to the chart, so the variant is declared here and selected
by name.

**The page does not reflow when data arrives.** Cards reserve their height,
numeric values reserve their width via tabular figures, and a chart with no data
yet renders its axes with an empty plot area rather than collapsing. A control
plane that jumps every two seconds as an SSE update lands is unusable for the
long sessions it exists to support.

**Content is width-capped at 1600px and centred.** Beyond that, an eight-card
row stretches into a shape where scanning left to right costs a head turn.

## Elevation & Depth

**Static surfaces do not need shadows.** Arco's neutral fills and 1px boundaries
carry hierarchy more reliably across both themes. Depth comes from three other
means, in order of preference:

1. **Surface lightness.** The four-step ladder is the primary depth cue. Raised
   things are lighter; inset things are darker than the page.
2. **1px borders** in `border`, for cards that must be distinct where a
   lightness step alone is too subtle (a card on `surface` inside a `surface`
   section). `border-strong` is for focus rings and the active item in a list.
3. **Nothing else for static content.** No blur or glass, and gradients stay in
   the brand mark rather than behind data. Popovers may use a restrained shadow
   when they have to separate from content in both themes.

Overlays (a video player, a metric picker) darken the page behind them with a
60%-opacity black scrim and sit on `surface-raised`. The scrim is the only
place a semi-transparent black is used.

## Shapes

Corners are uniformly rounded per role, never mixed within one component:

- `sm` (2px) — inline code and tiny inset chips
- `md` (4px) — buttons, inputs and select controls
- `lg` (8px) — cards, chart panels and modals
- `full` — status badges and progress bars, where a pill shape reads as "state"
  rather than "container"

`rounded.none` exists for one thing: the health bar spanning the full page
width. A pill-shaped full-bleed bar looks like a mistake.

The scale stops at `lg`. There is deliberately no larger step, because every
radius above is tied to a role and a spare one is what the next person reaches
for when they want something "rounder" — at which point two cards on one page
disagree and nothing in this file says which is right.

Data marks are square-cornered. A chart line, bar, or heatmap cell inherits
nothing from this scale — rounding a bar's top misrepresents its value at the
pixel level, and this console is read for values.

## Components

**Cards** are the Overview's unit. Each is a sentence-case label (`label-sm`,
faint), a value (`numeric-lg`), and an optional hint (`body-sm`, muted)
explaining or qualifying the value. The hint is where a health card
explains *why* it is amber. A badge with no explanation gets ignored after the
second time you see it, so the reason string the server already returns is
rendered, not dropped.

**Status badges** are a coloured word on a 14%-opacity wash of the same hue, in
a `full`-rounded pill. Four variants, one per health value, plus the same
treatment reused for lifecycle state (`running` / `finished` / `failed` /
`stopped` / `pending`). Lifecycle and health are visually parallel because they
are genuinely different questions — a `finished` run is `healthy` and must not
look alarming for being silent.

**The health bar** is a 4px full-bleed strip below the header, in the health
colour, present on every page. It is the fastest possible answer to the page's
central question: peripheral vision resolves a 4px colour band before the eye
finds a badge. It never animates. When health is `degraded` or worse, the reason
appears as text beside it — the bar alone says "look", the text says "at what".

**Progress** is a `full`-rounded 6px track in `surface-sunken` with a `primary`
fill, labelled with `step / max_steps` in `numeric-md` and the step semantics
(`RL iteration`, `minibatch`, `optimizer step`) in `label-sm` beside it. That
label is not decoration: the same "step 400" means an RL iteration in an
embodied run and a minibatch in a reasoning run, and the two are not comparable.
Where the server reports no horizon, the track renders indeterminate — a
striped static fill, not a fake percentage.

**Chart panels** carry a title, an optional unit, and the plot. Grid lines are
`border` (barely visible, enough to read a value against), axis labels are
`numeric-sm` in `text-faint`. Legends sit above the plot, not beside it, so plot
width does not depend on how long a metric key is. A crosshair is shared across
every chart in a group: moving the pointer over one reads out all of them at
that step, which is how a person correlates a loss spike with an entropy
collapse.

**Tables** (the run list, the checkpoint list) sit in one bordered surface, with
a neutral header fill and `surface-raised` hover row. Numeric columns are
right-aligned and monospaced; text columns left-aligned in Inter. Row height is
40px — dense enough to scan a node's runs, loose enough to hit with a pointer.

**Buttons** come in two weights only. `button-primary` in Arco blue is for the
one action a view has, if any; `button-ghost` for everything else. This console
mostly reads, so most controls stay neutral, and a page full of filled buttons
would imply a page full of consequences.

**Inline code** (`code-inline`) is `tertiary` on `surface-sunken`. Run IDs,
metric keys, paths and the assembled resume command all use it, since all four
are things a person copies into a terminal.

## Do's and Don'ts

**Do** render the server's health verdict and its reason verbatim. The
derivation lives in one pure function on the server precisely so a run cannot
look healthy in one view and dead in another; recomputing any part of it in the
browser reintroduces exactly that.

**Don't** invent a status the server cannot express. There are four health
values and five lifecycle states. `unknown` is not a loading state and must not
be styled as one.

**Do** reserve `primary` for action, selection and the first data series, and
`secondary` for the RLinf brand accent and second series. **Don't** use either
for status; success, warning, danger and unknown are a separate semantic axis.

**Do** pair every status colour with a word. Colour alone excludes readers with
red-green deficiency and is invisible in a screenshot pasted into a chat.

**Do** use tabular figures for every live number. **Don't** animate a number
between values — the intermediate digits are not data, and a value mid-tween
cannot be read.

**Don't** add blur or decorative gradients to a surface. Static depth is the
lightness ladder plus 1px borders; a restrained shadow is only for an overlay
that must visibly float above data.

**Do** keep the page geometry constant across SSE updates. **Don't** let a card
resize, a chart re-fit its axes on every push, or a list reorder itself while
someone is reading it.

**Do** label the step axis with the run's own `step_semantics`. **Don't** put
two runs with different step semantics on one axis without saying so — the
comparison is meaningless and the chart will be believed anyway.

**Do** hold WCAG AA (4.5:1) for body and label text and 3:1 for large text, chart
lines and functional UI boundaries. `text-muted`, `text-faint`, the `*-text`
badge roles, filled-button labels and light-theme series were chosen against
those bars; a new role colour has to clear its bar before it ships.
