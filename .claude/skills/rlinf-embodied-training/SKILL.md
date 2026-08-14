---
name: rlinf-embodied-training
description: General engineering discipline for embodied SFT/RL training in RLinf (VLA policies such as PI0/PI0.5/OpenVLA on ManiSkill or other sims). Use when designing a training recipe or reward, choosing warm-start checkpoints, building/calibrating a sim environment for a new robot, converting demonstration data, launching/stopping/monitoring long runs, or diagnosing zero-success and silent hangs. Principles first; robot- and machine-specific facts are in the appendices.
---

# RLinf Embodied Training — Engineering Discipline

General principles for training VLA policies (SFT + RL) in RLinf. Each rule is stated generally; **Evidence:** lines cite the incident that proved it (SO101+PI0.5 project, 2026-08). Appendices hold project/machine specifics.

---

## 1. Recipe design: check preconditions before training anything

**On-policy RL (PPO) is an amplifier of existing success, not a from-scratch discoverer.** Before any RL run, verify at least one of:
- the starting policy already succeeds sometimes in the TARGET environment (success_once > a few %), or
- exploration is strong enough to discover success by chance (from-scratch policy + high-entropy noise + huge env counts/steps).

A pretrained VLA has narrow exploration (approx_kl ~0.01–0.04/step under flow-noise) — it has NEITHER property unless you first SFT it on demonstrations **rendered in the target sim**. The proven pipeline is: *scripted/planner demos in sim → SFT (from the existing real-data SFT ckpt) → RL with nonzero initial success*.
> Evidence: real-image-SFT PI0.5 dropped into sim RL produced 0 grasps across 5 runs / ~2000 epochs; success appeared but stalled at 1–2/128 for 740 epochs (too sparse to amplify). The reference recipes that work (RLinf's own pi05+ManiSkill example; lerobot-sim2real) each hold one of the two preconditions.

**Frame-level premortem (mandatory, before every launch):** write down (a) which preconditions of successful reference recipes you have/lack — trace their FULL causal chain (starting-policy competence, exploration strength, data regime), not just their reward; (b) the most likely frame-level failure cause if this run fails; (c) a falsifiable verdict point (fixed epoch + expected metric decode). Never run open-ended.
> Evidence: every frame-level insight in the project came from the user forcing a pause; guards only monitored progress *within* the plan, never the plan.

**Verify the task is physically solvable before any long run**: drive the env with a scripted controller / motion planner to a full success. If a planner can't do it, RL never will.
> Evidence: a 12-hour RL run chased a task that was physically impossible (gripper couldn't close below 8.1cm on a 2.9cm cube); a 30-minute planner probe would have caught it. Later, the planner probe (100/150 success) proved the fixed env solvable in one hour.

**Nonzero initial success is NECESSARY but NOT SUFFICIENT.** A BC (behavior-cloned) warm start with high zero-shot success can still be DESTROYED by vanilla PPO within ~10 epochs: the cloned behavior is a narrow ridge; exploration noise knocks rollouts off it, failures near the ridge dominate the data, and advantages push the policy toward the noise-robust hover attractor. (Observed 3x: 50%-zero-shot starts collapsed to 0 by epoch 5–10 under default knobs, while an RL-grown 1.5% behavior — learned under the same noise — survived 740 epochs.) RL from a BC start needs protection — historically the **conservative-PPO bundle** (halved noise, lr ~2e-6, update_epoch 2, clip 0.1, entropy_bonus 0), but its validated regime is ONLY mid-competence planner-BC starts (~45-50%); outside that regime use the published reference recipe instead (§1b — fresh-sample-dominated updates, not smaller steps). **The bundle is calibrated in UPDATES PER EPOCH, not knob values: keep `global_batch_size = num_envs × (budget/num_action_chunks)` so each epoch does exactly ONE update.** Tripling the episode budget silently tripled updates/epoch and re-triggered soft BC-erosion (eval fell 71.9%→30% BELOW the 46.9% zero-shot while train rose — train↑/eval↓ converging is the fingerprint). Longer episodes also blow up return variance (value_loss 100+, kl→8e-4); and set save_interval ≤10 in fast-moving phases or the peak checkpoint is lost (71.9% peak at epoch 10 vs save_interval 50).

**RL-from-BC is a FINITE improvement window — plan the harvest, not just the run.** Even the correctly-calibrated conservative bundle (exactly 1 update/epoch) shows peak-then-decay: eval climbs for ~100 epochs, plateaus, then erodes monotonically while TRAIN success stays flat (the noise-taxed train metric cannot see the erosion; eval is the only truth signal — "train flat + eval decaying from peak" is the slow-erosion fingerprint, distinct from the fast train↑/eval↓ variant). Therefore: (a) treat the eval peak as the deliverable — set save_interval ≤ val_check_interval so the peak checkpoint is always captured; (b) build **auto-stop into the launcher itself** (stop when eval < best − 20 pts, or < zero-shot baseline for 3 consecutive evals) — a session-side verdict guard dies with the session/compaction, an in-launcher guard doesn't; (c) budget the run as "climb + margin", not open-ended.
> Evidence: pp4 (1 update/epoch, all knobs correct) rose 46.9%→75.0% by epoch 100, held ~65–72% to epoch 140, then decayed to 10.9% by epoch 320. The collapse threshold was crossed ~epoch 250 but no live guard was armed (prior watcher stale, new one lost to session compaction) — 180 epochs of GPU burned degrading a policy whose peak was already saved at step_100.

**The freeze test (cheap causal control):** run with `actor.optim.lr=1e-9` — if the behavior survives frozen (it did, 66 epochs) but dies under training, the destroyer IS the update step, not the env/reward/critic. Use this before theorizing.

**Expert iteration (collect deterministic successes on fresh seeds → gentle SFT) is a zero-risk gain at any competence; the hand-rolled conservative bundle is NOT safe at high competence.** One distillation round gained +18 pts (63.3%→81.6%); the same-day hand-rolled-PPO attempt on that sharp start lost 53 pts in 30 epochs (measured on saved ckpts). But high-start PPO is NOT inherently doomed — the published πRL recipe amplifies pi0.5 from 77%→98% (see §1b for the structural differences: update_epoch 1, big fresh-sample iterations, separate fast value_lr, default noise). Order of preference at high competence: (1) distill again, (2) reference-aligned PPO per §1b, (3) never the hand-rolled bundle outside its validated regime (planner-BC starts ~45-50%).
> Evidence: pp5 2026-08-09. Same-day hand-rolled PPO: −53 pts; same-day distillation: +18 pts; published reference: +21 pts from a 77% start.

**Before adopting ANY collapse diagnosis, check the refuted-diagnoses list (§6c) FIRST.** Cold-critic (huge value_loss/grad_norm at RL start) is a seductive and ALREADY-REFUTED primary explanation: v7's warmup held success and cut value_loss, yet v7b still collapsed — the destroyer was BC-brittleness. The cold-critic signature (value_loss 50–110, grad_norm 300–400 vs healthy ~23) is REAL but it is a co-symptom of a fresh value head on bimodal returns, not proof of causation. A diagnosis that was refuted once must clear a HIGHER bar (a controlled comparison, not just a matching fingerprint) before it justifies a new run.
> Evidence: pp5 collapse was initially re-diagnosed as cold-critic and a warmup phase was launched — repeating the v6→v7→v7b arc that the skill itself documents as a misdiagnosis. The user caught it ("你这是又把方向搞错了?").

**Expert-iteration protocol that works (validated twice):** (1) run deterministic evals of the current best policy on 6-8 NEVER-USED seeds with the rollout recorder on (`SO101_COLLECT_DIR`), which yields ~n_envs x n_seeds x success_rate trajectories for free while also producing an unbiased estimate of that policy; (2) convert those successes AND mix them with the original planner demos (iRe-VLA — pure self-distillation narrows the policy; v5 lost 53 pts that way); (3) SFT from the SAME policy at a GENTLE lr (1e-5, not the 2.5e-5 used for fresh behaviour), 2000 steps, save every 250; (4) gate on the usual seed-separated protocol. Historical gain: +18 pts (63.3% -> 81.6%).

## 1b. Published reference recipes — ALIGN FIRST, hand-roll knobs only after the reference fails

**Meta-rule: before inventing a knob bundle, find the published recipe closest to your (model, start-competence, sim) triple and replicate it exactly.** Hand-rolled bundles encode one project's failure modes; published recipes encode many. (This section exists because a hand-rolled "conservative bundle" was applied outside its validated regime and cost a day.)

- **πRL (RLinf's own paper, arXiv 2510.25889): PPO amplifies pi0.5 from HIGH starts — LIBERO 77.1%→98.3%, ManiSkill multi-task 41.9%→84.8%/90.9%.** High-start collapse is NOT inevitable; it's a recipe problem. The in-tree reference configs are the ground truth: `libero_spatial_ppo_openpi_pi05.yaml` (flow-noise per its own comment; the high-start one) and `maniskill_ppo_openpi_pi05.yaml` (flow-noise, mid-start).
- **The LIBERO high-start recipe's key structure vs our failed conservative bundle:** `update_epoch: 1` (every sample used ONCE — off-policy reuse is the poison, not lr); `rollout_epoch: 8` with fewer envs (≈24k FRESH samples per policy iteration → ~12 minibatch updates per iteration, all on-policy); policy `lr: 5e-6` with separate `value_lr: 1e-4` (critic learns fast on its own lr — a fresh value head is handled by DESIGN, no warmup phase); standard `clip 0.2`, `entropy_bonus 0`, `entropy_type: token_level`, default noise `[0.16,0.12,200]` (NOT halved, even at 77% start); `rewards_lower_bound/upper_bound 0.1/0.9`; `value_clip 0.2`, `huber_delta 10`.
- Contrast with the hand-rolled bundle that collapsed an 81.6% start (noise halved, lr 2e-6, update_epoch 2, clip 0.1, 6k samples/iteration): less data, reused twice, tighter clip — MORE off-policy pressure per fresh sample, not less.
- **iRe-VLA (arXiv 2501.16664): alternate RL and SFT stages.** RL stage: VLM frozen, train action head (+critic). SFT stage: train on **original expert demos + newly collected successful trajectories together** (the mix prevents catastrophic forgetting / mode narrowing — pure self-distillation sharpens but narrows). Validates the expert-iteration loop as a first-class published method, not a fallback.
- RLinf also ships DAgger (`libero_spatial_dagger_openpi.yaml`), DSRL, async-PPO variants — check the config directory for an existing implementation before hand-building any training loop.
- **lerobot-sim2real's ordering is the canonical real2sim sequence: CALIBRATE FIRST, TRAIN SECOND.** Visual/geometric alignment against the real setup (task-spec audit, camera parity, spawn distributions) comes BEFORE the first training run. This project inverted that order and paid ~2 days of GPU on a mis-specified task — the §4a audit is that step, mandatorily first.
- **Sensor/temporal parity with the real dataset is part of the recipe, not a finishing touch**: sim cameras at the REAL camera's native resolution & aspect (both domains through the identical resize path), and sim `control_freq` = the real dataset's fps (a 15Hz sim under a 30fps dataset halves temporal resolution and fights the pretrained VLA's action-rhythm prior). Both were inherited from reference examples here and went unquestioned for a week.

**Success-termination arithmetic:** if success terminates the episode, succeeding can FORFEIT more dense reward than it earns (measured: hold-without-lift ≈ 196 return vs succeed-and-terminate ≈ 35 → PPO rationally learns grasp-and-hold-never-lift; fingerprint = reward RISES while success → 0). For hold-style success criteria set `env.train.ignore_terminations=True` so the success state pays per-step and strictly dominates. Necessary, but fixing it alone did not stop the BC-collapse above — both are real, independent traps.

**Escalation ladder for zero-success** (in order, never blind-restart):
1. Env solvable? (planner probe)
2. Actuation physically correct? (command→travel measurement)
3. Warm-start still contains the needed behavior? (instrument and measure it)
4. Does the reward give a gradient into the missing behavior? (add a bridge term if not)
5. Still zero → the recipe lacks preconditions → build the sim-demo SFT pipeline.
6. Success exists but collapses under RL → freeze test → conservative-PPO bundle → if still collapsing, STOP iterating versions and verify the reference recipe itself (does the official example actually amplify from a BC start?) before any further variant.

## 2. Reward design

- Start from the proven manipulation recipe: `reach(1−tanh(5d)) + is_grasped + progress·is_grasped`, success = constant. **No gripper open/close micromanagement** — successful reference rewards leave actuator timing to the policy; the binary contact-jump makes success dominate hovering.
- Before accepting any shaping term, compute its **per-episode time integral**, not its per-step weight. A small always-on term beats a large rarely-on term.
  > Evidence: `0.3·openness` while far (~60 steps) crushed `0.5·closedness` while near (~5 steps) → policy learned permanent hover-open.
- Map shaping semantics to the demonstrated motion sequence before coding. Never invent a phase the demos don't contain.
  > Evidence: a "keep open while approaching" term was the exact OPPOSITE of the demos (gripper closed in transport, opens only above the object).
- If exploration provably cannot reach the binary jump (measure it: count contact events in rollouts), add ONE gradient-bridge term (`w·(d<thresh)·behavior_progress`) — nothing more.
- Decode rewards into behavior levels ("0.20 = reach-only, 0.28 = closing, >0.4 = grasping") and monitor against those levels, not raw numbers.

## 3. Warm-start / checkpoint lineage

- **Weights choice > reward choice.** PPO on a VLA barely moves behavior per step; a behavior absent from the warm-start will not be conjured by a better reward.
- **Never warm-start from a checkpoint trained under a defective reward.** Bad shaping bakes attractors into the weights; every descendant inherits them ("poisoned lineage").
  > Evidence: two runs with corrected rewards, warm-started from a bad-shaping checkpoint, both converged to the identical hover behavior; the untainted SFT start with the same reward produced first success within 14 epochs.
- Before reusing any checkpoint, MEASURE whether the target behavior still exists in it (instrument the env, log the relevant joint/action traces during a short eval).

## 4. Sim environment & calibration (new robot/embodiment)

- **Unit audit first**: identify the units of dataset actions/states vs sim controller (normalized vs radians vs degrees) before wiring anything.
- **Every conversion must round-trip** (`norm→rad→norm` error ≈ 1e-6) and be validated against the FULL dataset range vs sim joint limits (all joints, not just the suspicious one — audit siblings together).
- **Never reuse a conversion across different mechanisms.** For every actuator (especially grippers, which have different linkages than arm servos), measure the command→physical-travel curve in sim (`set_qpos` sweep + measure the physical quantity) before trusting any derived mapping.
  > Evidence: the arm servo tick→rad conversion applied to the gripper left an 8.1cm minimum jaw gap vs a 2.9cm cube — undetectable from code review, obvious from a 10-line measurement sweep.
- Clip commanded targets to URDF joint limits (real calibrations often include uncalibrated full-turn ranges).
- **Object physical parameters (mass/density at minimum) come from real bounds, not sim defaults.** ManiSkill's default density (1000) gave a 24.4g cube where the real one is <10g — a 2.4× dynamics error sitting silently under every grasp. Ask the user for bounds when unmeasurable; declare friction/etc. as unverified defaults when nobody can say.
- **Per-episode buffers in batched envs must be full-batch tensors updated via `buf[env_idx]=…`** — `_initialize_episode` receives PARTIAL env_idx on auto-reset after early termination. This bug stays dormant while episodes only truncate, then detonates at the first early termination — which for grasp tasks is the first SUCCESS. Test explicitly: `env.reset(options=dict(env_idx=tensor([...])))` then `evaluate()`.
- Scene fidelity: iterate against REAL photos side-by-side; the user's eye is the judge. Cameras mounted on robot links move when you change joint offsets — re-point them after any calibration change.
- Sign/offset calibration method: replay a real episode's actions in sim, compare frame-by-frame against the real video (read ALL frames — behavioral claims from 3–4 sampled frames were wrong once already).
- Wiring a new robot into RLinf usually needs NO new env type: drop the task into the sim's task dir (auto-registered), name cameras to match the wrapper's expectations, and add small branches for control mode and action formatting (see Appendix A file map).

**DENSITY LAW: the BC floor is set by demonstration spacing relative to the task's positional tolerance — and it is a THRESHOLD, not a gradient.** Compute `spacing = sqrt(spawn_area / n_demos)` and compare it against the tolerance the task actually needs (for a 2.9 cm cube with a parallel gripper, ≈ ±0.7 cm). Above the tolerance the nearest demonstration is not close enough to imitate, so BC must genuinely generalise and collapses; below it, BC interpolates and works. Fix the floor by shrinking the spawn area or adding demonstrations — NOT by changing recipes.
> Evidence (3 points, same env, same recipe, same warm start, 2026-08-12): spacing 1.01 cm → 12.5% (full board, 420 demos); 0.91 cm → 7.0% (curriculum band, 384 demos); **0.44 cm → 56.7% honest (small box, 247 demos)**. The 0.44 cm run also beat the historical pp-era floor (46.9%) under a STRICTER success criterion. Removing unreachable regions without changing density (the 0.91 cm run) did NOT help — achievability mass is a much weaker lever than spacing.

## 4a. Task-spec facts: measured or user-confirmed, NEVER assumed

Every sim task encodes real-world facts: where objects can appear, scene geometry, success semantics, data-collection protocol. **Each such fact must carry a provenance tag: MEASURED (from the real dataset — read ALL episodes, not a few frames) or USER-CONFIRMED. An eyeballed guess is not a provenance.** Before training on a new/changed task, list the facts and their provenance in the report; batch the unknowns into ONE question set for the user; measure everything the dataset can answer instead of asking.
> Evidence: the cube-spawn region was eyeballed from a few calibration frames at 6×8cm; extracting ALL 87 real first frames (a one-hour CPU job) would have shown on DAY ONE that real starts cover the ENTIRE board — the sim box covered 9% of them. It was done days late. **Price of the skipped hour: ~2 days of 8×H200 training compute (10+ SFT/RL runs, incl. an entire "reach 85%" campaign) optimized and celebrated a subset task whose numbers did not transfer (honest 80% on the subset ≈ 22.7% on the true task).** This is the single most expensive mistake of the project. The user had to point it out ("方块可能出现在棕色区域的任何一个地方"). Same audit later caught: wrong success semantics (missing the return-to-home the user required), wrong board size, wrong base-board gap, and an initial arm pose (extended) that contradicted all 87 first frames (folded, gripper closed). **Operational rule: the task-spec audit (this section) is the FIRST gate of any new env — before the first training run, not after the first plateau.**

**When two measurements CONTRADICT, show the contradiction to the user (annotated image + both numbers) — do not synthesize a theory that reconciles them.** Two invented theories (a 9.3cm phantom black board-end; an "environment changed between dataset and now" era story) each cost an iteration before the user resolved the contradiction in one sentence ("按第一个图片"). The dataset frames are the binding spec when the policy will be validated against dataset-era conditions.

**Image-pipeline parity (calibrate the CAMERA to the PIPELINE, not to your guess):** before designing sim cameras, read how the training pipeline actually processes real images (openpi: `resize_with_pad` = aspect-preserving LETTERBOX, never squash). Sim must render the real camera's aspect ratio (e.g. 160×120 for a 640×480 source) so both domains pass through the IDENTICAL transform. NEVER introduce a sim-side distortion to mimic an unverified pipeline assumption — an anisotropic-intrinsics hack built on "the pipeline squashes to square" put a fake distortion into sim until the user noticed the geometry looked wrong. Cheap camera validations: a known-size cube must render square (pixel-aspect check); a known object length + measured camera height pins the focal; board/object rect fractions real-vs-sim within ~2% is the convergence test.

**Any change to env facts (reset pose, object positions) silently invalidates downstream TOOL assumptions — rerun the planner probe first after every such change, before interpreting anything else.** Setting the true measured home pose (gripper nearly closed, wrist_roll 0) broke the demo generator's implicit preconditions (open jaws, wrist_roll π/2) → 1/60 probe with misleading per-stage failure labels. Demo generators should start with an explicit raise-arm-to-ready prefix from the true home (which also matches real demo semantics) rather than assuming the ready pose is the reset pose.

**Provenance applies to EVERY numeric parameter, not just real-world facts.** Legal provenances: derived-from-requirement / measured / user-confirmed / published-reference / EXPLICITLY-DECLARED-arbitrary (listed in the report with its risk). The failure mode this kills: a value gets filled in to keep moving ("reference example used 128", "closest 4:3 to the current value"), then later steps inherit it as if validated — provenance laundered by repetition. Verification instinct is crash-triggered (syntax, divisibility) but the expensive errors are the SILENT ones (resolution, spawn ranges) where the pipeline runs fine and just produces worse results — so the audit must be run as a checklist, not left to instinct.
> Evidence: render resolution 128² was copied from a reference example, then "minimally changed" to 160×120 for aspect ratio — never derived from the requirement (the pipeline feeds the policy 224×168 of real content; sim was delivering ~70% of the real detail, cube 10px vs 14px). Caught only when the user asked why results degrade ("数据集里应该是640x480的"). Fix: render at the real camera's native 640×480 so both domains share the identical resize path.

**A parameter has ONE source of truth — when changing it, grep for EVERY consumer before declaring the change done.** Shared quantities (control frequency, resolution, fps) are consumed by parallel paths that don't read each other: the env YAML (RL/eval), the demo generator's own gym.make, the dataset converter's metadata, the task register. Changing one path silently forks the parameter.
> Evidence: the approved 15→30Hz change was applied to the env yaml only; the demo generator's gym.make (no sim_config → ManiSkill default 20Hz) and the converter (fps=15 hardcoded) kept their own values — one pipeline ran THREE different frequencies (20Hz data, labeled 15, evaluated at 30) and burned a full S1-S4 cycle before the demo-length telltale (median 220, unchanged from the 15Hz era) exposed it. Corollary: the same audit retroactively showed ALL v3-era data was 20Hz-collected/15-labeled/15-evaluated.

**Every artifact sent to the user states its processing chain** (what was cropped/resized/squashed/detected), and comparisons must be like-for-like (both raw, or both pipeline-processed). A "real vs sim" collage with an undisclosed square-squash on one side misled a whole calibration round until the user asked what had been done to the image.

## 4b. MANDATORY pre-flight gates (run per stage, BEFORE launching — checklists are read-at-build-time, not write-at-retrospective)

Two Phase-2 failures were ALREADY-documented Phase-1 lessons that were not consulted when building the next stage (demo-coverage ceiling; demo length vs episode budget). Rule: before launching any stage, walk its gate and put the checked items in the report.

- **Demo generation gate:** (1) demo LENGTH vs env `max_episode_steps` — median demo length must fit with ≥20% headroom (150-step demos under an 80-step budget made the task impossible: zero-shot 3.1%→29.7% after fixing); (2) generator success RATE and coverage plan — the BC ceiling ≈ generator competence (observed 3x: 67%→68.8%, 43%→38-44%); add retry-on-failure / parameter fallbacks BEFORE the batch run, not after the plateau; (3) probe N≥6 (a 2-sample probe false-aborts 11% of the time at p=0.67).
- **SFT gate:** dataset unit/semantic sanity vs live env; ALL dataset-pointing config fields swapped in cloned configs; norm_stats computed AND copied into the ckpt dir.
- **RL gate:** frame-level premortem (section 1) + reward per-episode time integrals + success-termination arithmetic + episode budget sanity vs demo lengths. Episode budgets (`max_episode_steps`, `max_steps_per_rollout_epoch`) must be DIVISIBLE by `num_action_chunks` (validate_cfg asserts; 224%5 cost one failed eval), AND the batch arithmetic must close: `num_envs × (budget/num_action_chunks)` must be a multiple of `global_batch_size`, whose per-rank slice must divide the per-rank sample count (225-step budget gave 720 samples/rank vs 256 slice → crash; 240 gives 6144=3×2048 ✓). Pick budgets that keep ALL three constraints simultaneously.
- **Any launch:** lifecycle (section 6) + startup deadline + falsifiable verdict + **MANDATORY preflight**: `python -m toolkits.preflight_config --config-path <dir> --config-name <name> <EXACT launcher overrides, verbatim>` with the EXACT launcher env exported. It runs hydra compose + rlinf validate_cfg + path existence + batch arithmetic (samples/epoch vs global_batch vs per-rank micro) on CPU in seconds — the class of wiring failures that killed ~half of all first launches (bad override paths, %chunks asserts, updates-per-epoch drift, missing env vars, missing ckpts). Overrides must be copied verbatim from the launcher: a preflight with a partial override set validates a DIFFERENT config.
- **Interrogate every derived quantity you recompute for a gate.** When a change makes a number move (e.g. samples/epoch 2048→6144 while checking divisibility), ask "what ELSE does this number control?" before launching — the 3×-updates-per-epoch regression was computed by hand at gate time and waved through because only divisibility was being checked.

## 5. Demonstration data pipelines

- Scripted/planner demos: record with the sim's trajectory recorder; keep per-episode success flags; convert ONLY successful episodes.
- Convert units at the boundary (sim radians → dataset convention) with the SAME calibration module used at RL time, so SFT data and RL observations are self-consistent.
- **Never merge datasets with heterogeneous fps or resolution** — fps mismatch corrupts action-chunk time semantics. Keep the sim dataset separate at its true control frequency; continue SFT from the existing real-data checkpoint instead of merging files.
- After conversion, sanity-check semantics numerically (e.g. gripper open/close values match the real dataset's q01/q99 meaning) before training on it.
- norm_stats: compute per dataset; RLinf rollout workers ALSO look for norm_stats inside the checkpoint dir — copy it into every new warm-start checkpoint.
- **norm_stats are FROZEN across a checkpoint LINEAGE — never recompute mid-lineage.** A policy's action decoder is calibrated to the stats it was trained under; recomputing stats on an enlarged dataset and continuing SFT from an existing ckpt trains under a shifted convention and degrades monotonically. Recompute ONLY when starting a fresh lineage (new base weights). The per-ckpt stats copy is the lineage's ground truth — restore from there if the assets file was overwritten.
  > Evidence: v3→v3b (2026-08-10): dataset 209→472 eps triggered a stats recompute; continued SFT then DECAYED 22.7%→15%→1.6% across steps, and the stats swap ALONE cut the unchanged parent ckpt from 19.5% to 9.4% (controlled A/B, seed 777).
- **Contact-force grasp flags can be TRUE for useless pinches.** `is_grasping` fires on any contact force — including edge/corner pinches that cannot lift the object (observed: grasped=True while the cube dragged along the table, never rising 1mm). After closing, run a MICRO-LIFT verification (+3cm; payload must rise >1.5cm) before transporting; on failure, regrasp with a jittered grasp point rather than proceeding.
- **Transported-payload targeting: aim at the PAYLOAD, not the TCP.** A grasped object hangs offset from the TCP (direction set by the grasp yaw, 2–4cm for a small cube); planning the drop by TCP position misses containers whose acceptance window is comparable to the offset. Measure `payload_xy − tcp_xy` once after the grasp (rigid during transport) and subtract it from the target. Blind ±1cm retry variants cannot fix a systematic 2–4cm direction-specific offset.
  > Evidence: pick-and-place drop-miss rate: probe 3/8 with variants-roulette; all failures landed 2–4cm off in +y, exactly the measured hang offset.

## 6. Run lifecycle & operations

- **Process truth comes from /proc, never string matching.** `pgrep -f`/`ps|grep` match your own command line — this produced fake "alive" readings and cleanup loops that killed THEMSELVES mid-run (three separate incidents). Pattern:
  ```bash
  for p in $(pgrep -f 'PATTERN'); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink /proc/$p/exe 2>/dev/null)
    case "$exe" in */python*|*raylet*|*gcs_server*)
      st=$(awk '{print $3}' /proc/$p/stat 2>/dev/null); [ "$st" != "Z" ] && kill -9 "$p";; esac
  done
  ```
- **Stop sequence**: graceful `ray stop --force` → kill verified leftovers → remove `/tmp/ray/session_*` → **verify clean as the LAST step immediately before launch** (0 live procs, GPU memory empty). Verifying mid-cleanup and then killing more re-dirties state.
- **Launch detached** (`setsid bash launcher </dev/null >/dev/null 2>&1 &`) — plain background children die with the controlling session. No `sleep` in launch chains (harness blocks foreground sleep → silent abort).
- **Startup deadline (mandatory)**: after every launch, background-check that a FULL first training step (metrics line, not a progress bar) appears within ~30 min; otherwise declare dead, extract the traceback, root-cause. Milestone monitors alone cannot distinguish a hang from silence — a 51-minute blind hang proved it.
- **No root cause without a traceback or measurement.** "Cleanup fixed it" without a mechanism is luck + misattribution — the real bug (port-family mismatch, Appendix B) was blamed on "dirty state" three times.
- Monitors: filter only actionable signals (not per-epoch noise); fresh log file per attempt (stale tails reported old failures); cover failure signatures, not just success.
- **Every pipeline STAGE needs its own `timeout` + one retry — especially evals.** Rendezvous port collisions (EADDRINUSE / TCPStore flakes) kill one rank and leave the rest silently hung; a pipeline whose only deadline is the session-side watcher's total timeout burns hours on a 40-second eval. Wrap each eval/train invocation in `timeout <2-3x expected>` and retry once on empty result (retries have cleared every rendezvous flake so far).
  > Evidence: pp6b gate eval hung 2.4h on EADDRINUSE port 29691 (eval itself takes ~3 min); the stage had no timeout, the watcher's 3h total expired uselessly.
- Report "running" only with evidence: GPU util > 0 AND advancing step counts.
- **`ncclSystemError: ... Broadcast failed` is usually /dev/shm, NOT GPU memory.** NCCL allocates ~7MB shared-memory segments per communicator; containers default to a 64MB `/dev/shm`, and every crashed run leaves `cuda.shm.*` files behind — so each relaunch has LESS room than the last and the failure ratchets. Read the `Last error:` line (it names the shm path and size) before touching any memory knob. Fix: delete stale `cuda.shm.*`/`nccl-*`, `mount -o remount,size=16G /dev/shm`, and gate every launcher on free shm.
  > Evidence: 2026-08-12, four v6 launches "OOM-diagnosed" and shrunk (micro_batch 128→32→16, envs 64→32→16, no_shard→full_shard, rollout offload on) all kept failing; a minimal-scale diagnostic failed at 53/143GB — disproving the memory theory — and the `Last error` line read `Error while creating shared memory segment /dev/shm/nccl-... (size 7340384)` with `/dev/shm` at 64MB/94% full and 2882 stale segments. A night was lost to the wrong diagnosis.
- **Client-attached sessions FREEZE the agent's reactions when the client disconnects** — only already-detached (setsid) processes keep running; the agent's next tool call queues until the session resumes (observed: a corrective relaunch composed at 00:40 executed at 05:43). Overnight autonomy therefore lives ENTIRELY in the pipeline scripts: gates, timeouts, auto-stops, and pre-written fallback branches (e.g. "if gate < X, retry with start B") — anything left to the agent's live judgment will wait until morning.
- Clone-and-edit configs: grep the clone for EVERY reference to the original's dataset/checkpoint paths — one missed field cost a failed launch.
- **Never edit in-tree env/model code while a run that imports it is active.** Workers import the shared tree at PROCESS START — a pipeline that launches evals sequentially picks up mid-run edits in its later stages, silently mixing configurations within one result table. Either wait for the pipeline, stop it first, or have pipelines snapshot/pin the code they run.
  > Evidence: the 640×480 camera edit landed at 09:50:44 while the v3d gate was mid-run; every eval started after it fed 640-rendered frames to 160-trained policies — step_4000 read 0.0%, the verify read 0.78% vs its 11.3% gate, and the contamination was only caught by matching file mtime against eval timestamps.

- **A Ray worker can die mid-eval with `SYSTEM_ERROR: Worker unexpectedly exits` while the driver waits forever** — this is neither shm nor GPU memory (both were free when it happened). Every eval must therefore run under `timeout` AND retry (3 tries), with a full clean between attempts; a pipeline whose only guard is a long total timeout will stall for tens of minutes.
- **Never build launchers by sed line-range surgery on other scripts** — three incidents of dangling references (a variable defined in a cut-away section, an EXIT check pointing at the previous run's log, an `export EMBODIED_PATH` left behind in a removed stage → hydra `${oc.env:...}` KeyError). Write each launcher as a COMPLETE standalone file, then validate with `bash -n` AND a hydra dry-run (`--cfg job`) before the detached launch.

## 6b. OpenPI/PI0.5 architecture facts (verify before theorizing about training dynamics)

- `train_expert_only: True` (default in model/pi0_5.yaml) → `freeze_vlm()` freezes PaliGemma. Trainables = action expert + value head + noise head. Consequence: with `value_after_vlm: True` the value head sits on FROZEN VLM features — its gradients cannot damage the policy (a "critic corrupts backbone" theory is impossible under this config; check freezing FIRST).
- `detach_critic_input` originally only covered the suffix path; the VLM value path was fixed in-tree to honor it too (openpi_action_model.py ~line 1350).
- `noise_params: [start, end, anneal_steps]` = rollout exploration noise level annealed linearly (default 0.16→0.12 over 200). This is the BC-brittleness lever.
- Stable `actor/entropy_loss` (≈ −0.31…−0.34) means the learned noise head is NOT growing — rules out entropy/noise-explosion theories.
- Healthy-run reference magnitudes: `critic/value_loss ≈ 1`, `actor/grad_norm ≈ 23`. Sustained value_loss ≥ 15–45 means the critic cannot fit the returns (bimodal success/failure returns on frozen features) — expect biased advantages.
- Config diffs against the official reference (`maniskill_ppo_openpi_pi05.yaml`) are the fastest way to rule config in/out — our failing setup was knob-identical to the official one, which localized the fault to data/dynamics, not settings.

## 6c. Experiment methodology (how 12-hour lessons became 40-minute lessons)

- Every run states, BEFORE launch: hypothesis → falsifiable prediction → verdict epoch. Guards auto-fire the verdict (collapse threshold / growth threshold / death / max-epoch).
- Say "current binding constraint (evidence grade: X)" — never "root cause found". Two misdiagnoses (dirty-state, cold-critic) came from claiming certainty without a traceback or a controlled comparison; the refuting run is itself valuable evidence when the prediction was written down first.
- Budget rule: a wrong theory should cost ~1 verdict window (≈40 min), not a night. Stop-rule: after ~3 refuted variants of the same class, stop iterating and question the frame (see ladder step 6).
- **Every lesson must be tagged MECHANIZED or DOCUMENTED-ONLY, and DOCUMENTED-ONLY needs a stated reason why no automated check is possible.** Writing a rule into this file feels like closure but prevents nothing — three of the most expensive incidents (spawn-box coverage, frequency chain, render resolution) each had a written rule that was never consulted at build time. Post-mortems end with one mandatory question: *which automated check would have caught this?* If none exists, add it to `toolkits/invariant_audit.py` (silent wrong-result class) or `toolkits/preflight_config.py` (launch-blocking class) before moving on.
  > Evidence: the invariant auditor was written only after the user asked "不就是定期检查出来吧"; on its FIRST run it found two live defects (stale 15fps converters; sim spawn box covering only 89% of real cube starts) that had been sitting undetected behind written-but-unmechanized rules.

- **Failure loop (user-mandated standing workflow): after EVERY failure — (1) distill the lesson INTO this skill immediately, (2) RE-LOAD the updated skill, (3) walk the relevant 4b gate with it before building the next stage.** Checklists that are only written at retrospectives and never read at build time do not prevent repeats (two Phase-1 lessons were repeated verbatim in Phase-2 because the doc wasn't consulted).

## 7. Verification & reporting

- **Gate seeds and verification seeds must be DISJOINT, and the number you report is the verification number.** Selecting the best checkpoint on seeds A,B and reporting its A,B score bakes in selection bias (measured: gate 83.2% vs fresh-seed 77-80%; and a fixed eval episode set read a 63% policy as 75%). Protocol: select on gate seeds → re-measure on ≥2 never-used seeds → report both, verification first.
- **On spatial tasks report a PER-REGION breakdown, never just the mean** — a mean hides a dead corner (58% corner inside an "80%" policy; 0% band inside a "22%" one). Use spawn-restricted evals (sub-region sampling) or spawn-vs-outcome logging.
- **Judge a run by its BEST checkpoint, never by its first two.** Early checkpoints are undertrained; on precision tasks the useful policy can appear anywhere in the schedule and the series oscillates violently. Direction verdicts must wait for the peak of the screened set.
  > Evidence: v8 read 7.8% / 0.8% at steps 250/500 and was flagged DIRECTION-SUSPECT — then hit 54.7% at step_1000 and 61.7% at step_2500 (honest 56.7%). The premature flag would have killed the project's best run.
- **A single checkpoint's score is a SAMPLE from a noisy process, not a property of the recipe.** SFT on sparse-coverage precision tasks oscillates wildly between checkpoints (5.5%→22.7%→12.5% at 1000-step spacing). Gate EVERY saved checkpoint, and before claiming a recipe-level conclusion from one number, ask what checkpoint-to-checkpoint variance is.
- Claims require evidence: measurements for physics claims, full-frame reads for behavior claims, curves + video frames for training claims.
- Any image you cite must be SENT to the user (reading it only puts it in your own context).
- Eval videos in RLinf are often N×N env-grid montages — crop a single tile and zoom before judging behavior; the per-step reward burned into frames decodes distance.
- Never promise "this won't happen again." Offer bounded damage (deadline checks, verdict points) and falsifiable criteria instead.
- When a run plateaus: look at actual behavior (video frames) before theorizing.

---

## Appendix A — SO101 + PI0.5 project specifics (current as of 2026-08-10)

**TRUE task spec (user-confirmed / measured — supersedes all earlier "phases"):** red + blue 2.9cm cubes each independently uniform over the BROWN zone of the board (28.2×21.6cm; only anti-overlap constraint, no min separation); board 30.1×21.6cm with 1.9cm black band at −y (dataset frames; user's tape numbers set aside per "按第一个图片"); brown zone / tray / base share one centerline; base front to board near edge ~1.2cm ("紧贴"); tray 10.16×7.62×4.45cm (user-measured) flush at far edge; **success = cube in tray AND released AND 5 arm joints within 0.08 rad mean-abs of the measured home pose** (tolerance from the 87 real episodes: max deviation 0.076); home/reset pose = median of real first frames `[0.046,-0.880,1.013,0.586,-0.008,-0.931]` (folded, gripper near closed).

- Env: `SO101GrabRedCube-v1` (`rlinf/envs/maniskill/tasks/so101_pick_place.py`), robot uid `so101` (widened-limit so100, `so101_agent.py`), `pd_joint_pos`, budget 320 @15Hz. Reward ladder: reach + close-bridge + grasp + lift + 2·transport, placed = 6.0+1.5·homing, success = 8 (hold-hover maxes 5.4 — anti-hack arithmetic in comments). Cameras: front nadir 640×480, f=755.2, eye [-0.520,-0.007,0.559]; wrist 640×480 on Fixed_Jaw (pose visually calibrated only — numerical calibration pending). Env-var tools: `SO101_LOG_DIST` (grasp diagnostics), `SO101_SPAWN_MODE=legacy` (old 6×8cm box), `SO101_SPAWN_FRAC="x0,x1,y0,y1"` (sub-region targeting), `SO101_SPAWN_LOG=<csv>` (spawn-vs-outcome), `SO101_COLLECT_DIR` (ManiskillEnv rollout recorder, flush-on-first-success npz).
- **PENDING (user approval required per the every-parameter rule): control_freq 15→30Hz** (real fps=30; budget would go 320→640, samples 128×128=16384=8×2048), cube mass 24.4g→8g (user bound <10g), v4 demo regeneration at 640×480. The parameter approval table lives in the conversation of 2026-08-10.
- Calibration (`so101_calib.py`): normalized↔radians, homed frame tick 2048 (do NOT subtract homing_offset), SIGN all +1, OFFSET=[0,0,0,+0.6,0,0], gripper map norm0→−1.0 rad / norm100→+0.5 rad, clip to widened URDF limits.
- Integration: `rlinf/config.py::get_robot_control_mode`, `action_utils.prepare_actions_for_maniskill` (norm_to_rad), `maniskill_env._wrap_obs` (`so101_state_norm`, wrist_images). Preflight tool: `python -m toolkits.preflight_config` (§4b).
- Demo generator (`scratchpad/gen_so101_demos.py`): raise-arm prefix from real home → planner grasp (micro-lift verify) → FK-grid transport (payload compensation, closed-loop refine) → release → homing. True-task planner ceiling: **70% overall (45/64); near-base band ~25%, one corner ~0/22 (likely 5-DoF-infeasible)**.
- **Checkpoints that EXIST** (all others deleted in the 2026-08-09 disk purge): real-data SFT `so101_sft_openpi_pi05/global_step_8000`; subset-task best `so101_sft_pp6b/.../global_step_1000` (honest ~80% on the OLD 9%-area task, no-homing semantics — current warm start); `so101_sft_pp5/.../global_step_2000` (81.6% subset); true-task best `so101_sft_v3/.../global_step_3000` (**22.7%** gate=verify; bands 15.6/22.7/31.3; note SFT variance — neighbors read 5.5/12.5). v3b/v3c/v3d rounds: continued-SFT degradation + stats-poisoning + data-doubling-null results; ckpts deleted or obsolete.
- Sim datasets: `so101-sim-demos-v3` (472 eps, 160×120 — obsolete once 640×480 regeneration lands as `so101-sim-demos-v4`); older pp/sim sets deleted. TrainConfigs `pi05_so101_v3` / `pi05_so101_v4` in `dataconfig/__init__.py` (`HF_LEROBOT_HOME=/data08/henryg/pai/data`).
- Dataset facts: real set `henry-guo/so101-pick-place-v2` = 87 eps, **fps 30**, 640×480 front+wrist; LeRobot column `action` singular → `action_sequence_keys=("action",)`; openpi maybe_download takes local paths only; eval configs read `rollout.model`; absolute `--config-path` for hydra.
- **CURRENT BEST (2026-08-12): `so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500` — 56.7% honest** (seeds 1313/1414: 57.8/55.5; 8 further seeds 57.0-65.6, mean 61.3) in the pp-era 6x8cm spawn box, at FULL fidelity (640x480, 30Hz, measured geometry, 8g cube, success = in-box AND arm-home). Same ckpt on the full brown zone: **9.4%** — a box-trained policy does NOT transfer outward. Datasets: `so101-sim-demos-v8` (247 planner demos, 0.44cm spacing), `so101-sim-demos-v9` (v8 planner demos + 477 policy rollouts, ~0.26cm). Configs `so101_sft_v8.yaml` / `so101_sft_v9.yaml`; pipelines `scratchpad/{gen_v8_legacy,v8_pipeline,v8_verify,v9_expert_iter}.sh`. Full command-level runbooks: `V8_COMMANDS.md` / `V8_COMMANDS_ZH.md` (repo root).
- Open items: wrist camera never numerically calibrated (sim-vs-real board-mask IoU ~0; user specs say 106deg H-FOV native, sim has 86deg) — matters for sim2real, not for in-sim scores; spawn margin 2cm excludes 11% of real cube starts.
- Standing user rules: EVERY parameter pre-approved with provenance; GPU launches confirmed (overnight grants explicit); failure loop = update skill → reload → walk §4b gate.

## Appendix B — This machine (8×H200, IPv6)

- **IPv6 everywhere; two in-tree fixes, do not revert**: bracketed IPv6 in `collective_group.py` tcp:// URL; dual-stack AF_INET6 port probe in `cluster.py::find_free_port` (IPv4-only probing published ports already taken on IPv6 → flaky `TCPStore recvValue failed` → silent first-rollout hangs; worth upstreaming).
- Rendering on compute-only driver: exact-version driver libs symlinked into `.venv/nvidia_gl/` + private ICD; env vars `VK_ICD_FILENAMES`, `LD_LIBRARY_PATH`, `XDG_RUNTIME_DIR=/tmp/xdg-runtime`, `MUJOCO_GL=egl`. apt's libnvidia-gl is version-mismatched — never install it.
- `RAY_local_fs_capacity_threshold=0.99` (host /tmp ~96% full), `HF_HUB_OFFLINE=1` (everything local), proxy `http://[fdbd:dc61:d:297::16]:8888` (lowercase vars + `/root/.wgetrc`).
- Validated scale: **128 envs + global_batch 2048** (320 envs hangs). PID 1 is `sleep infinity` → zombies accumulate, harmless.
- Isaac Lab is effectively unavailable here (needs full RTX graphics stack; this box has a compute-only driver).

## Appendix C — Incident registry

The 50-item symptom→root-cause→prevention table backing these principles lives in the project work report: `SO101_WORK_REPORT.md` (repo root), §2 and the per-phase tables. Consult it when something here needs the original context.

- **This box: `/dev/shm` defaults to 64MB (container default) — remount to 16G before multi-worker runs** (`mount -o remount,size=16G /dev/shm`, root works here) and sweep stale `cuda.shm.*` between runs. Peak GPU memory for the true-task RL config (640×480 dual camera, 64 envs, micro_batch 32, no_shard) is ~90GB/card idle-to-rollout; micro_batch 128 at this resolution genuinely OOMs at 141GB.

## 6d. Pipeline handoff bugs waste GPU silently — three that actually happened

Long overnight pipelines are chains of `stage A finishes -> stage B starts`. Every joint in that chain is a place where the GPUs can go idle for hours with nothing crashing and nothing logging an error. Real incidents (SO101, 2026-08-12/13), each costing 1.5-4.5 h of 8xH200 idle:

- **Sentinel race.** A waiter written as `while ! grep -q DONE; do <producer alive?> || abort; sleep 60; done` aborts when the producer writes its marker and exits in the same instant — the liveness check sees a dead producer and calls it a failure. **Rule: after observing the producer is gone, re-check the completion marker before declaring failure.** Also never put the sentinel string inside your own log messages (`"died before S1 DONE"` makes every later `grep 'S1 DONE'` match your own error line).
- **`pgrep -f` self-match.** A watchdog running `pgrep -f 'bash .*run.sh'` matches ITSELF (the pattern is in its own command line), so it never reports "the job exited" — it reports nothing, forever, which is indistinguishable from a healthy run. Match on a pid captured at launch (checking `/proc/<pid>/stat` state, since a setsid-orphaned bash exits into a **zombie** whose `/proc/<pid>` directory still exists), or `pgrep -f` a pattern that cannot appear in the watcher's own argv.
- **Timeouts sized by round number instead of measured rate.** `timeout 10800` on a conversion stage whose measured rate (3.3 episodes/min x 724 episodes) predicts 3h20m would have killed it 25 min from the end and failed the whole pipeline. **Size every stage timeout from a measured rate with >=50% headroom, and state the arithmetic in a comment.** If a stage is already running under a too-tight timeout, `kill -9` the `timeout` WRAPPER only: the child survives, is reparented to init, keeps its inherited stdout fd, and a takeover script can own the remaining stages. Never edit a running bash script in place — bash reads it incrementally.

**And the meta-rule these share: a stage that ends with no successor armed is a silent stall.** Arm the successor BEFORE the predecessor can finish (a waiter that polls costs nothing), and make the completion notification itself independent of the polling logic that might be buggy.

## 6e. Expert iteration compounds; measure the frontier before widening

Round 2 of the collect-own-successes-and-distil loop is planned from a MEASURED competence frontier, not a guess. Evaluate the current policy on the region you intend to expand into, then invert the measurement to size the data: with in-region success `p_in` over area A and success `p_ring` over 2A, the outer annulus rate is `2*p_ring - p_in`, which tells you how many self-collected successes will land out there and therefore how many planner demos you must add to hold the density-law spacing in the new territory. Self-collection is biased toward the region already learned; the annulus is exactly where it under-delivers.
> Evidence: v9 (76.6% honest, in-box) measured 51.6% over 2x area => outer half ~27% => ~136 self-collected successes out there = 0.59 cm spacing, short of the proven 0.44 cm, so ~112 targeted planner demos were generated in four boundary strips instead of uniformly over the ring.

## 6f. Refuted: "coarse IK grid causes the placement error" (2026-08-13)

Symptom: a scripted planner's failures were 100% `drop-missed`, with a measured
median release-to-target error of 3.9 cm and a 46% tail beyond 5 cm, growing
with spawn distance (r=+0.51 between spawn y and error x). The closed-loop
pre-drop refinement had a 1.2 cm stop threshold it never reached, and its
search grid was coarse (0.25 rad elbow steps) with unchecked return values.

Fix attempted: fine local grid around the current pose, 4 passes, return value
checked, 2 cm accept tolerance. Pre-registered acceptance: +10 points of demo
yield. **Result: 20/24 vs 20/24 — zero improvement; pooled release error got
slightly WORSE (3.9 -> 4.8 cm median, tail 46% -> 48%).**

What the refutation teaches: the refinement loop reported converging (3.1 cm
pre-drop error at its own last measurement) while the LANDED position was 4.8
cm off, so the error is generated **after the gripper opens** — release
dynamics (residual velocity/pose as the cube leaves the jaws), not position
control. Search resolution was never the binding constraint. Next hypotheses
to test would be release height and gripper-open timing, not IK.

Two method errors worth repeating out loud: (1) per-ATTEMPT failure counts were
read as per-DEMO yield — a generator that retries 3 variants makes those differ
by 2x (35% vs 74%), inverting the apparent severity; (2) "all failures share
one stage" was read as "that stage is the bottleneck", when the median error
was already inside the target and only the tail failed. Decompose the
distribution before naming a root cause, and always run the A/B against a
pre-registered acceptance threshold so a null result is a decision, not a
debate.

## 1c. The PPO precondition must be measured IN THE ROLLOUT DISTRIBUTION, not deterministically

§1 says PPO needs the start policy to "already succeed sometimes in the target environment". That sentence hid an ambiguity that cost a full run: **which success — the deterministic eval, or the noisy rollouts PPO actually learns from?** They can differ by 50+ points.

**Measured on 2026-08-13 (freeze test, lr=1e-9, real training path):** the same checkpoint scored `eval/success_once` **0.539** (deterministic) and `env/success_once` **0.010** (under the official flow-noise `[0.16,0.12,200]`). PPO saw ~1% success in 24,576 samples/iteration, had almost no success signal to amplify, optimised the dense reward instead, and by step 9 (~108 updates) had destroyed the policy — the deterministic eval fell to 0.0 too.

**The threshold was already in this project's own logs**, unread until now (tensorboard `env/success_once`, first epochs):

| run | noisy-rollout success at start | deterministic eval at start | outcome |
|---|---|---|---|
| pp4 | 5-9% | 36.7% | amplified to 75.0% |
| v10 | 10-15% | 61.7% | amplified to 68.8% |
| pp5 | ~20% | 31.2% | still collapsed (necessary, not sufficient) |
| v11 | **1.0%** | 53.9% | destroyed by step 9 |
| v6 | 0.5% | 0.0% | never left zero |

**Rule: before launching PPO, measure `env/success_once` under the exact rollout noise you will train with. Below ~5% do not launch.** Cheapest probe: the freeze test (`actor.optim.lr=actor.optim.value_lr=1e-9`, `runner.val_check_interval=1`) — it exercises the REAL training path (same workers, env creation, model construction, weight sync) while leaving weights unchanged, and it reports both numbers in one epoch (~15 min).

**Why long-horizon tasks are structurally harder here:** noise is injected per decision, so staying on the BC ridge is a product over decisions. Decisions per episode = `max_episode_steps / num_action_chunks`: official ManiSkill 80/5 = **16**, official LIBERO 240/5 = **48**, this SO101 task 640/5 = **128**. At a per-decision on-ridge probability of 0.97 that is 61% / 23% / **2%**. Copying a reference recipe's noise parameters onto an 8x longer horizon is not "aligning with the reference" — it is a materially different amount of injected noise per trajectory.

**Do NOT use `runner.only_eval=True` as the probe.** It is not a "skip training" switch: `rlinf/config.py:826-830` sources the model spec from `cfg.rollout.model` instead of `cfg.actor.model`, and `env_worker.py:108` / `huggingface_worker.py:70` skip training-env creation. Three coupled changes, so a training config run under it exercises a different path (and typically dies on missing keys — which is the lucky outcome; a config with just enough keys would silently build a DIFFERENT model and hand back a number you would believe).

## 1d. PPO from a BC start: the three knobs that decided it, and the one that was inert

A campaign that failed seven times and then worked, on the same task and start.
What separated the working run from the failing ones, in order of impact:

**Updates per epoch is the decisive quantity — recalibrate it, do not inherit it.**
`updates = num_envs × (max_episode_steps / num_action_chunks) × rollout_epoch / global_batch_size`.
The published recipe's 12 updates/epoch destroyed the policy (61.7% → 7.0% by
epoch 9); the same start at **1 update/epoch** climbed to 73.4%. Set
`global_batch_size = samples per epoch` to force exactly one. The published
value is calibrated on benchmarks with 16-48 noisy decisions per episode; a
task with 64-128 is a different regime, and this is the parameter that
expresses the difference.

**Check whether you are discarding half the model's predicted horizon.** The
policy was SFT-trained with `action_horizon: 10` but the eval/RL configs
executed `num_action_chunks: 5` — copied from the reference examples and never
derived. Executing all 10 raised the DETERMINISTIC score 55% → 66.4% (free, and
it applies to deployment too) and halved noisy decisions per episode, which
raised rollout success 1.0% → 4.7%. Whenever `num_action_chunks < action_horizon`,
ask why.

**Confirm a knob is live before sweeping it.** `noise_params` belongs to
flow-SDE; with `noise_method: flow_noise` the magnitude comes from
`noise_logvar_range` through a learned noise head (`openpi_action_model.py:47-58`).
An 8x sweep of `noise_params` moved nothing — the correct knob then took rollout
success 4.7% → 39.1%. This also retroactively voids the "halve the exploration
noise" item of the old conservative bundle: it never did anything, and the runs
it was credited with succeeded for other reasons.

**Order of operations:** freeze-probe the rollout precondition (§1c) → fix the
chunk/noise settings until `env/success_once` ≥ 5% → then and only then tune the
update schedule. Fixing the precondition alone is not enough: a variant with 39%
rollout success still collapsed at 12 updates/epoch.

## 7. sim2real: measure the gap offline before the robot moves

A sim-trained policy has no right to be trusted on real observations, and you
can find out for free. Feed recorded REAL observations (images + proprioception)
to the policy and compare its predicted actions against what the human
teleoperator actually did. Two controls make the number interpretable: the same
measurement on SIM episodes (in-distribution reference) and a "hold still"
predictor (action = current state), which is the scale of motion. Report the
ratio policy-error / hold-still-error: below 1 the policy beats doing nothing,
at or above 1 it does not.
> Evidence: the SO101 policy measured 0.10 on sim and **4.47 on real** — its
> actions were 4.5x worse than freezing the arm. Cost: 20 minutes, no hardware.
> `tools_so101_session/offline_replay_check.py` is the implementation.

**Do not stop at the first suspect.** The obvious culprit was a known defect
(the sim wrist camera pointed at the robot's own body, so that input channel was
near-constant in training while it carries real information on the robot). But
re-running with the wrist channel removed changed nothing (4.47 → 4.59): the gap
is broader than the one defect. Had the check stopped at "fix the wrist camera",
a 30-hour retrain would have bought nothing.

**Render your training data and LOOK at it.** The wrist defect was invisible in
code review and in every metric for weeks; it took one grid of frames sampled
from actual training episodes. Do this once per camera when a new env is built.
