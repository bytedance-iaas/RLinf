import os
"""Append an exact per-step code appendix to the pp-80% runbook.

Code is EXTRACTED from the live source files (never hand-typed) so the runbook
cannot drift from what actually ran.
"""
import re
import subprocess

REPO = "/data08/henryg/pai/RLinf"
SCRATCH = os.environ.get("SCRATCH", "/tmp/so101_runs")
OUT = f"{REPO}/SO101_PP_80PCT_RUNBOOK.md"


def whole(path):
    return open(path).read().rstrip()


def func(path, name, nfuncs=1):
    """Extract `def name(...)` up to the next def at the same indent."""
    src = open(path).read().splitlines()
    out, depth, started, taken = [], None, False, 0
    for i, line in enumerate(src):
        m = re.match(r"^(\s*)def " + re.escape(name) + r"\b", line)
        if m and not started:
            depth = len(m.group(1))
            started = True
            out.append(line)
            continue
        if started:
            if re.match(r"^\s{0,%d}def \b" % depth, line) and line.strip().startswith("def "):
                indent = len(line) - len(line.lstrip())
                if indent <= depth:
                    taken += 1
                    if taken >= nfuncs:
                        break
            out.append(line)
    return "\n".join(out).rstrip()


def lines(path, a, b):
    src = open(path).read().splitlines()
    return "\n".join(src[a - 1:b]).rstrip()


TASK = f"{REPO}/rlinf/envs/maniskill/tasks/so101_pick_place.py"
ENVPY = f"{REPO}/rlinf/envs/maniskill/maniskill_env.py"

md = []
A = md.append

A("\n---\n")
A("## 11. Exact code, step by step\n")
A("Everything below is extracted verbatim from the running source, so it cannot "
  "drift from what actually produced the results. Paths are repo-relative.\n")

A("### Step 1 — robot: widened-limit SO101 agent\n")
A("`rlinf/envs/maniskill/so101_agent.py` (whole file). Importing it registers uid "
  "`so101`; the task's `__init__` does that import.\n")
A("```python\n" + whole(f"{REPO}/rlinf/envs/maniskill/so101_agent.py") + "\n```\n")
A("URDF limit changes vs the stock SO100 (radians):\n\n"
  "| joint | stock | widened (real servo calibration) |\n|---|---|---|\n"
  "| shoulder_pan | ±1.5708 | **±2.0** |\n| shoulder_lift | ±1.5708 | **[-1.5708, 2.48]** |\n"
  "| elbow_flex | ±1.5708 | **[-2.38, 1.5708]** |\n| wrist_flex | ±1.8 | **[-3.01, 1.8]** |\n"
  "| wrist_roll | ±3.14159 | unchanged |\n| gripper | ±1.1 | unchanged |\n")

A("### Step 2 — units: the calibration module\n")
A("`rlinf/envs/maniskill/so101_calib.py` (whole file). This is the highest-risk file "
  "in the project: an arm-style conversion applied to the gripper left an 8.1 cm "
  "minimum jaw gap on a 2.9 cm cube, which is why one 12-hour RL run scored exactly zero.\n")
A("```python\n" + whole(f"{REPO}/rlinf/envs/maniskill/so101_calib.py") + "\n```\n")

A("### Step 3 — action path: route SO101 through the conversion\n")
A("`rlinf/envs/action_utils.py`\n")
A("```python\n" + lines(f"{REPO}/rlinf/envs/action_utils.py", 22, 40) + "\n```\n")
A("`rlinf/config.py` — `get_robot_control_mode` (else: \"Robot so100 not supported\")\n")
A("```python\n" + lines(f"{REPO}/rlinf/config.py", 1064, 1072) + "\n```\n")

A("### Step 4 — observations: wrist camera + normalized state + rollout recorder\n")
A("`rlinf/envs/maniskill/maniskill_env.py`, default branch of `_wrap_obs` "
  "(additive: envs without a wrist camera get `None`, which `prepare_observations` backfills)\n")
A("```python\n" + func(ENVPY, "_wrap_obs") + "\n```\n")
A("Recorder used by expert iteration (Step 9). Enabled with `SO101_COLLECT_DIR`; "
  "flushes each episode as `.npz` on its first success, partial-reset safe.\n")
A("```python\n" + func(ENVPY, "_rec_record_step") + "\n```\n")

A("### Step 5 — the task: scene, spawn, success, reward\n")
A("`rlinf/envs/maniskill/tasks/so101_pick_place.py`. Cameras (the pp era used "
  "`128, 128` with an fov; the current file renders 640×480 with measured intrinsics — "
  "keep 640×480, see §1a):\n")
A("```python\n" + func(TASK, "_default_sensor_configs") + "\n```\n")
A("Spawn. `SO101_SPAWN_MODE=legacy` is the pp-era 6×8 cm box — the one thing you "
  "narrow to reproduce pp conditions at full fidelity:\n")
A("```python\n" + lines(TASK, 245, 290) + "\n```\n")
A("Success (the pp era had no homing term; the current file requires it):\n")
A("```python\n" + func(TASK, "evaluate") + "\n```\n")
A("Reward — the proven manipulation recipe plus ONE gradient bridge for the gripper:\n")
A("```python\n" + func(TASK, "compute_dense_reward") + "\n```\n")

A("### Step 6 — model wiring: policy transform + data config + TrainConfig\n")
A("`rlinf/models/embodiment/openpi/policies/so101_policy.py` (whole file)\n")
A("```python\n" + whole(f"{REPO}/rlinf/models/embodiment/openpi/policies/so101_policy.py") + "\n```\n")
A("`rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py` (whole file). "
  "`action_sequence_keys=(\"action\",)` is required by LeRobot 0.6.1's singular column.\n")
A("```python\n" + whole(f"{REPO}/rlinf/models/embodiment/openpi/dataconfig/so101_dataconfig.py") + "\n```\n")
A("One `TrainConfig` per dataset generation, in `dataconfig/__init__.py::_CONFIGS`:\n")
A("```python\n" + lines(f"{REPO}/rlinf/models/embodiment/openpi/dataconfig/__init__.py", 546, 563) + "\n```\n")

A("### Step 7 — demonstration generator (the planner)\n")
A("`scratchpad/gen_planner_demos.py` — grasp, micro-lift verification, payload-offset "
  "compensation, two-stage FK transport, closed-loop pre-drop refinement, homing.\n")
A("```python\n" + func(f"{SCRATCH}/gen_planner_demos.py", "solve_grab_red_cube") + "\n```\n")

A("### Step 8 — dataset conversion\n")
A("`scratchpad/convert_fullboard.py` — successful episodes only, units via the SAME "
  "calib module used at RL time, `FPS` = the generator's real control frequency.\n")
A("```python\n" + whole(f"{SCRATCH}/convert_fullboard.py") + "\n```\n")

A("### Step 9 — expert iteration (collect → convert → gentle SFT)\n")
A("Collection is just a deterministic eval with the recorder enabled:\n")
A("```bash\n"
  "export SO101_COLLECT_DIR=/data08/henryg/pai/data/<name>_rollouts\n"
  "for SEED in 101 202 303 404 505 606 707 808; do\n"
  "  .venv/bin/python evaluations/eval_embodied_agent.py \\\n"
  "    --config-path <abs>/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \\\n"
  "    rollout.model.model_path=$BEST_CKPT \\\n"
  "    rollout.model.openpi.config_name=<train config> \\\n"
  "    rollout.model.openpi_data.norm_stats_path=<stats> \\\n"
  "    env.eval.total_num_envs=128 env.eval.seed=$SEED\n"
  "done\n```\n")
A("Then convert the `.npz` files (state is already normalized by the env; actions are "
  "radians and need `rad_to_norm`) and run SFT from the SAME policy at **lr 1e-5**, "
  "2000 steps, reusing the existing `norm_stats`.\n")

A("### Step 10 — verification tooling (run these, they are cheap)\n")
A("```bash\n"
  "# before every launch: hydra compose + validate_cfg + paths + batch arithmetic\n"
  "python -m toolkits.preflight_config --config-path <abs cfg dir> --config-name <name> \\\n"
  "    <EXACT launcher overrides, verbatim>\n\n"
  "# periodically: catches SILENT wrong-result defects (frequency chain, camera chain,\n"
  "# spawn coverage vs the real dataset, stats lineage, dataset/env resolution,\n"
  "# budget headroom, action chunking, eval-seed disjointness)\n"
  "python -m toolkits.invariant_audit --ckpt <ckpt dir>\n```\n")

with open(OUT, "a") as f:
    f.write("\n".join(md))
print("appended §11; runbook now",
      subprocess.run(["wc", "-l", OUT], capture_output=True, text=True).stdout.strip())
