# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Launch preflight: compose the EXACT hydra config a launcher would use and
run RLinf's validate_cfg on it, plus existence checks for referenced paths —
all on CPU in seconds, so wiring errors (bad override paths, divisibility
asserts, missing env vars, missing checkpoints/norm_stats) are caught BEFORE
burning a GPU launch.

Usage:
  python -m toolkits.preflight_config --config-path <abs dir> \
      --config-name <name> [hydra overrides...]

Exit 0 = safe to launch; nonzero = fix before launching.
"""
import os
import sys

from hydra import compose, initialize_config_dir


def main() -> int:
    args = sys.argv[1:]
    cfg_dir = cfg_name = None
    overrides = []
    i = 0
    while i < len(args):
        if args[i] == "--config-path":
            cfg_dir = os.path.abspath(args[i + 1])
            i += 2
        elif args[i] == "--config-name":
            cfg_name = args[i + 1]
            i += 2
        else:
            overrides.append(args[i])
            i += 1
    if not cfg_dir or not cfg_name:
        print("PREFLIGHT FAIL: --config-path and --config-name required")
        return 2

    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name=cfg_name, overrides=overrides)

    from rlinf.config import validate_cfg

    cfg = validate_cfg(cfg)
    print("validate_cfg OK")

    # existence checks for the paths this run depends on
    problems = []
    for dotted in (
        "actor.model.model_path",
        "rollout.model.model_path",
        "actor.model.openpi_data.norm_stats_path",
        "rollout.model.openpi_data.norm_stats_path",
    ):
        node = cfg
        try:
            for part in dotted.split("."):
                node = node[part]
        except Exception:
            continue
        if isinstance(node, str) and node and not os.path.exists(node):
            problems.append(f"missing path: {dotted} = {node}")
    # batch arithmetic (embodied): these asserts otherwise fire only at RUNTIME
    # inside workers (e.g. "720 is not divisible by 256").
    try:
        envs = int(cfg.env.train.total_num_envs)
        steps = int(cfg.env.train.max_steps_per_rollout_epoch)
        chunks = int(cfg.actor.model.num_action_chunks)
        gbs = int(cfg.actor.global_batch_size)
        mbs = int(cfg.actor.micro_batch_size)
        world = 8  # single-node H200 box
        samples = envs * (steps // chunks)
        if steps % chunks != 0:
            problems.append(f"steps {steps} % num_action_chunks {chunks} != 0")
        if samples % gbs != 0:
            problems.append(
                f"samples/epoch {samples} % global_batch {gbs} != 0 "
                f"(-> {samples / gbs:.2f} updates/epoch; keep it an integer, "
                f"ideally exactly 1 for conservative-PPO warm starts)"
            )
        per_rank = samples // world
        if per_rank % mbs != 0:
            problems.append(f"per-rank samples {per_rank} % micro_batch {mbs} != 0")
        updates = samples // gbs if gbs and samples % gbs == 0 else None
        if updates is not None:
            print(f"batch arithmetic: {samples} samples/epoch -> {updates} update(s)/epoch")
    except Exception as e:  # reasoning/non-embodied configs lack these keys
        print(f"(batch arithmetic skipped: {e})")

    if problems:
        for p in problems:
            print(f"PREFLIGHT FAIL: {p}")
        return 1
    print("PREFLIGHT OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
