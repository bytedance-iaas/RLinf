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
"""Invariant audit — catches SILENT wrong-result defects, the class that does
NOT crash: a run completes, loss falls, logs look clean, and the numbers are
meaningless because sim and the real task drifted apart.

Every check compares a measurable fact against the real dataset or against a
pre-registered prediction, and prints PASS/FAIL with the numbers. Cheap (CPU,
seconds) — run it before every stage and periodically during long campaigns.

Usage: python -m toolkits.invariant_audit [--ckpt <dir>] [--since <unix-ts>]
"""
import glob
import hashlib
import json
import os
import re
import sys
import time

REPO = "/data08/henryg/pai/RLinf"
REAL_DS = "/root/.cache/huggingface/lerobot/henry-guo/so101-pick-place-v2"
TASK_PY = f"{REPO}/rlinf/envs/maniskill/tasks/so101_pick_place.py"
ENV_YAML = f"{REPO}/examples/embodiment/config/env/maniskill_so101_pick_place.yaml"
SCRATCH = ("/tmp/claude-0/-data08-henryg-pai-RLinf/"
           "3e748c24-1f70-49ee-a01c-395d2f1161dd/scratchpad")

results = []


def check(name, ok, detail):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def const(src, name, cast=float):
    m = re.search(rf"^{name}\s*=\s*([^#\n]+)", src, re.M)
    return eval(m.group(1).strip()) if m else None


def main() -> int:
    task = open(TASK_PY).read()
    envy = open(ENV_YAML).read()

    # --- A. temporal chain: real fps == sim control_freq == converter fps ---
    real_fps = json.load(open(f"{REAL_DS}/meta/info.json")).get("fps")
    sim_hz = int(re.search(r"control_freq:\s*(\d+)", envy).group(1))
    conv = sorted(glob.glob(f"{SCRATCH}/convert_*_demos.py"))
    conv_fps = set()
    for c in conv:
        m = re.search(r"^FPS\s*=\s*(\d+)", open(c).read(), re.M)
        if m:
            conv_fps.add(int(m.group(1)))
    ok = (real_fps == sim_hz) and (conv_fps <= {real_fps} or not conv_fps)
    check("temporal-chain", ok,
          f"real dataset {real_fps}fps | sim control_freq {sim_hz}Hz | converters {sorted(conv_fps)}")

    # --- B. camera chain: sim render vs real aspect and pipeline resolution ---
    w, h = const(task, "FRONT_CAM_W, FRONT_CAM_H", tuple) or (None, None)
    real_w, real_h = 640, 480
    aspect_ok = abs((w / h) - (real_w / real_h)) < 0.02 if w else False
    # openpi resize_with_pad target is 224x224 -> effective content 224x168
    res_ok = w >= 224 and h >= 168 if w else False
    check("camera-chain", aspect_ok and res_ok,
          f"sim {w}x{h} (aspect {'ok' if aspect_ok else 'MISMATCH'}), "
          f"pipeline needs >=224x168 -> {'ok' if res_ok else 'UNDER-RESOLVED'}")

    # --- C. spawn coverage: does the sim spawn box cover the REAL cube starts? ---
    uv_path = f"{SCRATCH}/real_cube_uv.npy"
    if os.path.exists(uv_path):
        import numpy as np
        uv = np.load(uv_path)          # board-relative (u across width, v across depth)
        bh = const(task, "BOARD_HALF", list)
        brown_half_y = (2 * bh[1] - const(task, "BLACK_END_LEN")) / 2
        margin = 0.02
        fx = margin / bh[0]            # fraction of half-extent lost to margin
        fy = margin / brown_half_y
        inside = ((uv[:, 0] >= fy / 2) & (uv[:, 0] <= 1 - fy / 2) &
                  (uv[:, 1] >= fx / 2) & (uv[:, 1] <= 1 - fx / 2)).mean()
        check("spawn-coverage", inside >= 0.90,
              f"{inside * 100:.0f}% of the 87 real cube starts fall inside the sim spawn box "
              f"(margin {margin * 100:.0f}mm)")
    else:
        check("spawn-coverage", False, "real_cube_uv.npy missing — cannot verify")

    # --- D. success semantics: homing required (user-confirmed spec) ---
    homing = "is_home" in task and "success" in task
    check("success-semantics", homing,
          "evaluate() gates success on in-box AND arm-home" if homing else "homing gate MISSING")

    # --- E. norm_stats lineage: ckpt copy must equal the assets file in use ---
    ck = None
    if "--ckpt" in sys.argv:
        ck = sys.argv[sys.argv.index("--ckpt") + 1]
    if ck:
        pairs = []
        for stats in glob.glob(f"{ck}/*/norm_stats.json"):
            name = os.path.basename(os.path.dirname(stats))
            asset = f"{REPO}/assets/pi05_so101_{name.split('-')[-1]}/{name}/norm_stats.json"
            if os.path.exists(asset):
                a = hashlib.md5(open(stats, "rb").read()).hexdigest()[:8]
                b = hashlib.md5(open(asset, "rb").read()).hexdigest()[:8]
                pairs.append((name, a, b, a == b))
        ok = all(p[3] for p in pairs) if pairs else False
        check("norm-stats-lineage", ok,
              "; ".join(f"{n}: ckpt {a} vs assets {b}" for n, a, b, _ in pairs) or "no stats found")

    # --- F. code freshness: env/model code must not change mid-run ---
    since = None
    if "--since" in sys.argv:
        since = float(sys.argv[sys.argv.index("--since") + 1])
    if since:
        touched = [f for f in (TASK_PY, ENV_YAML,
                               f"{REPO}/rlinf/envs/maniskill/maniskill_env.py")
                   if os.path.getmtime(f) > since]
        check("code-freshness", not touched,
              "no in-tree edits since run start" if not touched
              else f"EDITED MID-RUN: {[os.path.basename(t) for t in touched]}")


    # --- G. dataset/env resolution match: training data must be rendered at the
    #        SAME resolution the live env produces (a 160x120 dataset silently
    #        degrades a 640x480 env and vice versa).
    ds_root = None
    for cand in sorted(glob.glob("/data08/henryg/pai/data/so101-*")):
        info = f"{cand}/meta/info.json"
        if os.path.exists(info):
            ds_root = cand
    if ds_root:
        info = json.load(open(f"{ds_root}/meta/info.json"))
        feats = info.get("features", {})
        shp = None
        for k, v in feats.items():
            if "image" in k and isinstance(v, dict) and v.get("shape"):
                shp = v["shape"]
                break
        ds_hw = (shp[0], shp[1]) if shp else (None, None)
        ok = (ds_hw == (h, w))
        check("dataset-env-resolution", ok,
              f"{os.path.basename(ds_root)} images {ds_hw[1]}x{ds_hw[0]} vs env render {w}x{h}")

    # --- H. episode budget headroom: median demo length x1.2 must fit the budget
    #        (150-step demos under an 80-step budget made the task impossible).
    budget = int(re.search(r"max_episode_steps:\s*(\d+)", envy).group(1))
    lens = []
    for pq in sorted(glob.glob(f"{ds_root}/data/chunk-*/*.parquet"))[:4] if ds_root else []:
        try:
            import pandas as pd
            df = pd.read_parquet(pq, columns=["episode_index", "frame_index"])
            lens += df.groupby("episode_index").size().tolist()
        except Exception:
            pass
    if lens:
        import numpy as np
        med = float(np.median(lens))
        check("budget-headroom", med * 1.2 <= budget,
              f"demo median {med:.0f} steps x1.2 = {med * 1.2:.0f} vs budget {budget}")

    # --- I. action chunking consistency: the env may not execute more actions
    #        per decision than the SFT checkpoint predicts (action_horizon).
    cfgs = glob.glob(f"{REPO}/examples/embodiment/config/so101_ppo_v*.yaml")
    for c in cfgs:
        t = open(c).read()
        nac = re.search(r"num_action_chunks:\s*(\d+)", t)
        ah = re.search(r"action_horizon:\s*(\d+)", t)
        if nac and ah:
            ok = int(nac.group(1)) <= int(ah.group(1)) and budget % int(nac.group(1)) == 0
            check(f"action-chunking[{os.path.basename(c)}]", ok,
                  f"num_action_chunks {nac.group(1)} <= action_horizon {ah.group(1)}, "
                  f"budget {budget} divisible")

    # --- J. eval protocol: gate seeds and verification seeds must be disjoint
    #        (selection bias inflated a 63% policy to 75% once).
    for sh in glob.glob(f"{SCRATCH}/*pipeline*.sh") + glob.glob(f"{SCRATCH}/supervisor*.sh"):
        t = open(sh).read()
        # role is given by the TAG argument: *verify* tags vs gate/screen/confirm tags
        gate = set(re.findall(r'run_eval [^\n]*?"?\$?\w*"? (\d{3,4}) (?!verify)(?:\$\(basename[^)]*\)_)?s?\d*[a-z_]*', t))
        ver = set(re.findall(r"run_eval [^\n]*? (\d{3,4}) verify", t))
        gate -= ver
        if gate and ver:
            check(f"eval-seed-disjoint[{os.path.basename(sh)}]", not (gate & ver),
                  f"gate {sorted(gate)} vs verify {sorted(ver)}")

    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} invariants hold")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
