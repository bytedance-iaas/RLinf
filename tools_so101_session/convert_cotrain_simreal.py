# STATUS: ACTIVE — 当前流程在用。 阶段 G，协同训练数据集第一轮（真机全 87 集）
"""Build the sim+real co-training dataset.

Why: the sim-trained policy scores 4.47 on the offline real-observation check
(worse than holding still) while the real-trained policy scores 0.22 on the same
check. The gap is visual-domain, symmetric, and neither policy has seen the
other's images. Co-training is the cheapest way to give one policy both.

Method: copy the sim dataset and append the real episodes, upsampled, so real
frames carry roughly a fifth of the gradient instead of a sixteenth.

The real episodes need NO unit conversion -- they are already 640x480 at 30 fps
with 6-dim LeRobot-normalized state/action, exactly like the converted sim ones.
That is what makes this cheap. (The sim sources do need conversion: recorder npz
stores normalized state but RADIAN actions, h5 stores radians for both.)

The real dataset is LeRobot v3.0 (one parquet for all episodes, videos
concatenated per chunk with per-episode timestamps in meta/episodes); the sim
dataset is v2.0 (one file per episode). Both layouts are handled.
"""

import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, "/data08/henryg/pai/RLinf")
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

SIM_ROOT = "/data08/henryg/pai/data/so101-sim-demos-v10"
REAL_ROOT = "/data08/henryg/pai/data/so101-pick-place-v1-trimmed"   # == henry-guo/so101-pick-place-v2
OUT_REPO = "so101-cotrain-v14"
OUT_ROOT = "/data08/henryg/pai/data/so101-cotrain-v14"
TASK = "Grab the red cube"
# 87 x REPEAT real episodes against 1292 sim ones. Frame-wise, real episodes are
# ~575 frames vs the sim median 335, so REPEAT=2 gives 174 x 575 = 100k real
# frames against 433k sim = ~19% of the gradient.
REPEAT = 2
# Length filter for the REAL episodes only. 580 is the SIM budget (it drops
# overlong "still struggling" planner rollouts), but human teleop is slower:
# the 87 real episodes run 395-825 frames, median 575, so 580 would silently
# discard 46% of them (47/87 kept) and quietly halve the point of this run.
# 1000 keeps all 87 and still guards against a corrupt outlier.
MIN_LEN, MAX_LEN = 80, 1000


def read_real_episode(root, idx):
    """Decode one real episode into memory: (states, actions, front_frames, wrist_frames).

    Reads the video SEQUENTIALLY. The real dataset is LeRobot v3.0, where every
    episode lives inside one big concatenated mp4 (29k frames), and the obvious
    `reader.get_data(i + offset)` seeks per frame -- that is what made the first
    attempt run at 0.73 episodes/min instead of the ~2 the encoder alone costs,
    and it hit the 4 h timeout at 176/261 episodes. Iterating and skipping is
    ~3x faster, and decoding once here lets the caller write the episode REPEAT
    times without paying the decode again.
    """
    import imageio.v2 as imageio
    import pandas as pd

    meta = pd.read_parquet(glob.glob(os.path.join(root, "meta", "episodes", "*", "*.parquet"))[0])
    row = meta[meta["episode_index"] == idx].iloc[0]
    df = pd.read_parquet(
        os.path.join(root, "data", f"chunk-{int(row['data/chunk_index']):03d}",
                     f"file-{int(row['data/file_index']):03d}.parquet")
    )
    df = df[df["episode_index"] == idx]
    states = np.stack(df["observation.state"].to_numpy())
    actions = np.stack(df["action"].to_numpy())
    n_meta = len(states)
    fps = json.load(open(os.path.join(root, "meta", "info.json")))["fps"]

    frames = {}
    for key in ("observation.images.front", "observation.images.wrist"):
        path = os.path.join(root, "videos", key,
                            f"chunk-{int(row[f'videos/{key}/chunk_index']):03d}",
                            f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4")
        off = int(round(float(row[f"videos/{key}/from_timestamp"]) * fps))
        reader = imageio.get_reader(path)
        out = []
        for i, frame in enumerate(reader):
            if i < off:
                continue
            out.append(frame)
            if len(out) >= n_meta:
                break
        reader.close()
        frames[key] = out
    n = min(n_meta, len(frames["observation.images.front"]), len(frames["observation.images.wrist"]))
    return states[:n], actions[:n], frames["observation.images.front"][:n], frames["observation.images.wrist"][:n]


def main():
    shutil.rmtree(OUT_ROOT, ignore_errors=True)
    shutil.copytree(SIM_ROOT, OUT_ROOT)
    ds = LeRobotDataset(repo_id=OUT_REPO, root=OUT_ROOT)
    n_sim = ds.meta.total_episodes
    print(f"sim base: {n_sim} episodes", flush=True)

    n_real_eps = json.load(open(os.path.join(REAL_ROOT, "meta", "info.json")))["total_episodes"]
    added = 0
    for ep in range(n_real_eps):
        try:
            states, actions, front, wrist = read_real_episode(REAL_ROOT, ep)
        except Exception as exc:  # noqa: BLE001
            print(f"  real ep {ep} unreadable: {exc}", flush=True)
            continue
        n = len(states)
        if not (MIN_LEN <= n <= MAX_LEN):
            print(f"  real ep {ep} skipped: {n} frames outside [{MIN_LEN},{MAX_LEN}]", flush=True)
            continue
        # decoded once, written REPEAT times -- the whole point of holding it in memory
        for _ in range(REPEAT):
            for i in range(n):
                ds.add_frame({
                    "observation.state": states[i].astype(np.float32),
                    "action": actions[i].astype(np.float32),
                    "observation.images.front": front[i],
                    "observation.images.wrist": wrist[i],
                    "task": TASK,
                })
            ds.save_episode()
            added += 1
        del front, wrist
        if ep % 10 == 0:
            print(f"  real ep {ep}/{n_real_eps} done, {added} episodes appended", flush=True)

    total = n_sim + added
    print(f"DONE: {total} episodes (sim {n_sim} + real {added} = {added // max(REPEAT,1)} unique x {REPEAT}) -> {OUT_ROOT}")
    print(f"real share of episodes: {added / max(total, 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
