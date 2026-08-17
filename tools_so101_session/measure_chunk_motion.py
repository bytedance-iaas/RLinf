# STATUS: TOOL — 通用工具，与具体阶段无关。 测相邻动作块之间图像/关节的变化量
"""Full-coverage measurement: inter-chunk visual + joint motion across ALL
successful v4 demos (16 spawn cells, 420 episodes)."""
import glob, json, h5py, numpy as np

h5s = sorted(glob.glob("/data08/henryg/pai/data/v4_demos_cell_*/**/*.h5", recursive=True))
acc = {"front": {1: [], 5: [], 10: []}, "wrist": {1: [], 5: [], 10: []}}
per_cell = {}
zero5 = {"front": 0, "wrist": 0}; tot5 = 0
qd5 = []
n_ep = 0
for h5path in h5s:
    cell = h5path.split("/")[-2].replace("v4_demos_cell_", "")
    meta = json.load(open(h5path.replace(".h5", ".json")))
    ok = [e["episode_id"] for e in meta["episodes"] if e["success"]]
    f = h5py.File(h5path, "r")
    cell_front5 = []
    for eid in ok:
        t = f[f"traj_{eid}"]
        front = np.asarray(t["obs/sensor_data/3rd_view_camera/rgb"]).astype(np.int16)
        wrist = np.asarray(t["obs/sensor_data/wrist_camera/rgb"]).astype(np.int16)
        qpos = np.asarray(t["obs/agent/qpos"])
        T = min(len(front), len(wrist), len(qpos))
        for stride in (1, 5, 10):
            for cam, arr in (("front", front), ("wrist", wrist)):
                d = np.abs(arr[stride:T] - arr[:T - stride]).mean(axis=(1, 2, 3))
                acc[cam][stride].append(d.astype(np.float32))
                if cam == "front" and stride == 5:
                    cell_front5.append(d.astype(np.float32))
        d5f = np.abs(front[5:T] - front[:T - 5]).mean(axis=(1, 2, 3))
        d5w = np.abs(wrist[5:T] - wrist[:T - 5]).mean(axis=(1, 2, 3))
        zero5["front"] += int((d5f < 0.05).sum()); zero5["wrist"] += int((d5w < 0.05).sum())
        tot5 += len(d5f)
        qd5.append(np.abs(qpos[5:T] - qpos[:T - 5]).mean(axis=1).astype(np.float32))
        n_ep += 1
    f.close()
    if cell_front5:
        v = np.concatenate(cell_front5)
        per_cell[cell] = (float(v.mean()), float(np.percentile(v, 10)), len(v))
    print(f"done cell {cell}: {len(ok)} eps", flush=True)

print(f"\n=== FULL SWEEP: {n_ep} episodes, {tot5} five-step windows ===")
for cam in ("front", "wrist"):
    for s in (1, 5, 10):
        v = np.concatenate(acc[cam][s])
        print(f"{cam:6s} stride {s:2d}: mean={v.mean():.2f} median={np.median(v):.2f} "
              f"p10={np.percentile(v,10):.2f} p01={np.percentile(v,1):.2f} frac<0.1={(v<0.1).mean():.4f}")
print(f"\n5-step pairs visually identical (<0.05): front {zero5['front']/tot5:.5f}, wrist {zero5['wrist']/tot5:.5f}")
q = np.concatenate(qd5)
print(f"joint |dq| per 5 steps (rad): mean={q.mean():.4f} median={np.median(q):.4f} "
      f"p10={np.percentile(q,10):.4f} p01={np.percentile(q,1):.4f} frac<0.001={(q<0.001).mean():.4f}")
print("\n=== per-cell front stride-5 (mean, p10, n) ===")
for c in sorted(per_cell):
    m, p10, n = per_cell[c]
    print(f"  {c:26s} mean={m:5.2f} p10={p10:5.2f} n={n}")
