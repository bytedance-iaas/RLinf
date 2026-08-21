# STATUS: ACTIVE — 当前流程在用。 部署路线 B：Mac 侧控制回路（真机 + 双相机 + websocket 推理）
"""Mac-side control loop for the real robot, talking to deploy_policy_server.py.

Why this exists rather than lerobot's own robot_client: that one speaks to
lerobot's policy_server, which loads only LeRobot-layout checkpoints. Exporting
to that layout did not reproduce the policy (0.83 against a 0.70 reference), so
deployment goes through the RLinf checkpoint directly -- the same code path the
0.70 was measured on. The cost is this file.

Hardware is still lerobot's: SO101Follower drives the arm and OpenCVCamera reads
the frames, so nothing here reimplements a robot driver.

  --dry-run          robot and cameras OFFLINE. Sends synthetic observations and
                     reports round-trip latency and action shape. This is stage 1
                     of the deployment sequence: prove the loop before the arm
                     can move.
  --max-rel N        clamp per-step joint change to N normalised units. Stage 2
                     asks for 2. This is passed to the follower, which enforces
                     it in the driver rather than here.
  --episodes N       run N episodes, logging each one for failure classification.

Every episode is written to a .jsonl: the observation state, the action chunk
returned, and the timing. Stage 5 asks for failures to be sorted into four kinds
(no grasp / dropped / misplaced / did not return home) and that is not
reconstructable from memory afterwards.

    python tools_so101_session/robot_client_websocket.py --dry-run
    python tools_so101_session/robot_client_websocket.py \
        --port-name /dev/tty.usbmodem58FA0828301 --robot-id my_so101 \
        --front-cam 0 --wrist-cam 1 --max-rel 2 --episodes 5
"""

import argparse
import json
import threading
import time
from datetime import datetime, timezone

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PROMPT = "Grab the red cube"
HZ = 30


def build_robot(args):
    """lerobot owns the hardware. Returns a connected follower with two cameras."""
    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SOFollower, SOFollowerConfig

    cams = {
        "front": OpenCVCameraConfig(index_or_path=args.front_cam, width=640, height=480, fps=HZ),
        "wrist": OpenCVCameraConfig(index_or_path=args.wrist_cam, width=640, height=480, fps=HZ),
    }
    cfg = SOFollowerConfig(
        port=args.port_name, id=args.robot_id, cameras=cams,
        # The driver enforces the clamp, so a bad action chunk cannot produce a
        # large jump even if this script has a bug.
        max_relative_target=args.max_rel,
    )
    robot = SOFollower(cfg)
    robot.connect()
    return robot


def obs_to_request(obs):
    """lerobot's observation dict -> what the policy server expects.

    The follower reports joints as '<motor>.pos' in normalised units (arm ±100,
    gripper 0–100), which is exactly the convention the policy was trained in --
    no conversion, only ordering into the 6-vector the model expects.
    """
    state = np.array([obs[f"{j}.pos"] for j in JOINTS], dtype=np.float32)
    return {
        "observation/image": np.asarray(obs["front"]),
        "observation/wrist_image": np.asarray(obs["wrist"]),
        "observation/state": state,
        "prompt": PROMPT,
    }


def fake_obs():
    return {
        **{f"{j}.pos": 0.0 for j in JOINTS},
        "front": np.zeros((480, 640, 3), np.uint8),
        "wrist": np.zeros((480, 640, 3), np.uint8),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--port-name", help="serial port of the follower arm")
    ap.add_argument("--robot-id", default="so101")
    ap.add_argument("--front-cam", default=0)
    ap.add_argument("--wrist-cam", default=1)
    ap.add_argument("--max-rel", type=float, default=None,
                    help="clamp per-step joint change, in normalised units (stage 2 asks for 2)")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=640, help="per episode, at 30 Hz = ~21 s")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    print(f"connected to {args.host}:{args.port}", flush=True)

    robot = None if args.dry_run else build_robot(args)
    if args.dry_run:
        print("DRY RUN: robot and cameras offline, sending synthetic observations", flush=True)
    elif args.max_rel is None:
        print("WARNING: no --max-rel. Stage 2 of the deployment sequence asks for 2.", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logpath = args.log or f"so101_run_{stamp}.jsonl"
    log = open(logpath, "w")
    print(f"logging to {logpath}", flush=True)

    for ep in range(args.episodes):
        print(f"\n=== episode {ep} ===", flush=True)
        rtts, t_ep = [], time.perf_counter()
        step = 0
        pending = {}          # holds the next chunk while the current one executes

        def request_chunk():
            """Ask for the next chunk in the background.

            Inference costs ~190 ms even on loopback, and a 10-action chunk only
            covers 10/30 s = 333 ms. Serialising the two gives 524 ms per chunk,
            i.e. 19 Hz for a policy trained and measured at 30 -- a temporal
            mismatch of exactly the kind the recipe warns about. Overlapping them
            is what lerobot's async client does; this is the minimal version.
            """
            obs = fake_obs() if args.dry_run else robot.get_observation()
            req = obs_to_request(obs)
            t0 = time.perf_counter()
            act = np.asarray(policy.infer(req)["actions"], dtype=np.float32)
            pending["rtt"] = (time.perf_counter() - t0) * 1e3
            pending["actions"] = act
            pending["state"] = req["observation/state"]

        request_chunk()                       # prime the pump
        while step < args.max_steps:
            actions, rtt, state = pending["actions"], pending["rtt"], pending["state"]
            rtts.append(rtt)
            if actions.ndim != 2 or actions.shape[1] != 6:
                raise SystemExit(f"unexpected action shape {actions.shape}, expected (chunk, 6)")

            log.write(json.dumps({
                "ep": ep, "step": step, "rtt_ms": round(rtt, 1),
                "state": state.tolist(), "actions": actions.tolist(),
            }) + "\n")

            # Fetch the NEXT chunk while this one is being executed. The
            # observation it is based on is the one current at fetch time, which
            # is what makes the overlap sound: the policy still sees a fresh frame.
            fetcher = threading.Thread(target=request_chunk, daemon=True)
            fetcher.start()

            # Execute the whole chunk. The horizon the policy was trained and
            # measured at is 10; truncating it is a different policy (§4.9).
            t_chunk = time.perf_counter()
            for k, a in enumerate(actions):
                if robot is not None:
                    robot.send_action({f"{j}.pos": float(v) for j, v in zip(JOINTS, a)})
                # absolute pacing, so a slow send does not accumulate drift
                target = t_chunk + (k + 1) / HZ
                time.sleep(max(0.0, target - time.perf_counter()))
                step += 1
                if step >= args.max_steps:
                    break

            if step % 100 < len(actions):
                print(f"  step {step:4d}  rtt {rtt:6.1f} ms  chunk {actions.shape}", flush=True)

            fetcher.join(timeout=5.0)
            if fetcher.is_alive():
                raise SystemExit("next chunk did not arrive within 5 s — server stalled")

        dur = time.perf_counter() - t_ep
        r = np.array(rtts)
        # The first chunk cannot be overlapped -- nothing is executing yet -- so a
        # short run reads low. Report the steady-state rate too, which is what a
        # full episode actually runs at.
        steady = (step - len(actions)) / max(dur - r[0] / 1000.0 - len(actions) / HZ, 1e-6) \
            if step > len(actions) else float("nan")
        print(f"  {step} steps in {dur:.1f}s ({step/dur:.1f} Hz overall, "
              f"{steady:.1f} Hz steady-state) | "
              f"rtt mean {r.mean():.0f} ms  p95 {np.percentile(r, 95):.0f} ms  max {r.max():.0f} ms",
              flush=True)
        # A chunk of 10 covers 10/30 s. If the round trip exceeds that, the arm
        # runs out of actions before the next chunk arrives and stutters.
        budget = 1000.0 * 10 / HZ
        if r.mean() > budget:
            print(f"  ⚠ mean rtt {r.mean():.0f} ms exceeds the {budget:.0f} ms a 10-action "
                  f"chunk covers — expect gaps between chunks", flush=True)

        if robot is not None and ep + 1 < args.episodes:
            input("  reposition the object, then press Enter for the next episode…")

    log.close()
    if robot is not None:
        robot.disconnect()
    print(f"\ndone. per-step log: {logpath}", flush=True)


if __name__ == "__main__":
    main()
