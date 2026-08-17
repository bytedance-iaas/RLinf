# STATUS: ACTIVE — 当前流程在用。 部署备用路线：RLinf 直接起 websocket 服务
"""Policy inference server for the real SO101.

Runs on the training node; the laptop that owns the arm connects over a
websocket, sends one observation, and gets back an action chunk.

Why it loads the checkpoint directly instead of going through sft2deploy: the
offline sim2real check already proves this exact loading path produces correct
actions from real camera frames, and sft2deploy additionally needs two
old-format reference model directories we do not have. Fewer moving parts
between the number we validated and the robot.

Server:
    python tools_so101_session/deploy_policy_server.py \
        --ckpt /data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000 \
        --port 8000

Client (laptop) sends a dict and receives {"actions": (chunks, 6)}:
    observation/image        (480, 640, 3) uint8   front camera
    observation/wrist_image  (480, 640, 3) uint8   wrist camera  -- REQUIRED, see below
    observation/state        (6,) float             LeRobot normalized units
    prompt                   str

Returns 10 actions per call, matching training. Execute all of them.

Both cameras matter now. Before co-training, dropping the wrist channel changed
nothing (4.47 -> 4.59 on the offline metric) because the sim wrist camera pointed
at the robot's own body. After co-training the model has seen real wrist images
and uses them: same checkpoint scores 0.90 with both cameras and 1.58 with the
front alone. Sending one camera roughly doubles the action error.
"""

import argparse
import sys

import numpy as np
import torch

sys.path.insert(0, "/data08/henryg/pai/RLinf")

DEFAULT_STATS = "/data08/henryg/pai/RLinf/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json"


class SO101Policy:
    """Adapts the RLinf openpi model to the openpi_client BasePolicy interface."""

    def __init__(self, ckpt: str, config_name: str, norm_stats: str, chunks: int, prompt: str):
        from omegaconf import OmegaConf

        from rlinf.models import get_model

        cfg = OmegaConf.create(
            {
                "model_type": "openpi",
                "model_path": ckpt,
                "precision": None,
                "num_action_chunks": chunks,
                "action_dim": 6,
                "num_steps": 4,
                "add_value_head": True,   # co-trained checkpoints carry one; harmless if absent
                "policy_setup": "so100",
                "is_lora": False,
                "load_to_device": True,
                "openpi": {
                    "config_name": config_name,
                    # MUST be set explicitly. In the YAML configs this is
                    # `${..num_action_chunks}`, so training and eval ran with 10.
                    # A hand-built config like this one silently falls back to the
                    # dataclass default of 5 and would emit half the horizon.
                    "action_chunk": chunks,
                    "num_images_in_input": 2,
                    "action_horizon": 10,
                    "noise_method": "flow_noise",
                    "value_after_vlm": True,
                    "joint_logprob": True,
                },
                "openpi_data": {"norm_stats_path": norm_stats},
            }
        )
        self.model = get_model(cfg)
        self.model.eval()
        self.prompt = prompt
        self.chunks = chunks

    def infer(self, obs: dict) -> dict:
        front = np.asarray(obs["observation/image"])
        wrist = np.asarray(obs["observation/wrist_image"])
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        env_obs = {
            "main_images": torch.from_numpy(front[None]),
            "wrist_images": torch.from_numpy(wrist[None]),
            "states": torch.from_numpy(state[None]),
            "task_descriptions": [obs.get("prompt", self.prompt)],
            "extra_view_images": None,   # obs_processor reads this key unconditionally
        }
        with torch.no_grad():
            actions, _ = self.model.predict_action_batch(env_obs=env_obs, mode="eval")
        return {"actions": actions[0].detach().float().cpu().numpy()}

    def reset(self) -> None:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="pi05_so101_v15")
    ap.add_argument("--norm-stats", default=DEFAULT_STATS)
    ap.add_argument("--chunks", type=int, default=10,
                    help="actions returned per call. 10 is what training and "
                         "sim eval used (the YAML interpolates action_chunk from "
                         "num_action_chunks); do not lower it.")
    ap.add_argument("--prompt", default="Grab the red cube")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    print(f"loading {args.ckpt}", flush=True)
    policy = SO101Policy(args.ckpt, args.config_name, args.norm_stats, args.chunks, args.prompt)

    # one self-test before accepting connections, so a broken load fails here
    dummy = {
        "observation/image": np.zeros((480, 640, 3), np.uint8),
        "observation/wrist_image": np.zeros((480, 640, 3), np.uint8),
        "observation/state": np.zeros(6, np.float32),
        "prompt": args.prompt,
    }
    out = policy.infer(dummy)["actions"]
    print(f"self-test OK: returns {out.shape} {out.dtype}", flush=True)

    print(f"serving on {args.host}:{args.port}", flush=True)
    WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port,
        metadata={"checkpoint": args.ckpt, "chunks": args.chunks, "prompt": args.prompt},
    ).serve_forever()


if __name__ == "__main__":
    main()
