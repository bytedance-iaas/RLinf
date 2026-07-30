"""Smoke-test fused Pi0.5 training with ``train_expert_only=False``.

The test loads a real Pi0.5 LIBERO checkpoint, replaces all prefix Gemma
layers with the fused implementation, runs the same action/log-prob path used
by PPO, and verifies that a prefix VLM parameter receives a finite non-zero
gradient and changes after an optimizer step.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from omegaconf import OmegaConf


def _config(model_path: str):
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": model_path,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 3,
            "add_value_head": True,
            "load_to_device": True,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": 3,
                "train_expert_only": False,
                "enable_fused_prefix": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": True,
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": None,
                "use_dsrl": False,
            },
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.repo)
    os.environ["RLINF_FUSED_BACKWARD_COUNTER"] = "1"
    torch.manual_seed(20260722)

    from openpi.models import model as openpi_model

    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi.fused_prefix_layer import (
        FusedGemmaPrefixLayer,
        _BackwardCounter,
    )

    model = get_model(_config(args.model))
    model.train()
    prefix = model.paligemma_with_expert.paligemma.language_model
    expert = model.paligemma_with_expert.gemma_expert.model
    fused_layers = sum(isinstance(layer, FusedGemmaPrefixLayer) for layer in prefix.layers)
    trainable_tensors = sum(parameter.requires_grad for parameter in prefix.parameters())
    total_tensors = sum(1 for _ in prefix.parameters())
    trainable_elements = sum(
        parameter.numel() for parameter in prefix.parameters() if parameter.requires_grad
    )
    total_elements = sum(parameter.numel() for parameter in prefix.parameters())
    prefix_weight = prefix.layers[0].self_attn.q_proj.weight
    expert_weight = expert.layers[0].self_attn.q_proj.weight

    print(f"config.train_expert_only={model.config.train_expert_only}")
    print(f"fused prefix layers={fused_layers}/{len(prefix.layers)}")
    print(
        f"prefix trainable tensors={trainable_tensors}/{total_tensors}; "
        f"elements={trainable_elements}/{total_elements}"
    )
    print(
        "requires_grad: "
        f"prefix_q={prefix_weight.requires_grad}, expert_q={expert_weight.requires_grad}"
    )

    device = torch.device("cuda")
    batch = 1
    observations = {
        "main_images": torch.randint(
            0, 256, (batch, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        "wrist_images": torch.randint(
            0, 256, (batch, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        "extra_view_images": None,
        "states": torch.randn(batch, 8, device=device),
        "task_descriptions": [
            "pick up the black bowl and place it on the plate"
        ]
        * batch,
    }

    with torch.no_grad():
        _, rollout = model.predict_action_batch(
            observations, mode="train", compute_values=True
        )
    forward_inputs = rollout["forward_inputs"]
    transformed = model.obs_processor(observations)
    transformed = model.precision_processor(
        model.input_transform(transformed, transpose=False)
    )
    openpi_observation = openpi_model.Observation.from_dict(transformed)
    images, image_masks, lang_tokens, lang_masks, states = model._preprocess_observation(
        openpi_observation, train=False
    )

    model.zero_grad(set_to_none=True)
    log_prob, value, entropy = model.get_log_prob_value(
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        states,
        forward_inputs["chains"],
        forward_inputs["denoise_inds"],
        compute_values=True,
    )
    loss = -log_prob.float().mean() + value.float().mean()
    loss.backward()
    torch.cuda.synchronize()

    prefix_grad = prefix_weight.grad
    expert_grad = expert_weight.grad
    prefix_grad_norm = (
        prefix_grad.float().norm().item() if prefix_grad is not None else 0.0
    )
    expert_grad_norm = (
        expert_grad.float().norm().item() if expert_grad is not None else 0.0
    )
    before = prefix_weight.detach().flatten()[:4096].float().clone()
    optimizer = torch.optim.SGD([prefix_weight], lr=0.1)
    optimizer.step()
    after = prefix_weight.detach().flatten()[:4096].float()
    changed = torch.count_nonzero(after != before).item()
    max_delta = (after - before).abs().max().item()

    finite = all(
        torch.isfinite(item).all().item() for item in (log_prob, value, entropy, loss)
    )
    passed = all(
        (
            model.config.train_expert_only is False,
            fused_layers == len(prefix.layers) == 18,
            trainable_tensors == total_tensors,
            prefix_weight.requires_grad,
            prefix_grad is not None,
            torch.isfinite(prefix_grad).all().item(),
            prefix_grad_norm > 0,
            expert_grad is not None,
            torch.isfinite(expert_grad).all().item(),
            expert_grad_norm > 0,
            _BackwardCounter.bwd > 0,
            changed > 0,
            finite,
        )
    )
    print(
        f"outputs finite={finite}; loss={loss.item():.6g}; "
        f"log_prob_shape={tuple(log_prob.shape)} value_shape={tuple(value.shape)}"
    )
    print(
        f"fused counter: forward={_BackwardCounter.fwd}, "
        f"backward={_BackwardCounter.bwd}"
    )
    print(
        f"grad norms: prefix_q={prefix_grad_norm:.6e}, "
        f"expert_q={expert_grad_norm:.6e}"
    )
    print(f"prefix optimizer update: changed={changed}/4096, max_delta={max_delta:.6e}")
    print(f"peak allocated memory={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
