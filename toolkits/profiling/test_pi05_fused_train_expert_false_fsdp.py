"""FSDP1 integration smoke for trainable Pi0.5 fused prefix layers."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from test_pi05_fused_train_expert_false import _config
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.repo)
    os.environ["RLINF_FUSED_BACKWARD_COUNTER"] = "1"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29622")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    torch.cuda.set_device(local_rank)
    torch.manual_seed(20260722)

    from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi.fused_prefix_layer import (
        FusedGemmaPrefixLayer,
        _BackwardCounter,
    )

    model = get_model(_config(args.model))
    model.train()
    prefix = model.paligemma_with_expert.paligemma.language_model
    fused_before = sum(
        isinstance(layer, FusedGemmaPrefixLayer) for layer in prefix.layers
    )

    batch = 1
    device = torch.device(f"cuda:{local_rank}")
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
        ],
    }
    with torch.no_grad():
        _, rollout = model.predict_action_batch(
            observations, mode="train", compute_values=True
        )
    forward_inputs = rollout["forward_inputs"]

    policy_cfg = OmegaConf.create({"use_orig_params": False, "wrap_policy": {}})
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=policy_cfg,
        is_lora=False,
        model_type="openpi",
    )
    wrapped = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
        sharding_strategy=ShardingStrategy.NO_SHARD,
        use_orig_params=False,
        sync_module_states=True,
    )
    nested_fsdp = sum(isinstance(module, FSDP) for module in wrapped.modules())
    print(
        f"rank={rank}; fused_before_wrap={fused_before}; "
        f"fsdp_modules={nested_fsdp}; "
        f"train_expert_only={wrapped.module.config.train_expert_only}"
    )

    output = wrapped(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        use_cache=False,
    )
    loss = -output["logprobs"].float().mean() + output["values"].float().mean()
    loss.backward()
    torch.cuda.synchronize()

    grads = [parameter.grad for parameter in wrapped.parameters() if parameter.grad is not None]
    nonzero_grads = sum(torch.count_nonzero(grad).item() > 0 for grad in grads)
    all_finite = all(torch.isfinite(grad).all().item() for grad in grads)
    outputs_finite = all(
        torch.isfinite(value).all().item()
        for value in output.values()
        if torch.is_tensor(value)
    )
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-4)
    optimizer.step()
    passed = all(
        (
            fused_before == 18,
            nested_fsdp > 1,
            _BackwardCounter.bwd == 18,
            len(grads) > 0,
            nonzero_grads > 0,
            all_finite,
            outputs_finite,
        )
    )
    print(
        f"rank={rank}; outputs_finite={outputs_finite}; loss={loss.item():.6g}; "
        f"fused_fwd={_BackwardCounter.fwd}; fused_bwd={_BackwardCounter.bwd}"
    )
    print(
        f"rank={rank}; flat grads={len(grads)}; nonzero={nonzero_grads}; "
        f"finite={all_finite}; "
        f"peak_memory={torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )
    result = torch.tensor(int(passed), device=device)
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    if rank == 0:
        print("PASS" if result.item() else "FAIL")
    dist.destroy_process_group()
    return 0 if result.item() else 1


if __name__ == "__main__":
    raise SystemExit(main())
