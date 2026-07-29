# Pi0.5 + LIBERO 4×H20 Quick Start

按照本指南，你可以启动 `rlinf-0`、准备 Pi0.5 模型，并在单机 4×H20 上运行
async + colocated PPO。源码、模型、日志和 checkpoint 都放在 `/workspace` PVC 中，
重建 Pod 后仍会保留。

## 运行环境

| 项目 | 值 |
|---|---|
| Kubernetes namespace | `rlinf` |
| StatefulSet / Pod | `rlinf` / `rlinf-0` |
| GPU | 单机 4×NVIDIA H20 |
| 镜像 | `iaas-us-cn-beijing.cr.volces.com/physicalai/rlinf:19ed26da621b8f51505a0b79d7924972f8820327` |
| PVC | `workspace-rlinf-0`，挂载到 `/workspace` |
| 源码 | `/workspace/rlinf` |
| Python | `/opt/venv/openpi/bin/python` |
| LIBERO | `/opt/envs/LIBERO` |
| 模型 | `/workspace/models/RLinf-Pi05-LIBERO-SFT` |
| 推荐配置 | `libero_spatial_async_ppo_openpi_pi05_best_4gpu` |

## 1. 配置容器并下载模型

本节命令都在 `rlinf-0` 内执行。

### 1.1 下载 Pi0.5 模型

模型已经存在时跳过下载。否则把模型直接下载到 PVC：

```bash
mkdir -p /workspace/models
cd /workspace/models
export BUCKET=ai-infra
oniond download model RLinf/RLinf-Pi05-LIBERO-SFT
```

## 2. 启动训练

### 2.1 推荐配置

使用
`examples/embodiment/config/libero_spatial_async_ppo_openpi_pi05_best_4gpu.yaml`。
它对应：

- 4 卡 async + colocated；
- 64 个训练环境，每卡 16 个；
- `group_size=2`、`update_epoch=2`；
- global batch 128、micro batch 32；
- `decoupled_actor_critic`；
- 每 40 step 保存一次 checkpoint。

### 2.2 运行

为每次训练使用独立目录：

```bash
cd /workspace/rlinf
source switch_env openpi
export EMBODIED_PATH=/workspace/rlinf/examples/embodiment
export RUN_NAME="$(date +%Y%m%d-%H%M%S)-pi05-libero-4gpu"
export LOG_DIR="/workspace/rlinf/logs/${RUN_NAME}"
mkdir -p "${LOG_DIR}"

/opt/venv/openpi/bin/python examples/embodiment/train_async.py \
  --config-path /workspace/rlinf/examples/embodiment/config \
  --config-name libero_spatial_async_ppo_openpi_pi05_best_4gpu \
  'actor.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT' \
  'rollout.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT' \
  'rollout.enable_torch_compile=true' \
  'actor.model.openpi.enable_fused_prefix=true' \
  runner.logger.log_path="${LOG_DIR}" \
  >& "${LOG_DIR}/run.log"

```

这条命令会解析 Hydra 配置、启动本地 Ray、创建 actor/rollout/env workers，然后开始
PPO。第一次 compile 会增加约 50 秒冷启动时间；长训练看稳态吞吐，不要用首步判断性能。

快速 smoke test 时，在命令末尾追加：

```text
runner.max_steps=2 runner.save_interval=-1
```

## 3. 查看训练进度和 checkpoint

### 3.1 查看日志

如果还在启动训练的 shell 中，`LOG_DIR` 已经设置。打开新的 shell 时，先指定本次目录：

```bash
export LOG_DIR=/workspace/rlinf/logs/<RUN_NAME>
```

查看主日志和结构化指标：

```bash
tail -f "${LOG_DIR}/run.log"
tail -f "${LOG_DIR}/metrics.log"
```

`metrics.log` 会在第一个训练 step 完成后创建。模型和环境初始化期间先看 `run.log`。

只看最新进度：

```bash
grep "Global Step:" "${LOG_DIR}/metrics.log" | tail -n 5
```

重点关注：

| 指标 | 含义 |
|---|---|
| `env/success_once` | 每批环境是否至少成功一次，训练效果的主要观察指标 |
| `env/return` | 环境回报 |
| `time/step` | 端到端每步耗时 |
| `time/actor_training` | actor 训练耗时 |
| `time/env/run_interact_once` | 完整 rollout/env 交互耗时 |
| `time/actor/wait_for_rollout_store_ready` | actor 等待 rollout batch 的时间 |
| `train/actor/policy_loss` | policy loss |
| `train/critic/value_loss` | value loss |

查看 GPU 和 Ray：

```bash
watch -n 1 nvidia-smi
/opt/venv/openpi/bin/ray status
```

### 3.2 使用 TensorBoard

在容器内启动 TensorBoard：

```bash
nohup /opt/venv/openpi/bin/tensorboard \
  --logdir /workspace/rlinf/logs \
  --host 0.0.0.0 \
  --port 6006 \
  >/workspace/rlinf/logs/tensorboard.log 2>&1 &
```

在本机另开终端转发端口：

```bash
kubectl port-forward -n rlinf pod/rlinf-0 6006:6006
```

浏览器打开 `http://localhost:6006`。本次 Hydra 最终配置保存在
`${LOG_DIR}/tensorboard/config.yaml`。

### 3.3 Checkpoint 路径

配置默认 `runner.save_interval=40`。训练会在每 40 step 和最后一个 step 保存：

```text
${LOG_DIR}/pi05_async_best_4gpu/checkpoints/global_step_<N>/actor/
```

关键内容包括：

```text
actor/dcp_checkpoint/                         # 模型、optimizer、scheduler 的训练状态
actor/model_state_dict/full_weights.pt        # rank 0 汇总的完整模型权重
```

恢复训练时，先保存旧 run 的 checkpoint 路径：

```bash
export RESUME_DIR=/workspace/rlinf/logs/<OLD_RUN_NAME>/pi05_async_best_4gpu/checkpoints/global_step_40
```

然后在原训练命令末尾追加：

```text
runner.resume_dir="${RESUME_DIR}"
```

`runner.resume_dir` 必须指向 `global_step_<N>`，不要指向其下的 `actor` 目录。恢复时可以
使用新的 `LOG_DIR` 写后续日志，但 `RESUME_DIR` 必须保留为旧 checkpoint 的绝对路径。
