# RLinf (Volcano Engine Version) Release Note

本版本基于上游 `RLinf/RLinf` 的 `release/v0.3` 分支构建，完整继承 RLinf v0.3
的模型、环境、算法与系统能力，并针对火山引擎环境和 Pi0.5 具身强化学习场景补充了
安装、数据下载和训练性能优化。

## 新功能

### 默认使用火山引擎软件源

下游 Docker 镜像默认将 Ubuntu APT 和 pip 切换到火山引擎镜像，
降低国内网络环境下安装系统包和 Python 依赖的等待时间及失败率。

### 预装 oniond，支持 TOS 高速下载

镜像默认预装 `oniond`（`onion-ai-data`），可从火山引擎 TOS 高速拉取模型和数据资产。
主流模型 （π0.5, GR00T）及 对应仿真资产 （LIBERO，ManiSkill，etc.）都已经接入。

### CUDA 13.1 + PyTorch 2.11

新增 CUDA 13.1 + PyTorch 2.11 安装支持，解决旧软件栈在 Hopper GPU 上运行
cuBLAS FP8 GEMM 时的兼容问题，并支持 Blackwell GPU。检测到 CUDA 13 主版本且未显式
指定 `--torch` 时，安装脚本会自动选择 PyTorch 2.11，并匹配 CUDA 13 版本的 nvCOMP
和 FlashAttention 依赖；显式指定的 PyTorch 版本仍具有更高优先级。

该环境已验证 Pi0.5 + LIBERO Sync GRPO、Pi0.5 + ManiSkill Async PPO、
GR00T N1.7 + LIBERO Sync PPO，以及 OpenVLA + ManiSkill Sync PPO。

## 功能增强

### Pi0.5 Rollout 编译

新增 Pi0.5 + LIBERO Async PPO rollout 编译配置。推荐的 H20 配置默认启用
`rollout.enable_torch_compile=true`，仅编译 rollout inference，actor training
仍保持 eager 模式。

在单机 4 张 NVIDIA H20 的测试中，独立 rollout 延迟降低约 12.53%，LIBERO
Async PPO 稳态 step time 降低约 5.59%。首次运行包含编译预热开销，短任务应结合
总墙钟时间评估收益。

### Pi0.5 Fused Prefix Kernel

Pi0.5 新增可选的 fused prefix decoder kernel，融合 RMSNorm、RoPE 和 projection
epilogue 等操作，同时保持 checkpoint 参数命名、FSDP 封装、prefix cache 和 backward
兼容。

在 4 张 H20 的测试中，Pi0.5 rollout 延迟降低约 7.36%，LIBERO Async PPO 稳态
step time 降低约 3.73%；与 rollout compilation 组合时，独立 rollout 测试提升约
14.54%。该能力默认关闭，可通过 `actor.model.openpi.enable_fused_prefix=true`
按需启用。

### Async PPO 权重同步重叠

Async PPO 支持通过 `actor.sync_weight_no_wait=true` 将 actor-to-rollout 权重同步
放到后台执行，与后续训练和 rollout 重叠。系统最多保留一个进行中的同步任务，并在
退出前等待最后一次同步完成，避免权重版本或资源回收不完整。

在单机 8 张 H20 的 collocated 测试中，runner 的阻塞权重同步区间从每 step 约
3.1 秒降至约 0.0018 秒。该结果表示串行等待被消除，实际端到端收益仍取决于 rollout
负载、placement 和异步队列状态。

### H20 Pi0.5 推荐配置

新增 4 套经过 profiling 调优的 Pi0.5 Async PPO 配置，可作为 H20 环境的起始配置：

| 场景 | GPU | Placement | 环境数 | Global batch |
| --- | ---: | --- | ---: | ---: |
| LIBERO Spatial | 4 | Collocated | 64 | 128 |
| LIBERO Spatial | 8 | Collocated | 128 | 256 |
| ManiSkill 25-main | 4 | Actor 2 卡，rollout/env 2 卡 | 160 | 2560 |
| ManiSkill 25-main | 8 | Collocated | 320 | 5120 |

这些配置默认启用 rollout compilation 和后台权重同步，并针对环境数、batch size、
FSDP 策略及组件 placement 做了联合调优。它们是 H20 上的推荐起点，不应直接视为
其他 GPU 型号或任务的通用最优配置。

## 交付与稳定性改进

- 镜像使用 `tini` 作为 PID 1，负责转发信号并回收退出的 Ray worker，修复长期运行
  容器中的僵尸进程残留问题。
- 优化 GitHub、Hugging Face 和 ManiSkill 资产下载路径；依赖仓库默认使用 shallow
  clone，降低大仓库在不稳定网络下的下载量和失败概率。

## 使用说明

- 本版本包含上游 RLinf v0.3 的全部功能；上游完整变更请参考 RLinf v0.3 release note。
- 上述性能数字来自特定 H20 workload 的稳态测试。环境数量、任务长度、placement 等会影响
  实际收益。