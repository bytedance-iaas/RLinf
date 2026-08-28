# Physical AI Kit -- RLinf Release Note

本版本基于上游 `RLinf/RLinf` 的 `main` 分支提交
[`a3816b59`](https://github.com/RLinf/RLinf/commit/a3816b596478dcd8a5c69a6ec1468c9519f77b5b)
（`docs(openpi): add RoboTwin adjust_bottle resources`，#1486，2026-08-22）构建，完整继承该
提交为止的上游模型、环境、算法与系统能力，并针对火山引擎环境和 π₀.₅ 具身强化学习场景补充
了部署、安装、数据下载和训练性能优化。

## 新功能

### 默认使用火山引擎软件源

下游 Docker 镜像默认将 Ubuntu APT 和 pip 切换到火山引擎镜像，
降低国内网络环境下安装系统包和 Python 依赖的等待时间及失败率。

### 预装 oniond，支持 TOS 高速下载

镜像默认预装 `oniond`（`onion-ai-data`），可从火山引擎 TOS 高速拉取模型和数据资产。
主流模型（π₀.₅、GR00T）及对应仿真资产（LIBERO、ManiSkill 等）都已经接入。

### RLinf Dashboard

新增自研的训练可视化面板，以 Sidecar 容器随训练环境一同部署，只读挂载日志目录，扫描
`_rlinf/runs/` 与 TensorBoard 曲线，无需引入 RLinf 依赖即可独立运行。

面板提供运行列表、概览、指标、视频、事件与多 run 对比六个视图，指标按 embodied 模板分组，
核心指标为 `env/success_once`。支持 HTTP Basic 认证与中英文界面切换；通过 API 网关对外发布
时强制要求启用认证，避免无鉴权页面暴露到公网。

### 一键部署 Helm Chart

新增 Helm Chart，一次安装即可拉起训练容器、面板 Sidecar、持久卷与可选的 API 网关入口。
源码、模型、日志与 checkpoint 统一落在 `/workspace` 持久卷，Pod 重建后不丢失。

### SO101 机械臂的 real2sim 强化学习

新增 SO101（SO-ARM101）低成本 6 自由度机械臂的 real2sim 任务：依据实测几何与录制的 LeRobot
数据集，在 ManiSkill 中重建真实抓取摆放工作台（`SO101GrabRedCube-v1`），并在其上对 π₀.₅ 做
PPO 微调。任务工作在关节空间，策略输出的 6 维动作直接对应机械臂关节，观测包含前视与腕部
两路相机。

随附的配置有三项偏离 π-RL 默认配方，都在该任务上做过对照：动作块长度取 10（等于检查点的
action horizon）、`noise_logvar_range` 取 `[0.02, 0.04]`（`flow_noise` 实际读取的字段）、
每轮恰好一次更新。默认组合在该任务上会崩塌，确定性评测在第 9 步跌到 7%，因此这三项应视为
任务要求而非调参旋钮。

### LeRobot 检查点转换

LeRobot 与 RLinf 的 π₀.₅ 检查点布局不同，直接加载会失败，需要先转换。检查点转换器新增
`lerobot_to_openpi_pytorch` 模式，把 LeRobot 的 `pi05` 检查点转成 RLinf 所加载的布局，权重
键名与归一化统计量都会一并处理。连同既有模式，转换器现覆盖六种转换方向：

```text
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode <mode> ...

    jax_to_openpi_rlinf              JAX Pi0/Pi05 checkpoint -> OpenPI_RLinf layout
    openpi_pytorch_to_openpi_rlinf   OpenPI PyTorch layout -> OpenPI_RLinf layout
    sft_to_openpi_rlinf              RLinf SFT full_weights.pt -> OpenPI_RLinf layout
    openpi_rlinf_to_openpi_pytorch   OpenPI_RLinf layout -> OpenPI PyTorch layout
    sft2deploy                       RLinf SFT -> OpenPI PyTorch deploy full_weights.pt
    lerobot_to_openpi_pytorch        LeRobot pi05 checkpoint -> OpenPI PyTorch layout
```

完整说明见 `--help` 与 `--mode <mode> --help`，用法与注意事项见
[检查点转换器说明](rlinf/utils/ckpt_convertor/openpi/README.md)。

## 功能增强

### π₀.₅ Fused Prefix Kernel

π₀.₅ 新增可选的 fused prefix decoder kernel，将 PaliGemma 前缀侧的 decoder layer 替换为
Triton 实现，同时保持 checkpoint 参数命名、FSDP 封装、prefix cache 和 backward 兼容。
当前实现是 flash attention 版本：additive mask 直接进入 kernel，前向与反向都不再落地
`[B, Hq, S, S]` 的分数矩阵，也不再依赖外部 `flash_attn`。

在 4 张 H20 的四个场景（LIBERO / ManiSkill × colocated / disaggregated）中，actor 训练耗时
`time/actor_training` 一致下降 **6.98% 至 10.13%**，全部 95% 区间不跨零。kernel 级测试中，
带 mask 的前向峰值激活显存下降 15% 至 24%，构建 prefix cache 的峰值显存下降 32% 至 39%。

该能力默认关闭，通过 `actor.model.openpi.enable_fused_prefix=true` 启用。

### π₀.₅ Rollout 图编译

新增 rollout 侧的 torch.compile 支持，仅编译 rollout inference，actor training 保持 eager
模式。在 4 张 H20 的三个场景中，rollout 推理耗时 `time/rollout/predict` 下降
**11.84% 至 12.93%**，对 actor 训练无影响。

端到端收益取决于 rollout 是否是瓶颈：disaggregated 下 LIBERO 稳态 step time 下降 4.76%、
ManiSkill 下降 6.99%；colocated 下 rollout 不是瓶颈，同一配置反而慢 8.85%。首次运行包含约
53 至 59 秒的编译预热开销，短任务应结合总墙钟时间评估收益。

该能力默认关闭，通过 `+rollout.enable_torch_compile=true` 与
`+rollout.torch_compile_mode=default` 启用。

### fused kernel 与图编译的作用域拆分

rollout 的模型配置由 actor 深拷贝而来，两项优化直接叠加时会同时落在 rollout 上并互相抵消：
融合层的自定义 autograd Function 无法被 torch.compile 追踪，rollout 推理收益从 12% 掉到
0% 至 2%。新增 `+rollout.model.openpi.enable_fused_prefix=false`，可让 actor 用 fused、
rollout 用 compile，两侧收益同时保留。

### Async PPO 权重同步重叠

Async PPO 支持通过 `+actor.sync_weight_no_wait=true` 将 actor-to-rollout 权重同步放到后台
执行，与后续训练和 rollout 重叠，消除 runner 每步的串行等待。系统最多保留一个进行中的同步
任务，并在退出前等待最后一次同步完成，避免权重版本或资源回收不完整。

单机场景下同机权重同步本身只有 1 至 2 秒，实测端到端无可测收益；该能力面向跨机权重传输，
跨机收益本轮硬件条件下未验证。

### H20 π₀.₅ 推荐配置

最优的优化组合取决于瓶颈侧，判据是比较 `time/actor_training` 与
`time/rollout/generate_one_epoch`，数值大的一侧是瓶颈。加速非瓶颈侧换不到收益，colocated
下还会因破坏流水线平衡而变慢。

| 场景 | 瓶颈侧 | 推荐开关 | 稳态 step time |
| --- | --- | --- | --- |
| LIBERO colocated | actor | 只开 fused | −8.0% ~ −19.0% |
| LIBERO disaggregated 2+2 | rollout | 只开 compile | −4.76% |
| ManiSkill disaggregated 2+2 | rollout | 只开 compile | −6.99% |
| 瓶颈侧不确定 | — | actor 开 fused、rollout 开 compile | 距当场最优 1 个百分点以内 |

colocated 的端到端数字随流水线停顿次数波动很大，上表第一行同配置的两批测试分别测到
−8.0% 与 −19.0%，不宜当作精确值，阶段指标才是稳定的信号。

单机 4 卡 H20 上经过验证的 LIBERO 起始配置为：async + colocated，64 个训练环境，horizon
120，`group_size=2`，`update_epoch=2`，global batch 128，micro batch 32。完整命令见快速上手
文档。这是 H20 上的推荐起点，不应直接视为其他 GPU 型号或任务的通用最优配置。

## 问题修复与稳定性改进

### auto resume 可能加载到写了一半的 checkpoint

`resume_dir: auto` 原先按目录名取最大的 `global_step_N`，而保存流程是先建目录、最后写
`data/data.pt`。崩溃、抢占或 `kill -9` 中断的保存会留下一个只有目录名的半成品，并被优先
选中。现在从新到旧遍历，选第一个真正完整的 checkpoint（以 dataloader 状态作为完成标记），
被跳过的会给出告警。

### 容器内的僵尸进程

镜像使用 `tini` 作为 PID 1，负责转发信号并回收退出的 Ray worker，修复长期运行容器中的
僵尸进程残留问题。

## 使用说明

- 上述性能数字来自特定 H20 workload 的稳态测试。环境数量、任务长度、placement 等会影响
  实际收益，建议在目标 workload 上复测后再决定是否启用。
- 部署、训练、面板与 SO101 任务的完整步骤见仓库根目录的 [快速上手文档](QUICKSTART.md)。
- 性能测试方法、四场景逐项数据与测量口径见
  [π₀.₅ 强化学习性能报告](docs/performance/pi05_rl_performance.md)。
