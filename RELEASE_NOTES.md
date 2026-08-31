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

### LeRobot 检查点互转

LeRobot 与 RLinf 的 π₀.₅ 检查点布局不同，直接加载会失败，需要先转换。检查点转换器新增
两个方向的模式，覆盖「拿 LeRobot 的策略进来训」和「把训好的策略交回真机」这两件事。

`lerobot_to_openpi_pytorch` 把 LeRobot 的 `pi05` 检查点转成 RLinf 所加载的布局。LeRobot
微调是为新机器人得到 π₀.₅ 策略的常见做法，公开发布的 π₀.₅ 检查点也大多是这个格式，而两处
差异都不会报错：键名带 LeRobot 的 `model.` 前缀，加载走 `strict=False`，不匹配时会静默丢弃
全部权重，策略实际跑在随机初始化上；归一化统计量存放在 `*_normalizer_processor.safetensors`
中，而 openpi 需要的是 `norm_stats.json`。两者转换器都会处理。

`sft_to_lerobot` 是它的逆向，把 RLinf 的检查点导出成 LeRobot 布局，交给 LeRobot 的异步推理
栈部署，机器人侧因此可以直接用现成的 `robot_client`，不必自己写控制循环。权重部分是纯键名
重写，两个方向往返后逐位一致；真正需要留意的是随权重同行的元数据，它们会被**原样带过来**而
不是转换出错，且同样不报错：RL 专用参数在 LeRobot 侧没有位置，会被丢弃；绑定的
`embed_tokens.weight` 以模板为准；推理去噪步数必须取 RLinf 的 4 而非 LeRobot 默认的 10，
沿用默认值等于导出了另一个策略，在 SO101 上实测每个关节的误差差约 26%；模板自带的归一化
统计量来自它自己的数据集，必须用导出策略所属血统的统计量覆盖。转换后的键集会与模板核对，
任何缺失、多余或形状不符都会让转换器拒绝写出。

连同既有模式，转换器现覆盖七种转换方向：

```text
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode <mode> ...

    jax_to_openpi_rlinf              JAX Pi0/Pi05 checkpoint -> OpenPI_RLinf layout
    openpi_pytorch_to_openpi_rlinf   OpenPI PyTorch layout -> OpenPI_RLinf layout
    sft_to_openpi_rlinf              RLinf SFT full_weights.pt -> OpenPI_RLinf layout
    openpi_rlinf_to_openpi_pytorch   OpenPI_RLinf layout -> OpenPI PyTorch layout
    sft2deploy                       RLinf SFT -> OpenPI PyTorch deploy full_weights.pt
    lerobot_to_openpi_pytorch        LeRobot pi05 checkpoint -> OpenPI PyTorch layout
    sft_to_lerobot                   RLinf SFT checkpoint -> LeRobot pi05 layout
```

需要说明的是，两个方向都验证过无损，但往返一致只能说明两个映射互为逆运算，无法排除 openpi
与 LeRobot 的分位数归一化约定存在差异——这类差异会在往返中相互抵消，却仍然在真机上是错的。
上硬件之前应对导出结果做一次离线评测，确认它能复现源检查点的成绩。

完整说明见 `--help` 与 `--mode <mode> --help`，用法与注意事项见
[检查点转换器说明](rlinf/utils/ckpt_convertor/openpi/README.md)。

## 功能增强

### π₀.₅ Fused Prefix Kernel

π₀.₅ 新增可选的 fused prefix decoder kernel，将 PaliGemma 前缀侧的 decoder layer 替换为
Triton 实现，同时保持 checkpoint 参数命名、FSDP 封装、prefix cache 和 backward 兼容。
当前实现是 flash attention 版本：additive mask 直接进入 kernel，前向与反向都不再落地
`[B, Hq, S, S]` 的分数矩阵，也不再依赖外部 `flash_attn`。

在单机 8 张 H20 的五个场景（LIBERO / ManiSkill × colocated / disaggregated，含 4 卡对比臂）中，
每组重复 2 至 3 次，actor 训练耗时 `time/actor_training` 一致下降 **6.6% 至 9.4%**，全部 95%
区间（基于 run 间方差）不跨零。8 卡为 −6.6%，4 卡为 −9.4%——收益随卡数缩水。kernel 级测试中，
带 mask 的前向峰值激活显存下降 15% 至 24%，构建 prefix cache 的峰值显存下降 32% 至 39%。

**显存收益仅限前向。** 上述显存数字测于 `train_expert_only: true`，此时 `freeze_vlm()` 使融合层
只执行前向。若改为可训练 prefix，梯度流经融合层后显存不降反升：实测在 95 GB 卡上 micro batch
取 8、4、1 均 OOM（峰值与 batch 无关），而未融合实现 81 GB 即可运行。该场景下不应启用 fused。
手写 backward 本身可正常执行，问题在显存而非正确性。

该能力默认关闭，通过 `actor.model.openpi.enable_fused_prefix=true` 启用。

### π₀.₅ Rollout 图编译

新增 rollout 侧的 torch.compile 支持，仅编译 rollout inference，actor training 保持 eager
模式。在 8 张 H20 的五个场景中，rollout 推理耗时 `time/rollout/predict` 下降
**12.1% 至 13.5%**，对 actor 训练无影响。该收益**与卡数无关**：4 卡与 8 卡分别为 −12.18% 与
−12.19%。

端到端收益小于阶段收益，取决于 rollout 是否是瓶颈：ManiSkill disaggregated 4+4 下稳态 step
time 下降 5.89%，LIBERO colocated 8 卡下为 3.14%。首次运行包含约 53 至 59 秒的编译预热
开销，短任务应结合总墙钟时间评估收益。


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

单机场景下同机权重同步本身只有 1 至 2 秒，实测端到端无可测收益。该结论已在 4 卡与 8 卡
disaggregated 两处确认——后者权重需跨 GPU 组传输，仍无可测收益。该能力面向跨机权重传输，
跨机收益两轮硬件条件下均未验证。

### H20 π₀.₅ 推荐配置

**推荐 split：actor 用 fused、rollout 用 compile。** 这是唯一同时拿到两侧收益的组合，在实测的
五个场景中端到端都不劣于当场最优的单项优化。

```text
actor.model.openpi.enable_fused_prefix=true \
+rollout.model.openpi.enable_fused_prefix=false \
+rollout.enable_torch_compile=true \
+rollout.torch_compile_mode=default
```

单机 8×H20 独占节点实测，每组重复 2 至 3 次，`±` 为 run 间标准差：

| 场景 | baseline 稳态 step | fused | compile | **split** |
| --- | --- | --- | --- | --- |
| **LIBERO colocated 8 卡** | 102.98±0.90 s | −6.91% | −3.14% | **−8.31%** |
| LIBERO disaggregated 4+4 | 103.23±0.24 s | −4.15% | −2.40% | +1.12% |
| ManiSkill disaggregated 4+4 | 221.85±1.14 s | −4.55% | −5.89% | **−6.24%** |
| LIBERO colocated 4 卡（对比） | 89.74±2.04 s | −11.53% | −2.83% | **−14.24%** |

split 相对当场最优单项的领先幅度，在每个场景都小于二者 run 间标准差之和，应理解为
**「不劣于最优单项」** 而非「严格更优」。它可靠的优势在阶段指标：`time/actor_training`
−6.6% 至 −9.4%，同时 `time/rollout/predict` −12.1% 至 −13.5%。

**收益不随卡数等比放大。** 每卡工作量不变时，4 卡到 8 卡的端到端收益接近腰斩（split 由
−14.24% 降到 −8.31%）。根因是 `no_shard` 的 all-reduce 随卡数变贵：同样的每卡工作量，
`time/actor_training` 由 74.63 s 涨到 86.38 s（+15.7%）。rollout 侧收益则与卡数无关。扩容时
应重新评估，不要按卡数外推收益。

经过验证的 LIBERO 起始配置，两组每卡工作量相同：

| | **8 卡（推荐起点）** | 4 卡 |
| --- | --- | --- |
| 训练环境数 | **128** | 64 |
| global batch / micro batch | **256** / 32 | 128 / 32 |
| horizon、`group_size`、`update_epoch` | 120、2、2 | 120、2、2 |

**4 卡取值不能直接搬到 8 卡**：actor 侧断言
`global_batch_size % (micro_batch_size × world_size) == 0`，8 卡下 `128 % 256 ≠ 0` 会在模型
加载与首轮 rollout 之后才失败，global batch 必须同步放大到 256。此外 `env.eval.total_num_envs`
默认 500，8 卡下不被整除，开启评测时需改为 504。完整命令见快速上手文档。

这是 H20 上的推荐起点，不应直接视为其他 GPU 型号或任务的通用最优配置。

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

- 上述性能数字来自特定 H20 workload 的稳态测试，每组重复 2 至 3 次并给出 run 间方差。环境
  数量、任务长度、placement 与卡数都会影响实际收益——尤其**卡数**：本轮实测 4 卡到 8 卡的
  端到端收益接近腰斩。建议在目标 workload 上复测后再决定是否启用。
- 部署、训练、面板与 SO101 任务的完整步骤见仓库根目录的 [快速上手文档](QUICKSTART.md)。
- 性能测试方法、四场景逐项数据与测量口径见
  [π₀.₅ 强化学习性能报告](docs/performance/pi05_rl_performance.md)。
