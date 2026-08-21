# π₀.₅ 强化学习性能报告

本报告给出 RLinf 三项自研性能优化在 π₀.₅ 上的实测数据：Fused Prefix Kernel、Rollout 图
编译与异步权重同步。第 1 至 5 节的数据来自 2026-08-19 的测试，硬件为单机 4×H20 97GB。

Fused Prefix Kernel 此后换代。第 1 至 5 节中该项的数据对应第一代内核，换代后的复测数据与
两代之间的对比见第 6 节。

面向使用者的配置建议见 [RLinf 快速上手](../../QUICKSTART.md) 第二部分第 5 节，本文给出
完整的测试方法与逐场景数据。

---

## 1. 结论

1. **最优优化项由瓶颈侧决定，没有通用最优。** 同模型同环境下仅改变 placement，最优项
   即发生翻转：LIBERO colocated 下 compile 使端到端 **+8.85%（更慢）**，同一配置在
   disaggregated 下为 **−4.76%（更快）**。
2. **fused 与 compile 不能同时作用于 rollout。** 二者直接叠加时 rollout 推理收益从
   −11.8%~−12.9% 塌陷到 −0.0%~−2.0%；把 fused 从 rollout 摘除后收益立即恢复。正确用法是
   **actor 用 fused、rollout 用 compile**。
3. **fused 的 actor 侧收益稳定**：四个场景跨两种环境、两种 placement，均落在
   **−5.75%~−6.93%** 的窄区间，全部 95% 区间不跨零。换代后的内核在同样四个场景为
   **−6.98%~−10.13%**，见第 6 节。
4. **compile 的 rollout 侧收益稳定**：三个场景均为 **−11.84%~−12.93%**，对 actor 无影响
   （−0.0%~−0.8%）。
5. **异步权重同步在单机无可测收益**：同机权重同步本身仅 1–2 s，无可隐藏的开销。该特性
   面向跨机场景，本轮硬件条件下未覆盖。

### 推荐配置

| 场景 | 瓶颈侧 | 推荐 | 端到端收益 |
|---|---|---|---|
| LIBERO colocated | actor | 只开 fused | −7.97% |
| LIBERO disaggregated 2+2 | rollout | 只开 compile | −4.76% |
| ManiSkill disaggregated 2+2 | rollout | 只开 compile | −6.99% |
| 瓶颈侧不确定 | — | split（actor fused + rollout compile） | 距当场最优 1 个百分点以内 |

瓶颈侧判据：比较 `time/actor_training` 与 `time/rollout/generate_one_epoch`，数值大者为瓶颈。

split 的完整写法：

```text
actor.model.openpi.enable_fused_prefix=true \
+rollout.model.openpi.enable_fused_prefix=false \
+rollout.enable_torch_compile=true \
+rollout.torch_compile_mode=default
```

---

## 2. 测试方法

### 2.1 环境

| 项 | 取值 |
|---|---|
| 硬件 | 单节点 4×H20 97GB |
| 模型 | π₀.₅（openpi），LIBERO SFT / ManiSkill SFT 检查点 |
| 训练模式 | async PPO（`train_async.py`） |
| 基线配置 | `examples/embodiment/config/` 下的对应 example config，仅修改 placement 与 `experiment_name` |
| 图编译模式 | `torch_compile_mode=default`（代码兜底值 `max-autotune-no-cudagraphs` 编译开销显著更高，未采用） |

### 2.2 场景矩阵

| 编号 | 环境 | placement | 优化组数 |
|---|---|---|---|
| S1 | LIBERO | colocated（actor/rollout/env 共用 4 卡） | 6 |
| S2 | ManiSkill | disaggregated 2+2 | 5 |
| S3 | LIBERO | disaggregated 2+2 | 4 |
| S4 | ManiSkill | colocated | 4 |

共 19 次 run，全部一次通过，无失败无重跑。

### 2.3 优化组定义

| 组 | actor fused | rollout fused | rollout compile | 异步权重同步 |
|---|---|---|---|---|
| `baseline` | 关 | 关 | 关 | 开 |
| `fused` | 开 | 开 | 关 | 开 |
| `compile` | 关 | 关 | 开 | 开 |
| `both` | 开 | 开 | 开 | 开 |
| `split` | 开 | **关** | 开 | 开 |
| `nomixin` | 关 | 关 | 关 | **关** |

`fused` / `both` 两组中 rollout 侧同时开启 fused，是因为 rollout 的模型配置由 actor 深拷贝
而来；`split` 组使用 `+rollout.model.openpi.enable_fused_prefix=false` 单独关闭 rollout 侧。

### 2.4 取数口径

- 数据源为 TensorBoard event 文件。`metrics.log` 是 rich 渲染的表格，数值会被截断，不可用于取数。
- 稳态窗口：LIBERO 取 12 步中的后 8 步，ManiSkill 取 10 步中的后 7 步，剔除初始化与首次编译开销。
- 每组 1 次 run，表中 `±` 为该 run 内的 step 间标准差。
- 端到端 `time/step` 在 colocated 下呈双峰分布，一律以**中位数**报告，不做显著性检验；
  阶段指标（`time/actor_training`、`time/rollout/predict`）为等工作量口径，可直接比较，
  报告 Welch 95% 区间。
- 每次 run 结束后断言开关真实生效（融合层替换日志的出现与否须与该组预期一致），不满足即判失败。

---

## 3. 瓶颈侧

`time/actor_training − time/rollout/generate_one_epoch` 是解释各场景收益差异的核心变量：
为正说明 actor 更慢，为负说明 rollout 更慢。

| 场景 | actor − rollout | 瓶颈侧 |
|---|---|---|
| LIBERO colocated | +2.70 s | actor |
| LIBERO disagg 2+2 | −22.92 s | rollout |
| ManiSkill disagg 2+2 | −77.63 s | rollout |
| ManiSkill colocated | −101.20 s | rollout（该场景端到端不可测，见第 5 节） |

**加速非瓶颈侧换不到端到端收益**；在 colocated 下还会因破坏流水线平衡而变慢。

---

## 4. 实测数据

### 4.1 Fused Prefix Kernel（actor 侧）

本节数据对应第一代内核。换代后的复测见第 6 节。

`time/actor_training`，单位秒：

| 场景 | baseline | fused | Δ | Welch 95% 区间（秒） |
|---|---|---|---|---|
| LIBERO colocated | 73.84 ± 0.46 | 68.72 ± 0.63 | **−6.93%** | [−5.66, −4.57] |
| ManiSkill disagg 2+2 | 144.15 ± 0.12 | 135.86 ± 0.16 | **−5.75%** | [−8.44, −8.14] |
| LIBERO disagg 2+2 | 74.27 ± 0.20 | 69.86 ± 0.24 | **−5.93%** | [−4.62, −4.19] |
| ManiSkill colocated | 113.60 ± 0.40 | 105.86 ± 0.69 | **−6.82%** | [−8.34, −7.15] |

四个场景跨两种环境、两种 placement，收益一致落在 −5.75%~−6.93%，全部区间不跨零。

### 4.2 Rollout 图编译（rollout 侧）

`time/rollout/predict`，单位秒：

| 场景 | baseline | compile | Δ predict | Δ actor |
|---|---|---|---|---|
| LIBERO colocated | 35.73 ± 2.18 | 31.24 ± 1.09 | **−12.58%** | −0.60% |
| ManiSkill disagg 2+2 | 116.29 ± 0.64 | 101.26 ± 0.77 | **−12.93%** | −0.54% |
| LIBERO disagg 2+2 | 35.48 ± 0.01 | 31.29 ± 0.02 | **−11.84%** | −0.04% |

编译只在 rollout worker 中接线，对 actor 训练无影响，符合设计。

### 4.3 fused 与 compile 的相互抵消

`time/rollout/predict` 相对 baseline 的变化：

| 场景 | compile 单开 | both | split |
|---|---|---|---|
| LIBERO colocated | −12.58% | **−0.04%** | −12.33% |
| ManiSkill disagg 2+2 | −12.93% | **−1.99%** | −13.09% |
| LIBERO disagg 2+2 | −11.84% | **−1.55%** | 未测 |

原因是融合层由自定义 `autograd.Function` 包裹，torch.compile 无法追踪，被编译的
`paligemma.model.language_model.forward` 图被打断。三个场景一致复现；把 fused 从 rollout
摘掉（split）后收益立即恢复，而 actor 侧的 fused 收益在 both 与 split 中均完整保留，说明
干扰只发生在 rollout 侧。

佐证：`both` 组的首步编译开销约为 compile 单开的一半（+27.7~+30.4 s vs +53.1~+59.3 s），
指向被编译的图确实变少。

### 4.4 端到端

`time/step` 中位数相对 baseline 的变化：

| 场景 | 瓶颈侧 | fused | compile | both | 最优 |
|---|---|---|---|---|---|
| LIBERO colocated | actor | **−7.97%** | +8.85% | −6.54% | fused |
| LIBERO disagg 2+2 | rollout | −2.74% | **−4.76%** | +0.05% | compile |
| ManiSkill disagg 2+2 | rollout | −3.50% | **−6.99%** | −1.18% | compile |

LIBERO colocated 各组明细。停顿步数指 `wait_for_rollout_store_ready` 出现明显跳变的步数，
其排序与「actor 是否比 rollout 慢」完全同序：

| 组 | actor − rollout | 停顿步数 | `time/step` 中位数 | 相对 baseline |
|---|---|---|---|---|
| compile | +6.95 s | 5 / 8 | 82.20 s | +8.85% |
| nomixin | +3.24 s | 3 / 8 | 79.47 s | +5.23% |
| baseline | +2.70 s | 3 / 8 | 75.52 s | — |
| split | +1.30 s | 3 / 8 | 70.26 s | −6.96% |
| both | −2.19 s | 0 / 8 | 70.58 s | −6.54% |
| **fused** | −1.55 s | 0 / 8 | **69.50 s** | **−7.97%** |

split 在本场景距最优（fused）1.01 个百分点，同时保留了 rollout 侧 −12.33% 的推理收益。

### 4.5 异步权重同步

关闭该特性的 `nomixin` 组，各项指标均落在 baseline 噪声内：`time/actor_training` −0.02%、
`time/rollout/predict` +0.64%，区间均跨零。单机同卡间权重同步本身仅 1–2 s，无可隐藏的开销。
该特性的目标场景是跨机权重传输，本轮硬件条件下未覆盖。

### 4.6 冷启动开销

| 项 | 实测 |
|---|---|
| compile 首步额外开销 | +53.1~+59.3 s |
| both 首步额外开销 | +27.7~+30.4 s |
| disaggregated 场景回本步数 | 约 8–12 步 |

torch.compile 产物缓存于 `/tmp/torchinductor_root` 且**跨进程持久**。同一机器上重复运行图与
shape 相同的配置时，后续 run 的首步开销会被大幅低估。比较冷启动成本前需清空该目录。

---

## 5. 指标可信度

两类指标的可信度差别很大，需分别对待：

| 类别 | 指标 | 等工作量依据 | step 间标准差 |
|---|---|---|---|
| **阶段指标** | `time/actor_training`、`time/rollout/predict` | actor 每步固定消费 1 个 store 条目；rollout 每 epoch 固定 env×horizon | ±0.01~0.9 s |
| **端到端** | `time/step`（= wait + actor_training） | — | colocated ±11~22 s；disagg ±0.7~3.9 s |

**ManiSkill + colocated 不可用于性能测量。** GPU simulator 与 actor/rollout 争抢同卡，
`time/rollout/generate_one_epoch` 的 step 间标准差达 ±15.6~±31.3 s（同环境 disaggregated 下
仅 ±0.85 s），所有 rollout 侧区间跨零。该场景端到端出现的 +17%~+55% 波动是噪声，不可读作
优化使训练变慢。该场景中只有 `time/actor_training` 有效，它给出的 fused −6.82% 与另外三个
场景一致，反过来说明噪声源确实在 rollout/env 侧的 GPU 争用。

---

## 6. Fused Prefix Kernel 换代复测（2026-08-21）

### 6.1 换代内容

第一代内核在带 attention mask 时会退回到物化 `[B, Hq, S, S]` 的 fp32 softmax，并把概率矩阵
保留到反向；openpi 每次调用都会传 4D additive mask，所以这条路径是常态。第二代内核自带
Triton flash attention，前向与反向都直接接收 additive mask，两个方向都不再物化打分矩阵，
GQA 的 KV 头也不再 `repeat_interleave`；同时不再依赖 `flash_attn`。

### 6.2 与前五节的方法差异

沿用同一套测试脚手架、同样的场景与 workload、同样的稳态窗口（LIBERO 取 12 步中的后 8 步，
ManiSkill 取 10 步中的后 7 步）与同样的 Welch 95% 区间口径。新增的唯一维度是内核版本：它不
是配置项，而是 `rlinf/models/embodiment/openpi/fused_kernels/` 下的文件，因此驱动脚本在每个
run 开始前安装对应版本、结束后再次校验 sha256，任何一组都不可能测到非预期的版本。

三组的差异只有一处：prefix decoder 的实现。`baseline` 关闭 fused，`fused_v1` 与 `fused_v2`
开启 fused 并分别装载两代内核。

本轮共 12 个 run，其中两个按下列依据判为离群并重跑，两次尝试的原始数据均已留档：

| run | 现象 | 依据 | 重跑结果 |
|---|---|---|---|
| ManiSkill disagg / fused_v2 | `time/actor_training` 前 4 步 134.0，第 5 步起阶跃至 190–203 并保持 | 三组每步收到的轨迹张量同为 `[16, 80, 5, 7]`，工作量未变；run 日志无报错、无 OOM、无重编译；阶跃为突变而非渐变 | 逐步 134.1–134.4，平坦 |
| LIBERO colocated / baseline | `time/actor_training` 75.22 ± 1.04，停顿 7/8 步 | 2026-08-19 同配置的三个 fused-off run 为 73.39 / 73.82 / 73.84，run 间 spread 仅 0.45 s；停顿 3/8 步 | 73.26 ± 0.38，停顿 4/8 步 |

### 6.3 Fused Prefix Kernel 两代对比（actor 侧）

`time/actor_training`，单位秒，括号内为相对 baseline 的变化：

| 场景 | baseline | 第一代 | 第二代 |
|---|---|---|---|
| LIBERO colocated | 73.26 ± 0.38 | 66.98 ± 0.84（−8.57%） | 65.84 ± 2.22（**−10.13%**） |
| LIBERO disagg 2+2 | 74.76 ± 0.49 | 69.93 ± 0.30（−6.46%） | 69.54 ± 0.22（**−6.98%**） |
| ManiSkill disagg 2+2 | 144.36 ± 0.24 | 136.63 ± 0.16（−5.35%） | 134.19 ± 0.10（**−7.05%**） |
| ManiSkill colocated | 113.22 ± 0.46 | 106.01 ± 0.43（−6.37%） | 104.45 ± 0.41（**−7.74%**） |

八个区间全部不跨零。第一代在本轮的 −5.35%~−8.57% 与 4.1 节的 −5.75%~−6.93% 大体一致，其中
disaggregated 两场复现得最好（本轮 −6.46% 对 4.1 节 −5.93%，本轮 −5.35% 对 −5.75%）；
colocated 两场偏大：该 placement 下 actor 与 rollout 共用同一批卡，停顿期间的争用会抬高
`time/actor_training`，其 run 内标准差（±0.38~±2.22 s）也明显大于 disaggregated
（±0.10~±0.49 s）。第 5 节对这一点有更完整的说明。

### 6.4 两代之差

跨批次比较绝对百分比会受当日机器状态影响，而两代的差值是同一批次内背靠背测得的，是更稳的量：

| 场景 | `time/actor_training` | Welch 95% 区间（秒） | `time/rollout/predict` | Welch 95% 区间（秒） |
|---|---|---|---|---|
| LIBERO colocated | −1.71% | [−2.79, +0.50] | **−3.85%** | [−1.68, −0.92] |
| LIBERO disagg 2+2 | **−0.55%** | [−0.65, −0.13] | **−2.32%** | [−0.78, −0.75] |
| ManiSkill disagg 2+2 | **−1.79%** | [−2.58, −2.31] | **−3.00%** | [−3.82, −2.66] |
| ManiSkill colocated | **−1.47%** | [−2.00, −1.12] | +6.39% | [−2.41, +11.80] |

加粗为区间不跨零的项。actor 侧四场景一致为负，三场显著；rollout 推理侧三场显著为负，唯一
不显著的 ManiSkill colocated 正是第 5 节判定的 rollout 侧不可测场景。

绝对幅度不大，与内核本身的定位一致：融合层只覆盖 prefix VLM，而 actor 每步的大头是动作专家
的去噪；prefix 在 rollout 推理中的占比更高，所以 rollout 侧的差值更明显。

本轮未重新推导端到端 `time/step`。该量在 colocated 下呈双峰分布、由停顿次数主导（第 5 节），
每组一次 run 不足以在两代之间分辨出上表这个量级的差值；第 4.4 节与「推荐配置」表给出的端到端
数字仍以 2026-08-19 那批为准，两代之间的选择也不影响那里的结论。

### 6.5 适用范围

四个场景的 resolved config 均为 `train_expert_only: true`，此时 `freeze_vlm()` 会把整个
paligemma（SigLIP 视觉编码器与 Gemma）的 `requires_grad` 置为 False，梯度不会进入融合层，
只执行前向。上述数据因此全部是前向收益。第二代内核在反向路径上的改动——整行被 mask 的
query 行的 softmax 统计量精度，openpi 的 `pad_2d_masks` 在 prompt 不等长时必然产生这类行——
只在 `train_expert_only: false`，即需要训练 prefix 的配置下才会体现。

### 6.6 正确性

**前向对比。** prefix 冻结时，融合层的输出只经两条路径进入模型其余部分：动作专家所消费的
逐层 prefix KV cache，以及 value head 读取的 prefix 最终隐状态（这些配置为
`value_after_vlm: true`，见 `openpi_action_model.py` 的 `values_vlm`）。两条路径都做了比较，
输入取自真实 LIBERO SFT 检查点、由模型自身 `embed_prefix` 产生，配 openpi 的真实 mask 与不等长
prompt（2240 个 query 行中 63 行整行被 mask）：

| 对比 | 逐层 K cache | 逐层 V cache | prefix 最终隐状态 |
|---|---|---|---|
| 第一代 vs 关闭 fused | 0.993221 ~ 0.998853 | 0.999060 ~ 1.000000 | 0.998983 |
| 第二代 vs 关闭 fused | 0.993221 ~ 0.998851 | 0.999059 ~ 1.000000 | 0.998933 |
| 第二代 vs 第一代 | 0.999844 ~ 1.000000 | 0.999659 ~ 1.000000 | 0.999581 |

数值为余弦相似度，仅统计非 padding 的真实 token。两代与未融合实现的差距在每一层上都相同到
小数点后 5 位，而两代彼此之间比各自与未融合实现之间更接近——这正是两个都正确、仅舍入顺序
不同的实现应有的形态。

与未融合实现之间的 K 差异来自 RoPE：融合内核在 kernel 内以 fp32 计算 cos/sin，未融合实现先
转 bf16。两代在这一项上偏差数值完全相同（第 0 层最大绝对误差同为 3.146），是既有属性而非
换代引入。

**学习信号。** `env/success_once` 与 `env/return` 在四个场景中，两代相对关闭 fused 的 Welch
区间全部跨零。部分二阶指标（`train/actor/entropy_loss`、`train/critic/explained_variance`）
区间不跨零，但被标记的对两代是同一批指标，且 `entropy_loss` 的 run 内标准差仅约 1e-4，
Welch 区间只反映 run 内方差、不含 run 间方差，在每组一次 run 的设计下会标记任何 run 间差异。
这类标记不构成两代之间的差异证据，前向逐层对比才是。
