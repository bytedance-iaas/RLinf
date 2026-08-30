# π₀.₅ 强化学习性能报告

本报告给出 RLinf 三项自研性能优化在 π₀.₅ 上的实测数据：Fused Prefix Kernel、Rollout 图编译
与异步权重同步。

数据来自两批测试：

| 批次 | 硬件 | 规模 | 说明 |
|---|---|---|---|
| **2026-08-30（主）** | 单机 **8×H20 97GB，独占节点** | 51 run，每组 2～3 次重复 | 本报告的主要依据 |
| 2026-08-19 / 08-21 | 单机 4×H20，与其它负载共享节点 | 19 + 8 run，每组 1 次 | 仅保留新批未覆盖的结论 |

新批次相对旧批次有三点方法学改进：**每组重复 2～3 次**（旧批每组仅 1 次，只能报 run 内 step
抖动，无法给出 run 间方差）；**节点独占**（旧批同节点另有 4 个 GPU Pod）；**同批次内包含 4 卡
对照臂**，使 4→8 的比较不掺入代码与环境差异。

面向使用者的配置建议见 [RLinf 快速上手](../../QUICKSTART.md) 第二部分第 5 节。

---

## 1. 结论

1. **收益不随卡数等比放大。** 每卡工作量不变时，4→8 卡的端到端收益接近腰斩：split 从
   −14.24% 降到 −8.31%，fused 从 −11.53% 降到 −6.91%。根因是 `no_shard` 的 all-reduce 随
   卡数变贵——同样的每卡工作量，`time/actor_training` 从 74.63 s 涨到 86.38 s（**+15.7%**）。
2. **rollout 侧收益与卡数无关，actor 侧收益随卡数缩水。** compile 的 `rollout/predict` 收益
   在 4 卡与 8 卡分别为 −12.18% 与 −12.19%；fused 的 `actor_training` 收益从 −9.42% 降到 −6.56%。
3. **split（actor 用 fused、rollout 用 compile）是各场景的安全默认。** 它在全部 5 个场景中
   同时拿到两侧的阶段收益，端到端从不劣于当场最优单项。但需注意：**端到端上它相对最优单项的
   领先幅度在每个场景都落在 run 间方差之内**，所以应表述为"不劣于"，而非"严格最优"。
4. **fused 的显存收益仅限前向。** 一旦梯度真正流经融合层（可训练 prefix），显存不降反升，
   在 95 GB 卡上任何 micro batch 都放不下，而未融合实现 81 GB 即可运行。见 4.7。
5. **异步权重同步在单机仍无可测收益**，本轮把这个边界从 4 卡推进到 8 卡 disaggregated。

### 与旧批次矛盾的两处结论

| 项 | 旧批（4 卡，每组 1 run，共享节点） | 新批（4 卡对照，每组 3 run，独占节点） |
|---|---|---|
| colocated 开图编译 | **+8.85%（更慢）** | **−2.83%（更快）**，三次 run 分别 87.0 / 85.5 / 89.1 s，baseline 87.9 / 89.3 / 92.0 s |
| colocated 最优项 | 只开 fused（−7.97%），split 差约 1 pp | split −14.24% ≥ fused −11.53% |

两处都以新批为准。旧批那两个数字来自单次 run，且当时节点与其它 GPU 负载共享——旧报告自身
已记录该 caveat。

### 推荐配置

| 场景 | 推荐 | 端到端收益 |
|---|---|---|
| **colocated（4 卡或 8 卡）** | **split** | 4 卡 −14.24%，8 卡 −8.31% |
| **disaggregated** | **split**（或只开 compile，二者不可分辨） | ManiSkill 4+4：split −6.24%、compile −5.89% |
| 任何场景 | ❌ 不要把 fused 与 compile 同时落在 rollout 上（见 4.4） | |

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
| 硬件 | 单节点 8×H20 97GB，**本轮独占**（旧批为 4 卡且与其它 Pod 共享） |
| 模型 | π₀.₅（openpi），LIBERO SFT / ManiSkill SFT 检查点 |
| 训练模式 | async PPO（`train_async.py`） |
| 基线配置 | `examples/embodiment/config/` 下的对应 example config，仅修改 placement 与 `experiment_name`；工作量与开关全部走 CLI override |
| 图编译模式 | `torch_compile_mode=default`（代码兜底值 `max-autotune-no-cudagraphs` 编译开销显著更高，未采用） |

### 2.2 场景矩阵（2026-08-30）

| 编号 | 环境 | placement | 工作量 | 重复 |
|---|---|---|---|---|
| a1 | LIBERO | colocated 8 卡 | envs 128 / horizon 120 / gbs 256 / micro 32 | 3 |
| a4 | LIBERO | colocated 4 卡（**同批对照**） | envs 64 / horizon 120 / gbs 128 / micro 32 | 3 |
| a2 | LIBERO | disaggregated 4+4 | 同 a1 | 2 |
| a3 | ManiSkill | disaggregated 4+4 | envs 320 / gbs 5120 / micro 32（出厂配置） | 2 |
| a5 | LIBERO | colocated 8 卡，**出厂形状** | envs 128 / horizon 240 / gbs 2048 / **micro 128** | 2 |
| b2 | LIBERO | disaggregated 4+4，**可训练 prefix** | micro 4，`train_expert_only=false` | 1 |

a1 与 a4 采用弱扩展：每卡 per-rank batch 均为 32、每卡 16 个 env，因此每卡工作量完全一致，
4→8 的差异只来自并行规模本身。a5 用于检验收益是否对 batch 形状敏感（micro 32 vs 128）。

### 2.3 优化组定义

| 组 | actor fused | rollout fused | rollout compile | 异步权重同步 |
|---|---|---|---|---|
| `baseline` | 关 | 关 | 关 | 开 |
| `fused` | 开 | 开 | 关 | 开 |
| `compile` | 关 | 关 | 开 | 开 |
| `split` | 开 | **关** | 开 | 开 |
| `nomixin`（仅 a2） | 关 | 关 | 关 | **关** |

`fused` 组中 rollout 侧同时开启 fused，是因为 rollout 的模型配置由 actor 深拷贝而来；`split`
组使用 `+rollout.model.openpi.enable_fused_prefix=false` 单独关闭 rollout 侧。

### 2.4 取数口径

- 数据源为 TensorBoard event 文件。`metrics.log` 是 rich 渲染的表格，数值会被截断，不可用于取数。
- 稳态窗口：每个 run 的最后 8 步（12 步配置）或最后 7 步（10 步配置）。
- **每个 run 先归约为一个稳态均值，再跨 run 统计**；表中 `±` 为 **run 间**标准差。
- 95% 区间按 Welch 差值计算，自由度取 `(n_a−1)+(n_b−1)`：n=3 时 t=2.776，n=2 时 t=4.303。
- 每次 run 结束后断言开关真实生效（融合层替换日志的出现与否须与该组预期一致），不满足即判失败重跑。
- 51 个 run 全部一次通过，无重跑。

---

## 3. 瓶颈侧

`time/actor_training − time/rollout/generate_one_epoch` 为正说明 actor 更慢。

| 场景 | actor − rollout | 瓶颈侧 |
|---|---|---|
| a1 LIBERO colocated 8 卡 | **+13.18 s** | actor |
| a4 LIBERO colocated 4 卡 | +3.90 s | actor |
| a5 LIBERO colocated 8 卡（出厂形状） | +4.00 s | actor |
| a2 LIBERO disagg 4+4 | −23.54 s | rollout |
| a3 ManiSkill disagg 4+4 | −77.63 s | rollout |

**卡数从 4 增到 8 会把 colocated 更深地推向 actor 瓶颈**（+3.90 → +13.18），因为 actor 侧的
`no_shard` all-reduce 变贵而 rollout 侧基本不变。

> 需要修正旧报告的一处推论：旧报告由"加速非瓶颈侧换不到收益"推出"colocated 下开 compile
> 会更慢"。新批不支持该推论——a1 是全部场景中 actor 瓶颈最深的（+13.18 s），compile 仍给出
> −3.14% 的端到端收益。加速非瓶颈侧的收益确实较小，但并非负值。

---

## 4. 实测数据

### 4.1 Fused Prefix Kernel（actor 侧）

`time/actor_training`，单位秒，`±` 为 run 间标准差：

| 场景 | baseline | fused | Δ | split | Δ |
|---|---|---|---|---|---|
| a1 colocated 8 卡 | 86.38±0.93 | 80.71±0.52 | **−6.56%** | 81.32±0.54 | −5.86% |
| a4 colocated 4 卡 | 74.63±0.24 | 67.59±0.74 | **−9.42%** | 67.65±0.56 | −9.35% |
| a2 disagg 4+4 | 78.28±0.16 | 72.56±0.45 | **−7.31%** | 73.06±0.52 | −6.67% |
| a3 ManiSkill disagg 4+4 | 144.51±0.97 | 134.39±1.28 | **−7.00%** | 134.69±0.78 | −6.79% |
| a5 colocated 8 卡（micro 128） | 131.16±0.65 | 119.04±0.40 | **−9.24%** | 118.65±0.04 | −9.54% |

五个场景一致为负，落在 **−6.56%～−9.42%**，全部 95% 区间不跨零。图编译对 actor 无影响
（−1.42%～+0.54%，区间多数跨零），符合"编译只在 rollout worker 中接线"的设计。

### 4.2 Rollout 图编译（rollout 侧）

`time/rollout/predict`，单位秒：

| 场景 | baseline | compile | Δ | split | Δ |
|---|---|---|---|---|---|
| a1 colocated 8 卡 | 34.59±0.19 | 30.37±0.20 | **−12.19%** | 29.62±0.19 | −14.35% |
| a4 colocated 4 卡 | 35.42±0.45 | 31.11±0.20 | **−12.18%** | 31.12±0.39 | −12.16% |
| a2 disagg 4+4 | 35.45±0.03 | 31.08±0.03 | **−12.34%** | 31.02±0.02 | −12.49% |
| a3 ManiSkill disagg 4+4 | 115.95±0.10 | 101.94±0.06 | **−12.08%** | 101.85±0.01 | −12.16% |
| a5 colocated 8 卡（micro 128） | 70.67±0.37 | 61.10±0.21 | **−13.54%** | 61.13±0.89 | −13.49% |

收益高度稳定在 **−12.08%～−13.54%**，且**与卡数无关**（4 卡 −12.18% vs 8 卡 −12.19%）。
`split` 拿到的 rollout 收益与 `compile` 单开等同或更好。

### 4.3 收益对 batch 形状不敏感

a5 使用出厂配置形状（micro 128、horizon 240），是 a1 的 4 倍：

| 指标 | a1（micro 32） | a5（micro 128） |
|---|---|---|
| fused `actor_training` | −6.56% | **−9.24%** |
| compile `rollout/predict` | −12.19% | **−13.54%** |

方向一致、量级相近，在更大形状上甚至更好。因此本报告给出的百分比可以合理外推到其它
batch 配置。

### 4.4 fused 与 compile 不能同时作用于 rollout

该结论来自 **2026-08-19 批次**（新批未设 `both` 组）。二者直接叠加时 rollout 推理收益从
−11.84%～−12.93% 塌陷到 −0.04%～−1.99%，三个场景一致复现；把 fused 从 rollout 摘除后收益
立即恢复。原因是融合层由自定义 `autograd.Function` 包裹，torch.compile 无法追踪，被编译的
`paligemma.model.language_model.forward` 图被打断。

正确用法即 `split`：**actor 用 fused、rollout 用 compile**。

### 4.5 端到端

`time/step`，`±` 为 run 间标准差：

| 场景 | baseline | fused | compile | split |
|---|---|---|---|---|
| a1 colocated 8 卡 | 102.98±0.90 | 95.87±1.94（−6.91%） | 99.75±1.75（−3.14%） | **94.43±0.96（−8.31%）** |
| a4 colocated 4 卡 | 89.74±2.04 | 79.40±5.00（−11.53%） | 87.20±1.82（−2.83%） | **76.96±4.75（−14.24%）** |
| a2 disagg 4+4 | 103.23±0.24 | 98.95±1.20（−4.15%） | 100.76±4.07（−2.40%） | 104.39±3.55（+1.12%） |
| a3 ManiSkill disagg 4+4 | 221.85±1.14 | 211.75±1.52（−4.55%） | 208.79±0.53（−5.89%） | **208.01±0.21（−6.24%）** |
| a5 colocated 8 卡（micro 128） | 167.91±3.13 | 153.87±2.83（−8.36%） | 166.86±2.19（−0.62%） | 155.58±1.26（−7.34%） |

**如何读这张表。** split 在 5 个场景中 3 个是点估计最优、2 个与最优单项并列，从不劣于最优
单项。但**在每一个场景中，split 与当场最优单项的差距都小于二者的 run 间标准差之和**，因此
统计上不能声称 split 严格更优。a2 是唯一 split 点估计为正的场景（+1.12%），而该场景
compile/split/nomixin 三组的 run 间标准差达 ±3.5～4.1 且仅 2 次重复，组间差异完全埋在噪声里，
不构成反例。

split 可靠的优势在**阶段指标**：它是唯一同时拿满 actor 侧与 rollout 侧收益的配置（4.1、4.2）。

### 4.6 异步权重同步

`nomixin` 组（关闭该特性，仅 a2）：`time/actor_training` −0.83%（区间 [−1.15, −0.15] 不跨零，
即**关闭后反而略快**），`rollout/predict` +0.04%（跨零），端到端 +1.20%（埋在 ±3.78 的噪声里）。

单机同节点内权重同步本身仅 1～2 s，无可隐藏的开销。旧批在 4 卡得出同样结论，本轮把边界推进到
**8 卡 disaggregated**（权重需跨 GPU 组传输）后仍无可测收益。该特性的目标场景是**跨机**权重
传输，两轮硬件条件下均未覆盖。

### 4.7 Fused Prefix Kernel 在可训练 prefix 下不可用

此前所有 fused 数据均在 `train_expert_only: true` 下测得，此时 `freeze_vlm()` 使融合层只执行
前向，**手写 backward 从未被任何性能数字覆盖**。b2 场景专门填补该盲区。

实测结果：

| 配置 | micro batch | 结果 | 进程峰值显存 |
|---|---|---|---|
| baseline（未融合） | 4 | ✅ 运行，301.7 s/step | 81 GiB |
| **fused** | 8 | ❌ OOM | 95.29 GiB |
| **fused** | 4 | ❌ OOM | 95.29 GiB |
| **fused** | 1 | ❌ OOM | 94.11 GiB |

两点结论：

1. **手写 backward 确实会被执行**，日志出现 `[fused-prefix] backward IS used`，并完成 18 层替换。
2. **融合层在训练 prefix 时的显存开销与 batch 无关。** micro batch 从 8 降到 1（八分之一）峰值
   几乎不动，说明开销不由激活值主导。未融合实现 81 GiB 即可运行，融合实现超出 95 GB 卡容量。

因此 **RELEASE_NOTES 中"带 mask 的前向峰值激活显存下降 15%～24%"是仅前向场景的结论**；一旦
梯度流经融合层，显存优势消失并反超。在 95 GB 卡 + 出厂 `no_shard` FSDP 设置下，fused 无法用于
可训练 prefix，因而也拿不到该场景的速度对比。

> 边界：以上结论基于 `sharding_strategy: no_shard`（每个 rank 持有完整优化器状态）。改用
> `full_shard` / `hybrid_shard` 后每 rank 显存会显著下降，fused 是否即可容纳**本轮未测**，
> 不能据此断言该组合永远不可用。

### 4.8 数值一致性（2026-08-19 批次）

融合层的输出只经两条路径进入模型其余部分：动作专家消费的逐层 prefix KV cache，以及 value
head 读取的 prefix 最终隐状态。两条路径与关闭 fused 的实现比较，输入取自真实 LIBERO SFT
检查点：

| 对比 | 逐层 K cache | 逐层 V cache | prefix 最终隐状态 |
|---|---|---|---|
| fused 开 vs 关 | 0.993221～0.998851 | 0.999059～1.000000 | 0.998933 |

数值为余弦相似度，仅统计非 padding 的真实 token。K 上的差异来自 RoPE：融合内核在 kernel 内
以 fp32 计算 cos/sin，未融合实现先转 bf16。

新批次的学习行为哨兵与之一致：五个场景中 `env/success_once` 各组之间无系统性差异。

### 4.9 冷启动开销（2026-08-19 批次）

| 项 | 实测 |
|---|---|
| compile 首步额外开销 | +53.1～+59.3 s |
| disaggregated 场景回本步数 | 约 8～12 步 |

torch.compile 产物缓存于 `/tmp/torchinductor_root` 且**跨进程持久**。同一机器上重复运行图与
shape 相同的配置时，后续 run 的首步开销会被大幅低估。比较冷启动成本前需清空该目录。

---

## 5. 指标可信度

| 类别 | 指标 | 等工作量依据 | run 间标准差 |
|---|---|---|---|
| **阶段指标** | `time/actor_training`、`time/rollout/predict` | actor 每步固定消费 1 个 store 条目；rollout 每 epoch 固定 env×horizon | ±0.01～1.28 s |
| **端到端** | `time/step` | 不适用 | ±0.21～5.00 s |

端到端的 run 间方差按场景差别很大，直接决定了各场景结论的强度：

| 场景 | 端到端 run 间标准差 | 结论强度 |
|---|---|---|
| a3 ManiSkill disagg 4+4 | ±0.21～1.52 | 最强，组间可分辨 |
| a1 colocated 8 卡 | ±0.90～1.94 | 强 |
| a5 colocated 8 卡（micro 128） | ±1.26～3.13 | 中 |
| a4 colocated 4 卡 | ±1.82～5.00 | 弱，仅可比较量级 |
| a2 disagg 4+4 | ±0.24～4.07（n=2） | 最弱，组间不可分辨 |

**8 卡 colocated 的端到端比 4 卡稳定得多**（±0.90～1.94 vs ±1.82～5.00）。旧报告曾记录 4 卡
colocated 同配置两批相差 11 个百分点（−8.0% 与 −19.0%），新批以 3 次重复复现了该场景的高方差
特性——4 卡 colocated 是本组场景中最难测的一个，而非 8 卡。

**ManiSkill + colocated 不可用于性能测量**（2026-08-19 批次结论，新批未重测）：GPU simulator
与 actor/rollout 争抢同卡，`rollout/generate_one_epoch` 的 step 间标准差达 ±15.6～31.3 s，同
环境 disaggregated 下仅 ±0.85 s。

---

## 6. 数据与复现

| 文件 | 内容 |
|---|---|
| `PI05_8GPU_A{1..5}_*_AUDIT.csv`、`PI05_8GPU_B2_*_AUDIT.csv` | 逐 run 逐 step 原始数据 |
| `RUN_SUMMARY.csv` | 51 个 run 的状态、墙钟、首步耗时、开关断言结果 |
| `APPENDIX_REPRO.md` | 每个用例的完整启动命令与场景配置差异 |

每个 run 的产物（`command.sh`、`scenario.yaml`、`resolved_config.yaml`、`env_snapshot.txt`、
`facts.json`）随 run 一同归档，`command.sh` 可逐字重放。

> **一个环境前提**：ManiSkill 场景需要 `rlinf/envs/maniskill/assets/`（carrot / partnet_mobility，
> 约 80 MB）。`requirements/embodied/download_assets.sh` 只负责 `~/.maniskill` 下的 bridge/widowx
> 资产，不含这批。缺失时在 env worker 初始化阶段报单行 `FileNotFoundError`。
