# Pi0.5 PPO 性能数据简报（更新至 2026-07-23）

## 测试口径

- 硬件：单机 4×NVIDIA H20。
- 四组：baseline、compile-only、fused-only、fused+compile。
- 稳态口径：每组 9 step，统计第 4–9 step（最后 6 个点）。
- `±` 为 6 个 step 的样本标准差。
- compile 只作用于 rollout；fused 同时作用于 rollout 和 actor 的 18 个 prefix VLM layer。

## 局部算子与模型数据

| 模式 | rollout | actor recompute | rollout vs eager |
|---|---:|---:|---:|
| baseline | 750.90 ms | 688.84 ms | — |
| compile-only | 656.81 ms | 604.80 ms | -12.53% |
| fused-only | 695.64 ms | 629.33 ms | -7.36% |
| fused+compile | 641.70 ms | 586.20 ms | -14.54% |

| profiler 指标 | eager | compile | 变化 |
|---|---:|---:|---:|
| CUDA makespan | 750.85 ms | 658.21 ms | -92.64 ms |
| raw kernel sum | 718.14 ms | 623.95 ms | -94.19 ms |
| stream gap | 32.71 ms | 34.25 ms | +1.54 ms |
| raw launches | 6496 | 3787 | -41.70% |

结论：两个优化都能加速模型；组合不线性相加。compile 的收益来自 kernel 工作量下降，
不是 stream gap 被压缩。

## LIBERO async、4 卡 colocated

工作量：64 env、`group_size=2`、`update_epoch=2`、global batch 128、micro batch 32、
`decoupled_actor_critic`、训练录像开启。

| 组 | actor | rollout epoch | 等 rollout batch | step | step 变化 |
|---|---:|---:|---:|---:|---:|
| baseline | 66.10 s | 78.04 s | 10.09 s | 76.99 s | — |
| compile-only | 62.95 s | 73.82 s | 9.09 s | 72.69 s | **-5.59%** |
| fused-only | 61.38 s | 74.47 s | 11.94 s | 74.12 s | **-3.73%** |
| fused+compile | 63.63 s | 74.38 s | 8.59 s | 72.88 s | **-5.34%** |

结论：compile 和 fused 都有端到端收益；组合与 compile-only 基本持平。async 的 GPU 竞争、
流水线相位和队列等待会重新分配局部节省。

## LIBERO async、4 卡 colocated（CUDA 13 + Torch 2.11）

镜像：
`iaas-us-cn-beijing.cr.volces.com/physicalai/rlinf:90f692bec360b23f23c058439de43dc6bc8df3a6`；
Torch `2.11.0+cu130`、CUDA 13.0、Triton 3.6.0。代码、PVC、配置和工作量均与上一节一致。
四组均完成 9 step，统计第 4–9 步。

| 组 | actor | rollout epoch | 等 rollout batch | step | 相对本镜像 baseline | 相对旧镜像同组 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 64.87±2.08 s | 77.69±1.03 s | 9.38±1.22 s | 74.97±2.00 s | — | **-2.63%** |
| compile-only | 67.53±1.65 s | 72.66±0.43 s | ≈3.35 s | 71.44±3.05 s | **-4.72%** | **-1.72%** |
| fused-only | 61.15±1.33 s | 74.51±1.42 s | 11.55±0.84 s | 73.45±1.69 s | **-2.03%** | **-0.91%** |
| fused+compile | 65.06±1.40 s | 76.75±0.79 s | 11.38±1.04 s | 77.24±2.07 s | +3.02% | +5.97% |

本镜像最佳仍是 compile-only：71.44 s，比旧镜像最佳 72.69 s 快 1.72%。baseline、
compile-only、fused-only 分别比旧镜像快 2.63%、1.72%、0.91%，没有出现大幅框架升级
红利；组合反而慢 5.97%。单轮 6 个稳态样本下，组合回退不宜外推为普遍回归，但足以说明
它不是该 colocated 流水线的可靠最优配置。

compile-only 使 rollout 快 6.47%，但同卡并发下 actor 反而慢 4.09%；fused-only 使 actor
快 5.75%，但等待 rollout batch 增加 23.20%。组合相对 compile-only 的 rollout 慢 5.62%、
等待从约 3.35 s 增到 11.38 s，step 慢 8.12%。因此两项优化改变 GPU 竞争、队列水位和
流水线相位，端到端不能相加。

首步 `time/step` 为 baseline 138.39 s、compile-only 189.44 s、fused-only 125.15 s、
组合 162.46 s。compile-only 冷编译比 baseline 多 51.04 s，按稳态每步节省 3.54 s，
约需 16 个总 step 才能回本。组合在 compile-only 之后运行并复用了 Inductor cache，其
首步不是冷启动可比数据。9 步 `time/step` 之和分别为 741.54、763.19、710.88、772.87 s。

## LIBERO async、2+2 disaggregated

工作量与上一节相同；actor 使用 GPU 0–1，rollout/env 使用 GPU 2–3。

| 组 | actor | rollout epoch | 等 rollout batch | step | step 变化 |
|---|---:|---:|---:|---:|---:|
| baseline | 98.90±1.00 s | 100.85±0.53 s | 2.09 s | 101.64±2.07 s | — |
| compile-only | 97.48±0.72 s | 97.72±0.87 s | 0.09 s | 97.58±0.72 s | **-4.00%** |
| fused-only | 92.15±0.89 s | 98.75±0.74 s | 5.97 s | 98.30±1.77 s | **-3.29%** |
| fused+compile | 93.95±1.96 s | 95.96±0.64 s | 3.43 s | 97.46±1.34 s | **-4.11%** |

结论：2+2 下三种优化仍有约 3%–4% 收益；组合不优于 compile-only。baseline 的 rollout
略慢于 actor，优化后瓶颈转移到 actor/队列侧。

## LIBERO sync、4 卡 colocated（旧工作量，非 async 对齐）

工作量：64 env、`group_size=1`、`update_epoch=1`、global batch 128、micro batch 32、
`actor_critic`、训练录像关闭。该组用于串行阶段归因，不能与上面的 async 吞吐直接比较。

| 组 | rollout predict | 完整 rollout | actor | weight sync | step | step 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 25.95±0.38 s | 55.45±0.66 s | 19.77±0.20 s | 2.01±0.16 s | 77.23±0.82 s | — |
| compile-only | 23.59±0.72 s | 52.69±1.03 s | 19.60±0.09 s | 2.00±0.04 s | 74.30±1.09 s | **-3.80%** |
| fused-only | 24.70±0.35 s | 54.69±0.50 s | 18.74±0.03 s | 1.91±0.02 s | 75.35±0.50 s | **-2.44%** |
| fused+compile | 22.62±0.50 s | 52.81±0.64 s | 18.84±0.07 s | 2.00±0.01 s | 73.66±0.67 s | **-4.63%** |

结论：串行模式中阶段节省会直接累加到 step；Amdahl 固定部分和优化覆盖重叠仍使组合收益
小于两项独立收益之和。compile 首次编译约增加 29–32 s，约需 11 个总 step 才能回本。

## LIBERO sync、4 卡 colocated（与 async 工作量对齐）

工作量与 async 4 卡 colocated 完全对齐：64 env、`group_size=2`、`update_epoch=2`、
global batch 128、micro batch 32、`decoupled_actor_critic`、训练录像开启。四组均 9/9、
status 0；统计第 4–9 步。

| 组 | rollout predict | 完整 rollout | actor | weight sync | step | step 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 25.81±0.63 s | 61.74±0.67 s | 38.78±0.10 s | 1.98±0.04 s | 102.50±0.68 s | — |
| compile-only | 23.05±0.54 s | 58.65±0.95 s | 38.75±0.14 s | 2.03±0.02 s | 99.44±0.94 s | **-2.99%** |
| fused-only | 24.88±0.69 s | 60.50±0.78 s | 37.19±0.27 s | 2.07±0.13 s | 99.77±0.79 s | **-2.66%** |
| fused+compile | 23.46±0.51 s | 57.08±0.94 s | 37.23±0.11 s | 2.04±0.04 s | 96.36±1.01 s | **-5.99%** |

粗略独立样本 Welch 95% 区间：compile 1.95%–4.03%、fused 1.74%–3.59%、组合
4.89%–7.09%，均不跨零。串行恒等式
`step=sync_weights+generate_rollouts+cal_adv_and_returns+actor_training` 在四组中的均值误差均
小于 0.002 s。

| 组 | rollout 节省 | actor 节省 | sync/adv 变化 | 合计 | 实测 step 节省 |
|---|---:|---:|---:|---:|---:|
| compile-only | 3.09 s | 0.03 s | -0.05 s | 3.07 s | 3.07 s |
| fused-only | 1.24 s | 1.58 s | -0.09 s | 2.73 s | 2.73 s |
| fused+compile | 4.65 s | 1.55 s | -0.06 s | 6.14 s | 6.14 s |

结论：对齐工作量后，sync baseline 是 102.50 s，不是旧工作量的 77.23 s。compile 使
predict 快 10.71%、完整 rollout 快 5.00%、step 快 2.99%；fused 使 predict 快 3.61%、
actor 快 4.08%、step 快 2.66%；组合使 step 快 5.99%。组合的 predict 收益只有 9.11%，
仍证明 kernel 覆盖不线性叠加；本次完整 rollout 的非 predict 时间同时下降，使串行 step 的
点估计接近两项 E2E 收益之和，不能把它外推成稳定的算子可加性。

### 与 async 的 apple-to-apple 对比

| 组 | sync step | async step | async 时间下降 | async 吞吐提升 |
|---|---:|---:|---:|---:|
| baseline | 102.50 s | 76.99 s | 24.89% | 33.13% |
| compile-only | 99.44 s | 72.69 s | 26.90% | 36.80% |
| fused-only | 99.77 s | 74.12 s | 25.71% | 34.61% |
| fused+compile | 96.36 s | 72.88 s | 24.37% | 32.21% |

结论：相同工作量下 async 明确更快。async baseline 中 actor/rollout 分别为 66.10/78.04 s，
高于 sync 的 38.78/61.74 s，说明同卡并发竞争使单阶段变慢；但流水线重叠把周期降到
76.99 s，最终吞吐仍比 sync 高 33.13%。sync 的价值是做阶段归因，不是获得更高吞吐。

compile 首步为 134.86 s，较 baseline 首步多 29.30 s；组合首步多 34.72 s。9 步总计：
baseline 925.37 s、compile 932.99 s（慢 0.82%）、fused 900.18 s（快 2.72%）、组合
915.74 s（快 1.04%）。compile-only 约需 11 个总 step 回本，组合约需 7 个。

## ManiSkill async、4 卡 colocated

工作量：160 env、global batch 2560、micro batch 32、`joint_logprob=true`。

| 组 | actor | rollout epoch | env interact | 等 rollout batch | step | step 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 111.90±1.22 s | 230.93±35.24 s | 142.94±27.47 s | 183.10±85.94 s | 297.98±86.63 s | — |
| compile-only | 111.63±0.92 s | 249.67±31.72 s | 166.07±25.87 s | 197.55±46.38 s | 310.85±46.62 s | +4.32% |
| fused-only | 104.75±0.55 s | 248.03±35.10 s | 162.35±28.37 s | 206.60±77.15 s | 313.03±77.64 s | +5.05% |
| fused+compile | 104.22±0.55 s | 235.55±46.32 s | 155.52±36.81 s | 173.55±86.18 s | 278.73±86.62 s | -6.46% |

结论：fused 稳定加速 actor 约 6.5%，但 step 标准差为 47–87 s，四组端到端排序均无统计
可信度。GPU simulator 与 actor/rollout 同卡竞争，并由最慢 rank 放大尾延迟。

## ManiSkill async、2+2 disaggregated

工作量与上一节相同；actor 使用 GPU 0–1，rollout/env 使用 GPU 2–3。

| 组 | actor | rollout epoch | env interact | 等 rollout batch | step | step 变化 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 147.88±0.88 s | 223.95±0.22 s | 92.88±0.12 s | 71.12±0.52 s | 224.00±0.39 s | — |
| compile-only | 146.97±0.28 s | 211.32±0.37 s | 94.26±0.33 s | 61.40±2.65 s | 209.42±2.60 s | **-6.51%** |
| fused-only | 138.22±0.13 s | 217.33±0.77 s | 94.64±0.34 s | 76.67±3.60 s | 215.98±2.61 s | **-3.58%** |
| fused+compile | 137.87±0.29 s | 209.40±1.41 s | 94.09±0.32 s | 66.12±1.75 s | 209.13±2.08 s | **-6.64%** |

结论：2+2 baseline 比 4 卡 colocated 快 24.8%，step 标准差从 86.63 s 降到 0.39 s。
compile 约 6.5%、fused 约 3.6%；组合与 compile-only 不可分辨。

## ManiSkill async、2+1+1 disaggregated

placement：actor 使用 GPU 0–1，rollout 使用 GPU 2，ManiSkill 渲染/仿真独占 GPU 3。
工作量仍为 160 env、global batch 2560、micro batch 32、`joint_logprob=true`。四组均
9/9、status 0；统计第 4–9 步。单 rollout/env rank 使 step 呈稳定的短/长交替，因此同时
给出原始 step 样本标准差和三组完整两步周期的标准差。

| 组 | actor | rollout predict | env interact | 完整 env 交互 | step（原始） | 两步周期折算 | step 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 147.73±0.18 s | 234.28±0.03 s | 139.79±0.34 s | 398.71±0.39 s | 398.28±274.20 s | 398.28±0.40 s | — |
| compile-only | 145.51±0.23 s | 206.67±0.10 s | 139.07±0.46 s | 369.67±0.34 s | 369.00±238.28 s | 369.00±0.52 s | **-7.35%** |
| fused-only（int64 fix） | 138.38±0.16 s | 219.99±0.05 s | 138.98±0.43 s | 383.51±0.37 s | 383.50±266.24 s | 383.50±0.52 s | **-3.71%** |
| fused+compile（int64 fix） | 138.39±0.21 s | 204.81±0.02 s | 139.27±0.50 s | 368.05±0.51 s | 367.55±250.79 s | 367.55±0.61 s | **-7.72%** |

最后 6 步总时间分别为 2389.67、2213.99、2301.01、2205.29 s。组合只比
compile-only 再快 0.39%，说明二者仍覆盖同一段 rollout 计算，不能相加。

原始 fused kernel 在单 rollout rank 的 batch 160 首次 prefix MLP 上稳定崩溃：
`M=154880`、`N=16384`，展平输出元素数为 2,537,553,920，超过 int32 上限
2,147,483,647，触发 Triton illegal memory access。将大矩阵的行/线性 pointer offset
提升到 int64 后，1-step smoke test 和 fused/both 两组 9-step 测试均通过。表中 fused
结果均使用该 correctness fix；未修复版本不支持此 placement/workload。

与 2+2 baseline 相比，2+1+1 baseline 的 actor 基本不变（147.73 vs 147.88 s），但完整
env 交互从 223.95 增至 398.71 s，step 从 224.00 增至 398.28 s（慢 77.8%）。渲染独占
GPU 确实改善了单卡归一化仿真效率：单 rank 环境数从 80 翻倍到 160 时，纯 env interact
仅从 92.88 增至 139.79 s，而不是线性翻倍到 185.76 s；但单 rollout/env rank 的并行度
损失更大。最佳 both 配置仍比 2+2 both 的 209.13 s 慢 75.8%。

结论：4 卡 ManiSkill 不建议 2+1+1。渲染专卡能消除模型争用，却无法抵消 rollout/env
rank 减半；2+2 仍是更优 placement。2+1+1 的优化比例略高于 2+2，但绝对吞吐显著更差。

## Fused kernel 正确性与 `train_expert_only=false`

验证对象：实际 profiling 使用的 fused kernel 加 int64 pointer-offset fix，以及当前 combined
branch `2eb25cfd`。配置字段的准确名称是 `train_expert_only`，不是
`train_with_expert_only`。

| 验证 | 结果 |
|---|---|
| Gemma-2B 真实维度 forward/backward 对 PyTorch reference | PASS；forward 相对误差 0.714%，input grad 1.04%；9 组参数梯度 0.402%–0.932% |
| prefix zero/block mask、suffix adaRMS+block | 3/3 PASS；forward/grad-input 均低于 1.34% |
| int32 边界生产规模 `M=154880,N=16384,K=2048` | 7/7 PASS；抽样覆盖第 131071/131072 行，forward/backward 相对误差 0.202%–0.288%，峰值 24.44 GiB |
| 真实 LIBERO checkpoint，`train_expert_only=false` | PASS；18/18 fused layer，VLM 164/164 参数张量、2,508,531,712/2,508,531,712 元素可训练 |
| 真实 PPO log-prob/value 路径 | PASS；fused forward/backward counter 36/18；prefix/expert q-proj 梯度均非零且有限，prefix 权重 optimizer step 后发生变化 |
| 2-rank、2-H20、FSDP1 `NO_SHARD/use_orig_params=false` | PASS；每 rank 121 个 FSDP module、117/117 FlatParameter 梯度非零且有限，loss 均为 0.6205，峰值各 27.58 GiB |

结论：fused patch 的前向、反向、超 int32 索引和 `train_expert_only=false` 的真实模型/FSDP
路径均已验证；false 模式不是只完成初始化，梯度确实穿过 18 层 prefix VLM 并完成参数更新。
本次是单步正确性 smoke，不等价于多步 loss/收敛曲线验证。

## 16 卡条件预测（非实测）

假设双机 8+8 H20、actor 与 rollout/env 各占一机、弱扩展、RDMA 正常且权重同步可隐藏：

| 组 | 预测稳态 step 收益 |
|---|---:|
| compile-only | 2%–4% |
| fused-only | 2%–3% |
| fused+compile | 3%–5% |

结论：跨机通信或 env CPU/渲染饱和会进一步稀释收益；需双机实测确认。

## 总结论

1. compile 与 fused 的局部收益都可靠。
2. LIBERO async 的稳态端到端收益为 compile 5.59%、fused 3.73%、组合 5.34%；matched
   sync 为 2.99%、2.66%、5.99%。收益取决于执行模式，不能跨模式直接套用。
3. ManiSkill colocated 的主要问题是 GPU simulator 争用和 rank 尾延迟；2+2 更快且稳定。
4. compile 有明显冷启动成本，短任务可能没有总墙钟收益。
5. 相同工作量下 async baseline 比 sync 少 24.89% step 时间、吞吐高 33.13%；sync 更适合
   算子到端到端的串行归因。
6. ManiSkill 2+1+1 虽隔离了渲染，baseline 仍比 2+2 慢 77.8%；单 rollout/env rank 是主因。
   batch 160 还暴露了 fused kernel 的 int32 pointer-offset overflow，需 int64 fix。
7. CUDA 13 + Torch 2.11 下的 LIBERO async 4 卡最佳仍为 compile-only，稳态 71.44 s；
   比旧镜像最佳快 1.72%。fused+compile 本次为 77.24 s，不是可靠最优配置。

## 数据文件

- `PI05_RL_PERF_AUDIT_20260718.csv`
- `PI05_RL_PERF_DISAGG_AUDIT_20260719.csv`
- `PI05_RL_PERF_MANISKILL_AUDIT_20260719.csv`
- `PI05_RL_PERF_MANISKILL_DISAGG_AUDIT_20260719.csv`
- `PI05_RL_PERF_SYNC_LIBERO_AUDIT_20260720.csv`
- `PI05_RL_PERF_SYNC_MATCHED_LIBERO_AUDIT_20260720.csv`
- `PI05_RL_PERF_MANISKILL_211_AUDIT_20260721.csv`
- `PI05_RL_PERF_CUDA13_TORCH211_COLOCATED_AUDIT_20260723.csv`
