# Pi0.5 async PPO 性能结论复核（2026-07-18）

本文复核 `PI05_RL_PERF_SUMMARY.md` 的关键数据与端到端结论。复测环境是
`rlinf-0` 上的 4×H20、PyTorch 2.6.0+cu124、Pi0.5、LIBERO-spatial、async PPO、
actor/env/rollout collocated。2026-07-19 又以相同 workload 补跑了 actor 2 卡、
rollout/env 2 卡的 disaggregated 4-way 对照。原始运行日志保存在容器
`/workspace/audit_{A,B,C,D}_*.log` 和 `/workspace/audit_disagg_{A,B,C,D}_*.log`；逐 step
数据见 `PI05_RL_PERF_AUDIT_20260718.csv` 与
`PI05_RL_PERF_DISAGG_AUDIT_20260719.csv`。
同日还补跑了 ManiSkill GPU-sim、4 卡 async+collocated 最优配置的 4-way 对照，见第 8 节
和 `PI05_RL_PERF_MANISKILL_AUDIT_20260719.csv`。
2026-07-20 又补跑了 4 卡 LIBERO sync 串行 4-way，用于直接观察局部算子节省如何累加到
rollout、actor 和完整 step，见第 10 节与 `PI05_RL_PERF_SYNC_LIBERO_AUDIT_20260720.csv`。

## 结论摘要

1. **两个局部优化都真实有效。** 单层 fused、完整模型 fused、rollout
   `torch.compile` 的收益均复现。
2. **原文“compile 在真实 async 中为 0%、端到端为 0%”不成立。** 严格 off/on
   复测中，rollout compile 使稳态 step 改善约 **4%–6%**；fused-only 改善约
   **3%–5%**。
3. **“compile 主要填 launch gap，而 gap 被 collocation 填掉”的解释是错误的。**
   修正 profiler 的父子事件重复计数后，compile 的约 92.6 ms 收益全部来自 raw
   kernel 时间下降；stream gap 没有下降。
4. **两项优化不能线性叠加。** 它们覆盖了相同的 elementwise/copy/launch 胶水；更重要的
   是 async collocation 会重排 actor/rollout 相位、GPU 竞争和 rollout-store 等待。
   组合组的 step 约 -5%，与 compile-only 基本相同。
5. 当前问题不是“完全没有端到端收益”，而是 **收益只有约 5%、单 step 抖动很大、旧 A/B
   开关错误、两项优化高度重叠**，因此短跑容易得出“无收益”。
6. **2+2 disaggregated 不会让收益归零。** 严格复测得到 compile -4.0%、fused -3.3%、
   组合 -4.1%。baseline 的 rollout 比 actor 慢约 2 s；compile 消掉这部分不平衡后，actor
   才成为限制，因此收益是部分兑现，不是简单的“单侧优化全部被 max 吃掉”。

## 1. 原始实验存在的关键问题

### 1.1 所谓 baseline 实际开启了 compile

以下历史日志的最终 Hydra 配置都记录了 `rollout.enable_torch_compile: true`：

- `/workspace/spd_A_baseline.log`
- `/workspace/spd_B_fused.log`
- `/workspace/rollout_bd_nocompile.log`
- `/workspace/spd_C_compile.log`
- `/workspace/rollout_bd_compile.log`

rollout worker 会直接按该最终配置调用 `hf_model.enable_torch_compile()`。因此
`spd_A` 对 `spd_C`、`rollout_bd_nocompile` 对 `rollout_bd_compile` 都不是有效的
compile off/on A/B，不能支持“真实 infer 0%”或“端到端 0%”。

### 1.2 launch-gap 计算重复计数

旧 `bench_actor_launchbound.py` 对 `prof.key_averages()` 中所有 `self_cuda_time_total`
求和，重复包含了父级 PyTorch op 和子级 CUDA kernel，得到 -98% 到 -206% 的
“GPU idle”。负 idle 在物理上不可能，所以该指标不能用于判断 launch-bound。

新增的 `audit_pi05_compile_gaps.py` 只累计 `DeviceType.CUDA` raw events：

| 模式 | CUDA makespan | raw kernel sum | stream gap | raw launches |
|---|---:|---:|---:|---:|
| eager rollout | 750.85 ms | 718.14 ms | 32.71 ms (4.4%) | 6496 |
| compiled rollout | 658.21 ms | 623.95 ms | 34.25 ms (5.2%) | 3787 |

compile 后 raw kernel 时间减少 **94.19 ms**，而 gap 增加 1.54 ms。它确实通过融合
减少了 elementwise/copy kernel 和显存流量；不是把可被 actor 填充的空隙压掉。

### 1.3 历史端到端样本太少

旧 A/B/C/D 通常只有首步加 3 个稳态点，而 async step 的标准差可达数秒。首步还包含
模型加载、Triton/Inductor 编译和管线灌入。3 个点不足以分辨约 4%–6% 的真实收益。

## 2. 局部数据复测

### 2.1 GEMM

`bench_cublas_gemm.py` 复测：

| shape | TFLOPS | 相对 148 TFLOPS |
|---|---:|---:|
| prefix 968，2048→2048 | 100.2 | 67.7% |
| prefix 968，2048→16384 | 124.0 | 83.8% |
| prefix 968，16384→2048 | 108.6 | 73.4% |
| batch16×968，2048→2048 | 139.1 | 94.0% |

原文“最佳大 batch GEMM 达 94%”可靠，但不应泛化成“所有 Pi0.5 GEMM 都达 94%、
没有任何 kernel 空间”。更准确的说法是：大 batch 投影接近峰值，小 shape 为峰值的
约 68%–84%；整体仍以高效 GEMM 为主，fused/compile 的主要空间在 GEMM 周边。

### 2.2 fused 与 compile microbench

同一张 GPU、batch=16、num_steps=3：

| 模式 | rollout | actor recompute | 相对 eager rollout |
|---|---:|---:|---:|
| eager | 750.90 ms | 688.84 ms | — |
| fused-only | 695.64 ms | 629.33 ms | -7.36% |
| compile-only | 656.81 ms | 604.80 ms | -12.53% |
| fused + compile | 641.70 ms | 586.20 ms | -14.54% |

单层 masked Gemma forward 也复现 `4.521→3.803 ms`（-15.9%）。两个优化局部都有
收益，但组合只比 compile-only 再快约 2.3%，说明覆盖高度重叠。

fused/eager 端到端 parity 复测通过：logprob 最大相对误差 `7.16e-8`，value 误差约
1% 量级。compile 的平均 logprob 误差较小，但不同复测中最大绝对误差为 0.17–0.47；
上线前还需要以 PPO importance ratio 的 p99/p99.9 而不是仅 max/mean 做门槛测试。

## 3. 严格 4-way 端到端复测

四组顺序运行 9 个 step，配置、seed、模型和硬件完全相同，只切换两个开关；关闭 checkpoint。
下面剔除首步和前两个 async 相位过渡点，统计最后 6 个 step：

| 组 | actor training | rollout epoch | 等 rollout batch | step | step vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 66.10 s | 78.04 s | 10.09 s | 76.99 s | — |
| compile-only | 62.95 s | 73.82 s | 9.09 s | 72.69 s | **-5.59%** |
| fused-only | 61.38 s | 74.47 s | 11.94 s | 74.12 s | **-3.73%** |
| fused + compile | 63.63 s | 74.38 s | 8.59 s | 72.88 s | **-5.34%** |

只剔除首步、使用 8 个点时，step 改善分别为 compile -6.14%、fused -4.56%、组合
-5.53%。只看最后 5 个点时分别为 -4.33%、-2.55%、-4.93%。不同窗口下幅度有变化，
但 compile 和组合稳定落在约 4%–6%；fused-only 更容易受 async 相位抖动影响。

首步不能混入稳态：compile-only 的 `construct_rollout_batch` 首次等待为 93.46 s，组合组
为 97.46 s，baseline 为 62.33 s。短任务可能被首次编译开销完全抵消。

## 4. 为什么局部收益在端到端被稀释、组合不叠加

### 4.1 覆盖率，而不是“compile gap 蒸发”

compile-only 只作用在 rollout 模型；单次模型调用快 12.5%，但 rollout epoch 还包含
LIBERO env 物理/渲染和通信，最终 rollout epoch 只快 5.4%。这符合 Amdahl 定律。

fused 同时覆盖 actor/rollout 的 prefix VLM，actor 快 7.1%、rollout 快 4.6%，但仍只覆盖
prefix 层的部分工作量；它不是整 step 16% 或 8% 的优化。

### 4.2 async step 不是简单的 `max(actor, rollout)`

主线程的 step 近似为：

```text
construct_rollout_batch 等待 + actor_training + advantage/调度杂项
```

rollout/env 是长驻异步循环，`run_interact_once` 属于另一个时间轴。rollout-store 还受
staleness threshold、版本分布、1 秒轮询和跨 rank barrier 影响。因此优化改变生产/消费速率后，
会改变下一步的数据可用时刻和 actor/rollout 的重叠相位。

直接证据是：

- fused-only 把 actor 从 66.10 s 降到 61.38 s（省 4.72 s），但等 rollout batch 从
  10.09 s 增到 11.94 s（多 1.85 s），step 最终只省 2.87 s。
- compile-only 虽未 compile actor，却把 actor wall time降到 62.95 s；原因是 rollout
  少占 GPU 后，同卡 actor 的排队/竞争也减少。
- 组合组的 actor 又回到 63.63 s，同时等待降到 8.59 s。总 step 与 compile-only 相近，
  但省下的时间分布在不同阶段。

所以 collocation 不会让“减少实际 kernel 工作”失效；它会让组件级 wall time和队列等待
发生非线性重分配。

### 4.3 两个优化覆盖重叠

fused 和 Inductor 都在减少 RMSNorm、GELU、gate/mul、cast、clone/copy、residual 等胶水。
microbench 中两者独立为 -7.4% 和 -12.5%，组合只有 -14.5%，不是约 -19%。端到端再经过
env、队列和相位稀释，组合自然与 compile-only 接近。

### 4.4 噪声和统计口径会掩盖约 5% 的收益

单个 async step 在本次复测中可从 61 s 波动到 92 s；进入后半稳态后仍有约 1–3 s 抖动。
如果只取 3 个点、把首步编译混入、或对比不同的 sync/compile 配置，约 3–5 s 的收益很容易
被判为 0%。

## 5. 建议

1. 对当前 4×H20、flow_sde、固定 shape 配置，**保留 rollout compile，预期稳态 step
   收益约 4%–6%**；记录并摊销首次编译开销。不要推广到已知会 recompile 的
   `joint_logprob=True` 动态 shape 配置。
2. fused 自身可靠，适合 actor、eval 和 rollout-only；但在当前组合中与 compile 重叠，
   不应预期线性叠加。若只追求此配置的 step time，compile-only 与组合基本相当。
3. 在设为默认前增加 compile 的数值门槛：固定输入比较 eager rollout logprob 与 eager actor
   recompute，报告 `exp(new_logp-old_logp)` 的 p50/p99/p99.9/max 及 clip fraction。
4. 后续性能 A/B 至少跑 20 个稳态 step，使用 ABBA 顺序，并同时报告：连续 N 步总墙钟、
   `step`、`actor_training`、`construct_rollout_batch`、`run_interact_once` 和 GPU active time。
5. profiler 只能累计 raw CUDA device events；不要对包含父子层级的 `key_averages()` CUDA
   时间直接求和。
6. 本次 pod 只有 4 张 H20。16 卡结论只能基于 2+2 数据和一份很短的历史 4+4 日志做条件
   外推；跨机正式上线前仍需按相同矩阵复跑。

## 6. 2+2 disaggregated 严格复测（2026-07-19）

### 6.1 配置与结果

四组仍顺序运行 9 个 step。workload 与第 3 节完全相同：`total_num_envs=64`、
`global_batch_size=128`、`micro_batch_size=32`、seed 42；只把 placement 改为 actor
GPU 0–1、rollout/env GPU 2–3。为避免把 placement 与另一个变量混在一起，四组都显式设置
`actor.sync_weight_no_wait=true`，关闭 checkpoint。最后 6 个 step 的结果为：

| 组 | actor training | rollout epoch | 等 rollout batch | step | step vs baseline |
|---|---:|---:|---:|---:|---:|
| baseline | 98.90 ± 1.00 s | 100.85 ± 0.53 s | 2.09 s | 101.64 ± 2.07 s | — |
| compile-only | 97.48 ± 0.72 s | 97.72 ± 0.87 s | 0.09 s | 97.58 ± 0.72 s | **-4.00%** |
| fused-only | 92.15 ± 0.89 s | 98.75 ± 0.74 s | 5.97 s | 98.30 ± 1.77 s | **-3.29%** |
| fused + compile | 93.95 ± 1.96 s | 95.96 ± 0.64 s | 3.43 s | 97.46 ± 1.34 s | **-4.11%** |

表中的 `±` 是六个 step 的样本标准差，不是置信区间。连续六步总计分别为 609.86、
585.47、589.82、584.78 s。只剔除首步使用 8 个点时，step 改善分别为 -4.79%、
-3.44%、-5.42%；只看最后 5 个点时为 -4.13%、-3.40%、-4.52%。窗口变化不改变
“约 3%–4%、组合不叠加”的结论。

compile 与组合只差 0.12 s/step，远小于样本波动，不能认为组合优于 compile-only。
使用两组各 6 个点的独立样本近似，step 差的 95% 区间折算为收益约为：compile
2.0%–6.0%、fused 0.9%–5.7%、组合 1.9%–6.3%。该区间没有校正 async 自相关和顺序效应，
只能用于说明小于约 1% 的组间差异不可分辨。

首步 step 为 baseline 196.4 s、compile 257.5 s、fused 188.4 s、组合 238.4 s。按稳态
点估计，compile-only 约需再运行 16 个稳态 step（总计约 17 step）才能摊平相对 baseline
多出的首步成本；组合约需 11 个后续稳态 step。这个 break-even 只适用于本次缓存和 shape。

### 6.2 为什么 disaggregated 下仍有约 4% 收益

原文“`step=max(actor, rollout)`，所以单侧优化必然为 0%”过于绝对。本次 baseline 的两侧
并不完全相等：rollout epoch 100.85 s，actor 98.90 s，rollout 慢约 1.95 s，而且主线程还会
等 rollout batch 约 2.09 s。

- compile 把 rollout 降到 97.72 s，并把等 batch 降到几乎为零；此时 actor/rollout 基本
  平衡，所以 3.1% 的 rollout-epoch 改善最终兑现为约 4.0% step 改善。
- fused 把 actor 降 6.8%，但 rollout 只降 2.1%；rollout 成为明确瓶颈，actor 省下的时间
  大部分转成等 batch，step 只改善 3.3%。
- 组合把两侧都降约 5%，但 async 相位和 1 秒轮询让 step 仍约 97.5 s；相对 compile-only
  的增量小于噪声。局部覆盖重叠在 disaggregated 下仍然存在。

因此更准确的模型是 `step` 接近两条流水线较慢的一侧，再叠加队列/轮询相位；优化较慢侧可
兑现到重新平衡为止，优化较快侧则主要增加 slack。它不是严格的逐 step `max()`，更不是两侧
时间或加速比相加。

## 7. 双机 16 卡的条件外推

以下不是实测。主推断假设两台 8×H20：actor 独占一台 8 卡机，rollout/env 独占另一台；
采用弱扩展，把 `global_batch_size` 扩到 512、`total_num_envs` 扩到 256，保持每个 actor rank
64 个样本、每个 rollout rank 32 个 env，以及 `micro_batch_size=32`。同时假设 RDMA/NCCL
正常，`sync_weight_no_wait=true` 能把 actor→rollout patch sync 隐藏在约 100 s 的流水线内。

一份历史单机 8 卡 4+4 日志只能提供很弱的尺度锚点：它实际开启了 compile，只有 3 个较稳
step，actor 约 105 s、rollout 约 103–105 s、step 约 105 s。与本次 2+2 compile 的 97.6 s
相比，说明弱扩展并非理想常数时间，4+4 已有约 8% 的 host/调度/规模开销；不能拿 2+2 的
百分比原样复制到 8+8。

在上述假设下，8+8 双机的合理预期是：

| 组 | 2+2 实测 step 收益 | 8+8 双机预测 | 主要限制 |
|---|---:|---:|---|
| compile-only | 4.00% | **2%–4%** | rollout 加速后 actor/通信成为瓶颈 |
| fused-only | 3.29% | **2%–3%** | actor 更快，但 rollout/env 仍较慢 |
| fused + compile | 4.11% | **3%–5%** | 两侧都改善，但局部覆盖仍重叠 |

这个范围刻意没有相加。用历史 4+4 的约 105 s actor 和约 108 s eager-rollout 粗估，compile
会先把 rollout 拉回 104–105 s，收益约 2%–3%；fused 后 rollout 仍是瓶颈，约 2%；组合把
两侧都拉到约 100–103 s，约 4%–5%。额外的未隐藏跨机开销 `C` 会按
`gain ≈ saved_compute / (baseline_step + C)` 稀释收益。例如共同增加 5–10 s 时，4% 的
本地收益会降到约 3.6%–3.8%，而不是变成 0%。

风险主要有三个：

1. 当前 patch sync 的本机稳态 actor 计时通常约 1.0–1.4 s，且 no-wait 已隐藏；跨机若走
   健康 RDMA，即使变成数秒，仍远小于一个流水线周期。若退化到 TCP、超过周期或频繁
   coalesce，它会成为新瓶颈，以上范围会偏乐观。
2. 把 256 个 LIBERO env 全放在 rollout 节点可能先打满 CPU/渲染。这样 rollout 中不可被
   compile/fused 覆盖的比例上升，compile/fused-only 可能分别降到约 1%–3% 和 1%–2%，
   组合约 2%–4%。
3. 如果是固定 workload 的强扩展，而不是上述弱扩展，每卡只有更小 batch/env 数，GPU 利用率、
   microbatch 和通信占比都会改变；不应使用本表，且 8+8 很可能过度配置。

16 卡正式验证应至少采 20 个稳态 step，使用 ABBA 顺序，并额外记录 patch-sync 实际完成
时间、coalesced 次数、NIC 吞吐、rollout 节点 CPU 利用率。只有确认 sync 被隐藏且 env 未打满
CPU 后，才能把预期收紧到表中的上半区。

## 8. ManiSkill GPU-sim、4 卡 collocated 复测（2026-07-19）

### 8.1 配置与结果

使用 `maniskill_async_ppo_openpi_pi05_best_4gpu`：actor/env/rollout collocated 在 4 张 H20，
`total_num_envs=160`、`global_batch_size=2560`、`micro_batch_size=32`、每卡 40 个 env、
`num_steps=4`、`flow_noise`、`joint_logprob=true`、`ROBOT_PLATFORM=BRIDGE`。四组都显式设置
`sync_weight_no_wait=true`、关闭 checkpoint，各顺序运行 9 个 step；只切换 compile/fused。
原始容器日志是 `/workspace/audit_maniskill_{A,B,C,D}_*.log`。

最后 6 个 step 的结果为：

| 组 | actor training | rollout epoch | env interact | 等 rollout batch | step | step vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 111.90 ± 1.22 s | 230.93 ± 35.24 s | 142.94 ± 27.47 s | 183.10 ± 85.94 s | 297.98 ± 86.63 s | — |
| compile-only | 111.63 ± 0.92 s | 249.67 ± 31.72 s | 166.07 ± 25.87 s | 197.55 ± 46.38 s | 310.85 ± 46.62 s | **+4.32%** |
| fused-only | 104.75 ± 0.55 s | 248.03 ± 35.10 s | 162.35 ± 28.37 s | 206.60 ± 77.15 s | 313.03 ± 77.64 s | **+5.05%** |
| fused + compile | 104.22 ± 0.55 s | 235.55 ± 46.32 s | 155.52 ± 36.81 s | 173.55 ± 86.18 s | 278.73 ± 86.62 s | **-6.46%** |

正号表示变慢。六步 step 总计分别为 1787.9、1865.1、1878.2、1672.4 s。最后六步原始
step 数组是：

- baseline：`[228.3, 274.3, 299.5, 463.8, 230.4, 291.6]`
- compile：`[242.6, 337.5, 263.0, 327.2, 359.0, 335.8]`
- fused：`[175.8, 317.1, 347.6, 293.7, 334.8, 409.2]`
- both：`[204.5, 269.9, 299.5, 254.2, 438.6, 205.7]`

窗口点估计的方向相对稳定：只剔除首步的 8 个点为 compile +4.08%、fused +5.36%、
组合 -6.64%；最后 5 个点为 +4.03%、+9.16%、-5.88%。但这些窗口共享同一批分钟级
outlier，不能当作独立重复。

### 8.2 什么可以相信，什么不能

可以相信的是 **fused 确实稳定加速 actor**：fused-only 把 actor 从 111.90 s 降到
104.75 s（-6.39%），组合降到 104.22 s（-6.87%）。六步标准差只有约 0.5–1.2 s，
该收益远大于 actor 自身波动。compile-only 的 actor 为 111.63 s，与 baseline 一致，也符合
compile 只作用于 rollout 的实现。

不能相信的是当前四个端到端点估计的精确排序。step 标准差为 47–87 s，而要分辨的差只有
13–19 s。用两组各 6 个点做粗略独立样本估计，step 差的 95% 区间为：compile 相对
baseline `+12.9 s [-76.6, +102.4]`、fused `+15.1 s [-90.8, +120.9]`、组合
`-19.3 s [-130.7, +92.2]`，全部跨零。组合的 -6.46% 只能叫点估计，不能称为已证明收益；
同样也不能断言 compile/fused 端到端一定退化 4%–5%。

compile 首步为 304.0 s，baseline 为 240.5 s，多约 63.5 s；组合首步 276.2 s，多约
35.7 s。compile 没有 backend failure，日志也没有显式 recompile warning；但 compile-only
的稳态 rollout epoch 点估计反而比 baseline 慢 8.1%。仅凭当前计时不能证明原因一定是
`joint_logprob` 动态 shape 重编译，后续需要用 `TORCH_LOGS=recompiles` 或 Dynamo counters
做短诊断。

### 8.3 为什么 actor 收益没有稳定落到端到端

ManiSkill 与 LIBERO 最大的不同是 GPU simulator 也与 actor/rollout collocated。baseline
最后六步的 `env_interact_step` 从 96.8 s 到 178.0 s，`construct_rollout_batch` 从 113.6 s
到 346.8 s。运行中还直接观察到四个 rank 的轨迹到达相差约 1–2 分钟。主线程仍近似为：

```text
step ≈ actor_training + construct_rollout_batch + 少量调度开销
```

fused 节省的约 7.2–7.7 s actor 时间，只有 batch-wait 标准差的约十分之一，容易全部落入
rollout/sim 尾延迟。compile/fused/both 的 env-interact 点估计还分别比 baseline 高约 16%、
14%、9%；这可能来自同卡竞争、仿真轨迹差异、async 相位和顺序效应的组合，当前样本不能
进一步归因。关键事实是：端到端已经由最慢 env/rollout rank 主导，不再由稳定的 actor 均值主导。

### 8.4 建议

1. 当前 `joint_logprob=true` ManiSkill 配置 **不要默认开启 rollout compile**：存在显著首步
   成本，且本次 compile-only 没有任何端到端或 rollout 正收益证据。先用 recompilation
   counters 确认 shape 行为。
2. fused 可保留为 actor 侧的有效优化，但在当前 GPU-sim collocation 下不能声称端到端收益；
   先解决 env/rollout rank straggler，actor 的约 6.5% 才有机会兑现。
3. 下一轮先记录每个 env/rollout rank 的 `run_interact_once`、model predict 和 simulator step
   p50/p95/max，而不是立即把同样的 4-way 延长到更多 step。当前全局均值无法指出哪个 rank、
   哪个子阶段造成 1–2 分钟尾部。
4. 若必须给端到端置信结论，应使用 ABBA 顺序和至少 20 个稳态 step；但在当前约 3–5 分钟
   单步下成本很高，优先做 rank-level 诊断更划算。

## 9. ManiSkill GPU-sim、2+2 disaggregated 复测（2026-07-19）

### 9.1 配置与结果

workload 与第 8 节完全相同，只把 placement 改为 actor GPU 0–1、rollout/env GPU 2–3：
`total_num_envs=160`、`global_batch_size=2560`、`micro_batch_size=32`、
`joint_logprob=true`、`sync_weight_no_wait=true`。因此每个 rollout/env rank 从 40 个 env 增到
80 个，每个 actor rank 的训练样本也翻倍。这个实验的目的是隔离 placement，不是重新搜索一套
吞吐最优的 disaggregated workload。四组顺序运行 9 个 step，仍统计最后 6 个点。

| 组 | actor training | rollout epoch | env interact | 等 rollout batch | step | step vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 147.88 ± 0.88 s | 223.95 ± 0.22 s | 92.88 ± 0.12 s | 71.12 ± 0.52 s | 224.00 ± 0.39 s | — |
| compile-only | 146.97 ± 0.28 s | 211.32 ± 0.37 s | 94.26 ± 0.33 s | 61.40 ± 2.65 s | 209.42 ± 2.60 s | **-6.51%** |
| fused-only | 138.22 ± 0.13 s | 217.33 ± 0.77 s | 94.64 ± 0.34 s | 76.67 ± 3.60 s | 215.98 ± 2.61 s | **-3.58%** |
| fused + compile | 137.87 ± 0.29 s | 209.40 ± 1.41 s | 94.09 ± 0.32 s | 66.12 ± 1.75 s | 209.13 ± 2.08 s | **-6.64%** |

连续六步总计分别为 1344.0、1256.5、1295.9、1254.8 s。baseline 的六个 step 是
`[224.3, 224.1, 223.3, 224.0, 223.9, 224.4]`；compile 是
`[211.7, 204.3, 210.1, 209.6, 210.4, 210.4]`；fused 是
`[210.8, 217.2, 216.8, 217.9, 216.0, 217.2]`；组合是
`[204.9, 210.3, 210.1, 209.7, 209.9, 209.9]`。

用两组各 6 个点做与前文相同的粗略独立样本近似，step 收益的 95% 区间为：compile
5.57%–7.45%、fused 2.64%–4.52%、组合 5.88%–7.39%，三项都不跨零。组合相对
compile-only 只快 0.28 s（0.14%），差值区间为 `-2.95 s` 到 `+2.38 s`；换成组合的
相对收益约为 -1.14% 到 +1.41%，不能认为组合优于 compile-only。

原始逐步数据保存在 `PI05_RL_PERF_MANISKILL_DISAGG_AUDIT_20260719.csv`，容器日志为
`/workspace/audit_maniskill_disagg_{A,B,C,D}_*.log`。两组 fused 配置的两个 actor 和两个
rollout worker 都打印了“替换 18 层”的确认，compile 配置也在最终 Hydra dump 中为 true。

### 9.2 为什么“只换仿真后端”却会让 colocated 波动这么大

从算法接口看是换 backend；从 GPU 资源图看不是。LIBERO 的主要仿真压力在 CPU，而
ManiSkill GPU-sim 会在每张 placement GPU 上增加仿真 CUDA context、物理/渲染 kernel 和
reset 工作。4 卡 colocated 时 actor、rollout model 和 simulator 三类进程争用同一张 H20；
2+2 后 actor 与 rollout/simulator 的 CUDA context 被物理隔离。

`EnvWorker.env_interact_step` 的 timer 只包住 `chunk_step`，所以它没有包含等待 rollout action
的 channel 时间；但它记录的是墙钟时间，仍包含 simulator kernel 在 CUDA 队列中被 model
kernel 抢占或延迟调度的时间。ManiSkill 的 `chunk_step` 又会逐个执行 action chunk，并在一个
chunk 结束后只 reset 本轮 done 的 env 子集；不同轨迹的 done/reset 数会产生小幅真实变化。
这个小差异在 colocated 的 context 竞争下形成尾延迟，最后由四个 rollout/env rank 中最慢的一
个通过 async 队列和 batch join 放大到全局 step。

2+2 baseline 给出了比推测更强的反证：每个 env rank 的负载从 40 个翻到 80 个，
`env_interact_step` 却从 colocated 的 142.94 ± 27.47 s 降到 92.88 ± 0.12 s；step 从
297.98 ± 86.63 s 降到 224.00 ± 0.39 s。也就是说，env 每 rank 数量翻倍后仍快 35%，
env 的样本标准差缩小约 235 倍，step 标准差缩小约 222 倍。运行中两个 actor rank 的轨迹
到达也基本同步，不再出现 colocated 的 1–2 分钟 rank skew。因此主因不是 ManiSkill task
随机性或 reset 本身，而是 GPU collocation 的资源竞争和最慢 rank 放大。

2+2 的 actor 从 111.90 s 增到 147.88 s 是预期代价：actor GPU 数从 4 张减到 2 张，每 rank
样本翻倍。即便如此，总 step 仍快 24.8%，说明原 colocated placement 为了多给 actor 两张
逻辑 rank，付出了更大的 rollout/simulator 争用成本。

### 9.3 四项收益为什么仍不相加

- compile 基本不改变 actor（-0.62%），把 rollout epoch 降 5.64%，最终兑现为 6.51%
  step 收益；它优化了 baseline 中明确较慢的 rollout 侧。
- fused 把 actor 降 6.54%，rollout 降 2.95%，但等 batch 从 71.12 s 增到 76.67 s；actor
  节约的一部分变成流水线 slack，step 只改善 3.58%。
- 组合把 actor 降 6.77%、rollout 降 6.50%，step 改善 6.64%。它与 compile-only 的
  0.14% 差异不可分辨，而不是 6.51% + 3.58%。

原因仍是两层非加性。kernel 层上，compile 与 fused 都覆盖 prefix VLM 的同一片计算，组合时
不能把两份局部节省线性相加；系统层上，async step 由较慢流水线和队列相位决定，actor 侧的
额外节省在 rollout 仍约 209 s 时主要表现为等待。这里组合的 actor 确实比 compile-only 快约
9.1 s，但 rollout 只再快约 1.9 s，最终 step 只快 0.28 s。

首步分别为 baseline 361.0 s、compile 427.2 s、fused 344.4 s、组合 390.0 s。compile-only
相对 baseline 多 66.2 s，按每个稳态 step 节约 14.6 s 粗估，需要约 5 个后续稳态 step 才能
摊平。组合排在 compile-only 之后运行，可能复用了 Inductor 磁盘缓存，所以它的 390.0 s 不能
当作完全冷启动成本；要正式比较启动开销，需要清理或隔离 compile cache，并使用 ABBA 顺序。

### 9.4 当前结论

1. 对当前 4×H20 ManiSkill workload，2+2 比 4 卡 colocated 明显更快、更稳定；在重新搜索
   workload 前，它应作为性能基线和默认候选 placement。
2. 在 2+2 下 compile-only 有可靠的约 6.5% 稳态收益；fused-only 有可靠的约 3.6% E2E
   收益和约 6.5% actor 收益；组合约 6.6%，与 compile-only 不可分辨。只追求 step time 时
   优先 compile；fused 仍可保留给 actor/eval 或其他 placement。
3. 当前每组仅 6 个稳态点且顺序固定。绝对数值上线前仍应做至少 20 个稳态 step 的 ABBA；
   但 placement 对方差的改善超过两个数量级，关于 colocated 波动根因的结论已经很强。

## 10. LIBERO sync 串行 4-way 复测（2026-07-20）

### 10.1 为什么这组能直接看端到端兑现

使用标准 `EmbodiedRunner` sync PPO 路径，actor/env/rollout collocated 在 4 张 H20。固定
`total_num_envs=64`、`max_episode_steps=120`、`rollout_epoch=1`、
`global_batch_size=128`、`micro_batch_size=32`、seed 42、`flow_sde`、`num_steps=3`；关闭
训练/评测录像、validation 和 checkpoint。算法保持 sync 配置默认的 `actor_critic`、
`group_size=1`、`update_epoch=1`。四组顺序运行 9 个 step，统计第 4–9 步。

sync 的主线程严格串行：

```text
step = sync_weights + generate_rollouts + cal_adv_and_returns + actor.run_training
```

最后六步中，四组按上述四项相加与日志 `step` 的差都小于 0.003 s。因此这里不存在 async
流水线覆盖、staleness 或 rollout-store 相位变化；某阶段节约多少墙钟，原则上就直接从 step
中减去多少。

### 10.2 稳态结果

| 组 | rollout predict | 完整 rollout | actor training | weight sync | step | step vs baseline |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 25.95 ± 0.38 s | 55.45 ± 0.66 s | 19.77 ± 0.20 s | 2.01 ± 0.16 s | 77.23 ± 0.82 s | — |
| compile-only | 23.59 ± 0.72 s | 52.69 ± 1.03 s | 19.60 ± 0.09 s | 2.00 ± 0.04 s | 74.30 ± 1.09 s | **-3.80%** |
| fused-only | 24.70 ± 0.35 s | 54.69 ± 0.50 s | 18.74 ± 0.03 s | 1.91 ± 0.02 s | 75.35 ± 0.50 s | **-2.44%** |
| fused + compile | 22.62 ± 0.50 s | 52.81 ± 0.64 s | 18.84 ± 0.07 s | 2.00 ± 0.01 s | 73.66 ± 0.67 s | **-4.63%** |

连续六步总计分别为 463.40、445.79、452.11、441.94 s。粗略独立样本近似下，step
收益的 95% 区间为：compile 2.39%–5.21%、fused 1.44%–3.43%、组合
3.53%–5.73%，三项均不跨零。组合相对 compile-only 快 0.64 s（0.87%），但该增量的
粗略区间为 -0.51% 到 +2.24%，当前 6 个点不能证明组合稳定优于 compile-only。

逐阶段的收益与兑现关系是：

| 组 | predict 收益 | rollout 节省 | actor 节省 | sync 节省 | 三段总节省 | 实测 step 节省 |
|---|---:|---:|---:|---:|---:|---:|
| compile-only | 9.09% | 2.75 s | 0.17 s | 0.01 s | 2.94 s | 2.93 s |
| fused-only | 4.84% | 0.75 s | 1.03 s | 0.10 s | 1.88 s | 1.88 s |
| fused + compile | 12.84% | 2.63 s | 0.94 s | 0.01 s | 3.58 s | 3.58 s |

这张表给出了“算子本身到底在 E2E 中提升多少”的直接答案：当前 sync workload 中，fused
使 rollout 模型预测快 4.84%、actor training 快 5.19%，最终完整 step 快 2.44%；compile
使预测快 9.09%，最终 step 快 3.80%；组合预测快 12.84%、actor 快 4.73%，最终 step 快
4.63%。局部百分比不能直接相加，因为 baseline 中模型预测只占 step 的约 33.6%，actor 只占
25.6%，其余是仿真、图像/数据处理、通信和权重同步。

### 10.3 为什么组合仍小于两个 E2E 收益之和

compile 与 fused 的独立 step 点估计相加为 6.24%，组合只有 4.63%。sync 已排除流水线覆盖，
剩下的差异主要来自两点：

1. 两者在 rollout prefix VLM 上覆盖重叠。预测时间的独立收益为 9.09% 和 4.84%，组合是
   12.84%，本身就小于 13.93%。
2. 组合相对 compile-only 虽把 `rollout/predict` 再缩短 0.98 s，但完整 rollout 反而慢
   0.12 s（52.81 vs 52.69 s）；这约 1.1 s 被非 predict 的 env、预/后处理和 channel
   墙钟变化抵消。组合相对 compile 的可见增量最终主要来自 actor 快 0.77 s。

所以 async 不是非加性的唯一原因。即使完全串行，kernel 覆盖重叠和 Amdahl 固定部分仍会让
组合小于两个百分比之和；sync 只是让每个阶段已经产生的墙钟节省不再被并行 slack 吞掉。

### 10.4 冷启动与短任务

首步 step 为 baseline 80.78 s、compile 109.4 s、fused 78.41 s、组合 113.1 s。compile
和组合的首次图编译分别增加约 28.6 s 和 32.3 s。按稳态每步节省 2.93 s 和 3.58 s 估算，
两者都要约 10 个后续稳态 step、即总计约 11 step 才能摊平冷启动。

实际前 9 步总墙钟分别为 baseline 699.45 s、compile 704.69 s、fused 682.19 s、组合
704.11 s。因此只跑 9 step 时 compile 和组合仍分别比 baseline 慢约 0.75% 和 0.67%；
fused 因没有 Inductor 首编译成本，9 步总计已快约 2.47%。长任务应看稳态列，短任务必须把
首编译计入。

原始容器日志为 `/workspace/audit_sync_libero_20260720_{A,B,C,D}_*.log`，逐步数据保存在
`PI05_RL_PERF_SYNC_LIBERO_AUDIT_20260720.csv`。两组 fused 的 4 个 actor 和 4 个 rollout
worker 都确认替换 18 层；两组 compile 的最终 Hydra 配置均为 true；四组均 9/9、status 0。
