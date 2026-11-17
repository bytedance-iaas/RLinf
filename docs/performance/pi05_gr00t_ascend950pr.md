# Pi0.5 与 GR00T N1.7 在 Ascend 950PR 上的性能报告

本报告给出 RLinf 在 Ascend 950PR 上对 Pi0.5 和 GR00T N1.7 的优化前后实测数据。两个模型的瓶颈位置不同，所需的工作也完全不同：Pi0.5 的时间在模型计算，靠 CANN 融合算子；GR00T 的时间在环境，靠少渲染与少搬运。

硬件为单机 16× Ascend 950PR（131072 MiB HBM/die，可用约 123 GiB），本报告使用物理卡 8–11（4 卡，主线）与 8–15（8 卡，扩展）。基线为 `dev` 提交 `140feacc`。主线场景每个配置重复 3 次独立启动，表中 `±` 为跨启动的总体标准差，区间为 Welch 95% 区间。

面向使用者的推荐配置落在 `tests/e2e_tests/embodied/libero_spatial_ppo_openpi_pi05_ascend.yaml` 与 `libero_spatial_ppo_gr00t_n1d7_ascend.yaml`。

---

## 1. 结论

1. **Pi0.5 提升 1.273×**（57.41 ± 0.18 → 45.11 ± 0.05 秒/迭代，133.8 → 170.3 环境 step/s，4 卡，各 3 次独立启动，95% 区间 [−22.07%, −20.79%]）。全部来自模型计算：`predict` 与 `actor_training` 合计 36.60 → 24.73 秒（1.48×），环境步进与权重同步不变。
2. **GR00T N1.7 提升 1.542×**（59.47 ± 3.59 → 38.58 ± 0.43 秒/迭代，68.9 → 106.2 环境 step/s，区间 [−47.07%, −23.19%]）。主要来自环境：`env_interact_step` 45.30 → 28.22 秒。
3. **两个模型不能共用一套优化。** Pi0.5 的模型计算占整步 64%，GR00T 只占 21%——GR00T 的算子优化即使归零也只有 1.17×，所以本轮没有为它做任何算子工作。反过来，`shard_grad_op` 对 GR00T 的 actor 有 −15%，而 Pi0.5 被配置校验锁定在 `no_shard`，这个维度对它不存在。
4. **算子级倍数不能当端到端收益用。** 融合 attention 在 prefix 上是 9.30×，端到端是 1.085×；融合 RoPE 在 prefix 上是 4.92×，端到端的 95% 区间**跨零**——它在 `predict` 上 −3.5%、在 `actor_training` 上 −1.5% 都可分辨，但占整步太小，三次运行分辨不出整步的差别。本报告每一项都给算子级、阶段级、端到端三个层次。
5. **区间比点估计重要。** GR00T 的运行间标准差是 Pi0.5 的 20 倍（±3.59 对 ±0.18 秒），所以它的两项配置改动（关 offload、`shard_grad_op`）端到端都不可分辨，尽管各自把 actor 侧降了 14–15%。只作用于非瓶颈侧的改动需要用阶段指标判读。
6. **收益随卡数略有缩水**：同样工作量、同样三个开关下 4 卡 1.228×、8 卡 1.168×，与 RLinf 在 H20 上的既有结论一致。
7. **跨场景一致**：8 卡三开关下 LIBERO Spatial PPO 1.168×、LIBERO-10 PPO 1.185×、LIBERO Spatial GRPO 1.169×。第 7 节全部为三开关、各一次运行的筛选臂，只用来确认收益不随场景变号。
8. **未覆盖的是 async 与 disaggregated。** 本报告全部是同步 PPO + collocated，而 RLinf 在 H20 上的主线是 async + disaggregated。见 2.2 的缺口表。

### 推荐配置

| 场景 | 推荐 | 端到端收益 |
|---|---|---|
| **Pi0.5，LIBERO，Ascend** | `env.*.skip_reset_renders` + `openpi.npu_fused_attention` + `npu_fused_rms_norm` + `npu_fused_rope` + `skip_masked_images` | 4 卡 **1.273×** |
| **GR00T N1.7，LIBERO，Ascend** | `env.*.skip_reset_renders` + `actor.enable_offload=false` + `rollout.enable_offload=false` + `fsdp_config.sharding_strategy=shard_grad_op` | 4 卡 **1.542×** |
| 待定 | ⏸ `openpi.max_token_len=96` 另有 1.032×，但需先过训练行为验证（见第 11 节） | |

Pi0.5 的 mask 不物化与单专家不拷贝两项是 bit-exact 的，无开关、默认生效，已含在上面的基线对比里。

---

## 2. 测试方法

### 2.1 环境

| 项 | 取值 |
|---|---|
| 软件栈 | CANN 9.1.0，torch 2.10.0+cpu，torch_npu 2.10.0.post4，Ray 2.58.0，Python 3.11.14 |
| 驱动 | npu-smi 25.7.rc1.9 |
| 容器 | 非特权，仿 K8s device plugin 逐个 `--device` 映射 `/dev/davinci8..15` 与 `davinci_manager`、`hisi_hdc`；驱动、模型、数据只读挂载 |
| CPU | 绑定 128–255，即物理卡 8–15 所在的 NUMA node 1 |
| 仿真 | LIBERO，CPU PhysX + OSMesa 软件渲染（950PR 无图形栈） |

### 2.2 场景矩阵

| 编号 | 模型 / 环境 | 算法 | placement | 卡数 | 工作量 | 重复 |
|---|---|---|---|---|---:|---:|
| **a1** | Pi0.5 LIBERO Spatial | 同步 PPO | collocated | 4（物理 8–11） | 32 env，horizon 240，chunk 5，3 次 denoising，gbs 256 / micro 16，`no_shard`，bf16；每迭代 7,680 环境 step / 1,536 actor sample | **3** |
| **g1** | GR00T N1.7 LIBERO Spatial | 同步 PPO | collocated | 4 | 16 env，horizon 256，chunk 16，4 次 denoising，gbs 64 / micro 2，`full_shard` + 双边 offload，bf16；每迭代 4,096 环境 step | **3** |
| a2 | Pi0.5 LIBERO Spatial | 同步 PPO | collocated | 8（物理 8–15） | 同 a1，总量不变（强扩展） | 1 |
| a3 | Pi0.5 LIBERO-10 | 同步 PPO | collocated | 8 | 同 a1 | 1 |
| a4 | Pi0.5 LIBERO Spatial | GRPO | collocated | 8 | `group_size: 8`，64 env | 1 |
| a5 | Pi0.5 LIBERO Spatial | 同步 PPO | collocated | 8 | micro 8 / 16 / 32 | 1 |
| a6 | Pi0.5 LIBERO Spatial | 同步 PPO | collocated | 8 | 16 / 32 / 64 env（工作量随之变化，读容量不读加速比） | 1 |

a1 与 g1 是主线，结论只建立在它们之上。a2–a6 是筛选臂：各一次运行，且开的是
`skip_reset_renders` + `npu_fused_attention` + `npu_fused_rms_norm` **三个开关**，
因为这组实验在融合 RoPE 与 `skip_masked_images` 落地之前就排好了。它们回答「收益是否随场景变号」，
不回答「收益是多少」。

尚未覆盖的场景，按价值排序：

| 场景 | 为什么重要 | 状态 |
|---|---|---|
| async PPO（`train_async.py`） | RLinf 在 H20 上的主线是 async，本报告全部是同步 PPO。rollout 与 actor 重叠之后，只作用于一侧的改动端到端占比会变 | 配置已备好，未跑 |
| disaggregated 4+4 | collocated 下 actor / rollout / env 争抢同卡，分开之后瓶颈侧会移动 | 配置已备好，未跑 |
| 出厂 batch 形状（micro 128 / gbs 2048） | 本报告的 micro 16 是为了缩短单次迭代；H20 的同类对照显示大形状上收益更好，需要验证这里是否同向 | 未跑 |
| ManiSkill（CPU 物理 + 软件渲染） | 第二个仿真后端，`maniskill_ppo_openpi_pi05_ascend` 已存在 | 未跑 |
| 8 卡与扩展臂的五开关口径、各 3 次 | 把 a2–a6 从筛选升级为结论 | 未跑 |
| GR00T N1.7 的 8 卡 | GR00T 只有 4 卡数据 | 未跑 |

这些都卡在两个宿主侧问题上：`/data00` 的 docker 存储现在创建不出任何容器（overlay mount 报
`no space left on device`，而该文件系统有 299 GB 可用），以及容器内 `acl.init()` 返回 `500000`、
`acl.rt.get_device_count()` 返回 `Resource_Busy(EL0005)`，而宿主 `npu-smi` 正常、16 张卡都没有进程。
两者都与特权无关，证据与已排除的变量见 11 节。

### 2.3 优化项定义

| 开关 | 作用 | 平台 |
|---|---|---|
| mask 不物化 + 标量填充 | Gemma attention 的 mask 从 `(B,1,T,S)` 广播而不展开到 logits 形状，填充值用 Python 标量而不是每次新建的设备张量 | 全平台，bit-exact，无开关 |
| 单专家不拷贝 q/k/v | `torch.cat` 对单元素列表也分配，而每个 rollout pass 只有一个活跃专家 | 全平台，bit-exact，无开关 |
| `openpi.npu_fused_attention` | attention 走 `npu_fusion_attention`，原生 GQA 不展开 K/V | Ascend |
| `openpi.npu_fused_rms_norm` | 非 adaptive 的 norm 走 `npu_rms_norm`，gamma 保持 fp32 | Ascend |
| `openpi.npu_fused_rope` | RoPE 的旋转走 `npu_rotary_mul`，cos/sin 广播而不物化 | Ascend |
| `openpi.skip_masked_images` | 任务未填充的 `IMAGE_KEYS` 槽位跳过视觉编码器 | 全平台，需编码器冻结 |
| `env.*.skip_reset_renders` | LIBERO reset 的 settling 期间关相机，只在最后一步打开 | 全平台 |
| `enable_offload` | actor 与 rollout 在阶段之间把模型在 host 与 device 之间搬 | 全平台 |
| `fsdp_config.sharding_strategy` | `full_shard` 反向重新 all-gather 参数；`shard_grad_op` 保留 | 全平台 |

### 2.4 取数口径

- 数据源是 TensorBoard event 文件。`metrics.log` 是 rich 渲染的表格，列宽会截断数值，不作为取数来源。
- 稳态窗口：每 run 跑 7 次迭代，统计后 5 次，不开 profiler。
- **先把每次运行归约为一个稳态均值，再跨运行统计**；表中 `±` 是**运行间**总体标准差。
- 95% 区间按 Welch 差值计算，自由度取 `(n_a−1)+(n_b−1)`：n=3 对 n=3 时 t=2.776。区间跨零即判为不可分辨，报告里按「不可分辨」写，不按点估计写。
- 吞吐按相同工作量下的总环境 step 除以总时间；只对相同 placement、设备数、模型 shape、训练预算和精度的运行计算加速比。
- 每个 run 结束后从解析后的配置里断言每个开关的真实取值，与该臂声明的集合不符即判失败。汇总脚本再按这份开关指纹与 placement 指纹拒绝合并不同配置的运行——第 7 节的扩展臂一度被当成五开关，就是这个检查要防的事故。

每个 run 归档启动命令、解析后的完整配置、commit、源码补丁、依赖清单、开关与 placement 指纹、实际 rank 数与日志。汇总时按源码补丁的指纹校验同一配置的多次运行确实跑在同一棵树上——本轮有两次在 campaign 运行中改动过它正在使用的检出，这个校验是为了让那类事故可见而不是表现为噪声。

> 口径与 RLinf 在 H20 上的 [Pi0.5 报告](https://github.com/bytedance-iaas/RLinf/blob/release_v1/docs/performance/pi05_rl_performance.md) 有一处不同：那边的窗口是 10–12 步里的后 7–8 步，这边是 7 步里的后 5 步。本报告内部一致，但两份报告的百分比不要直接对读。

算子级测量每项 50 次采样，报中位数与 p10/p90 构成的比值区间。这一点有实际影响：最初用 10 次采样时 RMSNorm 读到 9.47×、attention 读到 9.93×，而 50 次采样的稳定值是 12.59× 与 9.30×——亚毫秒 kernel 在这个规模上 10 次中位数会偏移三成。融合项的被测形式与提交的代码一致（例如 RMSNorm 的 gamma 每次由 fp32 scale 现算、RoPE 的 cos/sin 表每次重建）。

> 算子级测量使用 712 token 的 prefix，而模型实际跑 968 token（3 个图像槽位 × 256 + 200 文本）。eager attention 的代价随 T² 增长而融合实现接近线性，所以表中的倍数在真实形状上是偏保守的。端到端数字来自真实运行，不受此影响。

---

## 3. 瓶颈侧

两个模型的时间构成（主线场景，Pi0.5 为 3 次均值，GR00T 为 3 次均值）：

| 阶段 | Pi0.5 | 占比 | GR00T N1.7 | 占比 |
|---|---:|---:|---:|---:|
| `time/step` | 57.41 s | 100% | 59.47 s | 100% |
| `time/generate_rollouts` | 40.78 s | 71.0% | 48.31 s | 81.2% |
| └ `time/rollout/predict` | 21.70 s | 37.8% | 2.87 s | 4.8% |
| └ `time/env/env/bootstrap_step` | **14.77 s** | **25.7%** | 0.00 s | 0% |
| └ `time/env/env_interact_step` | 11.75 s | 20.5% | **45.30 s** | **76.2%** |
| `time/actor_training` | 14.90 s | 26.0% | 8.80 s | 14.8% |
| `time/sync_weights` | 1.72 s | 3.0% | 2.36 s | 4.0% |

worker 之间的计时有重叠，不能相加当作总时间。

**Pi0.5 的算子优化作用面是 64%，不是 26%。** `predict` 与 actor 的前后向跑同一份模型代码，合计 36.60 秒，所以一次算子改动两处受益。

**Pi0.5 另有一个 25.7% 的环境开销此前未被识别。** `env/libero_spatial.yaml` 的 `auto_reset` 默认 `False`，而 Pi0.5 的 `env.train` 没有覆盖它，于是每个 rollout epoch 都整体 reset 一次全部 env：15 步 settling、每步全渲染。这比 actor 训练还大。GR00T 的 `env.train.auto_reset: True`，所以它的 `bootstrap_step` 为 0，reset 分散在步进过程中。

**GR00T 的算子优化不值得做。** 模型计算合计 11.67 秒 / 19.6%，归零也只有 1.24×；actor 单独归零只有 1.17×。本轮因此没有为 GR00T 做任何算子工作，只调环境与通信。

---

## 4. 算子级测量

真实 shape：prefix 712 token，suffix 6 token（1 state + 5 action），GQA 为 8 个 query head 对 1 个 KV head、head_dim 256，bf16。50 次采样，中位数与 p10/p90 区间。

### 4.1 Attention

| 实现 | prefix B=8 | prefix B=16 | suffix B=16 |
|---|---:|---:|---:|
| 现有 einsum 路径 | 1.7262 ms | 3.5689 ms | 0.1406 ms |
| mask 不物化 + 标量填充（bit-exact） | 1.17×（1.16–1.17） | 1.16×（1.16–1.16） | 1.53×（1.45–1.63） |
| `npu_fusion_attention`（原生 GQA） | **8.45×**（8.36–8.53） | **9.30×**（9.20–9.49） | **2.18×**（2.06–2.29） |

10 次采样的初筛另外给出：不升 FP32 的 QK 约 1.9×（被融合实现取代）；`F.scaled_dot_product_attention` 在 prefix 上约 6.3–6.6× 但在 suffix B=16 上是 0.88×（更慢）；`npu_fusion_attention` 若把 K/V 展开到 8 个 head 约 7.9×，比原生 GQA 慢且多占一份 8 倍大的 K/V。

**全屏蔽行**是两条路径唯一可能改变语义的地方。现有实现给被屏蔽的 logits 填有限负数 `-2.3819763e38` 而非 `-inf`，所以一个无处可看的 query 行 softmax 之后是均匀分布，返回 value 的均值：

| 实现 | 该行的范数 | 与现有实现的差 |
|---|---:|---:|
| 现有 einsum 路径 | 10.910 | — |
| mask 不物化 | 10.910 | 0 |
| `F.scaled_dot_product_attention` | **0.000** | **0.758（语义被改）** |
| `npu_fusion_attention` | 10.910 | 0.0078 |

SDPA 虽然也有 6.6×，但它把这样的行归零。这是选 `npu_fusion_attention` 而不选 SDPA 的决定性原因，两条行为都已写成测试。

### 4.2 RMSNorm

被测形式即提交的形式（gamma 每次由 fp32 scale 现算）：

| 形状 | 现有显式路径 | `npu_rms_norm` |
|---|---:|---:|
| prefix B=8（712×2048） | 0.1710 ms | **5.32×**（4.84–5.63） |
| prefix B=16（712×2048） | 0.5652 ms | **12.59×**（12.16–12.99） |
| suffix B=16（6×1024） | 0.0463 ms | **2.31×**（1.84–2.62） |

gamma 必须保持 fp32。`_apply_openpi_param_dtypes` 刻意把 norm 的 scale 留在 fp32 而 Gemma 权重用 bf16，注明这是为了不抬高 PPO KL。`npu_rms_norm` 接受 fp32 gamma 并仍返回输入 dtype：

| gamma dtype | 与显式路径的 rel_l2 | max_abs |
|---|---:|---:|
| bf16 | 2.86e-3 | 3.13e-2 |
| **fp32** | **1.24e-5** | 1.56e-2 |

1.56e-2 正是输出量级（max 5.16）上的一个 bf16 ULP，即 bf16 结果的精度下限。fp32 gamma 不比 bf16 慢（B=16 下 0.042 对 0.043 ms），所以保留精度是免费的。

### 4.3 RoPE

两项分开测。表构建在计时之内，因为 Pi0.5 的 `positions` 来自对 validity mask 的累加和，取决于一个 batch 带多少 padding，无法按序列长度缓存。

| 形状 | 现有路径 | 复用 cos/sin 表（bit-exact） | `npu_rotary_mul`（广播） | `npu_rotary_mul`（物化到 head 数） |
|---|---:|---:|---:|---:|
| prefix B=8 | 0.2221 ms | 1.12× | **2.68×** | 2.03× |
| prefix B=16 | 0.6190 ms | 1.11× | **4.92×** | 2.45× |
| suffix B=16 | 0.0726 ms | 1.65× | 1.04×（无收益） | 0.93× |

`npu_rotary_mul` 对 head 轴广播，所以 cos/sin 可以停在 `(B, T, 1, head_dim)`：物化到 query head 数在 B=16 的 712 token prefix 上是每张表约 47 MB，实测更慢。只有旋转走融合，表仍每次重建——这也是它只对长 prefix 有效的原因：suffix（T=6）的开销全部在建表上。

## 5. 实测数据：Pi0.5

4 卡，主线场景。每迭代固定 7,680 环境 step，所以吞吐比等于迭代耗时比。

### 5.1 单因子

第二行是两项 bit-exact 改动的叠加结果，后续各行分别单独叠在它之上。`±` 是三次独立启动之间的
总体标准差，区间是对基线差值的 Welch 95% 区间。

| 配置 | 次数 | 秒/迭代 | 环境 step/s | 相对 dev 基线 | 隔离效应（对第二行） | 95% 区间 |
|---|---:|---:|---:|---:|---:|---|
| dev 基线 | 3 | 57.41 ± 0.18 | 133.8 | 1.000× | — | — |
| mask 不物化 + 单专家不拷贝（bit-exact） | 3 | 54.90 ± 0.13 | 139.9 | **1.046×** | 1.046× | [−5.12%, −3.61%] |
| + `npu_fused_attention` | 3 | 50.61 ± 0.16 | 151.7 | **1.134×** | **1.085×** | [−8.54%, −7.10%] |
| + `skip_reset_renders` | 3 | 52.88 ± 0.02 | 145.2 | **1.086×** | 1.038× | [−4.14%, −3.22%] |
| + `skip_masked_images` | 3 | 53.95 ± 0.05 | 142.4 | **1.064×** | 1.018× | [−2.23%, −1.25%] |
| + `npu_fused_rms_norm` | 3 | 54.13 ± 0.14 | 141.9 | **1.061×** | 1.014× | [−2.08%, −0.76%] |
| + `npu_fused_rope` | 3 | 54.52 ± 0.18 | 140.9 | 1.053× | 1.007× | [−1.50%, **+0.09%**] |

前五项的区间都不跨零。**`npu_fused_rope` 是唯一跨零的一项**：它的端到端点估计是 1.007×，但三次运行
分辨不出它与基线的差别。它作用的两个阶段分辨得出来——`rollout/predict` −3.50%（[−5.65%, −1.35%]）、
`actor_training` −1.45%（[−2.77%, −0.13%]）——加上算子级 prefix 4.92×，所以这个开关确实在做事，
只是在这个场景里它占整步的比例太小。要把它的端到端效应测出来需要更多重复或更长的 prefix。

> 两项 bit-exact 改动单独拆开看：mask 不物化为 55.23 秒（1.039×），再加单专家不拷贝为 54.90 秒。
> 后者两个区间不重叠（54.74–55.05 对 55.20–55.29），是真实的 0.6%，与按访存量估算的 0.2–0.6% 一致。

**每一项只动它该动的阶段**，这是整套测量的内部一致性检查。下表是各阶段对 bit-exact 基线的差值与
95% 区间：

| 阶段 | 基线 | `npu_fused_attention` | `npu_fused_rms_norm` | `npu_fused_rope` | `skip_masked_images` | `skip_reset_renders` |
|---|---:|---:|---:|---:|---:|---:|
| `rollout/predict` | 20.72 ± 0.22 | **−17.79%** | −2.74% | −3.50% | −5.29% | +0.35%（跨零） |
| `actor_training` | 13.69 ± 0.09 | **−14.26%** | −3.07% | −1.45% | −3.12% | +1.62% |
| `env/bootstrap_step` | 14.59 ± 0.01 | +0.91% | +0.01%（跨零） | +1.23% | +1.73% | **−15.79%** |
| `env/env_interact_step` | 11.67 ± 0.04 | +0.03%（跨零） | +1.11% | +1.60%（跨零） | — | — |

四个算子开关只动 `predict` 与 `actor_training`，`skip_reset_renders` 只动 `bootstrap_step` 且对
`predict` 的影响跨零。环境步进在所有臂上都是 ±1.6% 以内的小正偏移，与被挤出的 CPU 时间一致。

### 5.2 推荐配置

| 配置 | 次数 | 秒/迭代 | 环境 step/s | 相对基线 | 95% 区间 |
|---|---:|---:|---:|---:|---|
| dev 基线 | 3 | 57.41 ± 0.18 | 133.8 | 1.000× | — |
| **五个开关全开** | 3 | **45.11 ± 0.05** | **170.3** | **1.273×** | [−22.07%, −20.79%] |

区间远离零，是本报告里最强的一个结论。五个开关为 `skip_reset_renders` + `npu_fused_attention` + `npu_fused_rms_norm` + `npu_fused_rope` + `skip_masked_images`。

### 5.3 阶段归因

| 阶段 | 基线 | 五个开关全开 | | 95% 区间（对 bit-exact 基线） |
|---|---:|---:|---:|---|
| `time/rollout/predict` | 21.70 ± 0.25 | 14.66 ± 0.07 | 1.48× | [−31.45%, −27.05%] |
| `time/actor_training` | 14.90 ± 0.04 | 10.07 ± 0.06 | 1.48× | [−27.98%, −24.95%] |
| `time/env/env/bootstrap_step` | 14.77 ± 0.11 | 12.13 ± 0.12 | 1.22× | [−18.51%, −15.17%] |
| `time/env/env_interact_step` | 11.75 ± 0.07 | 11.81 ± 0.03 | 不变 | 跨零 |
| `time/sync_weights` | 1.72 ± 0.01 | 1.71 ± 0.02 | 不变 | 跨零 |
| **`time/step`** | **57.41 ± 0.18** | **45.11 ± 0.05** | **1.273×** | [−22.07%, −20.79%] |

每一项都只动它该动的地方：`skip_reset_renders` 只改 `bootstrap_step`，四个算子开关只改 `predict` 与 `actor_training`。模型计算合计 36.60 → 24.73 秒 = 1.48×。

只开融合 attention 时的三组数字互相闭合，可作整套测量的交叉校验：算子级 prefix 9.30×；阶段级模型计算 34.47 → 28.57 秒 = 1.21×；端到端由模型计算占该步 62.4% 推出 1 / (0.376 + 0.624/1.21) = 1.088×，实测 1.089×。由 1.21× 反推 attention 约占模型计算 19%，与 trace 里 16.6%（actor）与 21.0%（rollout）的设备时间份额一致。

`skip_reset_renders` 把 `bootstrap_step` 降了 16.7%，而不是单独测 reset 时的 36%——这个 14.77 秒的块里渲染之外还有大头。追到 `libero_env._reconfigure`：训练时判断是 `if task_changed or not self.is_eval`，对 `is_eval=False` 恒为真，所以每次 reset 都把每个 env 整体重建（重载 BDDL、重建 MuJoCo 模型），即使采到的 task 没变。LIBERO Spatial 有 10 个 task、均匀采样，只有约 1/10 的 reset 能省掉重建，且重建可能承担必要的随机化，故记录不改。

## 6. 实测数据：GR00T N1.7

4 卡，主线场景。每迭代固定 4,096 环境 step。

GR00T 的 reset 发生在 episode 结束时，轨迹数随 policy 成功率浮动，于是 `time/step` 自带 14% 的固有方差，而 `actor_training` 只有 6%。判读规则见 9.2。

### 6.1 推荐配置与单因子

GR00T 的运行间标准差比 Pi0.5 大一个数量级（±3.59 s 对 ±0.18 s），所以端到端区间宽，判读必须看
阶段指标。下表同时给出两者。

| 配置 | 次数 | 秒/迭代 | 环境 step/s | 相对基线 | 端到端 95% 区间 |
|---|---:|---:|---:|---:|---|
| dev 基线 | 3 | 59.47 ± 3.59 | 68.9 | 1.000× | — |
| + `skip_reset_renders` | 3 | 42.95 ± 1.65 | 95.4 | **1.385×** | [−40.84%, −14.75%] |
| + 关闭双边 `enable_offload` | 3 | 57.79 ± 2.24 | 70.9 | 1.029× | [−16.80%, **+11.13%**] |
| + `sharding_strategy=shard_grad_op` | 3 | 59.21 ± 2.22 | 69.2 | 1.004× | [−14.37%, **+13.49%**] |
| **三项全开** | 3 | **38.58 ± 0.43** | **106.2** | **1.542×** | [−47.07%, −23.19%] |

只有 `skip_reset_renders` 与三项全开的端到端区间不跨零。另外两项的端到端**不可分辨**，但它们作用的
阶段可分辨：

| 改动 | 作用的阶段 | 基线 | 该臂 | 区间 |
|---|---|---:|---:|---|
| 关闭 `enable_offload` | `actor_training` | 8.80 ± 0.23 | 7.53 ± 0.02 | 不相交 |
| | `sync_weights` | 2.36 ± 0.02 | 1.42 ± 0.03 | 不相交 |
| `shard_grad_op` | `actor_training` | 8.80 ± 0.23 | 7.50 ± 0.16 | 不相交 |

所以这两项该写成「把 actor 侧降了 14–15%、权重同步降了 40%，但在一个环境占 76% 的步里这还不够
从端到端分辨出来」，而不是「1.029×」和「1.004×」。三项合起来的 1.542× 是可分辨的，因为
`skip_reset_renders` 同时把方差压下去了。

三项全开后标准差从 ±3.59 降到 ±0.43：`skip_reset_renders` 把 reset 开销拿掉后，轨迹数波动不再
显著影响环境时间，优化本身让测量变稳了。这也是为什么合起来反而比单项更可分辨。

`skip_reset_renders` 那一臂的轨迹数 28.20 与成功率 0.820 几乎与基线（28.73 / 0.820）相同，所以 −33.5% 不是协变量。按每次 reset 约 2.08 秒的斜率，0.53 条轨迹之差只解释 −1.1 秒，而实测环境降了 −15.17 秒。

### 6.2 通信：减少量有用，重排顺序没用

| 配置 | `actor_training` | 相对基线均值 8.80 s | 判读 |
|---|---:|---:|---|
| 基线三次 | 8.73 / 9.11 / 8.55 | — | 实际区间 8.55–9.11 |
| `shard_grad_op` | **7.50**（三次均值） | **−15%** | 低于整个基线区间 |
| `no_shard` | 8.51 | −3.3% | 区间下沿，无可分辨效应 |
| `forward_prefetch` + `backward_prefetch=pre` | 8.49 | −3.5% | 区间下沿，无可分辨效应 |
| 关闭 `gradient_checkpointing` | 8.94 | +1.6% | 区间内，无效应 |

trace 里 `hcom_allGather` 占 actor 设备时间 54.5%（5,518 次 / 5.05 秒），而同窗口设备时间合计 9.27 秒与 actor wall 9.31 秒几乎相等——这些 all-gather 串行阻塞、未与计算重叠。RLinf 的 `forward_prefetch` 默认 `False`、`backward_prefetch` 默认 `None`，正对应这个症状，但打开它们**没有可分辨效应**。

三组结果指向同一个解释：**瓶颈是带宽不是延迟**。该窗口的全部计算只有约 4 秒，没有足够的计算量去藏 5 秒的通信，所以预取只能重排顺序、造不出重叠容量。`no_shard` 看似最彻底，实际是把参数 gather 的量搬到梯度上——它用对 1.62 B 可训练参数的完整 all-reduce 换掉了分片的 reduce-scatter。只有 `shard_grad_op` 真的减少了总量：前向 gather 之后保留参数、反向不再 gather，gather 次数减半而 reduce 量不变。

`reshard_after_forward` 未列入对照：它只在 FSDP2 路径生效，而本配置用 `strategy: fsdp`。

关闭 `gradient_checkpointing` 的开关确实翻转了（基线日志 `[FSDP] Enabling gradient checkpointing`，候选日志 `Gradient checkpointing is disabled`），但 actor 落在基线区间正中。推测原因是 GR00T 的 backbone 冻结，`gradient_checkpointing_enable()` 作用的那些层不为反向重算、重计算无从省起，而可训练的 action head 在 `gr00t_n1d7` 里没有实现 checkpointing。这是推测——未逐模块验证哪些子模块收到 flag，但「开关有效翻转、性能零影响」是实测的。

### 6.3 关闭 offload 的显存代价

RLinf 不记录显存指标，运行结束后也无法补问，因此本轮用 `npu-smi` 采样器伴随 campaign 记录各卡 HBM。四张卡的峰值为 38.2–42.0 GiB，扣除驱动自身约 5.24 GiB 后净占用 **32.9–36.7 GiB**，而每卡可用约 123 GiB。关闭 offload 在容量上有充分余量。

---

## 7. 场景与参数扩展

### 7.1 卡数

同样工作量（32 env × horizon 240 = 7,680 环境 step/迭代）：

| 卡数 | 基线 | 三个开关 | 优化收益 |
|---|---|---|---|
| 4 | 57.41 s / 133.8 step/s | 46.74 s / 164.3 step/s | **1.228×** |
| 8 | 38.18 s / 201.2 step/s | 32.70 s / 234.9 step/s | **1.168×** |

本节全部扩展臂开的是 `skip_reset_renders` + `npu_fused_attention` + `npu_fused_rms_norm` **三个开关**——这组实验在融合 RoPE 与 `skip_masked_images` 落地之前就排好了，所以不能和第 5 节 1.273× 的五开关数字直接比。4 卡一行取的是同样三开关的 46.74 ± 0.13 秒（3 次运行），两行才是同一口径。

收益随卡数略有缩水（1.228× → 1.168×）：`no_shard` 的梯度 all-reduce 随卡数变贵，而被优化的模型计算占比相应下降。8 卡为单次运行，作筛选不作结论。

> 决定 world size 的是容器内可见设备数而不是 Ray 的 `--resources NPU:n`。本节全部 8 卡臂据此核对过实际 rank 数（最大 actor rank 为 7）。

### 7.2 micro batch 与 env 数（8 卡，三个开关）

| 变量 | 取值 | 秒/迭代 | 环境 step/s |
|---|---|---:|---:|
| micro batch | 8 | 33.44 | 229.6 |
| micro batch | 16 | 32.70 | 234.9 |
| micro batch | 32 | 32.45 | 236.7 |
| env 数 | 16 | 24.74 | 155.2 |
| env 数 | 32 | 32.70 | 234.9 |
| env 数 | 64 | 49.26 | 311.8 |

micro batch 的影响很小（8 → 32 只有 3%）。env 数改变工作量，应读容量：64 env 的吞吐是 32 env 的 1.33 倍，说明加 env 能有效摊薄固定开销。

### 7.3 另一个 suite 与另一个算法（8 卡）

| 场景 | 基线 | 三个开关 | 收益 |
|---|---:|---:|---:|
| LIBERO Spatial，PPO，32 env | 38.18 s / 201.2 step/s | 32.70 s / 234.9 step/s | 1.168× |
| LIBERO-10，PPO，32 env | 35.43 s / 216.8 step/s | 29.90 s / 256.9 step/s | **1.185×** |
| LIBERO Spatial，GRPO，64 env | 57.45 s / 267.4 step/s | 49.14 s / 312.6 step/s | **1.169×** |

跨 suite 与跨算法都在 1.17–1.19×，说明收益不依赖具体 task 集合，也不依赖是否有 value head。各为单次运行。

> GRPO 的 `group_size: 8` 要求 `total_num_envs // env_world_size // pipeline_stage_num` 能被它整除，所以 8 卡下用 64 env 而非 32。

---

## 8. 数值一致性

同一份已加载权重、同一个真实 LIBERO observation（batch 4）、逐变体重置全部随机流后切换开关。容差在看结果之前写定：action `atol=2e-3, rtol=2e-2`，logprob `atol=1e-2, rtol=1e-3`，value `atol=5e-3, rtol=1e-2`，全参数梯度 rel_l2 上限 2e-2。

| 变体 | actions max_abs | logprobs rel_l2 | values rel_l2 | 梯度 rel_l2 | 结论 |
|---|---:|---:|---:|---:|---|
| `npu_fused_attention` | 8.63e-3 | 1.47e-7 | 6.27e-3 | 1.18e-2 | 通过 |
| `npu_fused_rms_norm` | 5.03e-3 | 1.82e-7 | 3.98e-3 | 7.70e-3 | 通过 |
| 两者同时 | 6.41e-3 | 1.50e-7 | 7.55e-3 | 1.10e-2 | 通过 |

比较覆盖 rollout 采样的 action、log-probability、value，actor 在**同一条链**上的重算，以及一个诊断标量回传到全部可训练参数张量的梯度。log-probability 停在 1.5e-7 说明采样链没有变，其余差异就是 kernel 自身的舍入。

`npu_fused_rope` 的旋转与显式路径的 rel_l2 为 2.4e-3，前向与反向都有测试覆盖。非 adaptive RMSNorm 与显式路径的差异是其输出量级上的一个 bf16 ULP。

> 第一次跑这组对照是不通过的，原因在探针而不在 kernel，值得记录：`logprobs` 的 rel_l2 到了 1.1–1.4，即与参考基本不相关，但 `prev_values`（配置为 `value_after_vlm: True`，只依赖 prefix 不依赖采样链）始终对到 2–4e-3，且融合 RMSNorm 的偏差与融合 attention 一样大。两点都说明偏差不来自 kernel。追到 `tasks/rl.py:260`：denoising 循环里 `torch.randn(..., generator=rng)` 在 `rng=None` 时从全局加速器 RNG 取噪声，而 `predict_action_batch` 的显式 `noise` 参数只固定初始 x_t。

这同时说明框架层的随机流确实失控：`seed_everything`（`utils/utils.py:183`）已覆盖 Python / NumPy / torch / 加速器，但具身 actor 与 HF rollout worker 都没有调用它，`actor.seed` 只用于 batch shuffle 的 Generator，而上述逐步噪声和 `tasks/rl.py:215` 的 denoising 索引都未被播种。这是同配置两次短跑 success rate 可以相差 0.1875 的原因，属遗留项。

---

## 9. 指标可信度

### 9.1 两个模型的方差相差一个数量级，原因在 `auto_reset`

| | `env/num_trajectories` | `env_interact` 变动 | `actor_training` 变动 | `time/step` 变动 |
|---|---:|---:|---:|---:|
| Pi0.5（`env.train.auto_reset: False`） | **32.00，逐次相同** | 1.4% | 0.7% | **0.7%** |
| GR00T N1.7（`auto_reset: True`） | 28.73（单次 26.6–30.4） | **17.4%** | 6.3% | **14.1%** |

Pi0.5 每个 rollout epoch 恰好把全部 32 个 env 整体 reset 一次，轨迹数是常数，工作量逐次完全相同，所以 1% 以上的效应都能分辨。GR00T 的 reset 发生在 episode 结束时，于是轨迹数随成功率浮动：

| run | `success_once` | `episode_len` | 轨迹数 | `env_interact_step` |
|---|---:|---:|---:|---:|
| `gr00t-base-01` | 0.766 | 153.3 | 26.6 | 41.98 s |
| `gr00t-base-02` | 0.847 | 134.5 | 30.4 | 49.87 s |

成功率高 → episode 短 → 每迭代完成的轨迹多 → reset 次数多 → 环境时间长。环境步进总量由 horizon × env 数固定，变的是 reset 次数；由这两行算出每次额外 reset 约 2.08 秒。

真正的协变量是 reset 次数而不是成功率本身，Pi0.5 这边可以反证：它的成功率在各 run 之间从 0.425 波动到 0.662，而 `time/step` 只差 0.16%，因为 `auto_reset: False` 把轨迹数钉死在 32。

### 9.2 每个场景的方差决定它能支持多强的结论

| 场景 | 运行间标准差（`time/step`） | 可分辨的最小效应（n=3） | 结论强度 |
|---|---|---|---|
| a1 Pi0.5 LIBERO Spatial 4 卡 | ±0.02 ～ 0.25 s（0.04%～0.4%） | 约 1% | **最强**，七个单因子里有六个区间不跨零 |
| g1 GR00T N1.7 LIBERO Spatial 4 卡 | ±0.43 ～ 3.59 s（1.1%～6.0%） | 约 13% | **弱**，只有环境类改动和三项合开可分辨 |
| a2–a6 扩展臂 | 各 1 次运行，无方差估计 | 不适用 | **仅筛选**，只能读符号与量级 |

两个主线场景差 20 倍，根因在 2.2 的工作量里：Pi0.5 的 `num_trajectories` 被钉在 32（确定性工作量），
GR00T 的随 policy 成功率在 26.6–30.4 之间浮动。阶段指标的方差小得多（Pi0.5 ±0.01～0.36 s，
GR00T ±0.02～3.34 s），这就是为什么只作用于一侧的改动必须按阶段判读。

### 9.3 学习行为哨兵

性能改动不应该改学习行为。`env/success_once` 在各臂之间：

| 模型 | 基线 | 各优化臂 | 判读 |
|---|---|---|---|
| Pi0.5（4 卡，7 臂） | 0.61 ± 0.00 | 0.54 ～ 0.62 | 无系统性方向，波动量级与 7 次迭代的采样噪声一致 |
| GR00T N1.7（4 卡，5 臂） | 0.82 ± 0.04 | 0.80 ～ 0.86 | 同上 |

这只是哨兵，不是收敛性结论——7 次迭代太短。真正的收敛对照属于遗留项（见第 11 节的
`max_token_len` 一行）。

### 9.4 GR00T 的判读规则

- 只作用于 actor 的改动（通信、optimizer）按 `time/actor_training` 判，并与基线的**实际区间**比而不是与均值比。
- 作用于环境的改动要求对照两臂的轨迹数接近，或按轨迹数归一。
- 不以单次 run 的 `time/step` 对 GR00T 下结论。

这条规则有两处实际用处。`shard_grad_op` 的 `time/step` 为 61.39 秒、**高于**基线均值 59.47，只看这一列会把它读成 0.97× 的退化，而它的 actor 实际好了 15%。反过来，FSDP 预取那一臂的 `time/step` 为 53.38 秒、看起来是 1.114×，但它的轨迹数只有 26.20（基线 28.73）；按每次 reset 约 2.08 秒的斜率，少 2.53 条轨迹本身就解释 −5.26 秒，而实测环境只降了 −5.76 秒——那个 1.114× 基本是成功率协变量。

### 9.5 本轮已知的测量事故与处置

| 事故 | 处置 |
|---|---|
| 两次在 campaign 运行中改动了它正在使用的检出 | 汇总脚本增加源码补丁指纹列，使同一配置的多次运行是否跑在同一棵树上可见；现有结果经此校验同质 |
| 扩展 campaign 全部标为 4 卡，实际是 8 卡 | 决定 world size 的是可见设备数而非 Ray 的 `--resources`；已按实际 rank 数重新归类，见 7.1 |
| 算子级初筛用 10 次采样，倍数偏移达三成 | 改为 50 次采样并报 p10/p90 区间；提交信息与本报告均已按 50 次采样订正 |
| 汇总脚本按前缀收集一个臂的运行，而 `pi05-x-8card-` 也是 `pi05-x-8card-base-01` 的前缀 | 基线被并进候选臂，两个臂都读出 n=2。改成要求后缀恰好是 `-<数字>`；已发表的数字不受影响，核对后与原值一致 |
| 第 7 节的扩展臂一度被写成「五个开关全开」 | 实际只开了三个——这组实验在融合 RoPE 与 `skip_masked_images` 落地之前就排好了。已核对每个 run 的 composed config 并改成三开关口径，4 卡对照行也换成同样三开关的 46.74 秒 |
| `libero_10` 与 GRPO 四个 run 启动失败 | `libero_10` 的配置已在文件内声明 `skip_intermediate_renders`，而 launcher 总是用 `+` 追加，改为 `++`；GRPO 的 `group_size: 8` 需要 64 env。均为配置错误，已修并重跑 |

---

## 10. 复现

产物在 `950pr:/data01/yichen/pi05-perf/`：`repo-final/` 是本报告对应的自包含检出（`ca345709`），`runs/<run_id>/` 每个 run 存 `command.sh`、`composed_config.yaml`、`commit.txt`、`source.patch`、`packages.txt`、`workload.txt`、`switches.txt`、`placement.txt`、`max_actor_rank.txt`、`train.log` 与 tensorboard，`reports/` 存汇总与探针结果，`scripts/` 存 launcher 与分析脚本。

```bash
ssh 950pr
# 非特权容器，davinci 节点逐个 --device 映射，CPU 绑在 NUMA node 1
docker exec -it yichen-rlinf-perf2 bash

# 一个 run。NPUS / STEPS / ENVS / HORIZON / MICRO / GBS / ROLLOUT_EPOCH /
# CONFIG / CONFIG_PATH / ENTRY / MODEL_PATH 可覆盖，PERF_REPO 选代码树。
# EXPECT 列出该臂必须为 true 的开关，不符即判失败——这是防「五开关当成三开关」的检查。
# 注意 world size 由容器内可见设备数决定，不是 Ray 的 --resources。
PERF_REPO=/workspace/pi05-perf/repo-final \
  EXPECT=skip_reset_renders,npu_fused_attention,npu_fused_rms_norm,npu_fused_rope,skip_masked_images \
  bash /workspace/pi05-perf/scripts/run3.sh pi05 my-run-01 \
  ++env.train.skip_reset_renders=true \
  actor.model.openpi.npu_fused_attention=true \
  actor.model.openpi.npu_fused_rms_norm=true \
  actor.model.openpi.npu_fused_rope=true \
  actor.model.openpi.skip_masked_images=true

# 本报告全部统计量的来源：第一个前缀是基线，其余各臂给 mean ± 运行间 sd、
# Welch 95% 区间、逐阶段区间与学习哨兵。开关或 placement 指纹不一致时拒绝合并。
python /workspace/pi05-perf/scripts/measure3.py /workspace/pi05-perf/runs \
  pi05-base pi05-nocopy pi05-attn pi05-all3 --window 5 --envsteps 7680

# async / disaggregated 的派生配置（已备好，尚未跑，见 11 节）
#   CONFIG=perf_disagg_pi05
#   ENTRY=train_async.py CONFIG=libero_spatial_async_ppo_openpi_pi05

# 算子级与数值探针
python /workspace/pi05-perf/scripts/attn_probe.py reports/attn-probe.json
python /workspace/pi05-perf/scripts/norm_ffn_probe.py reports/norm-ffn-probe.json
python /workspace/pi05-perf/scripts/rope_probe2.py reports/rope-probe2.json
PI05_MODEL_PATH=/models/RLinf-Pi05-LIBERO-SFT \
  python /workspace/pi05-perf/scripts/parity_probe.py \
  libero_spatial_ppo_openpi_pi05 configs/libero_obs.pt reports/parity.json
PI05_MODEL_PATH=/models/RLinf-Pi05-LIBERO-SFT \
  python /workspace/pi05-perf/scripts/padding_probe.py \
  libero_spatial_ppo_openpi_pi05 reports/padding.json
```

NPU profiling 通过 `cluster.profiling.backend: torch_npu` 开启，用法见 [性能分析](../source-zh/rst_source/guides/profile.rst)。

---

## 11. 遗留项

第一项是环境阻塞，它挡住了 2.2 缺口表里的全部场景。

| 项 | 说明 |
|---|---|
| **宿主 docker 存储无法创建容器** | `/data00` 上每次创建容器都失败于 `error creating overlay mount … no space left on device`，而该文件系统有 299 GB 可用、inode 占用 2%、镜像只有 17 层、`dmesg` 无 ext4 报错。`tune2fs` 显示 ext4 的 `needs_recovery` 标志处于置位状态。清掉自己的三个陈旧测试容器（15.4 GB）后仍然失败，所以不是容量问题。这一条挡住任何新容器，与特权无关 |
| **容器内无法初始化 NPU** | 容器里 `acl.init()` 返回 `500000`、`acl.rt.get_device_count()` 返回 `(0, 507899)`（`Resource_Busy(EL0005)`），而 `/dev/davinci*` 节点映射正确、宿主 `npu-smi` 正常、16 张卡都没有进程、`lsof` 只显示宿主监控进程 `cloud-mon` 持有 `/dev/davinci_manager` 与 `/dev/hisi_hdc`。已排除：容器内僵尸进程（换 `--init` 重建）、`--cpuset-mems`、卡集合（8–15 与 12–15 相同）、device plugin 的四个 cfg 挂载（`hdcBasic.cfg` / `sys_version.conf` / `vnpu.cfg` / `hccn.conf` 补上后相同）、容器配置（照之前能跑的容器逐项复刻后相同）。`aclInit` 失败在写日志之前，`/root/ascend/log` 里没有对应的 plog |
| **特权容器换不来正确的卡** | `rlinf/scheduler/placement/flexible.py:216` 把 `visible_accelerators` 设成节点内的 local hardware rank（`0..N-1`），`ascend_npu.py:210` 再把它写进每个 worker 的 `ASCEND_RT_VISIBLE_DEVICES`，覆盖容器级设置。所以特权容器看到 16 张卡时，8 卡作业一定落在物理 0–7 上，没有环境变量能把它挪到 8–15。要钉住某一组卡只能走非特权的 `--device` 映射，依赖驱动把映射进来的卡重编号为 `0..N-1`——这也是前几轮 4 卡（物理 8–11）能正确工作的机制 |
| 五开关口径下的 8 卡与扩展对照 | 第 7 节现在是三开关、各一次运行。补齐需要 18 个 run（3 场景 × 2 臂 × 3 次），脚本与 job 列表已就绪 |
| async PPO 与 disaggregated 4+4 | 见 2.2。派生配置 `perf_disagg_pi05.yaml` 与 `perf_async_disagg_pi05.yaml` 已写好 |
| 出厂 batch 形状（micro 128 / gbs 2048）与 ManiSkill | 见 2.2 |
| GR00T N1.7 的 8 卡 | 只有 4 卡数据 |
| `max_token_len=96` 的训练行为验证 | 真实 LIBERO 指令只占 52–59 个 token，模板默认 200；缩到 96 把 prefix 从 968 列降到 864，实测 1.032×（三次运行，43.71 秒）。它改变采样 action，而经查 mask 没有泄漏——固定长度只改 padding 内容时输出逐位不变，只改长度则第 0 层即有 6.0e-4 相对误差并经 18 层放大到 3.3e-2，即 kernel 在不同列数下的累加次序差异。因此正确的判据是同训练预算下 reward / success rate 无系统性退化，这一步尚未做，默认值保持 200 |
| 框架层随机流播种 | `seed_everything` 已就绪，缺的是 worker 调用它，以及 `tasks/rl.py` 两处未播种的抽样。见第 8 节 |
| `libero_env._reconfigure` 训练时无条件重建 env | 见 5.3 |
| Pi0.5 actor 的梯度归约 | `hcom_allReduce` 1.258 秒 + `HcclLaunch` 0.645 秒 = actor 设备时间 13%、约合端到端 1.7%。已排除「每 micro-batch 归约」这个原因（`strategy/fsdp.py:498` 对非末个 micro-batch 用 `no_sync()`）。可疑的是启动开销本身：0.645 / 370 = 每次 1.74 毫秒，而 `HcclLaunch` 的代价与数据量无关。需要一份带算子尺寸的 trace 确认每次归约的实际字节数 |
| `patch_syncer.delta_decode` 的 AICPU 回落 | `torch.cummax` 走 AICPU，198 次 × 1.54 毫秒 = 0.305 秒，占 1.72 秒权重同步的 18%、端到端约 0.5%。避开它要用 cumsum + gather 重写这条跨平台共享路径并避免动态 shape 同步 |
| `get_grad_norm_for_mixed_precision` 的多余 FP32 拷贝 | `utils.py:886` 把全部梯度物化成一份 FP32 副本，而紧随其后的 `vector_norm(..., dtype=torch.float32)` 本就在 fp32 累加。GR00T 下是一次 6.48 GB 的瞬时分配；按访存量折算端到端 0.02–0.1% |
| CANN 工具链未采用 | `msprof` 就在 `/usr/local/Ascend/cann-9.1.0/bin/msprof`，本轮的 kernel 分析是手工从 `kernel_details.csv` 做的。其 advisor 专门标 AICPU 回落、格式转换与通信计算重叠差，而这三样本轮都是手工发现的 |
| 8 卡与扩展场景的重复次数 | 第 7 节各为单次运行，作筛选不作结论 |
