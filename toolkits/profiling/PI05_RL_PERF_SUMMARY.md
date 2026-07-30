# Pi0.5 RL 训练性能剖析 —— 组件拆分、Profiling 数据与端到端结论

> **2026-07-18 复核更正：** 本文的局部 microbench 数据大体可复现，但 compile
> 端到端 baseline 实际也开启了 `enable_torch_compile`，且 launch-gap 统计重复计算了
> profiler 父子事件。因此“真实 async compile=0%”和“compile 收益被 collocation 填掉”
> 两个核心结论不成立。严格 4-way 复测得到 compile/组合约 4%–6%、fused-only 约
> 3%–5% 的稳态 step 收益。后续严格 2+2 disaggregated 复测也得到 compile -4.0%、
> fused -3.3%、组合 -4.1%，所以“单侧优化必然被另一侧完全卡住”同样过于绝对。详见
> [PI05_RL_PERF_AUDIT_20260718.md](PI05_RL_PERF_AUDIT_20260718.md)。

本文档汇总在 VKE（8×H20，实测多为 4×H20 collocated）上对 RLinf Pi0.5 async PPO
（LIBERO-spatial）做的算子级性能剖析、两个优化（fused kernel、rollout
torch.compile）的实测收益，以及"为什么局部有收益、端到端看不到"的完整分析。

> **一句话结论（先说）**：Pi0.5 RL 训练是 **GEMM-bound**，而 GEMM 已达 H20
> bf16 峰值的 ~94%。collocated 下 GPU 被 actor+rollout 分时**打满（饱和）**，
> 此时 `step 时间 = GPU 总工作量 ÷ 吞吐`。局部算子加速（fused +8%、compile
> +12.5%）要么只覆盖总工作量的一小块（fused 减工作量 → 端到端 ~5%，落在噪声里），
> 要么其收益在 collocated 下蒸发（compile 的"填 launch 空隙"收益被 actor 的活填满，
> 实测端到端 **0%**）。**真正能降 step 的只有"减少总工作量（降 vision token）"或
> "加算力"，而非"让算子更高效"。**

---

## 0. 环境

| 项 | 值 |
|---|---|
| 硬件 | H20（SM90，bf16 峰值 ~148 TFLOPS —— 算力被阉割，显存带宽高） |
| 软件 | torch 2.6.0+cu124，CUDA 12.4，cudnn 9.1，容器 venv `/opt/venv/openpi` |
| 模型 | Pi0.5（openpi）：PaliGemma VLM（gemma_2b：hidden 2048/mlp 16384/heads 8/kv 1/hd 256/depth 18，`train_expert_only=True` → **VLM 全 freeze，0/164 可训练参数**）+ action expert（adaRMS，训练的部分） |
| 算法/环境 | async PPO，collocated（actor+env+rollout 共享卡）；LIBERO（CPU MuJoCo 物理 + EGL 渲染，每 env 一个子进程） |
| 训练后端 | FSDP1（`sharding_strategy: no_shard`，本质≈DDP，单卡放得下 7GB 模型） |
| 典型稳态（4卡 best_4gpu，g=128/envs=64） | step ≈ 72s；actor_training ≈ 62s；run_interact_once ≈ 70-76s |

---

## 1. 一个 RL step 有哪些步骤 / 算子 / 各占多少时间

### 1.1 整体流程（async collocated）

一个 step 里 **actor 训练**和 **rollout 采样**在同一批卡上**分时重叠**执行。
`step ≈ max(actor_training, run_interact_once)`（两者错峰用 GPU，见 §4）。

```
rollout (run_interact_once, ~70-76s):  等env(物理+渲染) ↔ 模型推理  交替
actor  (actor_training, ~62s):         update_epoch × global_batch × micro_batch × (fwd + bwd + opt)
```

### 1.2 rollout 一个 epoch 的拆分（实测，4卡，24 chunk steps，compile off）

| 段 | 时间 | 占比 | 说明 |
|---|---|---|---|
| **wait_env**（等 env：CPU 物理 + EGL 渲染） | ~37-39s | **~50%** | rollout worker 阻塞等 EnvGroup 发 obs |
| **infer**（模型推理 predict_action_batch） | ~36-38s | **~49%** | 1× prefix VLM 前向 + num_steps× denoise（走 gemma_expert） |
| send | 0.1s | ~0% | |

> rollout ≈ 一半等 env、一半模型推理，两者**串行不重叠**（worker: 等 env → 推理 → 等 env …）。
> env 内部"物理 vs EGL 渲染"的进一步拆分**未完成**（patch 撞到 robosuite 方法绑定层级问题，
> 64 env 子进程 hook 成功但触发点未命中；正确做法见 §6 未尽事项）。

### 1.3 actor 一个 step 的拆分（实测，4卡，24 micro-batch，RLINF_PROFILE_ACTOR_BREAKDOWN）

| 段 | 时间 | 占比 |
|---|---|---|
| **forward** | ~50-53s | **~82%** ← 绝对大头 |
| backward | ~9-11s | ~16% |
| optimizer | ~0.8-1s | ~1.5% |

> backward 只占 16%（远小于常规训练的 ~2×fwd）—— 因为 **VLM freeze，梯度只回传
> gemma_expert + value_head**（小）。所以 actor 的大头是 **forward**，forward 的大头是
> **prefix VLM 前向**。

### 1.4 单次模型前向的算子分解（torch profiler，rollout predict_action_batch，batch=16）

| 算子 | 占 CUDA 时间 | 归属 |
|---|---|---|
| **`aten::mm` / `addmm`（GEMM）** | **~62%**（单个 sm90 gemm kernel 就 49%） | q/k/v/o proj + MLP（gate/up/down）|
| `aten::bmm`（attention QKᵀ、attn·V） | **~4%** | attention 核 |
| copy_ / clone / to（dtype cast、显存搬运） | ~13% | 胶水 |
| mul / add（gated residual、逐元素） | ~8% | 胶水 |
| gelu-tanh | ~2% | MLP 激活 |
| RMSNorm / rope / softmax | 合计小 | 胶水 |

> **denoise 循环**：rollout 的 `sample_actions` 有 num_steps 次循环（走 gemma_expert，suffix
> token）；actor 的 `get_log_prob_value`（flow_sde/joint_logprob=False）**只重算 1 步**。
> denoise 每步 ~18ms（num_steps 3→10：rollout 751→877ms）。成本几乎全在 **prefix cache
> 构建（968 token 过 PaliGemma VLM，一次）**，denoise 循环本身很便宜。

### 1.5 关键硬件事实：GEMM 已打满，attention 无优化空间

- **cuBLAS bf16 GEMM 实测达 H20 峰值 93.9%**（139/148 TFLOPS，`bench_cublas_gemm.py`）。
  → 老 CUDA 12.4 不是瓶颈；GEMM **没有 kernel 级优化空间**。
- **attention 仅占 4%**，且是 prefix-LM **块状 mask**（`make_att_2d_masks`），FlashAttention
  kernel 不支持任意 dense mask；eager softmax 是 fp32 upcast，换 fused 会改数值破坏
  rollout/actor 一致性（PPO ratio 漂）。→ **flash attention 无收益，不该碰**。
- 唯一能动 GEMM 的方向是**减少工作量**（少算），非"加速 kernel"。

---

## 2. Rollout 开 torch.compile 的收益

openpi 已实现 `enable_torch_compile`（compile vision_tower / language_model /
gemma_expert / get_logprob_norm），由 `rollout.enable_torch_compile` 开关控制。

| 测法 | 收益 |
|---|---|
| **纯模型推理**（`bench_pi05_denoise.py`，GPU 独占，batch16 num_steps3） | **-12.5%**（rollout 751→658ms；num_steps=10 时 -14.6%） |
| **真实 async rollout 的 infer 段**（rollout breakdown，compile on vs off） | **~0%**（infer 37s → 37s，**没变**） |
| LIBERO 端到端 step time（A/B/C/D 对比） | 噪声内，无明显变化 |

> **这是最重要的反直觉发现**：compile 单独测有 12.5%，真实 collocated 训练里 **归零**。
> 原因见 §4：compile 的收益大半来自"填 kernel launch 空隙"，而 collocated 下这些空隙
> 早被 actor 训练填满了。
>
> **另一个坑**：ManiSkill config 用 `joint_logprob=True`（flow_noise，形状多变）时，rollout
> compile 触发反复 **recompile → 负优化**（run_interact_once 69.5s→200~296s）。所以 compile
> 只在 flow_sde/形状稳定配置下才可能正收益，**不是安全默认**（这也是 openpi 默认不开的原因）。

---

## 3. Actor / 模型用融合算子（fused kernel）—— 融了什么，收益多少

### 3.1 融合了哪些算子（KernelAgent v3，`fused_kernels/`）

替换 **prefix 侧 PaliGemma VLM 的 `GemmaDecoderLayer`**（标准 RMSNorm，18 层）。
把一层的算子重组成几个 Triton kernel：

| fused kernel | 融了原来的 |
|---|---|
| `_rmsnorm` | RMSNorm 的 rsqrt + (1+w) scale + dtype cast |
| `rope_kernel` | rotary（cos/sin + rotate_half）|
| `_matmul`（带 epilogue） | GEMM **+ 融合 residual 加法 + 融合 gelu** |
| `_twoout_mm` | MLP 的 gate GEMM + gelu + up GEMM + gate·up 相乘 |
| attention | masked 路：matmul + mask + fp32 softmax + matmul（materialize p 供 bwd）；无 mask 路：FlashAttn |

**融的是"GEMM 周边的胶水"（RMSNorm/gelu/gated-residual/rope/cast）+ 减少 kernel 数/显存往返。
GEMM 本身没变（已 94% 峰值）。** v3 还带**手写融合 backward** + **KV cache 输出**（prefix
cache build 需要），并 honor 任意 additive mask（actor 的块状 mask 必需）。

### 3.2 正确性（全部实测通过，对拍真实 openpi GemmaDecoderLayer）

- 单层 fwd + bwd + 全部 9 个参数梯度，zero-mask 和 block-mask（actor 真实场景）下 rel 均 <2e-2（bf16 正常）。
- 端到端 logprob parity（fused vs 原版，完整 get_log_prob_value）：**logp rel 7.2e-8**（近 bit-exact，
  对 PPO importance ratio 几乎零影响），value rel ~1e-2。
- **backward 实测用不上**：prefix VLM 全 freeze，集成后 `fwd_calls=36, bwd_calls=0`。

### 3.3 收益

| 测法 | 收益 |
|---|---|
| **单层 forward**（masked，真实 shape B×968，`prof_fused_vs_eager.py`） | **-16%**（eager 4.50ms → fused 3.76ms/层） |
| **纯模型前向**（`bench_pi05_denoise`，fused on vs off，单卡无抢占） | rollout **-7.5%**（750→694ms），actor **-8.6%**（688→629ms） |
| **端到端 step**（A/B/C/D/E 对比，4卡 collocated） | 噪声内，无明显变化（62-72s） |

> fused 单层/纯前向确实快（16%/8%，反复验证）。端到端没体现，因为它只减了"胶水"那部分
> 工作量（GEMM 大头动不了），且 prefix VLM 只占总工作量一部分 → 总工作量降 ~5% → step 降
> ~5%，恰好落在 ±3-5s 噪声里。

### 3.4 actor 侧开 fused/compile 的兼容性

- **fused kernel 兼容 FSDP1**（不走 Dynamo，无 writeback 冲突）；rollout+actor 都通过
  `get_model` 生效（`RLINF_FUSED_PREFIX=1`）。
- **actor 侧 torch.compile 在 FSDP1 下不可行**：`Cannot writeback when the parameter shape
  changes`（FSDP1 FlatParameter 1D 与 compile 的 2D 视图冲突，Dynamo 触发）。即使
  `use_orig_params=True` + `fullgraph=False` + fused 替换 prefix 也照样崩（实测 E 组）。
  要 actor compile 必须先迁 **FSDP2**（DTensor，与 compile 兼容）—— 独立大工作项。
- 只能 compile actor 里**无参数**的部分（get_logprob_norm 等逐元素），但占比 <1%，无意义。

---

## 4. 端到端收益的理解与结论（核心）

### 4.1 决定性实测：同一操作在真实训练里慢 2 倍

| | 单独跑（GPU 独占） | 真实 collocated 训练 |
|---|---|---|
| rollout predict_action_batch | 750ms | **1540ms**（慢 2×） |
| actor 一次 forward | ~1.4s | ~2.1s（慢 ~1.5×） |

多出来的时间不是操作变慢，是在**等 GPU** —— 因为 actor 和 rollout 分时抢同一张卡。
**说明 collocated 下 GPU 是 100% 饱和的。**

### 4.2 GPU 饱和 → step 由"总工作量"决定，不由单算子快慢决定

```
GPU 饱和时：  step 时间 = GPU 总工作量 ÷ GPU 吞吐
```

此时加速某个算子，只是让 GPU 少干一点活或让排队更顺；**step 降不降，取决于是否真的
减少了总工作量**。

### 4.3 两种"加速"，只有一种在 collocated 下活下来

1. **减少实际工作量**（fused 少算 kernel/显存往返）→ 饱和下仍有效，但 fused 只减"胶水"
   （GEMM 大头动不了）→ 端到端 ~5%，落噪声。
2. **填 GPU 的 launch 空隙**（compile 的主要收益）→ **collocated 下这些空隙早被 actor 的活
   填满**，压掉空隙不释放任何墙钟 → 实测 **0%**。

> 深刻的点：**collocated async 训练本身做的就是 compile 想做的事（把 GPU 塞满）。collocation
> 先到一步，compile 就没剩什么可给了。**

### 4.4 disaggregated 也救不了

> **2026-07-19 更正：** 本节的 `max()` 推导只能作为近似，不能推出单侧优化收益必为零。
> 相同 workload 的严格 2+2 复测中，baseline rollout 比 actor 慢约 2 s；compile 优化较慢侧
> 后，step 改善 4.0%。fused 和组合分别改善 3.3% 和 4.1%。完整数据与 16 卡条件外推见复核
> 文档第 6–7 节。

disagg 下 actor/rollout 分不同卡不抢算力，但 `step ≈ max(T_actor, T_rollout)`，且实测**两侧
几乎平衡**（4卡 actor 96.3s ≈ 采样 96.2s；8卡 105s ≈ 103s）。单侧优化被另一侧 max 卡住；
且 colocated 恒更快（4卡快 37%、8卡快 23%，因时分复用让两者都摸到全部卡）。

### 4.5 统一结论

我之前给过的"算力墙 / 稀释 / max() / 噪声 / compile 填空隙"都是**同一个模型的不同侧面**：

> **GPU 饱和 → step = 总工作量 ÷ 吞吐 → 要降 step 只能减少总工作量或加算力。**
> 你的优化里：compile 的大头（填空隙）在 collocated 蒸发（实测 0%）；fused 的真收益（减工作量）
> 只有 ~5% 且落在噪声里。**不是 kernel 写得不好，是这个 GEMM-bound + GPU 饱和的场景，
> "让算子更高效"的天花板就在这。**

### 4.6 优化不是白做 —— 场景对了就有效

fused 的 8% 在**这些场景**能落地：**纯推理服务 / eval**（rollout-only，GPU 不饱和、无 env、
无 actor 抢占）；或 disagg 且模型推理是明确瓶颈那侧。collocated 训练恰是最不利场景。

### 4.7 真正的杠杆（唯二能穿透这堵墙）

1. **减少总 GEMM 工作量（少算）** —— 唯一能穿透一切：
   - **降 vision token**（最实在）：LIBERO 现 2 图 = 512 vision + 200 lang = 712 token；prefix
     VLM 的 GEMM 量随 token 数线性。砍到 1 图 → prefix 工作量降 ~35%，实打实减工作量，不受
     94% 峰值天花板限制、不被 collocation 抢走。**（本 session 未实测，最值得下一步做。）**
   - 减 denoise 步数 / num_action_chunks 等。
2. **加算力**：换更强的卡（H20 算力被阉割）或加卡摊工作量。

---

## 5. 相关文件

| 文件 | 用途 |
|---|---|
| `bench_pi05_denoise.py` | denoise 热路径 micro-bench（rollout/actor 纯模型前向计时 + torch profiler + compile 对比 + 数值一致性） |
| `bench_cublas_gemm.py` | H20 cuBLAS bf16 GEMM 峰值效率（实测 94%） |
| `bench_te_gemma_layer.py` | 单层 fused vs eager 计时模板 |
| `prof_fused_vs_eager.py` | 单层 fused vs eager 算子级对比（含真实 shape） |
| `check_fused_gemma_layer.py` / `verify_v2_masked.py` | fused kernel 对拍真实 openpi layer（fwd+bwd, zero/block mask） |
| `verify_fused_forward_fsdp1.py` | 手写 fused forward + FSDP1 兼容性探针 |
| `FUSED_GEMMA_LAYER_KERNELAGENT*.md` | 给 KernelAgent 的 fused kernel 需求文档（含 3 版迭代 gap 记录） |
| `PI05_DENOISE_OPTIMIZATION.md` | denoise 优化早期背景 |

临时插桩（env-flag 门控，默认不生效）：
- `async_ppo_fsdp_worker.py`：`RLINF_PROFILE_ACTOR_BREAKDOWN` → actor fwd/bwd/opt 分段计时。
- `huggingface_worker.py`：`RLINF_PROFILE_ROLLOUT` → rollout wait_env/infer/send 分段计时。
- `libero/venv.py`：`RLINF_PROFILE_ENV` → env 子进程物理/渲染计时（触发点未命中，待修）。

---

## 6. 未尽事项 / 建议下一步

1. **降 vision token 的实测**（§4.7）—— 唯一未量化、且理论能真正降 step 的方向。**最高优先级。**
2. **env 细拆（物理 vs EGL 渲染）** —— wait_env 占 rollout 50% 且 rollout 是 step 瓶颈之一；
   要判断"加 CPU 核 / 优化渲染"有没有用，必须先拆开。正确做法：在 `_worker` 的 `env.step(data)`
   直接整体计时（肯定命中），再跑一次 `has_offscreen_renderer=False`（纯物理）对比，差值 = 渲染。
   （本 session 的 method-patch 方式撞了 robosuite 方法绑定层级，未命中。）
3. **FSDP2 迁移** —— 才能开 actor compile；但既然 collocated 下算子优化天花板 ~5%，性价比存疑，
   仅当有独立需求（训更大模型）时做。
4. **fused kernel 的落地场景**：eval / 纯推理服务（GPU 不饱和）才是它 8% 能体现的地方。
