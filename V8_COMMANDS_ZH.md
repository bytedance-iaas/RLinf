# V8 全流程命令与参数说明(中文)

> **路径说明**：脚本的持久位置是仓库内的 `tools_so101_session/`（用途索引见该目录的 `README.md`）。文中出现的 `scratchpad/*.status` 是运行时的状态日志，路径可自定。
**目标配置**:保持与真机完全同构的保真度(**640×480 @ 30 Hz**、实测板面几何、8 g 方块、
成功判据 = 方块入盒 **且** 机械臂回到初始位),**只把红方块的出生区收窄到 pp 时代的
6 × 8 cm 小框**。收窄是唯一的简化,目的是把示范密度恢复到 pp 时代水平
(间距 **0.44 cm**,而抓取容差约 ±0.7 cm)。

路径约定:`$REPO = /data08/henryg/pai/RLinf`、`$DATA = /data08/henryg/pai/data`、
`$RES = /data08/henryg/pai/results`。

---

## 第 0 步 — 会话环境变量(每一步之前都必须设置)

```bash
cd /data08/henryg/pai/RLinf
export REPO_PATH="$PWD" PYTHONPATH="$PWD" HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p "$XDG_RUNTIME_DIR"
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 机器前置条件 —— 漏掉这一条曾导致整夜训练全部失败
mount -o remount,size=16G /dev/shm
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
```

| 变量 | 取值 | 为什么 |
|---|---|---|
| `VK_ICD_FILENAMES`、`LD_LIBRARY_PATH` | `.venv/nvidia_gl` 下的私有驱动库 | 本机是纯计算驱动,apt 装的 libnvidia-gl 版本不匹配会直接破坏渲染 |
| `MUJOCO_GL` | `egl` | 无显示器的离屏渲染 |
| `RLINF_MASTER_ADDR_OVERRIDE`、`GLOO/NCCL_SOCKET_IFNAME` | `127.0.0.1`、`lo` | 本机节点 IP 是 IPv6;单机把所有集合通信钉到 IPv4 回环,消除一类"首轮 rollout 静默挂起" |
| `RAY_local_fs_capacity_threshold` | `0.99` | 宿主 `/tmp` 已用约 96%,否则 Ray 拒绝创建对象 |
| `/dev/shm` 扩容到 16 GB | 必需 | 容器默认只有 64 MB;NCCL 每个通信器要申请约 7 MB 共享内存段,不够就报 `ncclSystemError`(极易被误判为显存不足) |

---

## 第 1 步 — 生成示范数据(CPU,约 1.8 小时,8 进程并行)

```bash
export SO101_SPAWN_MODE=legacy          # 唯一的收窄项:6×8 cm 出生框

for W in 0 1 2 3 4 5 6 7; do
  SEED=$((90000 + W*1000))
  .venv/bin/python tools_so101_session/gen_so101_demos.py \
      --num 32 \
      --seed0 $SEED \
      --out /data08/henryg/pai/data/v8_demos_w$W &
done
wait
```

| 参数 | 取值 | 为什么 |
|---|---|---|
| `SO101_SPAWN_MODE` | `legacy` | 把红方块限制在 x ∈ [−0.534, −0.474]、y ∈ [0.020, 0.100](6×8 cm = 48 cm²),已验证该框完全落在当前棕区内。其余一切(相机、30 Hz、几何、质量、回位判据)保持真任务取值 |
| `--num` | 每进程 32(合计 256) | 目标约 175 条成功;**实测 247/256 = 96.5%** |
| `--seed0` | 90000 + 1000·W | 各进程种子区间互不重叠,避免生成重复轨迹 |
| 并行进程数 | 8 | 纯 CPU 负载;ManiSkill 的 CPU 后端每进程单线程 |

脚本内部的关键机制(已内置):夹爪闭合指令 `-0.8`;**微提验证**(抬 3 cm,方块必须升高
>1.5 cm,否则抖动抓取点重试 —— 接触力标志会对"夹不住的边角捏合"误报成功);
**负载偏移补偿**(抓住后方块悬挂在 TCP 外 2–4 cm,投放要瞄准方块而不是 TCP);
**两段式 FK 运输**(先垂直提升再平移;5 自由度手臂无法到达任意 6 自由度位姿,不要试图用
笛卡尔 IK);**投放前闭环修正**(最多 2 次);**回位段**(30 步插值 + 12 帧安定,成功判据要求);
每条轨迹 3 种重试变体。

**门槛**:成功 ≥120 条、示范中位长度 ≤530 步。**实测:247 条,中位 357 步。**

---

## 第 2 步 — 转换成 LeRobot 数据集(CPU,约 1 小时)

```bash
.venv/bin/python tools_so101_session/convert_v8_demos.py
```

| 脚本内参数 | 取值 | 为什么 |
|---|---|---|
| `SRC_GLOB` | `$DATA/v8_demos_w*/**/*.h5` | 汇总 8 个进程的输出 |
| `OUT_REPO` / `OUT_ROOT` | `so101-sim-demos-v8` | 新数据集 id,由 TrainConfig 引用 |
| `FPS` | **30** | 必须等于生成器真实的 `control_freq`。曾经生成器实跑 20 Hz、却把数据标成 15 fps,静默毁掉一整天的工作 |
| 图像尺寸 | `(480, 640, 3)` | 与真机数据集完全一致,两者才会走同一条 `resize_with_pad` 路径 |
| `MIN_LEN` / `MAX_LEN` | 80 / 580 | 580 = 回合预算 640 ÷ 1.1 余量;短于 80 帧属退化轨迹 |
| 单位换算 | `so101_calib.rad_to_norm` | 与 RL 阶段**同一个模块**,保证 SFT 数据与 RL 观测使用同一套约定 |

**结果:247 条、8.8 万帧、中位 357 步,示范间距 √(48 cm² / 247) = 0.44 cm**
(pp 时代 0.51 cm;抓取容差约 ±0.7 cm)。

---

## 第 3 步 — 归一化统计量(**故意不执行任何命令**)

```bash
# v8 不运行这条:
#   python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101_v8 --repo-id so101-sim-demos-v8
# 而是复用 v4 的统计量:
#   $REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
```

**原因**:v8 延续的是 **v4 的权重血统**(热身权重 = `so101_sft_v4/global_step_1000`)。
策略的动作解码器是按训练时的统计量标定的;在血统中途重算统计量会让续训单调劣化 ——
实测:仅仅替换统计量文件,就把一个**未改动**的权重从 19.5% 打到 9.4%。
只有从全新基座权重开始时才应重算。

---

## 第 4 步 — 基线:热身权重在小框内的成绩(GPU,约 6 分钟)

```bash
SO101_SPAWN_MODE=legacy \
.venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
  --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v8 \
  rollout.model.model_path=$RES/so101_sft_v4/so101_sft_openpi_pi05/checkpoints/global_step_1000 \
  rollout.model.openpi.config_name=pi05_so101_v4 \
  rollout.model.openpi_data.norm_stats_path=$REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
  env.eval.total_num_envs=128 \
  env.eval.seed=777
```

**用途**:这是 v8 必须超越的数字。同一权重在全板上是 12.5%,在小框内应更高;两者之差
才是"新示范买到的增量"。(该次运行曾因 Ray 连接瞬时错误失败,已排队补测。)

---

## 第 5 步 — 启动前校验 SFT 配置(CPU,数秒,绝不可跳过)

```bash
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v8 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v8
```

在 CPU 上完成 hydra 组装 + RLinf 的 `validate_cfg` + 路径存在性 + 批量算术检查。
本项目约**一半的首次启动失败**属于这一类(覆盖路径写错、`% num_action_chunks` 断言、
每轮更新次数漂移、文件缺失)。**覆盖参数必须与启动器逐字一致** —— 用不完整的覆盖集做校验,
等于校验了另一个配置。

---

## 第 6 步 — SFT 训练(8×H200,约 85 分钟)

```bash
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v8 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v8
```

配置文件 `examples/sft/config/so101_sft_v8.yaml` 的关键项:

| 参数 | 取值 | 为什么 |
|---|---|---|
| `actor.model.model_path` | `$RES/so101_sft_v4/.../global_step_1000` | 在**相同视觉域**(640×480、30 Hz)里的热身权重。曾试过从纯真机数据权重起步(v5),结果差得多 |
| `data.train_data_paths` | `so101-sim-demos-v8` | 小框内的 247 条示范 |
| `actor.model.openpi.config_name` | `pi05_so101_v8` | TrainConfig,绑定数据集 id 与 `Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True)` |
| `openpi_data.norm_stats_path` | v4 的统计量 | 血统冻结,见第 3 步 |
| `optim.lr` / `min_lr` | 2.5e-5 / 2.5e-6,cosine,warmup 200 | 本项目所有成功的 SFT 轮次都用这一组 |
| `runner.max_steps` | 4000 | 对 8.8 万帧约 5.8 个 epoch |
| `runner.save_interval` | **250**(共 16 个存档) | 峰值常出现在不到 1 个 epoch 处,间隔太粗会整段错过。代价:每轮约 446 GB 磁盘,启动前先查空间 |
| `micro_batch_size` / `global_batch_size` | 16 / 128 | 纯显存旋钮;优化器步长不变(损失按累积份数缩放,模型中无 BatchNorm),**不影响训练正确性** |
| `train_expert_only` | False | 全量微调,与本项目历次 SFT 一致 |

---

## 第 7 步 — 门评第一轮:逐个存档在小框内筛查(GPU,每个约 6 分钟 × 16)

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
for CK in $RES/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_*; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp $REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"

  SO101_SPAWN_MODE=legacy \
  .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v8 \
    rollout.model.model_path=$CK \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 \
    env.eval.seed=777
done
```

| 要点 | 取值 | 为什么 |
|---|---|---|
| 把 `norm_stats.json` 拷进每个存档目录 | 必需 | RLinf 的 rollout worker 也会在 checkpoint 目录里找统计量 |
| `env.eval.total_num_envs` | 128 | 每次评估 128 条 episode;可靠性上"多种子"优于"多环境" |
| `env.eval.seed` | 777 | **筛选用**种子,只用于挑选 |
| 必须逐个存档评估 | 不能只评最后一个 | 单个存档的分数是噪声过程的一次抽样。**v8 实测:250→7.8%、500→0.8%、750→3.9%、1000→54.7%、1250→36.7%、1500→7.8%** —— 峰值只出现在一个点上 |

---

## 第 8 步 — 门评第二轮:前 3 名换第二个种子复检

```bash
# 命令与第 7 步相同,仅把 env.eval.seed 改为 888,且只跑得分最高的 3 个存档
```
最终选择依据 = 种子 777 与 888 的平均值。

---

## 第 9 步 — 诚实验收:从未用过的种子 + 全板参照

```bash
# 小框内,用于筛选之外的全新种子
for SEED in 1313 1414; do
  SO101_SPAWN_MODE=legacy .venv/bin/python evaluations/eval_embodied_agent.py \
    ... rollout.model.model_path=$BEST ... env.eval.seed=$SEED
done

# 参照:同一权重在全板上的成绩(不设 SO101_SPAWN_MODE)
.venv/bin/python evaluations/eval_embodied_agent.py \
    ... rollout.model.model_path=$BEST ... env.eval.seed=1313
```

| 规则 | 为什么 |
|---|---|
| 验收种子必须与筛选种子完全不重叠 | 在 A、B 上选优又报 A、B 的分数,曾把一个 63% 的策略读成 75% |
| **对外汇报的数字是验收值**,不是筛选值 | 筛选值天然带选择偏差 |
| 同时必须给出全板数字 | 防止把"窄任务成绩"当成"真任务成绩" |

---

## 第 10 步 — 存档清理

```bash
for C in $RES/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_*; do
  [ "$C" = "$BEST" ] || rm -rf "$C"
done
```
16 个存档约 446 GB。门评一结束立即删除非最优存档 —— 曾发生过磁盘写满导致训练在写
checkpoint 时崩溃。

---

## 第 11 步 — 不变量审计(CPU,数秒;训练前后各跑一次)

```bash
.venv/bin/python -m toolkits.invariant_audit --ckpt $BEST
```
专门抓"**不崩溃但结果是错的**"这一类缺陷:时间链一致性(真机 fps = sim `control_freq` =
转换器 `FPS`)、相机链(仿真长宽比与分辨率 vs 策略管道的 224×168)、出生区对真实 87 帧
方块位置的覆盖率、成功语义(回位判据是否还在)、统计量血统(存档副本 = 在用文件)、
数据集与环境分辨率是否一致、回合预算余量、动作块与 action_horizon 的关系、
筛选/验收种子是否重叠。

---

## 自动化执行

整个流程由两个脚本无人值守完成(全文见 `SO101_SESSION_LOG.md` 第 3 部分):

```bash
setsid bash tools_so101_session/gen_v8_legacy.sh </dev/null >/dev/null 2>&1 &   # 第 1 步
setsid bash tools_so101_session/v8_pipeline.sh   </dev/null >/dev/null 2>&1 &   # 第 2–10 步
```

每个阶段自带 `timeout` 与一次重试、数值门槛(示范条数、示范长度、episode 数、preflight),
状态写入 `scratchpad/v8.status`。**自主性必须写在脚本里,而不是依赖交互会话** ——
会话断开时智能体的反应会被冻结,而 `setsid` 启动的进程照常运行。

---

## 截至目前的实测结果(2026-08-12)

| 阶段 | 实测 |
|---|---|
| 规划器在小框内的成功率 | **247/256 = 96.5%**(全板时仅 58%) |
| 示范条数 / 中位长度 | 247 条 / 357 步 |
| **示范间距** | **0.44 cm**(pp 时代 0.51,抓取容差 ±0.7) |
| SFT | 4000 步,退出码 0,末尾 loss ≈ 0.0016 |
| 筛选(种子 777)250 / 500 / 750 | 7.8% / 0.8% / 3.9% |
| 筛选(种子 777)**1000** | **54.7%** |
| 筛选(种子 777)1250 / 1500 / 1750 / 2000 | 36.7% / 7.8% / 43.0% / 16.4% |
| 筛选(种子 777)2250 / **2500** / 2750 / 3000 | — / **61.7%** / 23.4% / 15.6% |
| 筛选最优 **global_step_2500**(777/888) | **61.7% / 56.3% → 59.0%** |
| **诚实验收(从未用过的种子 1313 / 1414)** | **57.8% / 55.5%** |
| 全板参照(同一权重) | 9.4% |

**预注册判据**:小框内 ≥40% 即确认"示范密度决定 BC 地板"(pp 时代同等密度对应 46.9%)。
**`global_step_2500` 达到 61.7%、`global_step_1000` 达到 54.7%,均已超过 pp 时代地板** —— 假说成立;早期存档的低分只是
训练不足。教训:仅凭前两个存档就在 15:58 判定"方向可疑",属于过早下结论。

注意:上表均为**筛选种子**数值,含选择偏差;诚实数字以第 9 步在从未用过的种子
1313 / 1414 上的验收值为准,并附全板参照。

---

# 第二部分:V9 —— 专家迭代(放大器第一步)

## 为什么这样训(先读这一段)

v8 把**地板**建起来了(小框内诚实 56.7%),但 56.7% 不是终点。本项目有两个"放大器"
可以把地板变成成绩:

| 放大器 | 历史表现 | 风险 |
|---|---|---|
| **专家迭代**(采成功轨迹 → 轻量重训) | 63.3% → 81.6%(**+18 点**) | **零风险**:最坏情况是没提升,起点权重原封不动 |
| PPO 强化学习 | 46.9% → 75%(pp 时代) | 高:地板 12% 时曾 10 个迭代把策略打到 0 |

**先做专家迭代**的三条理由:
1. **零风险**且便宜(约 4 小时,大部分是 CPU);
2. 采集阶段**顺带产出无偏估计** —— 8 个从未用过的种子上的成绩(实测 57.0-65.6%,均值 61.3%),
   等于免费又验收了一遍 v8;
3. 它产生的数据同时把**示范密度**再推一档(0.44 cm → 0.26 cm),而密度正是本项目已证实的
   决定性变量。

**为什么要混合规划器示范,而不是纯自蒸馏**:iRe-VLA 论文的处方是"原始专家数据 + 新的成功
轨迹一起训"。纯自蒸馏会让策略越来越窄(本项目 v5 就是纯蒸馏,丢了 53 点)。混合后
= 247 条规划器 + 477 条策略轨迹 = 724 条。

**为什么学习率要降到 1e-5**:这一步是"锐化已有行为",不是"教新行为"。用训练新行为的
2.5e-5 会把已经学好的东西冲掉 —— pp 时代同一步也用的 1e-5。

---

## 第 1 步 — 采集策略自己的成功轨迹(GPU,约 55 分钟)

```bash
export SO101_COLLECT_DIR=/data08/henryg/pai/data/v9_rollouts
mkdir -p $SO101_COLLECT_DIR
V8=$RES/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500

for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  SO101_SPAWN_MODE=legacy \
  .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path /data08/henryg/pai/RLinf/examples/embodiment/config/ \
    --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v9 \
    rollout.model.model_path=$V8 \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$REPO/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 \
    env.eval.seed=$SEED
done
```

| 参数 | 取值 | 为什么 |
|---|---|---|
| `SO101_COLLECT_DIR` | 目录路径 | 打开 `ManiskillEnv` 里的采集器:每条 episode **首次成功时**把 (前视图, 腕视图, 状态, 动作) 落盘为 `.npz`,部分重置安全 |
| 种子 | **2001-2008**(8 个全新) | 与筛选(777/888)和验收(1313/1414)完全不重叠;这样这 8 个数字本身就是无偏估计 |
| `env.eval.total_num_envs` | 128 | 8 × 128 = 1024 条 episode |
| `SO101_SPAWN_MODE` | `legacy` | 与 v8 训练域一致 |

**实测**:8 个种子成功率 59.4 / 57.0 / 62.5 / 60.9 / 57.0 / 63.3 / 65.6 / 64.8%
(**均值 61.3%**),共收得 **477 条成功轨迹**。门槛:≥300 条。

---

## 第 2 步 — 混合转换(CPU,约 1.5 小时)

```bash
.venv/bin/python tools_so101_session/convert_v9_demos.py
```

| 数据源 | 条数 | 单位处理 |
|---|---|---|
| 规划器示范(v8 的 h5) | 247 | `qpos` 与 `actions` 都经 `rad_to_norm` |
| 策略轨迹(v9 的 npz) | 477 | **状态已是归一化**(采集器取自 `_wrap_obs`),**动作是弧度**需 `rad_to_norm` |

**注意这个不对称**:采集器记录的状态来自观测(已归一化),动作来自 `env.step()` 的入参
(弧度)。搞反任何一边都会静默毁掉数据集。

产物:`so101-sim-demos-v9`,724 条,示范间距约 **0.26 cm**(v8 为 0.44 cm)。

---

## 第 3 步 — 轻量 SFT(GPU,约 45 分钟)

```bash
export EMBODIED_PATH=$PWD/examples/sft
.venv/bin/python -m toolkits.preflight_config \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v9 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v9

.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path /data08/henryg/pai/RLinf/examples/sft/config/ \
  --config-name so101_sft_v9 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_v9
```

| 参数 | 取值 | 为什么 |
|---|---|---|
| `model_path` | **v8_step_2500**(不是 v4) | 从当前最优策略继续,专家迭代就是"自我锐化" |
| `train_data_paths` | `so101-sim-demos-v9` | 混合数据集 |
| `openpi.config_name` | `pi05_so101_v9` | 绑定新数据集 |
| `norm_stats_path` | **仍是 v4 的** | 血统冻结:v4 → v8 → v9 是同一条血统,中途换统计量会单调劣化 |
| `lr` / `min_lr` | **1e-5 / 1e-6** | 锐化用小学习率(见开头说明) |
| `max_steps` | **2000**(v8 是 4000) | 数据翻倍但目标是微调,步数减半防过拟合 |
| `save_interval` | 250 | 峰值可能出现在任意点 |

---

## 第 4 步 — 门评与诚实验收(GPU,约 1 小时)

```bash
# 4a 筛选:每个 ckpt,框内,种子 777
# 4b 复检:前 3 名,框内,种子 888
# 4c 诚实验收:最优 ckpt,框内,种子 2323 / 2424(从未用过)
# 4d 全板参照:同一 ckpt,不设 SO101_SPAWN_MODE
# 命令形式与 V8 第 7-9 步完全相同,仅 config_name 换成 pi05_so101_v9
```

| 规则 | 为什么 |
|---|---|
| 验收种子换成 2323/2424 | 1313/1414 已在 v8 验收中用过;**每一代都必须用全新种子**,否则选择偏差会累积 |
| 必须给全板参照 | v8 的全板只有 9.4%,说明小框策略不外推 —— 每一代都要如实呈现这个差距 |

**预注册判读**:
- 诚实值 **≥65%** → 专家迭代有效(pp 时代 +18 点),可再跑一轮或转 (b) 扩域;
- **57-65%** → 有增益但递减,做一轮就够,转 (b);
- **≤57%** → 已达该数据分布的天花板,转 πRL 官方配方的 PPO。

---

## 运维加固(针对本轮实际发生的故障)

```bash
# 每次评估前完整清理
.venv/bin/ray stop --force
pkill -9 -f 'ray::|raylet|gcs_server'      # 实际用 /proc 校验,勿用纯字符串匹配
rm -rf /tmp/ray/session_*
find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
# 每次评估:timeout 1800 + 最多 3 次重试
```

原因:一次评估中 Ray 的 rollout worker 猝死(`SYSTEM_ERROR: Worker unexpectedly exits`),
主进程无限等待,卡了 24 分钟才被发现。**不是显存也不是 shm**(当时两者都空闲)。
凡是评估都必须有超时 + 重试,不能只依赖流水线总超时。

---

## 全自动执行

```bash
setsid bash tools_so101_session/v9_expert_iter.sh </dev/null >/dev/null 2>&1 &
```
状态写入 `scratchpad/v9.status`;各阶段自带门槛(采集 ≥300 条、转换 DONE、preflight OK)。
