# SO101 + PI0.5：从真机数据集到 RL 微调策略的完整可复现流程

**输入**：`henry-guo/so101-pick-place-v2`（87 集真机遥操作数据）+ PI0.5 基座权重。
**输出**：一个在仿真里抓取红方块、放入托盘、并回到初始位姿的策略，**并且已通过上真机前的离线检验**。
**总耗时**：约 40 小时（8×H200），其中约 14 小时是纯 CPU 的示范生成与视频编码。
**代码**：分支 `henryg/pi05-maniskill-so101`（github.com/bytedance-iaas/RLinf）。

---

## 摘要

| 阶段 | 手段 | **测试区域** | 成绩（诚实口径） |
|---|---|---|---|
| 起点 | 仅用 87 集真机数据 SFT | 全板 426 cm² | **0.0%** |
| B | 全板仿真示范 SFT（420 条） | 全板 426 cm² | 12.5% |
| C | 生成区收窄，示范间距降到 0.44 cm | 框内 48 cm² | **56.7%** |
| D | 专家迭代（策略自采成功 + 规划器示范混合） | 框内 48 cm² | **76.6%** |
| E | 生成区扩到 96 cm² | 环 1 96 cm² | 55.1%（负结果） |
| **F** | **PPO 在线微调** | 环 1 96 cm² | **57.8%**（起点 52.0%，**+5.9**） |
| **G** | **真机协同训练**（打通 sim2real） | 环 1 96 cm² | 仿真 **约 60%**（不塌）；**离线真机比值 4.47 → 0.70** |

> ⚠️ **各行的测试区域不同，不能直接纵向比较。** 区域越大越难：同一个最终策略在框内是 **77.3%**、环 1 是 **57.8%**、全板是 **14.8%**。
> 按同一区域看：框内 0% → 76.6%；环 1 52.0% → 57.8%。

**任务判据**：红方块放入托盘 **且** 手臂回到初始位姿——比常见的"放进去即成功"更严。所有数字都来自**从未参与任何挑选的种子**。

**三个可复用的结论**：

1. **示范密度决定 BC 的地板**，而不是示范总量。间距 = √(生成面积 ÷ 示范条数)，与抓取容差（±0.7 cm）比较；实测 1.01 cm→12.5%、0.44 cm→56.7%，是阈值不是渐变。
2. **专家迭代是零风险的放大器**（+20 点），而 PPO 有条件才成立。
3. **PPO 能否放大，取决于一个此前没人量过的量**：策略在**带探索噪声的 rollout 分布**下的成功率（不是确定性评测下的）。二者在本任务上可以差 50 个点以上。
4. **仿真成绩再高也不代表能上真机**，这是两个独立的问题。仿真 57.8% 的策略在真实观测上比"完全不动"还差 4.5 倍；把 87 集真机数据掺进仿真数据集重训（约 1.5 h）就把它压到 0.70。**在碰机器人之前，这件事可以离线量出来。**

**当前状态**：仿真训练与 sim2real 协同训练均已完成；**上真机前的离线门已通过**（留出集比值 0.70 < 1），部署方案见 `SIM2REAL_PLAN_ZH.md`。剩余局限见 §13。

---

## 0. 这条流程在做什么

真机数据只有 87 集。**只用它做 SFT、跳过仿真示范训练、直接放进仿真环境**的策略，独立评测为 **0.0%**（`success_once` 与 `success_at_end` 均为 0）；从这个检查点起跑的两次 RL，一次全程为 0，另一次峰值仅 3–4%。原因是视觉域完全不同——同一个策略在**真机上**大概率可用，只是在仿真里为零。所以流程分三段：

```
真机数据 SFT（A）──→ 仿真示范 SFT（B–E，监督学习）──→ PPO（F）──→ 真机协同训练（G）
  建立语义           地板 0 → 57.8%                  +5.9 点      打通 sim2real
```

全流程共 **7 个阶段 A–G**，每一个都从上一个的检查点热启动，因此共用同一份归一化统计量（§4）。
A–F 解决"在仿真里会不会做这件事"，**G 解决"换成真实图像还认不认得"**——两者是不同的问题，缺一不可。

> **命令里的 `v4` / `v8` / `v9` / `v13` / `v15` 是什么？** 是磁盘上的真实名字，**不是版本号，也没有含义**——它们是当时实验的流水编号，恰好各阶段用了不同的数字。
>
> | 出现在 | 例子 | 你复现时 |
> |---|---|---|
> | 数据集目录 | `so101-sim-demos-v4` | 随便起名 |
> | 注册表条目 | `pi05_so101_v4` | 随便起名，但**必须与数据集名配套且全程不变**（§4） |
> | 结果目录 / yaml 文件名 | `so101_sft_v8`、`so101_ppo_v13` | 随便起名 |
>
> **脚本文件名里的编号已经全部去掉**，改成按功能命名（`convert_v4_demos.py` → `convert_fullboard.py`）。留下编号的只有上面这三类真实路径——因为它们被写进了已有的检查点内部，改名会让现有产物失效。想直接复用我们的检查点就照抄；从零复现就用你自己的名字。

**三个贯穿全程的原理**，每一个都由对照实验确立：

| 原理 | 内容 | 证据 |
|---|---|---|
| **密度律** | BC 的成绩下限由**示范间距** = √(生成面积 ÷ 示范条数) 决定，不是由总量决定。方块 2.9 cm、抓取容差 ±0.7 cm | 三个实测点：间距 1.01 cm→12.5%、0.91 cm→7.0%、**0.44 cm→56.7%**。是阈值不是渐变 |
| **血统冻结** | **血统 = 权重的继承链**：每个阶段都从上一阶段的检查点热启动，所以它们必须用同一套输入解释方式，也就是同一份 `norm_stats`（把关节数值缩放到模型习惯范围的均值/标准差）。**链内绝不重算** | A/B 实测：只换统计量、其余全不变，19.5%→9.4%——上游传下来的权重突然看到被换了刻度的输入，它学到的"关节角 0.3 意味着什么"整个错位 |
| **RL 是放大器** | PPO 放大已有的成功，不发现新成功；先决条件必须在 **rollout 分布**下测 | 起点为 0 时 RL 起不来（两次运行：全程 0、峰值 3–4%）；带噪成功率仅 1% 时，PPO 9 轮内摧毁策略 |

---

## 1. 前置条件

### 1.1 硬件

| 项 | 要求 |
|---|---|
| GPU | 8 × H200（141 GB） |
| 磁盘 | ≥ 500 GB |
| `/dev/shm` | **≥ 8 GB**（容器默认 64 MB 会让 NCCL 直接失败，报错看起来像显存不足）：`mount -o remount,size=16G /dev/shm` |
| CPU | ≥ 32 核（示范生成 8 进程并行） |

### 1.2 代码与数据

```bash
git clone https://github.com/bytedance-iaas/RLinf.git && cd RLinf
git checkout henryg/pi05-maniskill-so101
bash requirements/install.sh embodied --model openpi --env maniskill
```

| 资源 | 落地路径 |
|---|---|
| PI0.5 基座（LeRobot 格式） | `checkpoints/lerobot_pi05_base`（14 GB） |
| 真机数据集 87 集 | `$HF_LEROBOT_HOME/henry-guo/so101-pick-place-v2` |
| ManiSkill 资产 | `hf download --repo-type dataset RLinf/maniskill_assets` → 复制到 `rlinf/envs/maniskill/assets/`（`.gitignore` 排除了这个目录，clone 后一定没有） |

### 1.3 环境变量（每个终端）

```bash
export REPO_PATH=$PWD PYTHONPATH=$PWD HYDRA_FULL_ERROR=1
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export XDG_RUNTIME_DIR=/tmp/xdg-runtime; mkdir -p $XDG_RUNTIME_DIR
export MUJOCO_GL=egl TOKENIZERS_PARALLELISM=false
export HF_LEROBOT_HOME=/data08/henryg/pai/data
export RAY_local_fs_capacity_threshold=0.99     # Ray 默认阈值会因 / 盘满而阻塞对象创建
export RLINF_MASTER_ADDR_OVERRIDE=127.0.0.1 GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

### 1.4 注册表条目（`rlinf/models/embodiment/openpi/dataconfig/__init__.py`）

本流程用到 6 个，结构完全相同（`action_horizon=10`、`discrete_state_input=True`、`extra_delta_transform=False`），只有 `repo_id` 不同：

| 条目 | repo_id | 阶段 |
|---|---|---|
| `pi05_so101` | `henry-guo/so101-pick-place-v2` | A |
| `pi05_so101_v4` | `so101-sim-demos-v4` | B（**血统 norm_stats 在此确立**） |
| `pi05_so101_v8` | `so101-sim-demos-v8` | C |
| `pi05_so101_v9` | `so101-sim-demos-v9` | D |
| `pi05_so101_v10` | `so101-sim-demos-v10` | E、F |
| `pi05_so101_v15` | `so101-cotrain-v15` | G |

**`config_name` 选数据变换管线，`norm_stats_path` 单独由 yaml 指定**——正因为分离，各阶段才能用自己的数据集条目却共享同一份统计量。

### 1.5 两条贯穿全程的命令

阶段 B–G 的训练和评测都是这两条，**只换配置名、检查点路径、区域和种子**。后面各阶段不再重复抄，只列参数。

**① 训练（SFT）**

```bash
export EMBODIED_PATH=$PWD/examples/sft
CFG=so101_sft_v4                                    # 各阶段替换这一行
OUT=/data08/henryg/pai/results/$CFG

# 预检：必须看到 PREFLIGHT OK。它校验路径存在性、批量算术、模型-数据一致性
.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name $CFG \
  runner.logger.log_path=$OUT

# 每次训练/评测前都要把 Ray 清干净，否则会连到上一次的残留集群
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*

timeout 21600 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name $CFG \
  runner.logger.log_path=$OUT
```

**训练完必须把 norm_stats 复制进每个检查点目录**——openpi 按 `<检查点>/<数据集名>/norm_stats.json` 查找，不放进去后面评测和部署都会失败：

```bash
for CK in $OUT/*/checkpoints/global_step_*; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"
done
```

**② 评测（门评）**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*

SO101_SPAWN_FRAC="$REGION" timeout 1500 .venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval \
  rollout.model.model_path=$CK \
  rollout.model.openpi.config_name=$ENTRY \
  rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
  env.eval.total_num_envs=128 env.eval.seed=$SEED
```

成绩读日志里最后一个 `success_once=`。三点纪律：

- **不设 `SO101_SPAWN_FRAC` 就是全板**；区域字符串是 `x0,x1,y0,y1` 的归一化比例（环 1 = `0.4294,0.9115,0.5142,0.9817`）。
- **每次评测都要 `timeout` + 失败重试一次**：Ray worker 偶发 `SYSTEM_ERROR` 猝死后，驱动进程会永远等待。
- **`norm_stats_path` 全程写死同一份**（§4），不随 `config_name` 变。

---

## 2. 工具清单

全部在 `tools_so101_session/`（79 个脚本；`SO101_TOOLS_RUNBOOK_ZH.md` 是完整索引，含每个脚本"要解决什么问题"）。
下面只列本流程直接用到的，并说清**它做什么、怎么用、有什么坑**。

### 2.1 造数据

**`gen_planner_demos.py` —— 主力示范生成器（阶段 B、C、E）**

脚本化的运动规划器，不含任何神经网络。一条示范的完整流程：复位 → **抬臂到就绪位姿**（24 步插值）→ 用方块的定向包围盒算抓取位姿 → 上方 3cm 下降到 1cm → 闭合夹爪并**保持 10 帧让它物理上真的合拢** → **微提验证**（抬 3cm，方块必须升高 >1.5cm，否则判定抓取失败重试）→ 两段式 FK 搬运（先垂直提升再平移）→ 闭环修正落点 → 松爪 → 30 步插值回到初始位姿。**只有成功的才留下**。

```bash
SO101_SPAWN_FRAC="x0,x1,y0,y1" python tools_so101_session/gen_planner_demos.py     --num 30 --seed0 90000 --out $DATA/demos_w0
```

| 参数 | 含义 |
|---|---|
| `--num` | 要**多少条示范**，不是尝试多少次——每条最多试 3 个变体（投放点微调、投放高度、抓取点抖动） |
| `--seed0` | 起始种子；多进程并行时各进程种子区间必须不重叠，否则生成重复轨迹 |
| `SO101_SPAWN_MODE=legacy` | 把方块限制在 6×8cm 小框 |
| `SO101_SPAWN_FRAC="x0,x1,y0,y1"` | 按全板比例限制到任意子矩形 |

**两个关键设计**，都是踩坑后加的：**微提验证**——ManiSkill 的 `is_grasping` 靠接触力，夹住一个角、其实拿不起来也会返回 True，没有这一步会有一批"看起来抓住了、一搬运就掉"的假示范混进训练集，而模型会老实把这种失败模式学下来。**抬臂前缀**——真实复位姿态是手臂折叠、夹爪近闭合，而规划器的抓取例程假设张开的夹爪和腕滚 π/2；不加这个前缀，成功率从 45/64 掉到 1/60。

**`gen_demos_annulus.sh` —— 只补新增区域（阶段 E）**

扩大生成区时，旧区域已经有足够示范，重新生成整个区域是浪费。这个脚本把**环形带**（新区域减去旧区域）切成四条边带，按**面积配额**分配生成量（左33/右33/下23/上23），8 个 worker 并行。配额来自密度律：需要多少条 = 面积 ÷ 目标间距²。

**`collect_policy_successes.sh` —— 让策略给自己造数据（阶段 D、E）**

不是新写的采集程序，而是**在评测回路里挂一个录制器**：设 `SO101_COLLECT_DIR` 后，`ManiskillEnv` 每步把 (前视图, 腕视图, 状态, 动作) 追加进**内存缓冲区**；某一路环境**第一次成功**时，才把整段缓冲区**写成一个 `.npz` 文件存到磁盘**。失败的回合一直留在内存里，随下一次 reset 被丢弃，不产生文件。

```bash
SO101_COLLECT_DIR=$DATA/rollouts SO101_SPAWN_MODE=legacy python evaluations/eval_embodied_agent.py --config-name so101_eval_openpi_pi05     rollout.model.model_path=$CKPT env.eval.seed=2001 env.eval.total_num_envs=128
```

三条纪律：**只在第一次成功时写文件**（成功之后策略还会继续动，只保留到成功那一刻）；**必须用从未用过的种子**（采过的局面等于进了训练集）；**评测是确定性的**（录的是策略的最好水平，不是抖出来的偶然成功）。

### 2.2 转成训练数据集

**先说三种文件格式**，后面反复出现：

| 格式 | 谁产生 | 是什么 |
|---|---|---|
| **`.h5`**（HDF5） | 规划器（ManiSkill 原生） | 一个文件里装一个"文件系统"：`traj_0/obs/agent/qpos`、`traj_0/obs/sensor_data/3rd_view_camera/rgb`、`traj_0/actions`……一个文件放几十条轨迹，图像**未压缩**（一条 391 帧的双相机轨迹就 720 MB）。外带一个同名 `.json` 记录每条轨迹的 `episode_id` / `episode_seed` / **`success`**，转换器靠它挑出成功的那些 |
| **`.npz`** | 策略自采的录制器（numpy 原生） | 扁平的数组打包，一个文件一条轨迹 |
| **LeRobot**（parquet + mp4） | 转换器的产出 | 训练器只认这个。状态/动作进 parquet，图像编码成视频，体积小两个数量级 |

转换做四件事：**统一单位** → **长度过滤** → **编码成视频** → **写索引**。视频编码占 90% 的时间（约 3.3 集/分钟）。

| 脚本 | 用在 | 它的特殊之处 |
|---|---|---|
| `convert_fullboard.py` | 阶段 B | 纯规划器 h5；**同时产出整条血统共用的 norm_stats**（见下） |
| `convert_narrow_box.py` | 阶段 C | 窄框规划器 h5；不重算 norm_stats |
| `convert_expert_iter.py` | 阶段 D | **混合两种来源**：规划器 h5 + 策略 npz |
| `convert_append_region.py` | 阶段 E | **在上一版数据集副本上追加**，不重编码旧集 |
| `convert_cotrain_simreal.py` | 阶段 G（v14） | 把真机集追加进仿真数据集，真机全部 87 集 |
| `convert_cotrain_heldout.py` | 阶段 G（v15，**交付**） | 同上，但真机 0–69 训练、**70–86 留出**，门评才有意义 |

**它们不能互相取代**：`convert_append_region.py` 的追加技巧只在"已有数据集可追加"时成立，从零复现时前面没有可追加的东西。

**单位不对称是这里最容易出错的地方**：

| 来源 | `state` | `action` |
|---|---|---|
| 规划器 h5 | 弧度 → 要转 | 弧度 → 要转 |
| 策略自采 npz | **已归一化** → 直接用 | 弧度 → 要转 |
| 真机数据 | 已归一化 | 已归一化 |

搞错不会报错，只会让模型学错。

**长度过滤的上限要按数据来源定**：仿真示范用 580（挡掉超时挣扎的轨迹），但真机是人类遥操作、更慢，87 集是 395–825 帧、中位 575——**用 580 会静默丢掉 46% 的真机数据**。这个坑在 sim2real 那步实际发生过，靠启动前先量分布挡住。

### 2.3 成套流水线

| 脚本 | 串起哪几步 |
|---|---|
| `pipeline_expert_iteration.sh` | 采集策略成功 → 与规划器示范混合 → 轻量 SFT → 门评（阶段 D） |
| `pipeline_region_expand.sh` | 转换 → SFT → 环形门评 → 检查点清理（阶段 E） |
| `pipeline_cotrain_simreal.sh` | 建混合数据集 → SFT → **双轴门评**（sim2real） |

这些脚本都是 `setsid nohup bash <脚本> &` 启动、无人值守跑完。它们内部都遵守同一套约定：**超时按实测速率定**、**阶段间用哨兵串行**、**每次评测前完整清理 Ray 与 /dev/shm**、**评测失败重试 3 次**。

### 2.4 PPO 专用

**`ppo_freeze_probe.sh` —— 启动前的先决条件测量**

跑**真实的训练路径**（同样的 worker、环境创建、模型构建、权重同步），但把学习率设成 `1e-9`，权重实质不变。一轮就同时给出两个数：

- `env/success_once`：**带探索噪声的 rollout 成功率**——PPO 真正学习的分布，**门槛 ≥5%**；
- `eval/success_once`：确定性评测成功率——用来确认为了抬高前者没有把策略本身改坏。

这两个数在本任务上**可以差 50 个点以上**。不测这个直接启动 PPO，就是七次失败的原因。

**`ppo_train.sh` —— 启动器 + 自动停机守卫**

守卫每 5 分钟读 tensorboard 的 `eval/success_once`：低于历史峰值 20 点 → 立即杀训练；连续 3 次低于第一次评测 5 点以上 → 停。**必须写在启动器里**，写在会话里会随会话消失（历史上因此白烧 180 轮）。

### 2.5 验证与体检

| 脚本 | 回答什么问题 |
|---|---|
| `verify_honest_seeds.sh` | "这个成绩能对外说吗"——用**从未参与挑选**的种子重测 |
| `verify_baseline_control.sh` | "涨的是真本事还是评测集选择效应"——同种子同条件测**起点** |
| `offline_replay_check.py` | "策略在真实观测下还能不能用"——**上机前的门**，不碰机器人 |
| `toolkits/preflight_config.py` | "这次启动会不会白跑一夜"——路径存在性、**批量算术**、模型-数据一致性 |
| `toolkits/invariant_audit.py` | "有没有不报错但结果是错的地方"——9 项静默错误检查 |

`preflight_config.py` 的**批量算术**那一项对 PPO 是命门：`updates/epoch = num_envs × (预算÷块长) × rollout_epoch ÷ global_batch`，这个数决定 PPO 是放大还是摧毁策略。这个工具自己曾漏算 `rollout_epoch` 因子、把 12 次报成 4 次——守门的工具算错，等于没有门。

---

## 3. 阶段 A —— 真机数据 SFT

**为什么** 建立任务语义：让模型知道 `"Grab the red cube"` 这句话对应什么动作、6 维关节值是什么量纲。**它不是可用策略**——产物在仿真里独立评测是 **0.0%**，只作为阶段 B 的热启动权重。

**输入**

| | |
|---|---|
| 数据 | `$HF_LEROBOT_HOME/henry-guo/so101-pick-place-v2`（87 集真机遥操作，30 fps，640×480 双相机） |
| 权重 | `checkpoints/lerobot_pi05_base`（PI0.5 基座，LeRobot 格式，14 GB） |

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/sft

# ① 算真机域的归一化统计量
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats --config-name pi05_so101

# ② 训练
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
.venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_sft_openpi_pi05
```

**参数**（都写在 `examples/sft/config/so101_sft_openpi_pi05.yaml` 里，不用命令行覆盖）

| 参数 | 值 | 为什么是这个值 |
|---|---|---|
| `actor.model.model_path` | `checkpoints/lerobot_pi05_base` | PI0.5 基座 |
| `actor.model.openpi.config_name` | `pi05_so101` | 注册表条目，决定数据怎么进模型（repo_id = 真机数据集） |
| `actor.optim.lr` | 2.5e-5 | 全新任务的标准 BC 学习率 |
| `runner.max_steps` / `save_interval` | 20000 / 2000 | 存点密一些，好挑 |
| `micro_batch_size` / `global_batch_size` | 16 / 128 | 128 必须是 `world_size(8) × micro(16)` 的整数倍，否则 preflight 直接拦下 |

**输出**

| 产物 | 路径 |
|---|---|
| 归一化统计量（**只有阶段 A 用**） | `assets/pi05_so101/henry-guo/so101-pick-place-v2/norm_stats.json` |
| 权重（取用 step_8000） | `results/so101_sft_openpi_pi05/checkpoints/global_step_8000` |

**验收** 不做仿真评测（跑了也是 0.0%，这一步的价值不在成绩）。**耗时约 3 h。**

---

## 4. 阶段 B —— 全板仿真示范 + 冻结统计量

**为什么** 真机数据教不会"在仿真里怎么做"（视觉域完全不同）。这一步用**运动规划器**在仿真里造示范，把成功率从 0 抬起来；同时**确立整条血统共用的归一化统计量**。

**输入** 阶段 A 的 `global_step_8000` + 仿真环境（无需数据）。

### B1 规划器探针 —— 先证明任务可解

**为什么** 规划器做不到的事，BC 和 RL 都做不到。12 条的成本是分钟级，跳过它可能白花一整天。

```bash
.venv/bin/python tools_so101_session/gen_planner_demos.py \
  --num 12 --seed0 79000 --out $DATA/probe
```

| 参数 | 含义 |
|---|---|
| `--num` | 尝试次数（不是成功数） |
| `--seed0` | 起始种子，逐条 +1；**探针种子要与后面生成用的种子段错开**，避免用同一批局面 |
| `--out` | h5 输出目录 |

**输出** `$DATA/probe/*.h5`（轨迹）+ 同名 `.json`（每条是否成功、长度）。
**验收** ≥8/12 成功、成功轨迹中位长度 ≤530 步（预算 640 的 1.1 倍余量）。**不过就不要往下走。**

### B2 分层生成 420 条全板示范

**为什么** 均匀铺满整块板。分成 4×4 格逐格生成，是为了避免随机采样在某些格子上过密、某些格子空白——**BC 的地板由最稀的地方决定**。

```bash
SEED=80000
for XI in 0 1 2 3; do for YI in 0 1 2 3; do
  SO101_SPAWN_FRAC="$(echo "$XI*0.25"|bc -l),$(echo "($XI+1)*0.25"|bc -l),$(echo "$YI*0.25"|bc -l),$(echo "($YI+1)*0.25"|bc -l)" \
  .venv/bin/python tools_so101_session/gen_planner_demos.py --num 45 --seed0 $SEED \
      --out $DATA/v4_demos_cell_${XI}_${YI} &
  SEED=$((SEED+100)); done; done; wait
```

| 参数 | 含义 |
|---|---|
| `SO101_SPAWN_FRAC` | `x0,x1,y0,y1`，方块生成区在棕色板上的**归一化比例**；这里每格 1/4×1/4 |
| `--num 45` × 16 格 | 720 次尝试，实测约 420 条成功 |
| 输出目录名 | **必须是 `v4_demos_cell*`** —— 下一步的转换器按 `/data08/henryg/pai/data/v4_demos_cell*/**/*.h5` 找输入（`convert_fullboard.py:19`） |

**输出** 16 个目录的 h5，合计约 420 条成功轨迹。
**验收** 总成功数 ≥250。8 个 CPU 进程并行，**耗时约 4 h**。

### B3 转换 + 计算 norm_stats（仿真域只算这一次）

```bash
.venv/bin/python tools_so101_session/convert_fullboard.py
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats \
  --config-name pi05_so101_v4 --repo-id so101-sim-demos-v4
```

| 步骤 | 输入 | 输出 |
|---|---|---|
| `convert_fullboard.py` | `$DATA/v4_demos_cell*/**/*.h5` | LeRobot 数据集 `$DATA/so101-sim-demos-v4`（30 fps、640×480、双相机；轨迹长度 >580 帧的丢弃） |
| `calculate_norm_stats` | 上面那个数据集 | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` |

> **这是第二次调用 `calculate_norm_stats`，但不是重复劳动。** 归一化统计量定义"模型看到的数值世界"，真机与仿真是两个不同的分布：
>
> | 调用 | 产出 | 谁用 |
> |---|---|---|
> | 阶段 A 的 `--config-name pi05_so101` | `assets/pi05_so101/henry-guo/so101-pick-place-v2/norm_stats.json` | **只有阶段 A** |
> | 这一次 `--config-name pi05_so101_v4` | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` | **阶段 B 之后全部**，且从此冻结 |
>
> **此后 C/D/E/F/G 一律沿用这一份，绝不重算**（各阶段 yaml 里都硬写着这个路径，注释是 `lineage frozen: continuing v4`）。

> **关于名字里的 `v4`（§0 已说明它没有含义，这里说它为什么不能改）**：
>
> | 约束 | 为什么 |
> |---|---|
> | 阶段 B–G 必须自始至终用同一个名字 | 条目名决定 norm_stats 落在 `assets/<条目名>/<数据集名>/norm_stats.json`；中途换名 = 换了一份统计量 = 换了坐标系。**实测代价：19.5% → 9.4%** |
> | 名字被写进了检查点内部 | openpi 按 `<model_path>/<数据集名>/` 找 norm_stats，所以每个检查点目录里都有一个 `so101-sim-demos-v4/` 子目录（可以自己 `ls` 一下交付的那个检查点）。事后改名会让**已有的所有检查点失效** |

### B4 SFT

> ⚠️ **仓库里的 `so101_sft_v4.yaml` 的 `model_path` 指向一个已废弃的早期检查点（`so101_sft_pp6b/...`），你不会有它。** 必须在命令行覆盖成阶段 A 的产物——下面的命令已经带上了。（SFT 配置里**没有 `rollout` 节点**，所以只覆盖 `actor.model.model_path`；写 `rollout.model.model_path=` 会被 Hydra 直接拒绝。）

```bash
export EMBODIED_PATH=$PWD/examples/sft
CFG=so101_sft_v4; OUT=/data08/henryg/pai/results/$CFG
WARM=/data08/henryg/pai/results/so101_sft_openpi_pi05/checkpoints/global_step_8000

.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name $CFG \
  runner.logger.log_path=$OUT \
  actor.model.model_path=$WARM              # 必须看到 PREFLIGHT OK

.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
timeout 21600 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name $CFG \
  runner.logger.log_path=$OUT \
  actor.model.model_path=$WARM

# 必做：把 norm_stats 放进每个检查点，否则评测和部署都找不到它
for CK in $OUT/*/checkpoints/global_step_*; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"
done
```

| 参数 | 值 | 为什么 |
|---|---|---|
| `config_name` | `pi05_so101_v4` | 数据集换成仿真示范 |
| `norm_stats_path` | v4 那份 | 血统起点 |
| `lr` | 2.5e-5 | 与阶段 A 同量级；这仍是 BC |
| `max_steps` / `save_interval` | 4000 / 1000 | |

**输出** `results/so101_sft_v4/so101_sft_openpi_pi05/checkpoints/global_step_{1000,2000,3000,4000}`。

### B5 门评

**为什么** SFT 的 loss 会一路降，但成绩在早期就见顶——**必须逐个检查点评，不能只看最后一个**。

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
for CK in $OUT/*/checkpoints/global_step_{1000,2000,3000,4000}; do
  for SEED in 777 888; do
    .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
    timeout 1500 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v4 \
      rollout.model.model_path=$CK \
      rollout.model.openpi.config_name=pi05_so101_v4 \
      rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
      env.eval.total_num_envs=128 env.eval.seed=$SEED \
      2>&1 | grep -oE 'success_once=[0-9.]+' | tail -1
  done
done
```

| 参数 | 值 | 为什么 |
|---|---|---|
| 不设 `SO101_SPAWN_FRAC` | → 全板 426 cm² | 这一阶段的目标区域就是全板 |
| `env.eval.total_num_envs` | 128 | 配置默认只有 16，样本太少；128 集下 1 集 = 0.8 个百分点 |
| `env.eval.seed` | 777 / 888 | 两个种子取均值挑点 |
| `timeout 1500` | | Ray worker 偶发猝死后驱动会永远等待，**每个评测都要加，并失败重试一次** |

**输出** 日志里最后一个 `success_once=` 就是成功率。
**验收** 最优点是 `global_step_1000`，全板约 **12.5%**。之后单调下降（2.0→2.3→0.0）——SFT loss 降到 0.002 却越训越差，是**过拟合规划器习惯**。挑出最优点后**再用一个没用过的种子（909）复测**，门评种子参与了挑选，会偏乐观。

**耗时** B 全阶段约 7 h。

---

## 5. 阶段 C —— 收窄生成区（用密度换成绩）

**为什么** 全板 426 cm² 要达到 0.44 cm 示范间距需要约 2200 条示范（约 20 h 生成）。先把生成区收窄到 48 cm²，**用同样的数据量换密度**——密度决定 BC 的地板，而不是总量。

**输入** 阶段 B 的 `so101_sft_v4/.../global_step_1000` + v4 的 norm_stats。

```bash
export SO101_SPAWN_MODE=legacy      # 唯一的收窄项：6×8 cm = 48 cm²
for W in 0 1 2 3 4 5 6 7; do
  .venv/bin/python tools_so101_session/gen_planner_demos.py \
    --num 32 --seed0 $((90000 + W*1000)) --out $DATA/v8_demos_w$W & done; wait
.venv/bin/python tools_so101_session/convert_narrow_box.py     # 不重算 norm_stats
```

| 项 | 值 | 说明 |
|---|---|---|
| `SO101_SPAWN_MODE=legacy` | 6×8 cm 框 | 与 `SO101_SPAWN_FRAC` 互斥，是早期版本留下的固定窄框 |
| 输出目录 | `v8_demos_w*` | **必须**——`convert_narrow_box.py:19` 按这个 glob 找输入 |
| 转换输出 | `$DATA/so101-sim-demos-v8` | **不调用 `calculate_norm_stats`**，血统冻结 |

训练按 §1.5 ①、门评按 §1.5 ②，参数：

| 参数 | 值 | 依据 |
|---|---|---|
| 配置 / 输出目录 | `so101_sft_v8` | 该 yaml 的 `model_path` 已正确指向 v4 的 step_1000，无需覆盖 |
| 热启动 | v4 的 `global_step_1000` | 阶段 B 最优点 |
| `lr` / 步数 / **存点** | 2.5e-5 / 4000 / **250** | 250 是教训值：最优点在 step_2500，按 1000 存会**整个错过** |
| 门评区域 | `SO101_SPAWN_MODE=legacy` | 必须与生成区一致，否则测的是没训过的地方 |
| 门评条目 / 种子 | `pi05_so101_v8` / 1313、1414 | 这两个种子从未参与挑选，是"诚实口径" |

**输出** `results/so101_sft_v8/.../global_step_2500`（247 条示范、间距 **0.44 cm**）。
**验收** 框内诚实值 **57.8 / 55.5%**（两个种子），全板参照 9.4%。**耗时约 5 h。**

> **判读纪律**：step_250 只有 7.8%、step_500 只有 0.8%——**前两个点低不代表方向错**，别在这里判死刑。

---

## 6. 阶段 D —— 专家迭代（零风险放大器，+20 点）

**为什么** 规划器示范只覆盖它自己的解法；让**当前策略**在没见过的局面上跑，把它自己成功的轨迹收回来重训，等于在策略实际会走的分布上加密数据。这一步是本流程性价比最高的（+20 点，无超参风险）。

**输入** 阶段 C 的 `so101_sft_v8/.../global_step_2500`。

```bash
# D1 采集：就是 §1.5 ② 那条评测命令，多设一个 SO101_COLLECT_DIR 就会把成功轨迹落盘
export EMBODIED_PATH=$PWD/examples/embodiment
export SO101_COLLECT_DIR=$DATA/v9_rollouts
V8=/data08/henryg/pai/results/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500
for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  SO101_SPAWN_MODE=legacy timeout 1500 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v9 \
    rollout.model.model_path=$V8 \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 env.eval.seed=$SEED
done

# D2 混合转换：247 条规划器示范 + 425 条策略轨迹 = 672 集
.venv/bin/python tools_so101_session/convert_expert_iter.py
```

| 项 | 值 | 说明 |
|---|---|---|
| `SO101_COLLECT_DIR` | `$DATA/v9_rollouts` | **必须**——`convert_expert_iter.py:53` 按这个路径找 npz |
| 种子 2001–2008 | 8 个全新种子 | 用训练过的种子采集等于把已经会的再抄一遍 |
| 转换器输入 | `v8_demos_w*/**/*.h5` **和** `v9_rollouts/*.npz` | 两种来源混合 |
| 转换器输出 | `$DATA/so101-sim-demos-v9` | 仍不重算 norm_stats |

D3 训练按 §1.5 ①（配置 `so101_sft_v9`，其 yaml 已指向 v8 的 step_2500；**lr 降到 1e-5**、2000 步、存点 250），D4 门评按 §1.5 ②（条目 `pi05_so101_v9`、`SO101_SPAWN_MODE=legacy`、种子 1313/1414）。

**必须混合，不能纯自蒸馏**：只用策略自己的轨迹会让它越练越窄（历史上掉 53 点）。
**单位不对称陷阱**：录制器写出的 npz 里 `state` 已归一化而 `action` 是弧度；h5 两者都是弧度。转换器对两种来源分别处理，**自己写采集器时这是最容易错的一处**。

**输出** `results/so101_sft_v9/.../global_step_1250`。
**验收** 框内诚实值 **77.3 / 75.8%**，全板 19.5%（翻倍）。**峰值在中段**——最后一个检查点只有 7.8%，**必须全部检查点都评**。**耗时约 5 h。**

---

## 7. 阶段 E —— 环 1 扩域（负结果，但产出了 PPO 的起点）

**为什么** 想把 48 cm² 的能力扩到 96 cm²。**结论是没做到**，但这一阶段的产物是 PPO 的正确起点——因为它是在 PPO 将要训练的那个区域上训出来的。

**输入** 阶段 D 的 `so101_sft_v9/.../global_step_1250`。

```bash
export RING1="0.4294,0.9115,0.5142,0.9817"     # 环 1：8.48 × 11.31 cm = 96 cm²

# E1 用 v9 在环 1 上自采（同 D1，只是区域换成环 1、种子换一批）
export SO101_COLLECT_DIR=$DATA/v10_rollouts
V9=/data08/henryg/pai/results/so101_sft_v9/so101_sft_openpi_pi05/checkpoints/global_step_1250
for SEED in 3001 3002 3003 3004 3005 3006 3007 3008; do
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  SO101_SPAWN_FRAC=$RING1 timeout 1500 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v10 \
    rollout.model.model_path=$V9 rollout.model.openpi.config_name=pi05_so101_v9 \
    rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 env.eval.seed=$SEED
done

# E2 只在新增的环形带上补规划器示范（内框已经够密，不重复生成）
SCRATCH=/tmp/so101_runs bash tools_so101_session/gen_demos_annulus.sh

# E3 在 v9 数据集的副本上追加，不重编码旧集
.venv/bin/python tools_so101_session/convert_append_region.py
```

| 项 | 值 | 说明 |
|---|---|---|
| `SO101_SPAWN_FRAC` | `0.4294,0.9115,0.5142,0.9817` | 环 1 的归一化边界；后面 PPO 和真机摆放都用这一组数 |
| `SCRATCH=` | 任意可写目录 | 这些 `.sh` 会把日志和状态写进 `$SCRATCH`（默认 `/tmp/so101_runs`），**它们本来硬编码的是写作时的会话目录** |
| E2 输出 | `$DATA/v10_demos_w*` | `convert_append_region.py:67` 按此 glob 找 |
| E3 输入/输出 | 复制 `so101-sim-demos-v9` → 追加 → `so101-sim-demos-v10` | **在副本上追加**，不重编码旧集，省数小时 |

E4 训练按 §1.5 ①（配置 `so101_sft_v10`，lr 1e-5、2000 步、存点 250），E5 门评按 §1.5 ②（条目 `pi05_so101_v10`、`SO101_SPAWN_FRAC=$RING1`、种子 1313/1414）。

**输出** `results/so101_sft_v10/.../global_step_1000`（数据集 1292 集）。
**验收** 环 1 诚实 55.1%、框内 75.0%、全板 10.2%。**耗时约 7 h。**

> **这是负结果**：扩域没有在目标区带来提升（v9 在环 1 上本来就有 58.6%），全板还掉了一半。**密度律不能外推成"扩域只要补够密度"**——v9 在外环一条示范都没有却已经有 58.6%，说明外环从来不缺示范。
> 保留它是因为它是 PPO 的起点：PPO 在环 1 上训练，起点也应当在环 1 上训过。

---

## 8. 阶段 F —— PPO 在线微调

**为什么** 到这里为止全是监督学习——策略只会模仿示范。PPO 让它在**自己实际会遇到的局面**上试错并放大成功。完整细节见 `PPO_V13_RUNBOOK_ZH.md`。

**输入** 阶段 E 的 `so101_sft_v10/.../global_step_1000`（已写在 `so101_ppo_v11.yaml` 里）。

### 8.1 三个决定成败的参数

| 参数 | 官方/原值 | 改成 | 实测依据 |
|---|---|---|---|
| `actor.model.num_action_chunks` | 5 | **10** | 带噪 rollout 1.0%→**4.7%**，且**确定性成绩 55%→66.4%**（白捡 11 点）。模型 SFT 时 `action_horizon=10`，只执行 5 步等于丢一半预测 |
| `openpi.noise_logvar_range` | `[0.08,0.16]` | **`[0.02,0.04]`** | 带噪 rollout 4.7%→**39.1%**。⚠️ **别调 `noise_params`**——那是 flow-SDE 的参数，`flow_noise` 根本不读（`openpi_action_model.py:47-58`），调它八次都不会有任何变化 |
| **每轮更新次数** | 12 | **1** | 12 次/轮：step 9 从 61.7% 掉到 **7.0%**；1 次/轮：稳定爬到 **73.4%**。**决定性的那一个** |

更新次数不是直接参数，是**算出来的**：

```
每轮样本数 = num_envs × (max_episode_steps ÷ 动作块长) × rollout_epoch
           = 64 × (640 ÷ 10) × 1 = 4096
每轮更新数 = 每轮样本数 ÷ global_batch_size = 4096 ÷ 4096 = 1
```

所以 `rollout_epoch` 从 3 改成 1、`global_batch_size` 从 2048 改成 4096，两个一起才得到 1。**`toolkits.preflight_config` 会把这个算式打出来**——启动前一定要看：

```bash
EMBODIED_PATH=$PWD/examples/embodiment .venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v13 \
  actor.model.num_action_chunks=10 env.train.rollout_epoch=1 actor.global_batch_size=4096
```

应当看到这两行（实测输出）：

```
batch arithmetic: 64 envs x 64 chunks x rollout_epoch 1 = 4096 samples/epoch -> 1 update(s)/epoch
PREFLIGHT OK
```

**`-> 1 update(s)/epoch` 这一句就是成败所在**，不是 1 就别启动。

### 8.2 启动前必测：先决条件要在 rollout 分布下测

**为什么** PPO 是放大器不是发现器：它只能放大**已经偶尔发生**的成功。而"偶尔成功"必须在**带探索噪声的 rollout 分布**下测，不是确定性评测下——这两个数在本任务上能差 50 个点以上。

用**冻结探针**（`lr=1e-9`，走真实训练路径但权重几乎不动）跑一轮，同时得到两个数：

| 指标 | 含义 | 门槛 |
|---|---|---|
| `env/success_once` | **带噪**，PPO 真正学习的分布 | **≥5%** |
| `eval/success_once` | 确定性 | 不应比起点低太多 |

5% 来自本项目四次历史运行的实测分界：**两次放大成功的起始带噪成功率是 5–15%，两次从未起来的是 0.5–1.0%**。**必要但不充分**——本次探索中有一个变体带噪 39% 仍然崩塌，因为更新次数不对。

**不要用 `runner.only_eval=True` 当探针**：它同时切换模型规格来源并跳过训练环境创建（`config.py:830`、`env_worker.py:108`），是另一条代码路径，测的不是同一件事。

### 8.3 启动

```bash
SCRATCH=/tmp/so101_runs bash tools_so101_session/ppo_train.sh
```

这个脚本做了三件手工容易漏的事：设好环境变量、清干净 Ray、**带一个自动停机守卫**。它实际发出的命令是：

```bash
export SO101_SPAWN_FRAC="0.4294,0.9115,0.5142,0.9817"
setsid .venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v13 \
  runner.val_check_interval=5 runner.save_interval=5 runner.max_epochs=300 \
  actor.model.num_action_chunks=10 \
  env.train.rollout_epoch=1 \
  actor.global_batch_size=4096 \
  actor.optim.lr=2e-6 \
  "+actor.model.openpi.noise_logvar_range=[0.02,0.04]"
```

| 参数 | 值 | 为什么 |
|---|---|---|
| `--config-name so101_ppo_v11` | | 官方配方（`adv_type: gae`、`loss_type: actor_critic`、clip 0.2、entropy 0），起点已指向 v10 |
| `runner.val_check_interval` / `save_interval` | 5 / 5 | 每 5 轮评一次、存一次；峰值很窄，存疏了会错过 |
| `actor.optim.lr` | 2e-6 | 官方 5e-6 的一半——起点已经很好，步子迈大了会毁掉它 |
| `setsid` | | 脱离终端。**不这么做，SSH 一断训练就没了** |
| `SCRATCH` | 任意可写目录 | 脚本把日志/状态写在这里 |

**自动停机守卫**每 5 分钟读一次 `eval/success_once`：低于峰值 20 点 → 停；连续 3 次低于起点 5 点 → 停。**必须写在启动器里**，写在会话里会随会话消失（历史上因此白烧 180 轮）。

### 8.4 输出与验收

| step | 4 | 9 | 14 | 19 | 24 | **29** | 34 | 39 | 44 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval | 60.9 | 58.6 | 62.5 | 58.6 | 70.3 | **73.4** | 69.5 | 70.3 | 68.0 | 68.8 |

前 20 轮徘徊（**不要在这里判死刑**），step 24 起上台阶。

**输出** `results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30`。

**验收（诚实口径）**——种子 4141/4242 从未参与挑选，与起点同条件对照，**动作块长同为 10**：

| | 4141 | 4242 | 均值 |
|---|---|---|---|
| 起点（阶段 E 产物） | 48.4% | 55.5% | 52.0% |
| **PPO 后** | **58.6%** | **57.0%** | **57.8%** |
| 增益 | +10.2 | +1.5 | **+5.9** |

框内参照 77.3%、全板参照 14.8%。**耗时约 2 h。**

> **报告里请引用 +5.9 这个诚实增益。** 门评口径上是 61.7%→73.4%（+11.7），但峰值检查点正是在那套固定评测局面上挑出来的，会偏乐观。

> ⚠️ **动作块长会改变成绩，比较时必须对齐**：`so101_eval_openpi_pi05.yaml` 的默认值是 `num_action_chunks: 5`，阶段 B–E 的门评都用的这个默认；阶段 F 起改成 10。所以**跨阶段比数字前，先确认两边块长一致**——同一个检查点从 5 换到 10 可以白涨 11 点。

---

## 9. 阶段 G —— 真机协同训练（打通 sim2real）

**为什么** A–F 只回答了"在仿真里会不会做这件事"。**换成真实相机图像，策略还认不认得，是另一个问题**，而且答案一开始是否定的。

**输入** 阶段 F 的 `so101_ppo_v13/.../global_step_30` + 87 集真机数据 + 阶段 E 的仿真数据集 `so101-sim-demos-v10`。

> **数据集路径的一个现实问题**：HF 上的 repo id 是 `henry-guo/so101-pick-place-v2`，而本机上这份数据落在 `/data08/henryg/pai/data/so101-pick-place-v1-trimmed`（87 集、30 fps、LeRobot **v3.0** 布局）——**是同一份数据**。阶段 A 按 HF repo id 找它，而 G 阶段的转换脚本里 `REAL_ROOT` 硬写的是本机路径。换机器时改这一行，或者做个软链。

### 9.1 先把问题量出来（不碰机器人）

**为什么** 直接上机去试，代价是可能撞坏机械臂，而且失败了也不知道是感知问题还是控制问题。离线检验 20 分钟、零硬件，就能给出可判定的数字。

**做法**：把真机录制的（前视、腕视、关节状态）喂给策略，比较它预测的动作与人类当时的实际动作。两个对照让数字可解释——**仿真数据**（分布内参照）和**"完全不动"**（动作幅度的尺度）。比值 <1 表示优于什么都不做。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools_so101_session/offline_replay_check.py \
  --ckpt /data08/henryg/pai/results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30 \
  --config-name pi05_so101_v10 \
  --norm-stats $PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
  --chunks 10 --real-root $DATA/so101-pick-place-v1-trimmed \
  --episodes 5 --frames 10
```

| 参数 | 含义 |
|---|---|
| `--chunks` | 动作块长；**要与被测策略的训练/部署一致**，否则测的是别的东西 |
| `--real-root` | 真机数据集根目录（v2.0 和 v3.0 两种布局都支持） |
| `--episodes` / `--frames` | 取几集、每集采样几帧 |

**输出**（stdout）：逐关节的 policy MAE、hold-still MAE、比值，以及仿真侧和真机侧各一个总比值。

| 检查点 | 仿真比值 | 真机比值 |
|---|---|---|
| 阶段 F 产物（仿真训练） | 0.10 ✅ | **4.47** ❌ |
| 阶段 A 产物（真机训练） | 3.82 ❌ | **0.22** ✅ |

两者**互为镜像**：各自在自己的域里好、在对方的域里差。这既证明这是**视觉域鸿沟**而非某个部件的 bug，也证明这个指标本身可信。

**不要停在第一个嫌疑上。** 当时的首要嫌疑是仿真腕部相机指向错误（它拍的是机械臂自己）。但加 `--no-wrist` 把这一路砍掉后真机比值几乎不变（4.47 → 4.59）——差距比这一个缺陷更广。**如果当时收手去修相机，30 小时机时会白花。**

### 9.2 做法：把真机数据掺进仿真数据集重训

**这一阶段实际是两次训练**（v14 → v15），第二次才是交付物：

| 轮次 | 数据 | 热启动 | 真机比值 |
|---|---|---|---|
| v14 | 真机**全部 87 集** ×2 + 仿真 1292 集 | 阶段 F 的 `global_step_30` | 0.84（**训练集口径，不可用**） |
| **v15** | 真机 **0–69 集** ×3 + 仿真 1292 集，**70–86 留出** | **v14 的 `global_step_1750`** | **0.70（留出集）** |

**为什么要两次**：v14 把 87 集全部拿去训练，然后又用其中几集读离线指标——那是拿训练集当考卷。v15 留出 17 集从不参训，才是第一个能作为上机依据的数字。**从零复现可以直接跑 v15**（用 `convert_cotrain_heldout.py` 的数据集、从阶段 F 热启动），但那条路没有实测过；这里记录的是实际跑出 0.70 的那条。

```bash
# G1 建数据集（纯 CPU，约 5.5 h；会先复制一份仿真数据集再往里追加）
.venv/bin/python tools_so101_session/convert_cotrain_simreal.py   # -> so101-cotrain-v14
.venv/bin/python tools_so101_session/convert_cotrain_heldout.py   # -> so101-cotrain-v15

# G2 两次训练，都按 §1.5 ①
export EMBODIED_PATH=$PWD/examples/sft
for CFG in so101_sft_v14 so101_sft_v15; do
  OUT=/data08/henryg/pai/results/$CFG
  .venv/bin/python -m toolkits.preflight_config \
    --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
    --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT
  for CK in $OUT/*/checkpoints/global_step_*; do
    mkdir -p "$CK/so101-sim-demos-v4"
    cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"
  done
done
```

| 参数 | 值 | 理由 |
|---|---|---|
| 真机上采样 | v14 ×2 / v15 ×3 | 87 集对 1292 集，不上采样在梯度里几乎看不见；v15 只用 70 集训练，所以倍数提高 |
| `lr` | 1e-5 | 轻量微调，别把仿真能力洗掉 |
| 步数 / 存点 | 2000 / 250 | 门评逐个检查点做 |
| 长度过滤 | `MIN_LEN, MAX_LEN = 80, 1000` | **不能沿用仿真示范的 580**——人类遥操作更慢，87 集是 395–825 帧、中位 575，**用 580 会静默丢掉 46% 的真机数据** |

### 9.3 门评必须是双轴的

**为什么** 这一步**targets 真机表现，同时有能力破坏仿真能力**。单轴门评会拿仿真能力换真机能力而不自知——实测抓到过一次：750 步时真机比值已改善到 1.16，而仿真掉到 52.3%，只看目标轴会以为是进步。

| 轴 | 指标 | 怎么测 | 判据 |
|---|---|---|---|
| 目标 | 离线真机比值（**留出集**） | 下面这条命令 | 越低越好，必须 <1 |
| 约束 | 仿真环 1 成功率 | §1.5 ②，条目 `pi05_so101_v15`、`SO101_SPAWN_FRAC=$RING1`、**`rollout.model.num_action_chunks=10`** | **不得跌破 50%** |

**约束必须在选点时有否决权**，不是事后看看。

```bash
for CK in /data08/henryg/pai/results/so101_sft_v15/*/checkpoints/global_step_*; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools_so101_session/offline_replay_check.py \
    --ckpt $CK --config-name pi05_so101_v15 \
    --norm-stats $PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    --chunks 10 --real-root $DATA/so101-pick-place-v1-trimmed \
    --ep-start 70 --episodes 5 --frames 10        # ep-start 70 = 只读留出集
done
```

### 9.4 输出与验收

**输出** `results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000`。

| 指标 | 阶段 F 产物 | **阶段 G 产物** |
|---|---|---|
| 离线真机比值（留出集，10 步时域） | 4.47 | **0.70** ✅ |
| 仿真环 1 | 57.8% | **62.5% / 65.6%**（约 60%，未下降） |

**离线门通过，可以上真机。** 部署方案见 `SIM2REAL_PLAN_ZH.md`。**耗时约 7 h**（5.5 h 建数据集 + 1.5 h 训练）。

> **一个必须知道的顺带发现**：**腕部相机那一路的重要性在协同训练后反转了**。协同训练前砍掉它没区别（4.47→4.59），之后砍掉它比值从 0.90 涨到 1.58——模型现在真的在用真实腕视图像。**部署时两路相机都必须送。**

---

## 10. 全流程验收表

**任何一步显著低于预期就停下排查**，后面每一步都建立在前一步之上。

| 阶段 | 产物 | 验收数字 | 耗时 |
|---|---|---|---|
| A 真机 SFT | `global_step_8000` | 不评测 | ~3 h |
| B 全板 SFT | `v4/global_step_1000` | 全板 **12.5%** | ~7 h |
| C 收窄 | `v8/global_step_2500` | 诚实 **56.7%**，间距 0.44 cm | ~5 h |
| D 专家迭代 | `v9/global_step_1250` | 诚实 **76.6%**，全板 19.5% | ~5 h |
| E 环 1 扩域 | `v10/global_step_1000` | 环 1 **55.1%**（负结果） | ~7 h |
| **F PPO** | `so101_ppo_v13/.../global_step_30` | 环 1 诚实 **57.8%**（+5.9） | ~2 h |
| **G 协同训练** | `so101_sft_v15/.../global_step_1000` | 留出集真机比值 **0.70**，仿真不塌 | ~7 h |

> **这份文档的命令做过一次静态核对**（2026-08-15）：所有引用的脚本、配置、注册表条目、检查点目录、数据集目录都确认存在；每个 SFT 配置的 `preflight_config` 实跑到 `PREFLIGHT OK`；PPO 的批量算式实跑输出 `-> 1 update(s)/epoch`。**没有重跑训练本身**——成绩数字来自当时的运行记录。

---

## 11. 会浪费一整夜的坑

| 症状 | 真因 | 处置 |
|---|---|---|
| NCCL 直接失败，像显存不足 | `/dev/shm` 只有 64 MB | remount 16G；先看日志里 `Last error:` 那行 |
| 评测卡住、驱动进程永远等待 | Ray worker `SYSTEM_ERROR` 猝死 | 每次评测加 `timeout 1800` + 3 次重试 + 完整清理 |
| 改了参数但行为没变 | 参数在**三处**独立定义（env yaml、生成器自己的 `gym.make`、转换器硬编码） | 改任何参数前 `grep` 出所有消费方 |
| 调了噪声却没有任何变化 | `noise_params` 对 `flow_noise` 是空参数 | 用 `noise_logvar_range` |
| 换数据集后成绩掉一半 | 重算了 norm_stats | 血统内绝不重算 |
| 流水线在离终点很近处被杀 | 超时按整数拍脑袋 | 按**实测速率**定，留 ≥50% 余量 |
| 上一阶段跑完，下一阶段没启动，GPU 空转数小时 | 等待脚本先查完成标记再查上游存活；两者只差微秒 | 发现上游消失后**必须再查一次完成标记**；别把哨兵串写进自己的日志文案 |
| 监工从不报告"任务已结束" | `pgrep -f` **匹配到了自己** | 用启动时记下的 pid 读 `/proc/<pid>/stat`（僵尸进程 `/proc/<pid>` 仍在） |
| 多个后继同时启动、互相杀 Ray | 后继之间没有串行 | 后继必须串成链，尤其当各阶段都做全局清理时 |

---

## 12. 相关文档

| 文档 | 内容 |
|---|---|
| `PPO_V13_RUNBOOK_ZH.md` | PPO 成功配方的详细步骤与失败对照表 |
| `PPO_CODE_WALKTHROUGH_ZH.md` | PPO 的代码级全流程（带文件行号） |
| `SIM2REAL_PLAN_ZH.md` | 真机部署方案（离线门已通过） |
| `V10_REPRODUCTION_ZH.md` | 阶段 A–F 的更详细版本，另附「附录 P」记录一版从未跑通的 PPO 接线 |
| `SO101_TOOLS_RUNBOOK_ZH.md` | 每个脚本干什么、要解决什么问题 |

---

## 13. 局限与下一步（写报告时必须一并说明）

### 13.1 已经解决的：视觉域鸿沟

阶段 F 的产物在真实观测上比"完全不动"还差 4.5 倍（比值 4.47）。阶段 G 的协同训练把它压到 **0.70**（留出集口径），**离线门已通过**。详见 §9，部署方案见 `SIM2REAL_PLAN_ZH.md`。

复现命令：

```bash
python tools_so101_session/offline_replay_check.py \
  --ckpt <检查点> --real-root <真机数据集> --ep-start 70 --episodes 5 --frames 10
```

### 13.2 仍未解决的

| 项 | 内容 | 代价 |
|---|---|---|
| **离线指标 ≠ 能完成任务** | 0.70 只证明"策略在真实观测下动作合理"，**不证明**闭环能把方块放进托盘。完成任务还要求闭环稳定性，**这只有真机能测** | 上机 1 天 |
| 与上界仍有 3.2 倍差距 | 真机数据训出的策略是 0.22。但那个数字也是训练集口径，**真正的可比上界未知**；同一份数据多训 2000 步比值还能降到 0.68，说明未收敛 | 每轮约 1.5 h，但要盯住仿真那一轴 |
| 仿真腕部相机指向仍是错的 | 协同训练让模型学会了处理真实腕视图像，但**仿真数据里那一路仍然是废的**——两个域的信息量不对称 | 重录数据 + 从阶段 B 重训，约 30 h |
| 域随机化未做 | 仿真渲染时随机化光照/材质/相机位姿 | 需改环境 + 重新生成示范并重训 |

**任何改动都必须重新过 §9.1 那个离线检验，且按 §9.3 的双轴门评选点。**

### 13.3 其它已知局限

| 项 | 内容 |
|---|---|
| 覆盖区域 | 策略只在 96 cm² 的环 1 内训练；全板（426 cm²）成绩仅 14.8%。真机摆放必须限制在对应矩形内 |
| 生成区边距 | 2 cm 的边距排除了约 11% 的真实方块起始位置 |
| 扩域未解决 | 把生成区从 48 扩到 96 cm² 没有带来提升（阶段 E 是负结果）；密度律不能外推成"扩域只要补够密度" |
| 物理参数 | 摩擦系数等未经真机标定，属声明的默认值；方块质量按用户给的上界（<10 g）设定 |
| 单一任务 | 只验证了"抓红方块放托盘并回位"这一个任务 |
