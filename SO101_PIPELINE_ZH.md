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

> **正文一律用功能名**（"阶段 B 的最优检查点"、"窄框数据集"）。命令块里出现的 `v4`/`v8`/`v9`/`v10`/`v13`/`v14`/`v15` 是**磁盘上的真实名字**，那里必须精确；**它们没有含义**，只是当时实验的流水编号。
> 全部真实名字集中在**一张表 §1.5**里。从零复现时可以整套换成自己的名字，唯一的约束见 §4。
> 脚本文件名里的编号已经全部去掉，改成按功能命名（`convert_v4_demos.py` → `convert_fullboard.py`）。

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

### 1.4 每条训练/评测命令共有的五条规则

阶段 A–G 的命令都写在各自章节里，**不需要回到这里拼装**。这一节只讲那些命令里反复出现、但原因不在命令本身的东西——踩错任何一条，跑出来的数字都不可信。

| 规则 | 具体做法 | 不遵守的后果 |
|---|---|---|
| **① 训练前先 preflight** | `python -m toolkits.preflight_config --config-name <配置> runner.logger.log_path=<输出>`，必须看到 `PREFLIGHT OK` | 它校验路径存在性、批量算术、模型-数据一致性。**PPO 的成败就藏在批量算术里**（§8.1） |
| **② 训练/评测前清 Ray** | `.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*` | 会连上一次的残留集群，行为诡异且难查 |
| **③ 训练后把 norm_stats 复制进每个检查点** | `mkdir -p "$CK/so101-sim-demos-v4" && cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"` | openpi 按 `<检查点>/<数据集名>/norm_stats.json` 查找。**不放进去，后面的评测和部署全部失败** |
| **④ 每个评测都要 `timeout` + 失败重试一次** | `timeout 1800 ...`，并清 `/dev/shm` 里的 `cuda.shm.*` / `nccl-*` | Ray worker 偶发 `SYSTEM_ERROR` 猝死后驱动会**永远等待**；残留共享内存段会让下一次评测以"看起来像显存不足"的方式失败 |
| **⑤ `norm_stats_path` 全程写死同一份** | 永远是 `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` | 它**不随 `config_name` 变**——条目换了数据集，但坐标系必须不变（§4） |

**还有一条关于种子的纪律**，它决定了报出来的数字诚不诚实：

| 层 | 种子 | 用途 |
|---|---|---|
| 筛选 + 确认 | `777` / `888` | 用来**挑**检查点 |
| **诚实复测** | 每阶段各一对，**从不重用**：C `1313/1414`、D `2323/2424`、E `3131/3232`、F `4141/4242` | **报告只能引用这一层** |

用挑选种子报出来的数字必然偏乐观——峰值检查点正是在那套局面上选出来的。

**区域怎么指定**：`SO101_SPAWN_FRAC="x0,x1,y0,y1"`（归一化比例，环 1 = `0.4294,0.9115,0.5142,0.9817`）；`SO101_SPAWN_MODE=legacy` 是早期的 6×8 cm 固定窄框，与前者互斥；**两个都不设就是全板**。

### 1.5 各阶段产物一览（磁盘上的真实名字都在这里）

> **这张表可以机器核对，不用信我**：
>
> ```bash
> .venv/bin/python tools_so101_session/check_doc_consistency.py SO101_PIPELINE_ZH.md
> ```
>
> 它把文档里出现的每个名字按类别对到仓库/磁盘上——注册表条目查 `dataconfig/__init__.py`、训练配置查 `examples/*/config/*.yaml`、数据集查 `meta/info.json`、脚本和统计量查文件是否存在——并且**双向**核对这张表：文档让你用的条目必须在表里，表里列的条目也必须真的被用到。有一处对不上就退出码非零。
>
> 这个检查存在的原因很直接：本文档曾经有两张互相重叠的名字表，一张声称"本流程用到 6 个条目"，而阶段 G 实际用了第 7 个（`pi05_so101_v14`）。人眼复核连续几轮都没发现，机器一次就查出来了。现在只保留这一张表。


**正文里一律用功能名**（"阶段 B 的最优检查点"、"窄框数据集"），磁盘上的真实名字只出现在两个地方：**这张表**，和各阶段的命令块里（那里必须精确）。

编号 `v4/v8/v9/v10/v13/v14/v15` **没有含义**——是当时实验的流水编号，各阶段恰好用了不同的数字。从零复现时你可以全部换成自己的名字，唯一的约束见 §4（同一条血统内不能中途改名）。

| 阶段 | 功能名 | 数据集（`$HF_LEROBOT_HOME/` 下） | 注册表条目 | 训练配置 | 结果目录 | 最优检查点 |
|---|---|---|---|---|---|---|
| A | 真机 SFT | `henry-guo/so101-pick-place-v2` | `pi05_so101` | `so101_sft_openpi_pi05` | `so101_sft_openpi_pi05` | `global_step_8000` |
| B | 全板示范 | `so101-sim-demos-v4` | `pi05_so101_v4` ← **血统在此确立** | `so101_sft_v4` | `so101_sft_v4` | `global_step_1000` |
| C | 窄框（48 cm²） | `so101-sim-demos-v8` | `pi05_so101_v8` | `so101_sft_v8` | `so101_sft_v8` | `global_step_2500` |
| D | 专家迭代 | `so101-sim-demos-v9` | `pi05_so101_v9` | `so101_sft_v9` | `so101_sft_v9` | `global_step_1250` |
| E | 环 1 扩域 | `so101-sim-demos-v10` | `pi05_so101_v10` | `so101_sft_v10` | `so101_sft_v10` | `global_step_1000` |
| F | PPO | **无新数据集** | `pi05_so101_v10`（沿用 E 的） | `so101_ppo_v11` | `so101_ppo_v13` | `global_step_30` |
| G | 协同训练（第一轮） | `so101-cotrain-v14` | `pi05_so101_v14` | `so101_sft_v14` | `so101_sft_v14` | `global_step_1750` |
| **G** | **协同训练（交付）** | `so101-cotrain-v15` | `pi05_so101_v15` | `so101_sft_v15` | `so101_sft_v15` | **`global_step_1000`** |

三个容易踩的不一致，都在这张表里能看出来：

| 现象 | 说明 |
|---|---|
| 阶段 F 的**配置叫 `so101_ppo_v11`，结果目录却叫 `so101_ppo_v13`** | 配置文件是第 11 版写的，成功的那次运行是第 13 次尝试。启动命令里两个名字都要写对 |
| 阶段 F **没有自己的注册表条目** | PPO 不产生新数据集，评测时 `config_name` 仍填 `pi05_so101_v10` |
| 所有阶段的 `norm_stats` **都指向 B 的那一份** | 路径永远是 `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json`，不随条目变（§4） |

**注册表条目**（`rlinf/models/embodiment/openpi/dataconfig/__init__.py`）7 个结构完全相同（`action_horizon=10`、`discrete_state_input=True`、`extra_delta_transform=False`），只有 `repo_id` 不同。**`config_name` 选的是数据变换管线，`norm_stats_path` 由 yaml 单独指定**——正因为这两件事是分开的，各阶段才能用自己的数据集条目却共享同一份统计量。

**结果目录的完整路径**是 `/data08/henryg/pai/results/<结果目录>/<内层目录>/checkpoints/<检查点>`。内层目录 SFT 阶段一律是 `so101_sft_openpi_pi05`，PPO 阶段是 `so101_ppo_v11`（跟配置名走）——所以文中写成 `so101_sft_v9/.../global_step_1250` 这种形式，中间那段用 `*` 匹配即可。

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

| 项 | 值 |
|---|---|
| 数据 | `$HF_LEROBOT_HOME/henry-guo/so101-pick-place-v2`（87 集真机遥操作，30 fps，640×480 双相机） |
| 权重 | `checkpoints/lerobot_pi05_base`（PI0.5 基座，LeRobot 格式，14 GB） |
| 注册表条目 | `pi05_so101` |

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/sft
OUT=/data08/henryg/pai/results/so101_sft_openpi_pi05

# ① 算真机域的归一化统计量（--repo-id 是必填的，漏了会直接报 required）
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats \
  --config-name pi05_so101 --repo-id henry-guo/so101-pick-place-v2

# ② 训练
.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_openpi_pi05 \
  runner.logger.log_path=$OUT
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
timeout 43200 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name so101_sft_openpi_pi05 \
  runner.logger.log_path=$OUT
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

**验收** 训练 loss 正常下降即可，**不必做仿真评测**——这一步的价值不在成绩。想自己确认"真机数据训出来的策略在仿真里是 0"，用它自己的条目和统计量评一次（**这是唯一一次不用 v4 那份 norm_stats 的评测**，因为这个检查点属于阶段 A 的血统）：

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_eval_stageA \
  rollout.model.model_path=$OUT/checkpoints/global_step_8000 \
  rollout.model.openpi.config_name=pi05_so101 \
  rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101/henry-guo/so101-pick-place-v2/norm_stats.json \
  env.eval.total_num_envs=128 env.eval.seed=777 \
  2>&1 | grep -oE 'success_once=[0-9.]+' | tail -1
```

实测 `success_once=0.0`。**耗时约 3 h**（训练）。

---

## 4. 阶段 B —— 全板仿真示范 + 冻结统计量

**为什么** 真机数据教不会"在仿真里怎么做"（视觉域完全不同）。这一步用**运动规划器**在仿真里造示范，把成功率从 0 抬起来；同时**确立整条血统共用的归一化统计量**。

**输入** 阶段 A 的 `global_step_8000`；仿真环境（`SO101GrabRedCube-v1`，无需外部数据）。

**输出** 全板示范数据集、**这条血统的归一化统计量**、以及阶段 B 的最优检查点（名字见 §1.5）。

**验收** 全板约 **12.5%**。**耗时约 7 h。** 分五个子步骤，每步都有独立门槛。

### B1 规划器探针 —— 先证明任务可解

**为什么** 规划器做不到的事，BC 和 RL 都做不到。12 条的成本是分钟级，跳过它可能白花一整天。

**输入** 仿真环境；无数据依赖。

**命令**

```bash
.venv/bin/python tools_so101_session/gen_planner_demos.py \
  --num 12 --seed0 79000 --out $DATA/probe
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `--num` | 12 | 尝试次数（不是成功数）；只为判定"可解不可解" |
| `--seed0` | 79000 | 起始种子，逐条 +1。**要与后面生成用的种子段（80000+）错开**，避免拿同一批局面 |
| `--out` | `$DATA/probe` | h5 输出目录 |

**输出** `$DATA/probe/*.h5`（轨迹）+ 同名 `.json`（每条是否成功、长度）。生成器最后打印 `TOTAL success N/12`。

**验收** ≥8/12 成功、成功轨迹中位长度 ≤530 步（预算 640 的 1.1 倍余量）。**不过就不要往下走。**

```bash
# 成功数
grep -oE 'TOTAL success [0-9]+' <生成日志> | grep -oE '[0-9]+$'
# 成功轨迹的中位长度
.venv/bin/python - <<'PY'
import glob, json, h5py, numpy as np
h5 = sorted(glob.glob("/data08/henryg/pai/data/probe/*.h5"))[-1]
meta = json.load(open(h5.replace(".h5", ".json")))
ok = [e["episode_id"] for e in meta["episodes"] if e["success"]]
f = h5py.File(h5, "r")
print("median length:", int(np.median([f[f"traj_{i}"]["actions"].shape[0] for i in ok])))
PY
```

### B2 分层生成 420 条全板示范

**为什么** 要均匀铺满整块板。分成 4×4 格逐格生成，是为了避免随机采样在某些格子过密、某些格子空白——**BC 的地板由最稀的地方决定**。

**输入** 仿真环境；B1 已通过。

**命令**

```bash
SEED=80000
for XI in 0 1 2 3; do for YI in 0 1 2 3; do
  SO101_SPAWN_FRAC="$(echo "$XI*0.25"|bc -l),$(echo "($XI+1)*0.25"|bc -l),$(echo "$YI*0.25"|bc -l),$(echo "($YI+1)*0.25"|bc -l)" \
  .venv/bin/python tools_so101_session/gen_planner_demos.py --num 45 --seed0 $SEED \
      --out $DATA/v4_demos_cell_${XI}_${YI} > $SCRATCH/gen_${XI}_${YI}.out 2>&1 &
  SEED=$((SEED+100)); done; done; wait
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `SO101_SPAWN_FRAC` | 每格 `x0,x1,y0,y1` | 方块生成区在棕色板上的**归一化比例**；这里每格 1/4 × 1/4 |
| `--num 45` × 16 格 | 720 次尝试 | 实测约 420 条成功（成功率约 58%） |
| `--seed0` 每格 +100 | 80000 起 | 各格种子段不重叠 |
| 输出目录名 | **必须以 `v4_demos_cell` 开头** | 下一步的转换器按 `v4_demos_cell*/**/*.h5` 找输入（`convert_fullboard.py:19`）。后缀叫什么无所谓 |
| 并行度 | 16 个进程（8 核以上机器实测可行） | 纯 CPU，不占 GPU |

**输出** 16 个目录的 h5 + json，合计约 420 条成功轨迹。

**验收** 总成功数 ≥250。**耗时约 4 h。**

```bash
grep -hoE 'TOTAL success [0-9]+' $SCRATCH/gen_*.out | grep -oE '[0-9]+' | awk '{s+=$1} END{print s"/720"}'
```

### B3 转换 + 计算 norm_stats（仿真域只算这一次）

**为什么** h5 是生成器的原始格式，训练要的是 LeRobot 数据集；同时**这条血统的归一化统计量在这一步确立并从此冻结**。

**输入** `$DATA/v4_demos_cell*/**/*.h5`（B2 的产物）。

**命令**

```bash
.venv/bin/python tools_so101_session/convert_fullboard.py
.venv/bin/python -m toolkits.lerobot.calculate_norm_stats \
  --config-name pi05_so101_v4 --repo-id so101-sim-demos-v4
```

**参数**

| 步骤 | 输入 | 输出 |
|---|---|---|
| `convert_fullboard.py` | `$DATA/v4_demos_cell*/**/*.h5` | LeRobot 数据集 `$DATA/so101-sim-demos-v4`（30 fps、640×480、双相机；**轨迹长度 >580 帧的丢弃**，那是超时挣扎的轨迹） |
| `calculate_norm_stats` | 上面那个数据集 | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` |

> **`calculate_norm_stats` 在 `toolkits/lerobot/calculate_norm_stats.py`**，从仓库根目录以模块方式运行（`python -m toolkits.lerobot.calculate_norm_stats`）。
> 两个参数**都是必填**：`--config-name` 选注册表条目（决定数据怎么进模型），`--repo-id` 指数据集。
> 输出路径不是你指定的，是**算出来的**：`config.assets_dirs / repo_id`（`calculate_norm_stats.py:146`）——这就是 `assets/<条目名>/<数据集名>/norm_stats.json` 这个结构的由来，也是为什么条目名一改、统计量就换了地方。

**输出** 数据集约 420 集；统计量 JSON 一份（含 `state`/`actions` 各自的 mean/std/q01/q99）。

**验收**

```bash
python3 -c "import json; d=json.load(open('/data08/henryg/pai/data/so101-sim-demos-v4/meta/info.json')); print(d['total_episodes'],'集', d['fps'],'fps')"
ls -la assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
```

集数 ≥250、fps=30、统计量文件存在且非空。

> **`**` 是 Python 的递归通配符，不是 shell 的。** 它匹配**任意层数的子目录，包括零层**，而且**只有在 `glob.glob(..., recursive=True)` 下才生效**——转换器正是这么调的（`convert_fullboard.py:26`）。
>
> 本例里 h5 就直接躺在 `v4_demos_cell_*/` 下，所以 `**` 匹配的是零层，结果与 `v4_demos_cell*/*.h5` 完全相同（实测都是 16 个文件）。写 `**` 是留个余地：将来生成器若改成按子目录存放，转换器不用改。
>
> **两个都不会报错、只会静默匹配到 0 个文件的坑**：
>
> | 写法 | 实际含义 | 在本例中匹配到 |
> |---|---|---|
> | Python `glob(..., recursive=True)` | `**` = 任意层（含零层） | **16** ✅ |
> | Python 漏掉 `recursive=True` | `**` 退化成普通 `*` = 正好一层 | 0 ❌ |
> | bash 默认 | 同上，`**` 就是 `*` | 0 ❌ |
> | bash `shopt -s globstar` 之后 | `**` = 任意层 | 16 ✅ |
>
> 所以**别把这个模式直接抄进 shell 命令**——在 bash 里要先 `shopt -s globstar`，否则它会安静地找不到任何文件。

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

**为什么** 把仿真示范学进策略。热启动用阶段 A 的权重，这样任务语义（提示词、关节量纲）不用重学。

**输入** B3 产出的全板示范数据集 + 阶段 A 的最优检查点 + 冻结统计量。

> ⚠️ **仓库里的 `so101_sft_v4.yaml` 的 `model_path` 指向一个已废弃的早期检查点（`so101_sft_pp6b/...`），你不会有它。** 必须在命令行覆盖成阶段 A 的产物——下面的命令已经带上了。（SFT 配置里**没有 `rollout` 节点**，所以只覆盖 `actor.model.model_path`；写 `rollout.model.model_path=` 会被 Hydra 直接拒绝。）

**命令**

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

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `config_name` | `pi05_so101_v4` | 数据集换成仿真示范 |
| `norm_stats_path` | v4 那份 | 血统起点 |
| `lr` | 2.5e-5 | 与阶段 A 同量级；这仍是 BC |
| `max_steps` / `save_interval` | 4000 / 1000 | |

**输出** 4 个检查点（step 1000/2000/3000/4000），每个目录下都有一份 norm_stats 子目录。

**验收** 4 个检查点都产出、`ls` 能看到 norm_stats 子目录；成绩由 B5 判定。

### B5 门评

**为什么** SFT 的 loss 会一路降，但成绩在早期就见顶——**必须逐个检查点评，不能只看最后一个**。

**输入** B4 的 4 个检查点。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
for CK in $OUT/*/checkpoints/global_step_{1000,2000,3000,4000}; do
  for SEED in 777 888; do
    .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
    find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
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

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| 不设 `SO101_SPAWN_FRAC` | → 全板 426 cm² | 这一阶段的目标区域就是全板 |
| `env.eval.total_num_envs` | 128 | 配置默认只有 16，样本太少；128 集下 1 集 = 0.8 个百分点 |
| `env.eval.seed` | 777 / 888 | **挑选用**的两个种子，取均值 |
| `timeout 1500` | | Ray worker 偶发猝死后驱动会永远等待，**每个评测都要加，并失败重试一次** |

**输出** 每次评测的 `success_once=`；8 个数（4 检查点 × 2 种子）。

**验收** 最优点是 `global_step_1000`，全板约 **12.5%**。之后单调下降（2.0→2.3→0.0）——SFT loss 降到 0.002 却越训越差，是**过拟合规划器习惯**。挑出最优点后**再用一个没用过的种子（909）复测**，门评种子参与了挑选，会偏乐观。

---

## 5. 阶段 C —— 收窄生成区（用密度换成绩）

**为什么** 全板 426 cm² 要达到 0.44 cm 示范间距需要约 2200 条示范（约 20 h 生成）。先把生成区收窄到 48 cm²，**用同样的数据量换密度**——密度决定 BC 的地板，而不是总量。

**输入** 阶段 B 的最优检查点 + 冻结统计量。

**输出** 窄框示范数据集（247 集）+ 阶段 C 的最优检查点。

**验收** 框内诚实值 **57.8 / 55.5%**，间距 0.44 cm。**耗时约 5 h。**

### C1 生成窄框示范

**为什么** 同样的生成预算，区域小一半，间距就密一倍。

**输入** 仿真环境。

**命令**

```bash
export SO101_SPAWN_MODE=legacy      # 唯一的收窄项：6×8 cm = 48 cm²
for W in 0 1 2 3 4 5 6 7; do
  .venv/bin/python tools_so101_session/gen_planner_demos.py \
    --num 32 --seed0 $((90000 + W*1000)) --out $DATA/v8_demos_w$W \
    > $SCRATCH/gen_v8_w$W.out 2>&1 & done; wait
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `SO101_SPAWN_MODE=legacy` | 6×8 cm 框 | 与 `SO101_SPAWN_FRAC` 互斥，是早期版本留下的固定窄框 |
| `--num 32` × 8 worker | 256 次尝试 | 实测 247 条成功 |
| `--seed0` 每 worker +1000 | 90000 起 | 与 B 阶段的 80000 段错开 |
| 输出目录 | `v8_demos_w*` | **必须**——`convert_narrow_box.py:19` 按这个 glob 找输入 |

**输出** 8 个目录的 h5，合计约 247 条成功轨迹。

**验收** 总成功数 ≥200：`grep -hoE 'TOTAL success [0-9]+' $SCRATCH/gen_v8_w*.out | grep -oE '[0-9]+' | awk '{s+=$1} END{print s"/256"}'`

### C2 转换（不重算 norm_stats）

**为什么** 换数据集但**不能换坐标系**——血统冻结。

**输入** `$DATA/v8_demos_w*/**/*.h5`。

**命令**

```bash
.venv/bin/python tools_so101_session/convert_narrow_box.py
```

**参数**

| 项 | 值 | 为什么 |
|---|---|---|
| 输出数据集 | `$DATA/so101-sim-demos-v8` | 新条目 `pi05_so101_v8` 指向它 |
| **不调用** `calculate_norm_stats` | —— | 重算会让上游权重看到被换了刻度的输入，实测 19.5%→9.4% |
| `MAX_LEN` | 580 | 与 B 一致，丢弃超时挣扎的轨迹 |

**输出** 窄框 LeRobot 数据集，247 集。

**验收** 集数 ≥200，且 **`assets/` 下不应该多出这一阶段的统计量目录**——多出来了就说明误调了 `calculate_norm_stats`，血统已经断了。

> **这一条可以在任何时候自查，也是血统冻结唯一的硬证据：**
>
> ```bash
> ls assets/                 # 整条流程只应有两个 pi05_so101* 目录
> REF=$(md5sum assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json | cut -d' ' -f1)
> for f in /data08/henryg/pai/results/*/*/checkpoints/*/so101-sim-demos-v4/norm_stats.json; do
>   [ "$(md5sum $f | cut -d' ' -f1)" = "$REF" ] || echo "血统断了: $f"
> done
> ```
>
> 本仓库实测：`assets/` 下只有 `pi05_so101`（阶段 A 专用）和 `pi05_so101_v4`（血统），**没有 v8/v9/v10/v14/v15**——因为 C 之后再没算过统计量。B 到 G 全部 7 个训练产物里的 norm_stats **与基准逐字节相同**（md5 `10648366…`）。
> （`assets/` 下若还有 `pi05_so101_pp` / `_sim` / `_v3` / `_v5`，那是早期废弃实验留下的，与本流程无关。）

### C3 训练

**为什么** 在更密的数据上继续 BC。热启动用阶段 B 的最优点，不从头开始。

**输入** C2 产出的窄框数据集 + 阶段 B 的最优检查点 + 冻结统计量。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/sft
CFG=so101_sft_v8; OUT=/data08/henryg/pai/results/$CFG

.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
timeout 21600 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT

for CK in $OUT/*/checkpoints/global_step_*; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"
done
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| 配置 / 输出目录 | `so101_sft_v8` | 该 yaml 的 `model_path` 已正确指向 v4 的 step_1000，**无需覆盖** |
| `lr` | 2.5e-5 | 仍是纯规划器数据，可以用大一点的步子 |
| 步数 / **存点** | 4000 / **250** | 250 是教训值：最优点在 step_2500，按 1000 存会**整个错过** |

**输出** 16 个检查点（step_250 到 step_4000），每个都带 norm_stats 子目录。

**验收** 检查点齐全；成绩由 C4 判定。

### C4 门评：三层，别混为一谈

**为什么** 挑检查点的种子和报告成绩的种子必须分开，否则报的是自己挑出来的最好局面。

**输入** C3 的全部检查点。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
STATS=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
gate(){ # $1=检查点 $2=种子 $3=区域模式(legacy 或空)
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
  SO101_SPAWN_MODE="$3" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v8 \
    rollout.model.model_path="$1" \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$STATS \
    env.eval.total_num_envs=128 env.eval.seed=$2 \
    2>&1 | grep -oE 'success_once=[0-9.]+' | tail -1
}

# ① 筛选：全部检查点跑种子 777（峰值在中段，不能只评最后几个）
for CK in $OUT/*/checkpoints/global_step_*; do echo "$(basename $CK) $(gate $CK 777 legacy)"; done
# ② 确认：候选跑种子 888，按 (777+888)/2 选最优
for CK in <上一步靠前的几个>; do echo "$(basename $CK) $(gate $CK 888 legacy)"; done
# ③ 诚实复测：只对选中的那个，跑两个从未参与挑选的种子 —— 这才是报告里的数字
BEST=$OUT/so101_sft_openpi_pi05/checkpoints/global_step_2500
gate $BEST 1313 legacy;  gate $BEST 1414 legacy
gate $BEST 1313          # 不设 SPAWN_MODE = 全板参照
```

**参数**

| 层 | 种子 | 区域 | 作用 |
|---|---|---|---|
| 筛选 + 确认 | **777 / 888** | `legacy` 框内 | 用来**挑**检查点。挑出来的点在这套局面上必然偏乐观，不能拿去报告 |
| 诚实复测 | **1313 / 1414** | `legacy` 框内 | 从未参与挑选，**报告引用这两个** |
| 全板参照 | 1313 | 不设 `SO101_SPAWN_MODE` | 看窄框训练能不能外推 |

> **每个阶段的诚实种子都不重样**：C 用 1313/1414、D 用 2323/2424、E 用 3131/3232、F 用 4141/4242。**一个种子一旦参与过挑选，就永远不能再当诚实种子用。**

**输出** 阶段 C 的最优检查点（247 条示范、间距 **0.44 cm**）。

**验收** 框内诚实值 **57.8 / 55.5%**（种子 1313/1414），全板参照 9.4%。

> **判读纪律**：step_250 只有 7.8%、step_500 只有 0.8%——**前两个点低不代表方向错**，别在这里判死刑。

---

## 6. 阶段 D —— 专家迭代（零风险放大器，+20 点）

**为什么** 规划器示范只覆盖它自己的解法；让**当前策略**在没见过的局面上跑，把它自己成功的轨迹收回来重训，等于在策略实际会走的分布上加密数据。这一步是本流程性价比最高的（+20 点，无超参风险）。

**输入** 阶段 C 的最优检查点 + C1 的规划器 h5。

**输出** 混合数据集（672 集）+ 阶段 D 的最优检查点。

**验收** 框内诚实值 **77.3 / 75.8%**，全板 19.5%（翻倍）。**耗时约 5 h。**

### D1 采集策略自己的成功轨迹

**为什么** 让策略在**它自己会走到的状态分布**上产出数据——这正是纯规划器示范给不了的。

**输入** C3 的最优检查点。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
export SO101_COLLECT_DIR=$DATA/v9_rollouts
V8=/data08/henryg/pai/results/so101_sft_v8/so101_sft_openpi_pi05/checkpoints/global_step_2500
for SEED in 2001 2002 2003 2004 2005 2006 2007 2008; do
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  SO101_SPAWN_MODE=legacy timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v9 \
    rollout.model.model_path=$V8 \
    rollout.model.openpi.config_name=pi05_so101_v8 \
    rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 env.eval.seed=$SEED
done
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `SO101_COLLECT_DIR` | `$DATA/v9_rollouts` | **设了这个变量，评测才会把成功轨迹落盘**；`convert_expert_iter.py:53` 按这个路径找 npz |
| 种子 2001–2008 | 8 个全新种子 | 用训练过的种子采集等于把已经会的再抄一遍 |
| `SO101_SPAWN_MODE=legacy` | 框内 | 与 C 的训练区一致 |

**输出** `$DATA/v9_rollouts/*.npz`，实测 425 条成功轨迹（8 × 128 集，成功率约 42%）。

**验收** `ls $DATA/v9_rollouts/*.npz | wc -l` ≥300。少于这个数说明起点太弱，先回 C 排查。

### D2 混合转换

**为什么** **必须混合，不能纯自蒸馏**：只用策略自己的轨迹会让它越练越窄（历史上掉 53 点）。

**输入** C1 的规划器 h5 + D1 的策略 npz。

**命令**

```bash
.venv/bin/python tools_so101_session/convert_expert_iter.py
```

**参数**

| 项 | 值 | 为什么 |
|---|---|---|
| 输入 1 | `v8_demos_w*/**/*.h5`（247 条规划器） | 保住多样性 |
| 输入 2 | `v9_rollouts/*.npz`（425 条策略） | 在策略分布上加密 |
| 输出 | `$DATA/so101-sim-demos-v9`（672 集） | |
| `MIN_LEN, MAX_LEN` | 80, 580 | 太短的是误判成功，太长的是超时挣扎 |

> **单位不对称陷阱**：录制器写出的 npz 里 `state` **已归一化**而 `action` 是**弧度**；h5 两者都是弧度。转换器对两种来源分别处理，**自己写采集器时这是最容易错的一处**。

**输出** 混合 LeRobot 数据集，672 集。

**验收** `python3 -c "import json;print(json.load(open('/data08/henryg/pai/data/so101-sim-demos-v9/meta/info.json'))['total_episodes'])"` ≈ 672。

### D3 训练

**为什么** 把加密后的数据学进去。学习率要比 C 小，因为数据里有一半是策略自己的轨迹，步子迈大会把刚学会的抹掉。

**输入** D2 产出的混合数据集 + 阶段 C 的最优检查点 + 冻结统计量。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/sft
CFG=so101_sft_v9; OUT=/data08/henryg/pai/results/$CFG

.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT

for CK in $OUT/*/checkpoints/global_step_*; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"
done
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| 配置 | `so101_sft_v9` | 其 yaml 的 `model_path` 已指向 v8 的 `global_step_2500`，无需覆盖 |
| `lr` | **1e-5**（C 是 2.5e-5） | 数据里有一半是策略自己的轨迹，步子迈大会把刚学会的抹掉 |
| 步数 / 存点 | 2000 / 250 | |

**输出** 8 个检查点（step_250 到 step_2000），每个都带 norm_stats 子目录。

**验收** 检查点齐全；成绩由 D4 判定。

### D4 门评

**为什么** **必须评全部检查点**：v9 的峰值在 step_1250，而最后一个点只有 7.8%。只评最后几个会得出"这一阶段失败了"的相反结论。

**输入** D3 的全部检查点。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
STATS=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
gate(){ # $1=检查点 $2=种子 $3=区域模式(legacy 或空)
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
  SO101_SPAWN_MODE="$3" timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v9 \
    rollout.model.model_path="$1" \
    rollout.model.openpi.config_name=pi05_so101_v9 \
    rollout.model.openpi_data.norm_stats_path=$STATS \
    env.eval.total_num_envs=128 env.eval.seed=$2 \
    2>&1 | grep -oE 'success_once=[0-9.]+' | tail -1
}

# ① 筛选：全部 8 个检查点 × 种子 777
for CK in $OUT/*/checkpoints/global_step_*; do echo "$(basename $CK) $(gate $CK 777 legacy)"; done
# ② 确认：候选 × 种子 888，按均值选最优
# ③ 诚实复测：只对选中的那个
BEST=$OUT/so101_sft_openpi_pi05/checkpoints/global_step_1250
gate $BEST 2323 legacy;  gate $BEST 2424 legacy
gate $BEST 2323          # 全板参照
```

**参数**

| 层 | 种子 | 区域 | 作用 |
|---|---|---|---|
| 筛选 + 确认 | 777 / 888 | `legacy` 框内 | 挑检查点 |
| 诚实复测 | **2323 / 2424** | `legacy` 框内 | 报告引用这两个（**与 C 的 1313/1414 不重用**） |
| 全板参照 | 2323 | 不设 | 看外推能力，实测从 9.4% 翻倍到 19.5% |

**输出** 阶段 D 的最优检查点。

**验收** 框内诚实值 **77.3 / 75.8%**（种子 2323/2424），全板 19.5%。

---

## 7. 阶段 E —— 环 1 扩域（负结果，但产出了 PPO 的起点）

**为什么** 想把 48 cm² 的能力扩到 96 cm²。**结论是没做到**，但这一阶段的产物是 PPO 的正确起点——因为它是在 PPO 将要训练的那个区域上训出来的。

**输入** 阶段 D 的最优检查点 + D 的混合数据集。

**输出** 环 1 数据集（1292 集）+ 阶段 E 的最优检查点，后者是 PPO 的起点。

**验收** 环 1 诚实 55.1%、框内 75.0%、全板 10.2%——**是负结果**。**耗时约 7 h。**

### E1 用 v9 在环 1 上自采

**为什么** 与 D1 同理，只是区域换成即将成为目标区的环 1。

**输入** 阶段 D 的最优检查点。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
export RING1="0.4294,0.9115,0.5142,0.9817"     # 环 1：8.48 × 11.31 cm = 96 cm²
export SO101_COLLECT_DIR=$DATA/v10_rollouts
V9=/data08/henryg/pai/results/so101_sft_v9/so101_sft_openpi_pi05/checkpoints/global_step_1250
for SEED in 3001 3002 3003 3004 3005 3006 3007 3008; do
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  SO101_SPAWN_FRAC=$RING1 timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v10 \
    rollout.model.model_path=$V9 rollout.model.openpi.config_name=pi05_so101_v9 \
    rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    env.eval.total_num_envs=128 env.eval.seed=$SEED
done
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `SO101_SPAWN_FRAC` | `0.4294,0.9115,0.5142,0.9817` | 环 1 的归一化边界；**后面 PPO 和真机摆放都用这一组数** |
| `SO101_COLLECT_DIR` | `$DATA/v10_rollouts` | `convert_append_region.py:56` 按这个路径找 npz |
| 种子 3001–3008 | 又一批全新种子 | 与 D 的 2001–2008 不重用 |

**输出** `$DATA/v10_rollouts/*.npz`，实测 429 条成功轨迹。

**验收** `ls $DATA/v10_rollouts/*.npz | wc -l` ≥300；基线成功率（种子 3000）实测 51.6%。

### E2 只在新增的环形带上补规划器示范

**为什么** 内框已经够密，重新生成整片区域是浪费；只补**环 1 减去内框**那圈。

**输入** 仿真环境。

**命令**

```bash
SCRATCH=/tmp/so101_runs bash tools_so101_session/gen_demos_annulus.sh
```

**参数**

| 项 | 值 | 为什么 |
|---|---|---|
| `SCRATCH` | 任意可写目录 | 脚本把日志和状态写进 `$SCRATCH`（默认 `/tmp/so101_runs`）；**它本来硬编码的是写作时的会话目录** |
| 输出目录 | `$DATA/v10_demos_w*` | `convert_append_region.py:67` 按此 glob 找 |
| 依赖 | 等 E1 打出完成标记后才开工 | 脚本内部会等 `collect_policy_successes.sh` 结束 |

**输出** `$DATA/v10_demos_w*/**/*.h5`，实测 204 条环形带示范。

**验收** 总成功数 ≥150。

### E3 追加转换

**为什么** 在 v9 数据集的**副本**上追加，不重编码旧集——旧集重编码要多花数小时。

**输入** D 的混合数据集 + E1 的策略 npz + E2 的环形带 h5。

**命令**

```bash
.venv/bin/python tools_so101_session/convert_append_region.py
```

**参数**

| 项 | 值 | 为什么 |
|---|---|---|
| 源 | `so101-sim-demos-v9`（672 集） | 先整目录复制 |
| 追加 1 | `v10_rollouts/*.npz`（429 条） | 环 1 策略轨迹 |
| 追加 2 | `v10_demos_w*/**/*.h5`（204 条） | 环形带规划器示范 |
| 输出 | `so101-sim-demos-v10`（1292 集） | |

**输出** 环 1 LeRobot 数据集，1292 集。

**验收** 集数 ≈1292（672 + 429 + 204，允许少量长度过滤损耗）。

### E4 训练

**为什么** 把扩域后的数据学进去，产出 PPO 的起点。

**输入** E3 产出的环 1 数据集 + 阶段 D 的最优检查点 + 冻结统计量。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/sft
CFG=so101_sft_v10; OUT=/data08/henryg/pai/results/$CFG

.venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT
.venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
timeout 14400 .venv/bin/python examples/sft/train_vla_sft.py \
  --config-path $PWD/examples/sft/config/ --config-name $CFG runner.logger.log_path=$OUT

for CK in $OUT/*/checkpoints/global_step_*; do
  mkdir -p "$CK/so101-sim-demos-v4"
  cp assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json "$CK/so101-sim-demos-v4/"
done
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| 配置 | `so101_sft_v10` | 其 yaml 的 `model_path` 已指向 v9 的 `global_step_1250` |
| `lr` / 步数 / 存点 | 1e-5 / 2000 / 250 | 与 D 相同：数据里仍有大量策略自采轨迹 |

**输出** 8 个检查点，每个带 norm_stats 子目录；最优点是 `global_step_1000`。

**验收** 检查点齐全；成绩由 E5 判定。

### E5 门评——区域换成环 1

**为什么** 这是与 C/D 唯一的结构性差别：目标区变成了环 1。而且**三个参照缺一不可**——只看环 1 会以为扩域"没坏"，加上框内和全板才看得出它在原来擅长的区域退步了。

**输入** E4 的全部检查点。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
export RING1="0.4294,0.9115,0.5142,0.9817"
STATS=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
gate(){ # $1=检查点 $2=种子 $3=SPAWN_FRAC $4=SPAWN_MODE(可空)
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
  SO101_SPAWN_FRAC="$3" SO101_SPAWN_MODE="${4:-}" timeout 1800 \
    .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v10 \
      rollout.model.model_path="$1" \
      rollout.model.openpi.config_name=pi05_so101_v10 \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      env.eval.total_num_envs=128 env.eval.seed=$2 \
      2>&1 | grep -oE 'success_once=[0-9.]+' | tail -1
}

# ① 筛选 ② 确认：环 1，种子 777 / 888
for CK in $OUT/*/checkpoints/global_step_*; do echo "$(basename $CK) $(gate $CK 777 $RING1)"; done
# ③ 诚实复测 + 两个参照
BEST=$OUT/so101_sft_openpi_pi05/checkpoints/global_step_1000
gate $BEST 3131 "$RING1";  gate $BEST 3232 "$RING1"     # 环 1 诚实值
gate $BEST 3131 "0,1,0,1" legacy                        # 框内参照（与 v9 的 77.3% 可比）
gate $BEST 3131 "0,1,0,1"                               # 全板参照
```

**参数**

| 层 | 种子 | 区域 | 作用 |
|---|---|---|---|
| 筛选 + 确认 | 777 / 888 | 环 1 | 挑检查点 |
| 诚实复测 | **3131 / 3232** | 环 1 | 报告引用这两个（**与 C、D 的不重用**） |
| 框内参照 | 3131 | `legacy` | 与 v9 的 77.3% 直接可比 |
| 全板参照 | 3131 | 不设 | 与 v9 的 19.5% 直接可比 |

**输出** 阶段 E 的最优检查点（数据集 1292 集）。

**验收** 环 1 诚实 55.1%（种子 3131/3232）、框内 75.0%、全板 10.2%。

> **这是负结果**：扩域没有在目标区带来提升（v9 在环 1 上本来就有 58.6%），全板还掉了一半。**密度律不能外推成"扩域只要补够密度"**——v9 在外环一条示范都没有却已经有 58.6%，说明外环从来不缺示范。
> 保留它是因为它是 PPO 的起点：PPO 在环 1 上训练，起点也应当在环 1 上训过。

---

## 8. 阶段 F —— PPO 在线微调

**为什么** 到这里为止全是监督学习——策略只会模仿示范。PPO 让它在**自己实际会遇到的局面**上试错并放大成功。完整细节见 `PPO_V13_RUNBOOK_ZH.md`。

**输入** 阶段 E 的最优检查点（**已写死在 PPO 配置里，无需覆盖**）+ 冻结统计量 + 仿真环境。

**输出** 阶段 F 的峰值检查点（step 30）。

**验收** 环 1 诚实 **57.8%**（起点 52.0%，**+5.9**）。**耗时约 2 h。**

### 8.1 三个决定成败的参数

**为什么** 官方配方直接套在这个任务上会崩。这三个改动每一个都有对照实验，其中一个是决定性的。

**参数**

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

所以 `rollout_epoch` 从 3 改成 1、`global_batch_size` 从 2048 改成 4096，**两个一起**才得到 1。

**输入** 无（这一节是参数依据，不执行）。

**命令** 用 preflight 把算式打出来核对：

```bash
EMBODIED_PATH=$PWD/examples/embodiment .venv/bin/python -m toolkits.preflight_config \
  --config-path $PWD/examples/embodiment/config/ --config-name so101_ppo_v11 \
  runner.logger.log_path=/data08/henryg/pai/results/so101_ppo_v13 \
  actor.model.num_action_chunks=10 env.train.rollout_epoch=1 actor.global_batch_size=4096
```

**输出**（实测）：

```
batch arithmetic: 64 envs x 64 chunks x rollout_epoch 1 = 4096 samples/epoch -> 1 update(s)/epoch
PREFLIGHT OK
```

**验收** **`-> 1 update(s)/epoch` 这一句就是成败所在**，不是 1 就别启动。

### 8.2 启动前必测：先决条件要在 rollout 分布下测

**为什么** PPO 是放大器不是发现器：它只能放大**已经偶尔发生**的成功。而"偶尔成功"必须在**带探索噪声的 rollout 分布**下测，不是确定性评测下——这两个数在本任务上能差 50 个点以上。

**输入** 阶段 E 的检查点（即 PPO 的起点）。

**命令** 用**冻结探针**（`lr=1e-9`，走真实训练路径但权重几乎不动）跑一轮：

```bash
SCRATCH=/tmp/so101_runs bash tools_so101_session/ppo_freeze_probe.sh
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `actor.optim.lr` | **1e-9** | 走完整训练路径但权重不变——测的是"起点在训练分布下什么水平" |
| 其余 | 与正式启动完全一致 | 探针必须与真实运行同一条代码路径 |
| `SCRATCH` | 任意可写目录 | 脚本把日志写这里 |

**输出** 一轮之后日志里同时有两个数：

| 指标 | 含义 | 门槛 |
|---|---|---|
| `env/success_once` | **带噪**，PPO 真正学习的分布 | **≥5%** |
| `eval/success_once` | 确定性 | 不应比起点低太多 |

**验收** 带噪 ≥5%。这个门槛来自本项目四次历史运行的实测分界：**两次放大成功的起始带噪成功率是 5–15%，两次从未起来的是 0.5–1.0%**。**必要但不充分**——本次探索中有一个变体带噪 39% 仍然崩塌，因为更新次数不对。

> **不要用 `runner.only_eval=True` 当探针**：它同时切换模型规格来源并跳过训练环境创建（`config.py:830`、`env_worker.py:108`），是另一条代码路径，测的不是同一件事。

### 8.3 启动

**为什么** 手工敲这条命令容易漏掉环境变量、Ray 清理和守卫，而 PPO 跑崩是静默的——曲线掉下去不会有任何报错。

**输入** 8.1 的算式为 1、8.2 的带噪成功率 ≥5%。

**命令**

```bash
SCRATCH=/tmp/so101_runs bash tools_so101_session/ppo_train.sh
```

它实际发出的命令是：

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

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `--config-name so101_ppo_v11` | | 官方配方（`adv_type: gae`、`loss_type: actor_critic`、clip 0.2、entropy 0），起点已指向 v10 |
| `SO101_SPAWN_FRAC` | 环 1 | 训练区域，与阶段 E 一致 |
| `runner.val_check_interval` / `save_interval` | 5 / 5 | 每 5 轮评一次、存一次；峰值很窄，存疏了会错过 |
| `actor.optim.lr` | 2e-6 | 官方 5e-6 的一半——起点已经很好，步子迈大会毁掉它 |
| `setsid` | | 脱离终端。**不这么做，SSH 一断训练就没了** |

**输出** 检查点每 5 轮一个，落在 PPO 的结果目录下；tensorboard 事件文件在同一目录树里。

**验收** 启动后 30 分钟内日志要出现第一轮完整的 `success_once=`；没有就是挂了，去查根因，别盲目重启。

> **自动停机守卫**每 5 分钟读一次 `eval/success_once`：低于峰值 20 点 → 停；连续 3 次低于起点 5 点 → 停。**必须写在启动器里**，写在会话里会随会话消失（历史上因此白烧 180 轮）。

### 8.4 读训练曲线

**为什么** 训练中的成绩不在 stdout 里，在 tensorboard 事件文件里；不会读就只能盲等。

**输入** PPO 的训练输出目录。

**命令**

```bash
.venv/bin/python - <<'PY'
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
fs = sorted(glob.glob("/data08/henryg/pai/results/so101_ppo_v13/**/events.out.tfevents*", recursive=True))
ea = EventAccumulator(fs[-1], size_guidance={"scalars": 0}); ea.Reload()
for s in ea.Scalars("eval/success_once"):
    print(s.step, round(s.value, 4))
PY
```

**参数**

| 标量 | 含义 | 什么时候看 |
|---|---|---|
| `eval/success_once` | **确定性**评测，每 `val_check_interval`（5 轮）一次 | **挑检查点看这个** |
| `env/success_once` | **带噪** rollout，PPO 实际学习的分布 | 判断"还在不在有效区间"看这个 |

**输出**（实测曲线）：

| step | 4 | 9 | 14 | 19 | 24 | **29** | 34 | 39 | 44 | 49 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval | 60.9 | 58.6 | 62.5 | 58.6 | 70.3 | **73.4** | 69.5 | 70.3 | 68.0 | 68.8 |

**验收** 峰值出现在 step 29 → 取检查点 `global_step_30`。前 20 轮徘徊（**不要在这里判死刑**），step 24 起上台阶。若到 step 50 仍未超过起点，回 8.1/8.2 查参数。

### 8.5 验收：诚实口径要跑两组评测

**为什么** 峰值检查点是在门评那套固定局面上挑出来的，再用同一套局面报增益等于自己给自己打分。所以换**从未用过的种子**，并且**把起点用同样的种子、同样的块长重测一遍**——不这么做就分不清"PPO 的增益"和"这两个种子恰好更容易"。

**输入** PPO 峰值检查点 + 阶段 E 的起点检查点。

**命令**

```bash
export EMBODIED_PATH=$PWD/examples/embodiment
export RING1="0.4294,0.9115,0.5142,0.9817"
STATS=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json
PPO=/data08/henryg/pai/results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30
BASE=/data08/henryg/pai/results/so101_sft_v10/so101_sft_openpi_pi05/checkpoints/global_step_1000

for CK in $PPO $BASE; do              # 两个检查点，同样的种子、同样的块长
  for SEED in 4141 4242; do
    .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
    find /dev/shm -maxdepth 1 -type f \( -name 'cuda.shm.*' -o -name 'nccl-*' \) -delete
    SO101_SPAWN_FRAC=$RING1 timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
      --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
      runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v13 \
      rollout.model.model_path=$CK \
      rollout.model.openpi.config_name=pi05_so101_v10 \
      rollout.model.openpi_data.norm_stats_path=$STATS \
      rollout.model.num_action_chunks=10 \
      env.eval.total_num_envs=128 env.eval.seed=$SEED \
      2>&1 | grep -oE 'success_once=[0-9.]+' | tail -1
  done
done
```

也可以直接用当时的两个脚本（多了 3 次重试和 shm 清理）：

```bash
SCRATCH=/tmp/so101_runs bash tools_so101_session/verify_honest_seeds.sh      # PPO 峰值，另加框内/全板参照
SCRATCH=/tmp/so101_runs bash tools_so101_session/verify_baseline_control.sh  # 起点，同种子同块长
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `rollout.model.openpi.config_name` | `pi05_so101_v10` | **PPO 阶段没有产生新数据集**，沿用起点那个条目 |
| `rollout.model.num_action_chunks` | **10** | 两个检查点都必须是 10，否则比的不是同一件事（见本节末的警示） |
| `env.eval.seed` | 4141 / 4242 | 从未参与任何挑选 |
| `env.eval.total_num_envs` | 128 | 每集 0.8 个百分点；再少就读不出 5 个点的差别 |
| `find /dev/shm ... -delete` | | 上一次评测残留的共享内存段会让下一次以"看起来像显存不足"的方式失败 |

**输出** 4 个 `success_once=` 数字（2 检查点 × 2 种子）。

**验收（诚实口径）**——种子 4141/4242 从未参与挑选，与起点同条件对照，**动作块长同为 10**：

| | 4141 | 4242 | 均值 |
|---|---|---|---|
| 起点（阶段 E 产物） | 48.4% | 55.5% | 52.0% |
| **PPO 后** | **58.6%** | **57.0%** | **57.8%** |
| 增益 | +10.2 | +1.5 | **+5.9** |

框内参照 77.3%、全板参照 14.8%。

> **报告里请引用 +5.9 这个诚实增益。** 门评口径上是 61.7%→73.4%（+11.7），但峰值检查点正是在那套固定评测局面上挑出来的，会偏乐观。

> ⚠️ **动作块长会改变成绩，比较时必须对齐**：`so101_eval_openpi_pi05.yaml` 的默认值是 `num_action_chunks: 5`，阶段 B–E 的门评都用的这个默认；阶段 F 起改成 10。所以**跨阶段比数字前，先确认两边块长一致**——同一个检查点从 5 换到 10 可以白涨 11 点。

---

## 9. 阶段 G —— 真机协同训练（打通 sim2real）

**为什么** A–F 只回答了"在仿真里会不会做这件事"。**换成真实相机图像，策略还认不认得，是另一个问题**，而且答案一开始是否定的。

**输入** 阶段 F 的峰值检查点 + 87 集真机数据 + 阶段 E 的环 1 仿真数据集。

**输出** 两个协同训练数据集 + **交付检查点**（第二轮的 step 1000）。

**验收** 留出集真机比值 **0.70**（<1），仿真环 1 不塌（约 60%）。**耗时约 7 h。**

> **数据集路径的一个现实问题**：HF 上的 repo id 是 `henry-guo/so101-pick-place-v2`，而本机上这份数据落在 `/data08/henryg/pai/data/so101-pick-place-v1-trimmed`（87 集、30 fps、LeRobot **v3.0** 布局）——**是同一份数据**。阶段 A 按 HF repo id 找它，而 G 阶段的转换脚本里 `REAL_ROOT` 硬写的是本机路径。换机器时改这一行，或者做个软链。

### 9.1 先把问题量出来（不碰机器人）

**为什么** 直接上机去试，代价是可能撞坏机械臂，而且失败了也不知道是感知问题还是控制问题。离线检验 20 分钟、零硬件，就能给出可判定的数字。

**输入** 阶段 F 的检查点 + 真机数据集。

**命令**

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools_so101_session/offline_replay_check.py \
  --ckpt /data08/henryg/pai/results/so101_ppo_v13/so101_ppo_v11/checkpoints/global_step_30 \
  --config-name pi05_so101_v10 \
  --norm-stats $PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
  --chunks 10 --real-root $DATA/so101-pick-place-v1-trimmed \
  --episodes 5 --frames 10
```

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `--chunks` | 10 | 动作块长；**要与被测策略的训练/部署一致**，否则测的是别的东西 |
| `--real-root` | 真机数据集根目录 | v2.0 和 v3.0 两种 LeRobot 布局都支持 |
| `--episodes` / `--frames` | 5 / 10 | 取 5 集、每集采样 10 帧，共 50 个样本点 |
| `--no-wrist`（可选） | | 砍掉腕视那一路，用来判断某一路输入是不是根因 |
| `--ep-start`（可选） | | 从第几集开始读，**留出集检验靠它** |

**做法**：把真机录制的（前视、腕视、关节状态）喂给策略，比较它预测的动作与人类当时的实际动作。两个对照让数字可解释——**仿真数据**（分布内参照）和**"完全不动"**（动作幅度的尺度）。比值 <1 表示优于什么都不做。

**输出**（stdout）：逐关节的 policy MAE、hold-still MAE、比值，以及仿真侧和真机侧各一个总比值。

| 检查点 | 仿真比值 | 真机比值 |
|---|---|---|
| 阶段 F 产物（仿真训练） | 0.10 ✅ | **4.47** ❌ |
| 阶段 A 产物（真机训练） | 3.82 ❌ | **0.22** ✅ |

**验收** 这一步的"验收"是**确认问题存在且指标可信**：两个策略必须**互为镜像**——各自在自己的域里好、在对方的域里差。镜像成立，说明这是**视觉域鸿沟**而非某个部件的 bug，指标本身也是好的。

**不要停在第一个嫌疑上。** 当时的首要嫌疑是仿真腕部相机指向错误（它拍的是机械臂自己）。但加 `--no-wrist` 把这一路砍掉后真机比值几乎不变（4.47 → 4.59）——差距比这一个缺陷更广。**如果当时收手去修相机，30 小时机时会白花。**

### 9.2 建协同训练数据集

**为什么** 把真机图像掺进训练数据，让模型同时见过两个域。真机只有 87 集、仿真有 1292 集，不上采样的话真机在梯度里几乎看不见。

**输入** 阶段 E 的环 1 仿真数据集 + 真机数据集（87 集）。

**命令**

```bash
.venv/bin/python tools_so101_session/convert_cotrain_simreal.py   # -> so101-cotrain-v14
.venv/bin/python tools_so101_session/convert_cotrain_heldout.py   # -> so101-cotrain-v15
```

**参数**（写在脚本头部，改这几行即可）

| 项 | v14 | v15 | 为什么 |
|---|---|---|---|
| `SIM_ROOT` | `so101-sim-demos-v10` | 同左 | 阶段 E 的数据集 |
| `REAL_ROOT` | `so101-pick-place-v1-trimmed` | 同左 | 真机 87 集 |
| 真机用哪些集 | **全部 0–86** | **只用 0–69**，70–86 留出 | v14 的指标因此不可用，见下 |
| `REPEAT` | 2 | **3** | v15 只用 70 集训练，倍数提高才维持相近占比 |
| `MIN_LEN, MAX_LEN` | 80, **1000** | 同左 | **不能沿用仿真的 580**——人类遥操作更慢，87 集是 395–825 帧、中位 575，**用 580 会静默丢掉 46% 的真机数据** |

**输出** 第一轮数据集（1292 + 87×2）、**第二轮数据集**（1292 + 70×3 = 1502 集）。

**验收** 脚本最后打印 `DONE: <总集数> episodes (sim N + real M ...)`；核对总集数与上表相符。**耗时约 5.5 h**（纯 CPU，瓶颈是视频编码）。

### 9.3 两次训练

**为什么** **这一阶段实际是两次训练**（v14 → v15），第二次才是交付物：

| 轮次 | 数据 | 热启动 | 真机比值 |
|---|---|---|---|
| v14 | 真机**全部 87 集** ×2 + 仿真 1292 集 | 阶段 F 的 `global_step_30` | 0.84（**训练集口径，不可用**） |
| **v15** | 真机 **0–69 集** ×3 + 仿真 1292 集，**70–86 留出** | **v14 的 `global_step_1750`** | **0.70（留出集）** |

v14 把 87 集全部拿去训练，然后又用其中几集读离线指标——那是拿训练集当考卷。v15 留出 17 集从不参训，才是第一个能作为上机依据的数字。**从零复现可以直接跑 v15**（用 `convert_cotrain_heldout.py` 的数据集、从阶段 F 热启动），但那条路没有实测过；这里记录的是实际跑出 0.70 的那条。

**输入** 两个协同数据集 + 阶段 F 的检查点 + v4 统计量。

**命令**

```bash
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

**参数**

| 参数 | 值 | 为什么 |
|---|---|---|
| `config_name` | `pi05_so101_v14` / `pi05_so101_v15` | 各自指向对应的协同数据集 |
| 热启动 | v14 ← PPO 峰值；v15 ← v14 的 step_1750 | **两个 yaml 里已写死**，无需覆盖 |
| `lr` | 1e-5 | 轻量微调，别把仿真能力洗掉 |
| 步数 / 存点 | 2000 / 250 | 门评逐个检查点做 |

**输出** 各 8 个检查点；v14 的最优是 `global_step_1750`，v15 的最优是 `global_step_1000`。

**验收** 检查点齐全且带 norm_stats 子目录；成绩由 9.4 判定。**耗时约 1.5 h。**

### 9.4 门评必须是双轴的

**为什么** 这一步**targets 真机表现，同时有能力破坏仿真能力**。单轴门评会拿仿真能力换真机能力而不自知——实测抓到过一次：750 步时真机比值已改善到 1.16，而仿真掉到 52.3%，只看目标轴会以为是进步。

**输入** 9.3 的全部检查点 + 真机数据集的**留出段**（70–86）。

**命令**

```bash
# 目标轴：留出集上的离线真机比值
for CK in /data08/henryg/pai/results/so101_sft_v15/*/checkpoints/global_step_*; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python tools_so101_session/offline_replay_check.py \
    --ckpt $CK --config-name pi05_so101_v15 \
    --norm-stats $PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    --chunks 10 --real-root $DATA/so101-pick-place-v1-trimmed \
    --ep-start 70 --episodes 5 --frames 10        # ep-start 70 = 只读留出集
done

# 约束轴：同一批检查点的仿真环 1 成功率
export EMBODIED_PATH=$PWD/examples/embodiment
export RING1="0.4294,0.9115,0.5142,0.9817"
for CK in /data08/henryg/pai/results/so101_sft_v15/*/checkpoints/global_step_*; do
  .venv/bin/ray stop --force; rm -rf /tmp/ray/session_*
  SO101_SPAWN_FRAC=$RING1 timeout 1800 .venv/bin/python evaluations/eval_embodied_agent.py \
    --config-path $PWD/examples/embodiment/config/ --config-name so101_eval_openpi_pi05 \
    runner.logger.log_path=/data08/henryg/pai/results/so101_eval_v15 \
    rollout.model.model_path=$CK \
    rollout.model.openpi.config_name=pi05_so101_v15 \
    rollout.model.openpi_data.norm_stats_path=$PWD/assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
    rollout.model.num_action_chunks=10 \
    env.eval.total_num_envs=128 env.eval.seed=4141 \
    2>&1 | grep -oE 'success_once=[0-9.]+' | tail -1
done
```

**参数**

| 轴 | 指标 | 关键参数 | 判据 |
|---|---|---|---|
| 目标 | 离线真机比值 | `--ep-start 70`（**只读留出集**）、`--chunks 10` | 越低越好，**必须 <1** |
| 约束 | 仿真环 1 成功率 | `SO101_SPAWN_FRAC=$RING1`、**`num_action_chunks=10`** | **不得跌破 50%** |

**约束必须在选点时有否决权**，不是事后看看：先按约束轴筛掉不合格的检查点，再在剩下的里面挑目标轴最好的。

**输出** 每个检查点两个数；**交付物是第二轮的 step 1000**。

**验收**

| 指标 | 阶段 F 产物 | **阶段 G 产物** |
|---|---|---|
| 离线真机比值（留出集，10 步时域） | 4.47 | **0.70** ✅ |
| 仿真环 1 | 57.8% | **62.5% / 65.6%**（约 60%，未下降） |

**离线门通过，可以上真机。** 部署方案见 `SIM2REAL_PLAN_ZH.md`。

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

> 这一节**不是执行步骤**，没有命令——它是写报告时必须一并说明的边界。

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

**任何改动都必须重新过 §9.1 那个离线检验，且按 §9.4 的双轴门评选点。**

### 13.3 其它已知局限

| 项 | 内容 |
|---|---|
| 覆盖区域 | 策略只在 96 cm² 的环 1 内训练；全板（426 cm²）成绩仅 14.8%。真机摆放必须限制在对应矩形内 |
| 生成区边距 | 2 cm 的边距排除了约 11% 的真实方块起始位置 |
| 扩域未解决 | 把生成区从 48 扩到 96 cm² 没有带来提升（阶段 E 是负结果）；密度律不能外推成"扩域只要补够密度" |
| 物理参数 | 摩擦系数等未经真机标定，属声明的默认值；方块质量按用户给的上界（<10 g）设定 |
| 单一任务 | 只验证了"抓红方块放托盘并回位"这一个任务 |
