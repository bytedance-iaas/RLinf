# SO101 sim2real 部署方案

**离线门已通过，可以上真机了。** 判据、证据、以及仍然存在的限制都写在下面。

| 指标 | 协同训练前 | **现在** | 判据 |
|---|---|---|---|
| 离线真机比值（策略动作误差 ÷「原地不动」误差） | 4.47 | **0.70** | <1 ✅ |
| 该数字测在 | — | **从未参训的 17 集真机数据**，且按**部署时的 10 步动作时域** | 泛化，不是记忆 |
| 仿真环 1 成功率 | 57.8% | **62.5% / 65.6%** | 不塌 ✅ |

**产物**：`results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000`

硬件约束：真机 SO101 只接在 Mac 上，推理跑在 8×H200 训练节点。架构是 **Mac 采图/读关节 → H200 推理 → 返回动作块 → Mac 执行**。

---

## 1. 现阶段产物

| 产物 | 路径 | 说明 |
|---|---|---|
| **策略检查点** | `results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000` | 协同训练产物，32 GB |
| 归一化统计量 | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` | **全血统冻结**，部署必须用这一份 |
| 数据变换条目 | `pi05_so101_v15`（`dataconfig/__init__.py`） | 决定图像/状态如何进模型 |
| 推理服务端 | `tools_so101_session/deploy_policy_server.py` | 已实测跑通，自检返回 `(10, 6)` |
| 离线检验工具 | `tools_so101_session/offline_replay_check.py` | 上机前的门 |

**策略的输入输出契约**：

| 项 | 值 |
|---|---|
| 输入图像 | 前视 + 腕视，各 **640×480 RGB**，**两路都必须送** |
| 输入状态 | 6 维关节位置，**LeRobot 归一化单位**（臂 ±100、爪 0–100） |
| 提示词 | `"Grab the red cube"` |
| 输出 | `(10, 6)` 动作块，同样是归一化单位，**直接发给 LeRobot follower** |
| 控制频率 | **30 Hz** |

**仿真内成绩**（供参考，不是真机预期）：

| 区域 | 成功率 |
|---|---|
| 环 1（8.48×11.31 cm，训练区） | **62.5% / 65.6%**（种子 4141 / 4242，各 128 集） |
| 全板（21.6×30.1 cm 棕区） | 未重测（协同训练前是 14.8%） |

> **这些数字带 ±4 个百分点的抖动**：同一检查点、同一种子 4141 重跑一次读到 58.6%（对 62.5%）。GPU 物理仿真不是逐位可复现的，128 集样本下 5 集的差别就是 4 个点。所以"62.5 还是 65.6"不必细究，**真实水平在 60% 上下**；真机对比时也要按这个精度看。

---

## 2. 这个问题曾经是阻断项，以及它是怎么解决的

**离线检验**（`tools_so101_session/offline_replay_check.py`）：把真机录制的观测喂给策略，比较它预测的动作与人类当时的实际动作。两个对照让数字可解释——仿真数据（分布内参照）和"完全不动"（动作幅度的尺度）。比值 <1 表示优于什么都不做。

这个指标的可信度是用**互为镜像的两个策略**验证的：

| 检查点 | 仿真比值 | 真机比值 |
|---|---|---|
| 仿真训练的（PPO 峰值） | 0.10 ✅ | **4.47** ❌ |
| 真机训练的（阶段 A） | 3.82 ❌ | **0.22** ✅ |

各自在自己的域里好、在对方的域里差——所以这是**视觉域鸿沟**，不是某个部件的 bug，指标本身也不是坏的。

**曾经的首要嫌疑是腕部相机**：仿真里它拍的是机械臂自己（`so101_pick_place.py:86-88`，渲染训练数据的三帧证实），而真机腕视看得到工作区。但**把这一路砍掉后真机比值几乎不变**（4.47 → 4.59），证明差距比这一个缺陷更广。**如果当时收手去修相机，30 小时机时会白花。**

**解决办法是协同训练**：把 87 集真机数据掺进仿真数据集重训。两轮：

| 轮次 | 做法 | 真机比值 | 仿真环 1 |
|---|---|---|---|
| v14 | 真机全部 87 集 ×2 掺入 | 0.84 | 63.3% |
| **v15** | 真机 **0–69 集** ×3 掺入，**70–86 留出** | **0.70（留出集）** | **62.5% / 65.6%** |

**v14 的 0.84 不可用**——87 集全进了训练集，而检验读的是同一批数据，等于拿训练集当考卷。v15 才是第一个诚实的数字。

**0.70 与 0.79 是同一个检查点的两次测量**，差别在动作时域：0.79 测于 5 步，0.70 测于 10 步。10 步才是训练和部署实际用的（见 §4.4 的插值坑），所以**以 0.70 为准**。时域拉长反而更好，说明策略预测的是一条连贯轨迹，而不是只有头一两步像样。

**仍未解决**：仿真腕部相机指向仍然是错的。协同训练让模型学会了处理真实腕视图像，但**仿真数据里那一路仍然是废的**。这意味着仿真成绩里腕视贡献有限，而真机上它现在很重要——两边的信息量不对称。彻底修它要重录数据并从阶段 B 重训（约 30 小时）；在真机数据证明有必要之前不做。

---

## 3. 与理论上界的距离

真机数据训出的策略在同一指标上是 **0.22**，我们是 **0.70**——还差 3.2 倍。

**但 0.22 不是干净的上界**：那个策略训练时用了全部 87 集，它的 0.22 也是训练集口径。真正的可比上界未知。

还有下降空间的证据：同一份数据多训 2000 步，比值继续降到 0.68（训练集口径，且仿真掉到 53%）。所以**没有收敛**，代价是要盯住仿真那一轴。是否继续压，取决于真机上的实际表现——**真机数据才是最终判据**，再优化离线指标不如去拿第一手数据。

---

## 4. 部署

有两条路。**推荐第一条**——它让 Mac 侧直接用 lerobot 官方的异步客户端，不用自己写控制回路。

### 4.0 路线 A（推荐）：导出成 LeRobot 格式，走 lerobot 官方异步栈

lerobot 的 `async_inference` 是一对组件：`policy_server`（推理端）+ `robot_client`（机械臂端）。服务端加载策略只有一条路径（`policy_server.py:152`）：

```python
self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
```

**只吃 LeRobot 格式**。我们的产物是 RLinf 格式，所以要先导出。

**导出几乎是零成本的**——实测比对：RLinf 的 openpi 后端用的就是 LeRobot pi05 的模块命名，813 个键**逐个同名、逐个同形**，没有任何张量需要拆分/合并/转置。lerobot 的 PI05 实现自己也写着 "a direct port of the OpenPI implementation"，连图像预处理都是 `resize_with_pad_torch`（注释标明 "exact copy" of openpi）+ `img * 2.0 - 1.0`，两边一致。

```bash
.venv/bin/python tools_so101_session/convert_rlinf_to_lerobot.py   --ckpt /data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000   --template /data08/henryg/pai/outputs/train/2026-07-14/13-53-42_pi05/checkpoints/last/pretrained_model   --norm-stats assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json   --chunk-size 10 --out /data08/henryg/pai/results/so101_v15_lerobot
```

`--template` 提供 `config.json` 和两个 processor 文件的结构；要用**同一台机器人**的 LeRobot pi05 微调检查点，这样特征名和维度（`observation.images.front` / `.wrist` / `observation.state` (6) / `action` (6)）本来就对得上。

导出时**必须改**的三处：

| 项 | 从 | 到 | 为什么 |
|---|---|---|---|
| `chunk_size` / `n_action_steps` | 50 | **10** | 只在 10 步时域微调过，要 50 步后 40 步没有训练目标 |
| processor 里的归一化统计量 | 模板自带（来自真机数据集） | **v4 血统的 q01/q99** | 换统计量等于换坐标系，而且**不报错** |
| `--actions_per_chunk` | 50 | **10** | 同上 |

归一化能直接搬是因为两边约定一致：模板的 `normalization_mapping` 是 `STATE/ACTION: QUANTILES`，而 openpi 的 `norm_stats.json` 正好存 `q01/q99`。

**导出后必须做等价性验证**，键对上不代表行为对上：

```bash
/root/miniconda3/envs/lerobot/bin/python tools_so101_session/verify_lerobot_export.py   --lerobot-ckpt /data08/henryg/pai/results/so101_v15_lerobot   --real-root /data08/henryg/pai/data/so101-pick-place-v1-trimmed   --ep-start 70 --episodes 5 --frames 10
```

它在同一批留出集上重跑离线检验，比值应当落回 **0.70**。对不上就是导出改变了策略（最可能是统计量），**不能上机**。

**两个 Python 环境是分开的，这决定了必须走导出**：

| 环境 | lerobot | openpi | 能干什么 |
|---|---|---|---|
| RLinf `.venv` (py3.11) | 0.1.0，**无 `async_inference`** | ✅ | 能加载 RLinf ckpt，起不了 lerobot 服务 |
| lerobot conda (py3.12) | ✅ 新版 | ❌ | 能起服务，读不了 RLinf ckpt |

验证通过后，服务端和客户端都用 lerobot 原样的命令：

```bash
# H200 侧
/root/miniconda3/envs/lerobot/bin/python -m lerobot.async_inference.policy_server     --host=0.0.0.0 --port=18080 --fps=30

# Mac 侧（隧道到 18080 之后）
python -m lerobot.async_inference.robot_client     --robot.type=so101_follower --robot.port=${FOLLOWER_PORT} --robot.id=${FOLLOWER_ID}     --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                      wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --task="Grab the red cube" \
    --server_address=127.0.0.1:18080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=/data08/henryg/pai/results/so101_v15_lerobot \
    --policy_device=cuda --client_device=cpu \
    --actions_per_chunk=10 --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

**为什么这条路更好**：`robot_client` 是异步的——动作队列跌到 `chunk_size_threshold` 就提前发下一帧观测（`robot_client.py:406`），新旧动作块重叠部分用 `weighted_average` 聚合（`:224`）。下面路线 B 那个"执行完再要下一批"的同步回路有硬往返预算，块边界还会抖。

### 4.1 路线 B（备用）：RLinf 直接起 websocket 服务

只在路线 A 的等价性验证不通过时用。它绕开格式转换（少一个可能出错的环节），代价是 Mac 侧要自己写控制回路、没有块聚合。

```
┌─ Mac（真机侧）──────────────┐          ┌─ H200 节点（推理侧）──────────┐
│ LeRobot SO101 follower      │  图像+    │ deploy_policy_server.py       │
│ 2 × USB 相机（640×480@30Hz）│  状态 →   │  + v15 检查点 + v4 norm_stats │
│ 控制回路                     │  ← 动作块 │  websocket                    │
└─────────────────────────────┘          └───────────────────────────────┘
                        SSH 隧道
```

**不需要 `sft2deploy` 转换**。它要两个我们没有的旧格式参考模型目录，而且没必要：离线检验已经证明"直接用 `get_model` 加载检查点"这条路能在真实图像上产出正确动作，服务端复用同一条加载路径，中间少一个可能出错的环节。

#### B.1 H200 侧：起推理服务

```bash
cd /data08/henryg/pai/RLinf
export REPO_PATH=$PWD PYTHONPATH=$PWD HF_LEROBOT_HOME=/data08/henryg/pai/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VK_ICD_FILENAMES=$PWD/.venv/nvidia_gl/nvidia_icd.json
export LD_LIBRARY_PATH=$PWD/.venv/nvidia_gl
export CUDA_VISIBLE_DEVICES=0

.venv/bin/python tools_so101_session/deploy_policy_server.py \
  --ckpt /data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000 \
  --config-name pi05_so101_v15 \
  --port 8000
```

启动后会先做一次**自检**（用全零观测跑一次推理并打印返回形状），自检不过就不会开始监听——加载出问题会在这里失败，而不是等机械臂已经通电才发现。实测输出：

```
loading .../so101_sft_v15/.../global_step_1000
self-test OK: returns (10, 6) float32
serving on 0.0.0.0:8000
```

Mac 侧建隧道：

```bash
ssh -N -L 8000:localhost:8000 <user>@<h200-host>
```

#### B.2 Mac 侧：控制回路

```python
import time
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy
# LeRobot 的 SO101 follower 与两个相机，按你采集数据时的同一套配置初始化

policy = WebsocketClientPolicy(host="localhost", port=8000)
PROMPT = "Grab the red cube"
HZ = 30

while not done:
    front = cam_front.read()          # (480, 640, 3) uint8
    wrist = cam_wrist.read()          # (480, 640, 3) uint8  ← 必须送，见 §4.2
    state = robot.get_observation()   # (6,) LeRobot 归一化单位

    actions = policy.infer({
        "observation/image":       front,
        "observation/wrist_image": wrist,
        "observation/state":       state,
        "prompt":                  PROMPT,
    })["actions"]                     # (10, 6)

    for a in actions:                 # 收到几个就执行几个，不要截断也不要补齐
        robot.send_action(a)
        time.sleep(1 / HZ)
```

**动作单位不需要换算**：策略输出的就是 LeRobot 归一化单位（`shoulder_pan.pos` 等），与 follower 接受的一致。

### 4.2 必须送两路相机（两条路线都适用）

这一条**在协同训练后反转了**，不要沿用旧结论：

| | 双相机 | 只送前视 |
|---|---|---|
| 协同训练前 | 4.47 | 4.59（没区别） |
| **协同训练后** | **0.90** | **1.58** |

协同训练之前腕视那一路是废的（仿真里它拍的是机械臂自己），砍掉无所谓；**现在模型见过真实腕视图像并开始使用它**，只送前视会让动作误差差不多翻倍，甚至退回"不如原地不动"。

### 4.3 红方块的摆放范围

策略只在**环 1**（96 cm²）训练过。全板成绩仅 14.8%，所以真机上红方块**必须放在这个矩形内**，建议用胶带标出来：

| 方向 | 位置 |
|---|---|
| x（朝托盘方向） | 距**托盘那一侧的板边** 3.6 cm 到 12.0 cm（宽 8.5 cm） |
| y | 距**黑色窄条对面那条板边** 3.4 cm 到 14.7 cm（长 11.3 cm） |

蓝色干扰块可放在棕区任意位置。

### 4.4 参数对齐表

任何一项不一致都会掉成绩。

| 项 | 必须是 | 依据 |
|---|---|---|
| 相机分辨率 | 640×480，两路 | 与真机数据集一致 |
| 控制频率 | 30 Hz | 真机数据集 fps |
| 每次推理执行的步数 | **10**（收到几个执行几个，服务端返回 10） | 见下方说明 |
| 状态/动作单位 | LeRobot 归一化（臂 ±100，爪 0–100） | 数据集原生单位 |
| norm_stats | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` | 血统冻结 |
| 提示词 | `Grab the red cube` | 训练用的任务串 |
| 方块 | 2.9 cm、<10 g | 仿真物理按此设定 |
| 初始姿态 | 手臂折叠、夹爪近闭合 | 87 帧真机首帧的中位姿态 |

> **关于动作块长度的一个已查明的坑**：有两个容易混的同名概念——`num_action_chunks` 是 **worker 级**的（多久调用一次推理），`openpi.action_chunk` 是**模型级**的（每次返回几个动作）。
> 所有配置文件里 `model/pi0_5.yaml:26` 写的是 `action_chunk: ${..num_action_chunks}`，**是插值**，所以走 YAML 的训练和仿真评测在 `num_action_chunks=10` 时每次确实返回并执行 **10** 个动作。
> 但 `offline_replay_check.py` 和 `deploy_policy_server.py` 的 config 是用 `OmegaConf.create()` 手搓的字典，**不会继承这条插值**，会静默退回 dataclass 默认值 5。两个工具现在都显式写死了 `action_chunk`，与训练对齐。

---

## 5. 上机流程

每一步都有明确的通过条件；**不过就停下来，不要往下走**。

| 步骤 | 做什么 | 通过条件 |
|---|---|---|
| 1. 空跑 | 机械臂**断电/脱机**，只跑回路 | 30 Hz 稳定、每次拿到 **(10,6)**、往返延迟 <330 ms（一个动作块覆盖 10/30 s；超过就会出现动作断档） |
| 2. 限幅 | 通电，把每步动作变化量夹到 ≤2 归一化单位，人手放急停上 | 机械臂动作平缓、方向合理 |
| 3. 固定位置 | 红方块放环 1 正中，跑 5 次 | **这是最容易的局面**；这里都不成功就不必往下试 |
| 4. 环 1 随机 | 在标出的矩形内随机放 20 次 | 记录成功率——**与仿真的 65.6% 之差就是真实的 sim2real gap**，这才是要报告的数字 |
| 5. 失败归类 | 抓不住 / 抓住掉 / 放偏 / 不回位，四类分别计数 | 它们指向不同的修复方向 |

**建议同时录像**（前视 + 腕视 + 关节轨迹）。失败时"看起来不对"没法改进，逐帧对比才能定位是感知问题还是动作问题——这与仿真里"渲染训练数据看一眼"发现腕部相机缺陷是同一个道理。

---

## 6. 当前状态

| 项 | 状态 |
|---|---|
| 策略训练（仿真） | ✅ 完成，环 1 诚实 65.6% |
| **离线 sim2real 门** | ✅ **通过**（留出集 0.70 < 1，10 步时域） |
| 部署链路 B（websocket） | ✅ 服务端已实测跑通（自检返回 `(10, 6)`） |
| 部署链路 A（LeRobot 导出） | ⏳ 已导出且 `PI05Policy.from_pretrained` 加载通过（813/813 键），**等价性验证待跑** |
| 上机 | 🔜 按 §5 执行 |

**已知限制**（上机前应当知道）：

| 项 | 内容 |
|---|---|
| 覆盖区域 | 只在 96 cm² 的环 1 训练；全板成绩仅 14.8%，真机摆放必须限制在 §4.3 的矩形内 |
| 生成区边距 | 2 cm 边距排除了约 11% 的真实方块起始位置 |
| 仿真腕部相机 | 指向仍是错的，两个域的信息量不对称（§2 末） |
| 离线指标的预测力 | 它证明了"策略在真实观测下动作合理"，**不等于**"能完成任务"——完成任务还要求闭环稳定性，这只有真机能测 |
| 物理参数 | 摩擦等未经真机标定，属声明的默认值 |
