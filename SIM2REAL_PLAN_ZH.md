# SO101 sim2real 部署方案

**离线门已通过，可以上真机了。** 判据、证据、以及仍然存在的限制都写在下面。

| 指标 | 协同训练前 | **现在** | 判据 |
|---|---|---|---|
| 离线真机比值（策略动作误差 ÷「原地不动」误差） | 4.47 | **0.70** | <1 ✅ |
| 该数字测在 | — | **从未参训的 17 集真机数据**，且按**部署时的 10 步动作时域** | 泛化，不是记忆 |
| 仿真环 1 成功率 | 57.8% | **62.5% / 65.6%** | 不塌 ✅ |

**产物**：`results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000`

硬件约束：真机 SO101 只接在 Mac 上，推理跑在 8×H200 训练节点。架构是 **Mac 采图/读关节 → H200 推理 → 返回动作块 → Mac 执行**。部署走 **lerobot 官方的 `async_inference` 异步栈**，Mac 侧不写一行控制代码。

---

## 1. 现阶段产物

| 产物 | 路径 | 说明 |
|---|---|---|
| **策略检查点（RLinf 格式）** | `results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000` | 协同训练产物，32 GB |
| **策略检查点（LeRobot 格式）** | `results/so101_v15_lerobot` | 上面那个导出来的，部署用这个 |
| 归一化统计量 | `assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json` | **全血统冻结**，导出时写进 LeRobot 的 processor |
| 数据变换条目 | `pi05_so101_v15`（`dataconfig/__init__.py`） | 决定图像/状态如何进模型 |
| 格式导出工具 | `tools_so101_session/convert_rlinf_to_lerobot.py` | RLinf → LeRobot |
| 导出验证工具 | `tools_so101_session/verify_lerobot_export.py` | 证明导出没改变策略 |
| 离线检验工具 | `tools_so101_session/offline_replay_check.py` | 上机前的门 |
| 备用推理服务端 | `tools_so101_session/deploy_policy_server.py` | 路线 B，见 §4.10 |

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

**0.70 与 0.79 是同一个检查点的两次测量**，差别在动作时域：0.79 测于 5 步，0.70 测于 10 步。10 步才是训练和部署实际用的（见 §4.9 的插值坑），所以**以 0.70 为准**。时域拉长反而更好，说明策略预测的是一条连贯轨迹，而不是只有头一两步像样。

**仍未解决**：仿真腕部相机指向仍然是错的。协同训练让模型学会了处理真实腕视图像，但**仿真数据里那一路仍然是废的**。这意味着仿真成绩里腕视贡献有限，而真机上它现在很重要——两边的信息量不对称。彻底修它要重录数据并从阶段 B 重训（约 30 小时）；在真机数据证明有必要之前不做。

---

## 3. 与理论上界的距离

真机数据训出的策略在同一指标上是 **0.22**，我们是 **0.70**——还差 3.2 倍。

**但 0.22 不是干净的上界**：那个策略训练时用了全部 87 集，它的 0.22 也是训练集口径。真正的可比上界未知。

还有下降空间的证据：同一份数据多训 2000 步，比值继续降到 0.68（训练集口径，且仿真掉到 53%）。所以**没有收敛**，代价是要盯住仿真那一轴。是否继续压，取决于真机上的实际表现——**真机数据才是最终判据**，再优化离线指标不如去拿第一手数据。

---

## 4. 部署：走 lerobot 官方异步栈

```
┌─ Mac（真机侧）────────────────────┐         ┌─ H200 节点（推理侧）──────────────┐
│ python -m lerobot.async_inference │  观测 →  │ python -m lerobot.async_inference │
│        .robot_client              │          │        .policy_server             │
│  · SO101 follower                 │          │  · PI05Policy.from_pretrained     │
│  · 2 × USB 相机 640×480@30Hz      │ ← 动作块 │  · results/so101_v15_lerobot      │
│  · 动作队列 + 重叠块聚合          │  gRPC    │                                   │
└───────────────────────────────────┘         └───────────────────────────────────┘
                              SSH 隧道 18080
```

Mac 侧**不需要自己写控制回路**，用 lerobot 自带的 `robot_client` 就行。

### 4.1 为什么必须先做一次格式导出

`policy_server` 加载策略只有一条路径（`policy_server.py:152`）：

```python
self.policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
```

**只吃 LeRobot 布局**（`config.json` + `model.safetensors` + processor 文件）。我们的 RL 产物是 RLinf 布局（`actor/model_state_dict/full_weights.pt` + openpi 的 `norm_stats.json`），`from_pretrained` 读不了。

反过来"在 RLinf 环境里起 lerobot 服务"也不行，因为两个 Python 环境是分开的：

| 环境 | lerobot | openpi | 能干什么 |
|---|---|---|---|
| RLinf `.venv`（py3.11） | 0.1.0，**没有 `async_inference`** | ✅ | 能加载 RLinf ckpt，起不了 lerobot 服务 |
| lerobot conda（py3.12） | ✅ 新版 | ❌ | 能起服务，读不了 RLinf ckpt |

所以**导出是唯一干净的路**。好在它几乎是零成本的。

### 4.2 导出（几乎零成本，实测）

**RLinf 的 openpi 后端就是 LeRobot pi05 的模块树**——实测比对：

```
template 813 keys, converted 813, missing 0, unexpected 0, shape mismatch 0
```

813 个键**逐个同名、逐个同形**，没有任何张量需要拆分、合并或转置。lerobot 的 PI05 源码自己也写着 "a direct port of the OpenPI implementation"，连图像预处理都是 `resize_with_pad_torch`（注释标明是 openpi 的 "exact copy"，`modeling_pi05.py:1191`）加 `img * 2.0 - 1.0`（`:1194`），两边一致。

```bash
cd /data08/henryg/pai/RLinf
.venv/bin/python tools_so101_session/convert_rlinf_to_lerobot.py \
  --ckpt /data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000 \
  --template /data08/henryg/pai/outputs/train/2026-07-14/13-53-42_pi05/checkpoints/last/pretrained_model \
  --norm-stats assets/pi05_so101_v4/so101-sim-demos-v4/norm_stats.json \
  --chunk-size 10 \
  --out /data08/henryg/pai/results/so101_v15_lerobot
```

`--template` 只提供 `config.json` 和两个 processor 文件的**结构**。要用**同一台机器人**的 LeRobot pi05 微调检查点，这样特征名和维度本来就对得上：`observation.images.front` / `observation.images.wrist` / `observation.state` (6) / `action` (6)——正好是 `robot_client` 用 `--robot.cameras="{front: ..., wrist: ...}"` 送过来的那一组。

**权重不是风险点，这三处才是**：

| 项 | 从 | 到 | 不改的后果 |
|---|---|---|---|
| `chunk_size` / `n_action_steps` | 50 | **10** | 只在 10 步时域微调过，问它要 50 步，后 40 步没有训练目标 |
| processor 里的归一化统计量 | 模板自带（来自**它自己的**数据集） | **v4 血统的 q01/q99** | 等于换了坐标系，而且**不报错** |
| `--actions_per_chunk`（客户端） | 50 | **10** | 同上 |

统计量能直接搬，是因为两边约定一致：模板的 `normalization_mapping` 是 `STATE/ACTION: QUANTILES`，而 openpi 的 `norm_stats.json` 正好存 `q01/q99`。

导出脚本会在写盘前逐键比对，对不上就拒绝写。实测输出：

```
template key prefix: 'model.'
keys: template 813, converted 813, missing 0, unexpected 0, shape mismatch 0
config: chunk_size/n_action_steps (50, 50) -> 10
normalization_mapping = {'VISUAL': 'IDENTITY', 'STATE': 'QUANTILES', 'ACTION': 'QUANTILES'}
wrote policy_preprocessor_step_3_normalizer_processor.safetensors: replaced 8 stats
```

用 lerobot 加载确认：

```
✓ Loaded state dict from model.safetensors
All keys loaded successfully!
loaded OK: PI05Policy | chunk_size 10 | n_action_steps 10
```

### 4.3 等价性验证——**上机前必须做**

键对上**不代表**行为对上。统计量写错、量化归一化的约定不同，都不会报错，只会让策略安静地变成另一个策略。唯一诚实的检查是：把授权上机的那个指标，用导出后的模型在同一批留出集上重新测一遍。

```bash
/root/miniconda3/envs/lerobot/bin/python tools_so101_session/verify_lerobot_export.py \
  --lerobot-ckpt /data08/henryg/pai/results/so101_v15_lerobot \
  --real-root /data08/henryg/pai/data/so101-pick-place-v1-trimmed \
  --ep-start 70 --episodes 5 --frames 10
```

它复用 `offline_replay_check.py` 的数据读取，只把策略换成 `PI05Policy` + 它的 processor 管线，所以**唯一的变量就是导出本身**。

| 结果 | 含义 |
|---|---|
| 比值 ≈ **0.70** | 导出没有改变策略，可以上机 |
| 明显更差 | 导出改变了策略（最可能是统计量），**不能上机** |

### 4.4 起服务（H200 侧）

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/root/miniconda3/envs/lerobot/bin/python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=18080 \
    --fps=30
```

服务端启动时**不加载任何模型**——它等客户端把策略规格（类型、路径、设备、每块动作数）通过 `SendPolicyInstructions` 发过来再加载。所以 `--pretrained_name_or_path` 写在客户端命令里，但**那个路径必须在 H200 上存在**。

Mac 侧建隧道：

```bash
H200_HOST="user@10.0.0.5"        # 改成你的用户名@主机
ssh -N -L 18080:localhost:18080 "$H200_HOST"
```

### 4.5 起客户端（Mac 侧）

```bash
python -m lerobot.async_inference.robot_client \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=${FOLLOWER_ID} \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --task="Grab the red cube" \
    --server_address=127.0.0.1:18080 \
    --policy_type=pi05 \
    --pretrained_name_or_path=/data08/henryg/pai/results/so101_v15_lerobot \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk=10 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average
```

三个参数与我们这个策略绑定，其余按你原来的用法：

| 参数 | 值 | 说明 |
|---|---|---|
| `--pretrained_name_or_path` | `results/so101_v15_lerobot` | **H200 上的路径**，不是 Mac 上的 |
| `--actions_per_chunk` | **10** | 训练时域，不要用默认的 50 |
| `--task` | `Grab the red cube` | 训练用的任务串，逐字一致 |

**动作单位不需要换算**：策略输出的就是 LeRobot 归一化单位，与 follower 接受的一致。

### 4.6 为什么这套异步栈比同步回路好

`robot_client` 不是"发一帧、等一批、执行完再发下一帧"：

- **提前请求**：动作队列跌到 `chunk_size_threshold`（0.5，即半块）就发下一帧观测（`robot_client.py:406`），推理延迟被队列吸收，不占用执行时间。
- **重叠块聚合**：新旧动作块在时间上重叠的部分用 `weighted_average` 融合（`:224`），块边界不会突跳。
- **must-go 兜底**：队列真空了会标记该观测必须处理，不会静默停摆。

同步回路（路线 B）没有这些：一个 10 步动作块只覆盖 10/30 ≈ 333 ms，往返一旦超过这个数就会出现动作断档，且每个块边界都可能抖一下。

### 4.7 必须送两路相机

这一条**在协同训练后反转了**，不要沿用旧结论：

| | 双相机 | 只送前视 |
|---|---|---|
| 协同训练前 | 4.47 | 4.59（没区别） |
| **协同训练后** | **0.90** | **1.58** |

协同训练之前腕视那一路是废的（仿真里它拍的是机械臂自己），砍掉无所谓；**现在模型见过真实腕视图像并开始使用它**，只送前视会让动作误差差不多翻倍，甚至退回"不如原地不动"。

所以 `--robot.cameras` 里 `front` 和 `wrist` **两个都要有**，键名也要就是这两个词（要与 `config.json` 的 `observation.images.front` / `.wrist` 对上）。

### 4.8 红方块的摆放范围

策略只在**环 1**（96 cm²）训练过。全板成绩仅 14.8%，所以真机上红方块**必须放在这个矩形内**，建议用胶带标出来：

| 方向 | 位置 |
|---|---|
| x（朝托盘方向） | 距**托盘那一侧的板边** 3.6 cm 到 12.0 cm（宽 8.5 cm） |
| y | 距**黑色窄条对面那条板边** 3.4 cm 到 14.7 cm（长 11.3 cm） |

蓝色干扰块可放在棕区任意位置。

### 4.9 参数对齐表

任何一项不一致都会掉成绩。

| 项 | 必须是 | 依据 |
|---|---|---|
| 相机分辨率 | 640×480，两路 | 与真机数据集一致 |
| 控制频率 | 30 Hz | 真机数据集 fps |
| 每次返回/执行的动作数 | **10** | 训练时域 |
| 状态/动作单位 | LeRobot 归一化（臂 ±100，爪 0–100） | 数据集原生单位 |
| 归一化统计量 | v4 血统的 q01/q99（已写进导出的 processor） | 血统冻结 |
| 提示词 | `Grab the red cube` | 训练用的任务串 |
| 方块 | 2.9 cm、<10 g | 仿真物理按此设定 |
| 初始姿态 | 手臂折叠、夹爪近闭合 | 87 帧真机首帧的中位姿态 |

> **关于动作块长度的一个已查明的坑**：有两个容易混的同名概念——`num_action_chunks` 是 **worker 级**的（多久调用一次推理），`openpi.action_chunk` 是**模型级**的（每次返回几个动作）。
> `model/pi0_5.yaml:26` 写的是 `action_chunk: ${..num_action_chunks}`，**是插值**，所以走 YAML 的训练和仿真评测在 `num_action_chunks=10` 时每次确实返回并执行 **10** 个动作。
> 但 `offline_replay_check.py` 和 `deploy_policy_server.py` 的 config 是用 `OmegaConf.create()` 手搓的字典，**不继承这条插值**，会静默退回 dataclass 默认值 5。两个工具现在都显式写死了 `action_chunk`。**凡是手搓 config，都要和真 YAML 逐条核对插值项**——它们恰恰是不会报错的那一类。

### 4.10 路线 B（备用）：RLinf 直接起 websocket 服务

只在 §4.3 的等价性验证不通过时用。它绕开格式转换（少一个可能出错的环节），代价是 Mac 侧要自己写控制回路、没有块聚合、对网络延迟敏感。

H200 侧：

```bash
cd /data08/henryg/pai/RLinf
export REPO_PATH=$PWD PYTHONPATH=$PWD HF_LEROBOT_HOME=/data08/henryg/pai/data
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0

.venv/bin/python tools_so101_session/deploy_policy_server.py \
  --ckpt /data08/henryg/pai/results/so101_sft_v15/so101_sft_openpi_pi05/checkpoints/global_step_1000 \
  --config-name pi05_so101_v15 \
  --port 8000
```

监听前会用全零观测自检一次，加载有问题就在这里失败，而不是等机械臂通电才发现。实测输出 `self-test OK: returns (10, 6) float32`。

Mac 侧控制回路：

```python
import time
from openpi_client.websocket_client_policy import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="localhost", port=8000)

while not done:
    actions = policy.infer({
        "observation/image":       cam_front.read(),   # (480, 640, 3) uint8
        "observation/wrist_image": cam_wrist.read(),   # 必须送，见 §4.7
        "observation/state":       robot.get_observation(),  # (6,)
        "prompt":                  "Grab the red cube",
    })["actions"]                                      # (10, 6)

    for a in actions:            # 收到几个执行几个，不要截断也不要补齐
        robot.send_action(a)
        time.sleep(1 / 30)
```

---

## 5. 上机流程

每一步都有明确的通过条件；**不过就停下来，不要往下走**。

| 步骤 | 做什么 | 通过条件 |
|---|---|---|
| 1. 空跑 | 机械臂**断电/脱机**，只跑 client↔server | 30 Hz 稳定、每次拿到 (10,6)、服务端日志里的 one-way latency 稳定 |
| 2. 限幅 | 通电，把每步动作变化量夹到 ≤2 归一化单位，人手放急停上 | 机械臂动作平缓、方向合理 |
| 3. 固定位置 | 红方块放环 1 正中，跑 5 次 | **这是最容易的局面**；这里都不成功就不必往下试 |
| 4. 环 1 随机 | 在标出的矩形内随机放 20 次 | 记录成功率——**与仿真的约 60% 之差就是真实的 sim2real gap**，这才是要报告的数字 |
| 5. 失败归类 | 抓不住 / 抓住掉 / 放偏 / 不回位，四类分别计数 | 它们指向不同的修复方向 |

**建议同时录像**（前视 + 腕视 + 关节轨迹）。失败时"看起来不对"没法改进，逐帧对比才能定位是感知问题还是动作问题——这与仿真里"渲染训练数据看一眼"发现腕部相机缺陷是同一个道理。

---

## 6. 当前状态

| 项 | 状态 |
|---|---|
| 策略训练（仿真） | ✅ 完成，环 1 约 60%（±4 个点） |
| **离线 sim2real 门** | ✅ **通过**（留出集 0.70 < 1，10 步时域） |
| LeRobot 格式导出 | ✅ 813/813 键写出，`PI05Policy.from_pretrained` 加载通过 |
| **导出等价性验证（§4.3）** | ⏳ **未跑**，需 1 张 GPU 约 15 分钟——**这是上机前的最后一道门** |
| 备用 websocket 服务端 | ✅ 已实测跑通（自检返回 `(10, 6)`） |
| 上机 | 🔜 §4.3 通过后按 §5 执行 |

**已知限制**（上机前应当知道）：

| 项 | 内容 |
|---|---|
| 覆盖区域 | 只在 96 cm² 的环 1 训练；全板成绩仅 14.8%，真机摆放必须限制在 §4.8 的矩形内 |
| 生成区边距 | 2 cm 边距排除了约 11% 的真实方块起始位置 |
| 仿真腕部相机 | 指向仍是错的，两个域的信息量不对称（§2 末） |
| 离线指标的预测力 | 它证明了"策略在真实观测下动作合理"，**不等于**"能完成任务"——完成任务还要求闭环稳定性，这只有真机能测 |
| 物理参数 | 摩擦等未经真机标定，属声明的默认值 |
