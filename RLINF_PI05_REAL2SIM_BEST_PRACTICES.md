# RLinf + PI0.5 Real2Sim 最佳实践

> 适用场景:已有**真机演示数据**训练的 PI0.5(LeRobot/openpi),要把它带进**仿真环境**做 RL 强化(real2sim)。
> 来源:SO101 项目两天实战(10 个 RL 版本、4 次证伪、7 个绑定约束)蒸馏。事故细节见 `SO101_WORK_REPORT.md`,操作手册见 `.claude/skills/rlinf-embodied-training/`。
> 状态标注:✅=已实证 ⚠️=已实证的陷阱 🔬=当前最佳假说(v10 验证中)

---

## 0. 一页总纲

```
真机数据 SFT(已有)
   │
   ▼
[1] 建仿真环境 ──── 场景对照真机照片迭代;相机双路对齐
   │
   ▼
[2] 标定 ────────── 单位制、每执行器行程实测、全关节审计
   │
   ▼
[3] 可解性探针 ──── 运动规划器脚本抓取成功 = 环境合格(30 分钟,必做)
   │
   ▼
[4] sim 演示 ────── 规划器批量生成成功轨迹 → LeRobot 格式
   │
   ▼
[5] sim-SFT ─────── 从真机 SFT ckpt 继续,用 sim 演示微调
   │                 验收:zero-shot eval 成功率(我们:0% → 50%)
   ▼
[6] 保守 RL ─────── conservative-PPO 束 + ignore_terminations
   │                 验收:success 存活并上行(而非 <10 epochs 塌缩)
   ▼
[7] (round 2)sim2real:真机数据回炉 + 部署桥
```

**三条铁律:**
1. **每一步有可验证的验收判据**,不达标不进下一步(跳过 [3] 曾让我们白训 12 小时)。
2. **PPO 是已有成功的放大器,不是发现器** —— RL 起点必须已有可观成功率([5] 的意义)。
3. **BC 起点是易碎品** —— RL 阶段必须用保守配方([6] 的意义)。

---

## 1. 建仿真环境

- **不用新增 env 类型**:任务文件放 `rlinf/envs/maniskill/tasks/`(自动注册),策略相机命名 `3rd_view_camera` 复用 `ManiskillEnv` 默认 wrap;需要的分支只有三处:`get_robot_control_mode`、`prepare_actions_for_maniskill`、`_wrap_obs`(wrist 相机 + 状态归一化开关)。
- **场景保真是迭代出来的,用户的眼睛是裁判**:逐一对照真机照片(桌面颜色/板子尺寸方向/容器形态——实心块 vs 开口托盘曾错过一轮/物体生成区域)。
- **相机**:front 按真实高度与视场角摆(俯视);wrist 挂在夹爪 link 上 —— ⚠️ **改任何腕部关节的零位 OFFSET 都会带动 wrist 相机,必须重新指向**。指向用位姿扫描网格找,别猜。
- ⚠️ **partial reset 陷阱**:`_initialize_episode(env_idx)` 在提前终止后收到**部分**env_idx;所有 per-episode 缓冲必须全批量张量 + `buf[env_idx]=` 更新。这个 bug 只在**第一次 success** 时引爆(此前从没有提前终止)。显式测试:`env.reset(options=dict(env_idx=tensor([...])))`。

## 2. 标定(sim ↔ 真机单位)

- **先审计单位制**:LeRobot 数据集是归一化单位(臂 ±100,爪 0-100),ManiSkill 是弧度。喂错单位的症状是"动作方向全错/关节打满"。
- LeRobot 舵机标定语义:homing 已写进硬件寄存器,读数在以 tick 2048 为中心的 homed 坐标系 —— **软件里不要再减 homing_offset**。
- ✅ **每个执行器单独实测"指令→物理行程"曲线**(`set_qpos` 扫描 + 测量),不要跨机构复用换算 —— 夹爪(平行爪连杆)套用手臂舵机换算曾造成"最小开度 8.1cm vs 方块 2.9cm",物理上永不可能成功,潜伏了整整 12 小时训练。
- 换算必须**全数据集回验**(往返误差 ~1e-6)+ **全关节限位审计**(不要只查出问题的那个;真机标定常含未标定整圈的关节如 wrist_roll)。
- 方向/零位(SIGN/OFFSET)标定方法:真机 episode 动作在 sim 里回放,与真机视频**逐帧**对比(采样 3-4 帧下过错误结论)。

## 3. 可解性探针(不可跳过)

- 用 ManiSkill 自带运动规划器(`examples/motionplanning/<robot>/`)在**你的**环境里做完整抓取。**规划器都做不到 = 环境有问题;RL 更不可能。**
- 我们的数据点:标定修复后规划器 100/150 成功 —— 同时这就是 [4] 的演示生成器。

## 4. Sim 演示数据

- 生成:规划器 + `RecordEpisode`(CPU 仿真,mplib 要求单环境);只保留 success=True 的轨迹。
- 转 LeRobot:**单位在边界处转换**(sim 弧度 → 数据集归一化),用与 RL 侧**同一个**标定模块,保证 SFT 数据与 RL 观测自洽。
- ⚠️ **不要合并异构数据集**:sim(15fps/128²)与真机(30fps/640×480)的 fps 不同会扭曲 action-chunk 的时间语义。sim 数据独立成库,SFT 从真机 ckpt **继续训**而不是混数据。
- 转换后数值 sanity check:夹爪开/合的归一化值应与真机数据的 q01/q99 语义一致。
- norm_stats 按数据集各算各的;⚠️ rollout worker 会去 **checkpoint 目录里**找 norm_stats —— 每个新 warm-start ckpt 都要拷一份进去。

## 5. Sim-SFT(real2sim 的核心一步)

- **为什么必须有**:VLA 是视觉条件策略,真机数据教的是"看真机画面抓取";sim 画面是分布外 → 真机 SFT 模型在 sim 里"失明"(reach 偏 18cm,动作链完整但抓空气)。sim 演示 SFT 补的是"**在 sim 里看路**"。
- 配方:从真机 SFT ckpt 继续,纯 sim 演示,少量步数(我们 4000 步 ≈ 2 小时,loss ~0.001)。
- **验收 = zero-shot eval**(无 RL、无噪声,128 envs):这是配方正确性的最干净指标。我们:1.5%(无 sim-SFT,740 epochs RL 硬磨)→ **50%**(sim-SFT,零 RL)。
- 注意 SFT 配置里**所有**指向数据集的字段都要换(`train_data_paths` 曾漏改导致启动失败)。

## 6. RL 阶段

### 6.1 Reward(✅ 成熟结论)
- 用 proven 配方:`reach(1−tanh(5d)) + is_grasped + progress·is_grasped`,**不要**微观管理夹爪开合(所有成功参考案例皆如此;我们两次自创 shaping 全部翻车,其一还污染了权重血统)。
- 任何 shaping 项过审标准:**按每 episode 时间积分**算贡献(远端小系数 × 60 步 ≫ 近端大系数 × 5 步)。
- 探索确实撞不出接触时,只加一项梯度桥:`0.5·(d<4cm)·closedness`。

### 6.2 终止结构(⚠️ 已实证的陷阱)
- **success 提前终止 = 没收剩余稠密奖励**。算术:抓住不举 80 步 ≈ 196 回报 vs 成功终止 ≈ 35 → PPO 理性地学会"抓住但绝不举起"。指纹:**reward 上涨的同时 success 归零**。
- 对"持续保持"型成功判据:`env.train.ignore_terminations=True`,让 success 状态每步计分、严格占优。(eval 本来就该 True。)

### 6.3 RL-from-BC 保守束(🔬 当前最佳假说,v10 验证中)
- BC 克隆的行为是窄脊,默认噪声/步长的 PPO 会在 <10 epochs 内把它推下脊(三次复现:50% 起点 → 0;对照:RL 自己长出的 1.5% 行为在同样噪声下 740 epochs 不灭)。
- 处方(一起上,同一机制):`noise_params=[0.08,0.05,200]`(减半)、`actor.optim.lr=2e-6`、`update_epoch=2`、`clip=0.1`、`entropy_bonus=0`。
- **冻结测试**(廉价因果对照):`actor.optim.lr=1e-9` 跑几十 epochs —— 行为冻着不死、训着就死 ⇒ 破坏者是更新本身,不用再查环境/奖励/critic。

### 6.4 指标解码表(以 SO101 奖励为例,normalized=raw/5)
| 数值 | 含义 |
|---|---|
| reward ~0.20 | 纯 reach 悬停 |
| ~0.28 | 方块处闭合中 |
| ~0.37 + success=0 | ⚠️ 抓住不举(终止陷阱指纹) |
| >0.4 | 真抓取 |
| value_loss ~1 / ≥15 | critic 健康 / 拟合不了(回报双峰) |
| grad_norm ~23 / ≥65 | 健康 / 异常 |
| entropy_loss 稳定 −0.31~−0.34 | 噪声头没有膨胀 |

### 6.5 架构事实(theorize 之前先核对)
- `train_expert_only: True`(默认)→ VLM 冻结,可训的只有 action expert + value/noise 头 → **"critic 梯度毁主干"类理论在此配置下不成立**。
- `value_after_vlm: True` → value head 骑在冻结特征上,critic 学不好(value_loss 高)但**伤不了**策略。
- 与官方参考配置(`maniskill_ppo_openpi_pi05.yaml`)做 diff 是排除"配置问题"的最快手段。

## 7. 运维(浪费算力的头号来源)

- **进程真相只信 `/proc/<pid>/exe`**,永远不用字符串 pgrep 判断/杀进程(自匹配曾三次制造假象,包括清理脚本自杀)。
- 生命周期:`ray stop` → /proc 校验杀残留 → 清 `/tmp/ray/session_*` → **启动紧邻前**最后验证(0 存活、GPU 空、磁盘 >100G)→ `setsid` 脱离启动(否则随会话死)→ **30 分钟启动期限检查**(完整第一步,不是进度条)。
- 启动链禁 `sleep`;短阶段 `save_interval` 必须远小于阶段长度(曾因 34<50 全丢);共享 GPU 上跑长任务要接受"别人上机 → OOM"的风险并有 ckpt 兜底。
- 本机特有:IPv6 双修复(端口探测双栈 + tcp URL 加括号,勿回退)、Vulkan 软链、`RAY_local_fs_capacity_threshold=0.99`、128 envs 上限(320 挂)。

## 8. 实验方法论(把 12 小时的学费降到 40 分钟)

1. 每次启动前写下:**假说 → 可证伪预言 → 判决 epoch**;守卫自动宣判(塌缩阈值/增长阈值/死亡/超时)。
2. 用"当前绑定约束(证据等级)"的措辞,**永不宣布"根因找到了"** —— 两次误诊(脏状态、冷 critic)都死于证据不足时的过度自信;预言写在前面,证伪的 run 本身就是有价值的对照实验。
3. 止损规则:同类变体连败 ~3 次 → 停止换版本,质疑框架本身(参考配方的前提是否真的成立)。
4. 失败面在管线上单调下移、修过的不复发,才是"在进步"的标志;原地打转的信号是同一站反复塌。

## 9. 开放问题(截至本文)

- 🔬 v10(保守束)能否让 BC 起点存活并放大 —— 判决中。
- 官方示例"从 BC 起点放大"是否被上游实证过 —— 未验证;若 v10 败,这是下一个要核实的前提。
- critic 在冻结 VLM 特征上拟合双峰回报的能力上限(value_loss 收敛不到 10 以下)。
- sim2real(round 2)未开始:真机数据回炉、部署桥、域随机化,均在此文范围之外。
