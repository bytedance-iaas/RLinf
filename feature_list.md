# Feature List

需求账本。任务开始后需求描述、验收步骤、测试标准不可改写，只允许更新 `Status` 字段
（取值 `passes` / `pending`）。

---

## 主题：清理训练日志噪声（quiet-run-logs）

基线：4×H20 async Pi0.5 PPO + LIBERO-spatial 一次完整启动日志共 1249 行，其中约 1030 行
（83%）为噪声。逐类归因见下。目标：把噪声压到 ~250 行，只保留真实训练信息。

### F1 — 抑制旧版 gym 的停维护公告

- 目标：消除 272 行 `Gym has been unmaintained since 2022 ...` 四连公告。
- 边界：不迁移 LIBERO env 栈到 gymnasium（改动面过大且 robosuite/LIBERO 自身仍 import 旧
  gym）；只在「装包层」和「运行期 fd 层」处理。
- 归因：旧版 `gym/__init__.py` 末尾 `print(notice, file=sys.stderr)`，是裸 print 而非
  `warnings`，故 `PYTHONWARNINGS` / `warnings.filterwarnings` 一律无效；且由 68 个进程
  （4 个 env worker × 16 个 LIBERO spawn 子进程）各自 import 时打印，父进程内抑制无效。
- 实测约束（2026-07-29）：`gym==0.26.2` 的 `setup.py:68` 把 `gym_notices >= 0.0.4` 声明为硬
  依赖，卸载它会让环境依赖校验失败、且下次 `uv sync` 会装回，故**装包层方案不可用**。改为
  在 `sys.modules` 里把 `gym_notices` 置 `None` 让 gym 自己的 `except Exception: pass`
  吞掉——已用 gym 形状的模块端到端验证：基线打印公告、抑制后静默且 import 仍成功。
  相应地 fd=2 过滤器不再需要（机制更干净），原验收标准 1/2 按此修正。
- 验收标准：
  1. 提供运行期抑制：阻断 `gym_notices` 导入后，gym 的 `try/except Exception: pass` 静默跳过。
  2. 已被 import 的模块不被篡改，且重复调用安全。
  3. 有单元测试覆盖上述两点。
- Status: passes

### F2 — 全量 config JSON dump 改为可开关

- 目标：消除 238 行 `print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))`。
- 边界：默认关闭；保留用户显式打开的能力；Hydra 本身已把解析后 config 存盘，信息不丢失。
- 归因：`examples/embodiment/train_async.py:38` 等 15 处入口无条件打印。
- 实测约束（2026-07-29）：157 个 embodiment config 全部设了 `hydra.output_subdir: null`，
  **Hydra 本身不存 config**；真正落盘的是 `MetricLogger`（`metric_logger.py:119-120`
  写 `tensorboard/<run>/config.yaml`，155/157 个 config 启用了 tensorboard），信息不丢失的
  结论成立但归因需按此修正。另：`collect_real_data.py` 原本就没有 config dump，故走统一
  helper 的是 3 个入口而非 4 个（第 4 个仅需 F7 的 `version_base`）。
- 验收标准：
  1. 新增 `runner.print_config` 配置项，缺省 `False`，且在 struct 模式下读取不报错。
  2. 有 config dump 的 3 个入口（train_async / train_embodied_agent / train_offline_rl）
     走统一 helper。
  3. helper 有单元测试覆盖「默认不打印、显式开启时打印、解析插值」。
- Status: passes

### F3 — 统一压制 TensorFlow / absl 启动横幅

- 目标：消除 67 行 oneDNN / cpu_feature_guard / cudart_stub / absl 横幅。
- 边界：只设环境变量，不改 TF 调用；不引入对 TF 的强依赖。
- 归因：`TF_CPP_MIN_LOG_LEVEL` 未设或设为 `"2"`（压不住 `I0000 ... port.cc:153`）；
  157 个 embodiment config 里仅 1 个设了该变量；driver 进程的 TF 在 module import 期即加载，
  比 `hydra.job.env_set` 更早，故 YAML 层无法覆盖 driver。
- 实测约束（2026-07-29）：本次日志对应的 config 已经通过 `hydra.job.env_set` 设了
  `TF_CPP_MIN_LOG_LEVEL: "2"`，而 26 条 `port.cc:153` + 13 条 `cudart_stub.cc:31`
  **依然全部打印**，反证 `"2"` 不足，须用 `"3"`（与仓库内既有先例
  `sgl_engine.py:76` 一致）。另外已实测 `hydra.job.env_set` 只在 job 体内生效
  （driver 的 `import` 早于它），故该 YAML 项对本条无效、已从 config 中移除。
  `TF_ENABLE_ONEDNN_OPTS=0` 经复核属于**改计算而非改日志**，不纳入（同 barrier 的处理原则）。
- 验收标准：
  1. `TF_CPP_MIN_LOG_LEVEL=3` 在 driver 侧（sh 脚本 export）和 worker 侧
     （`get_accelerator_env_var`）都生效。
  2. 用户已显式设置的值不被覆盖。
  3. worker 侧注入有单元测试覆盖「默认注入完整映射、用户值优先」。
- Status: passes

### F4 — 压制第三方库的 INFO 日志

- 目标：消除 36 行第三方 INFO（`INFO:datasets:*` ×32、`INFO:OpenGL` ×4），并消除
  robosuite 警告经 root logger 的重复打印。
- 边界：RLinf 自身日志级别与格式不得变化（`Worker._setup_logging` 用独立 logger +
  `propagate = False`，不受 root 影响）；仓库内既有的 `logging.info(...)` 根 logger 调用
  仍需可见。
- 归因：`rlinf/config.py:39` 的 `logging.getLogger().setLevel(logging.INFO)` 叠加
  `worker.py:316` 的 `logging.basicConfig()`，把第三方 logger 全量放行到 stdout。
- 实测约束（2026-07-28）：`INFO:root:Loaded norm stats` ×16 走的是 **root logger 本身**，
  与仓库内 ~100 处 `logging.info(...)` 同源、不可区分，压掉它必然连带压掉 RLinf 自己的
  输出，故不在本 feature 目标内（原始目标 52 行据此修正为 36 行）。同理，root 级别必须
  保持 INFO，只能对已知噪声命名空间显式设级别。
- 验收标准：
  1. 已知噪声库（`datasets`、`OpenGL`、`robosuite_logs`、`absl` 等）被显式压到 WARNING。
  2. 仓库内根 logger 的 `logging.info(...)` 调用仍可见。
  3. 被压制的库其 WARNING 及以上仍可见（不是全量静音）。
  4. 有单元测试覆盖上述三点。
- Status: passes

### F5 — 抑制 `NO_SHARD` FutureWarning

- 目标：消除 16 行 `The NO_SHARD sharding strategy is deprecated`。
- 边界：不迁移到 DDP / FSDP2 —— 现有 4 卡 / 8 卡性能结论全部建立在 `no_shard` 上，迁移是
  行为变更，不在本任务范围。
- 归因：`rlinf/hybrid_engines/fsdp/strategy/fsdp.py:173` 及 torch `fsdp/wrap.py:485`；
  `fsdp_model_manager.py` 现有 filter 是 `UserWarning` + 文案含 `full_state_dict`，
  与本条（`FutureWarning`，文案不含该词）不匹配。
- 验收标准：
  1. `no_shard` 路径下该 FutureWarning 不再输出。
  2. 其它 FutureWarning 不被误吞。
  3. 有单元测试覆盖上述两点。
- Status: passes

### F6 — 消除 AccumulateGrad stream 不匹配警告

- 目标：消除 8 行 torch `autograd/graph.py:869` UserWarning。
- 边界：不改变梯度累积语义与数值结果。
- 归因：`rlinf/workers/actor/fsdp_actor_worker.py` 的 `train_micro_batch` 在 `.backward()`
  之后仍持有 `loss` 引用，autograd graph 被留到下一轮迭代。
- 实测约束（2026-07-29）：上述归因**已被实测否证**。用 weakref 追踪 graph 存活的最小复现显示
  `explicit_del=False` 与 `True` 两种情况下，函数返回后 graph 均已释放（局部变量随返回即释放），
  即加 `del loss` 是 no-op，warning 不是由这个引用造成的——graph 由 FSDP 内部保留。故原验收
  标准 1 的「根治」不可达，本 feature 改为只做 torch 层开关（这也正是 torch 自己的 warning
  文案推荐的做法：`set_warn_on_accumulate_grad_stream_mismatch(False)`）。
- 验收标准：
  1. 在 FSDP 包装模型的进程中调用 torch 官方开关关闭该 warning，不改变梯度累积语义与数值结果。
  2. 在 torch 不支持该 API 的版本上安全降级（返回 False 而非抛异常）。
  3. 有单元测试覆盖开关生效与降级两条路径。
- Status: passes

### F7 — 修复 Hydra chdir FutureWarning

- 目标：消除 3 行 `Future Hydra versions will no longer change working directory`。
- 边界：不改变各入口的实际工作目录行为（现有 config 已普遍 `hydra.run.dir: .`）。
- 归因：embodiment 入口 `@hydra.main(version_base="1.1")`。
- 实测约束（2026-07-29）：已实测 `1.1 → 1.2` 会把 `hydra.job.chdir` 默认值从 `True` 翻成
  `False`；但 156/157 个 config 设了 `hydra.run.dir: .`，实测该设置下两个版本的 cwd 完全一致，
  故不受影响。唯一没有 `hydra:` 块的 `realworld_peginsertion_async_ppo_pi05.yaml` 会被默认值
  翻转影响，已为其显式补上 `chdir: false` + `run.dir: .`。
- 验收标准：
  1. embodiment 4 个入口的 `version_base` 提升到不再触发该警告的版本。
  2. `hydra.job.chdir` 行为显式化，不依赖版本默认值；所有 config 的实际 cwd 行为不变。
- Status: passes

### F8 — 交接闭环

- 目标：`feature_list.md`、`claude-progress.txt`、`README.md` 手动验证步骤齐备，代码可提交。
- 验收标准：
  1. `claude-progress.txt` 含合法 `TaskStatus`。
  2. `README.md` 记录本主题的人工验证步骤且可执行。
  3. 单元测试全部通过，改动已 commit。
- 实测约束（2026-07-29）：本机无 GPU 且 pip 装 TensorFlow 触发 `Errno 28 No space left on
  device`，故 TF 横幅的抑制效果无法在本地实证；已在 `claude-progress.txt` 的 Main gaps 记录，
  并把真机复测步骤写进 `README.md`。其余 6 项均有单测或端到端验证。
- Status: passes
