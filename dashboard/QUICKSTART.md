# Dashboard Quick Start

在 `rlinf-0` 容器里把 dashboard 跑起来，看一个真实训练 run 的状态、曲线和视频。

本文所有命令都在容器内执行，并且都在一个**全新 pod**（镜像已含
`build(docker): build the dashboard frontend into the image`）上逐条验证过。
日志与 venv 都放在 `/workspace` PVC 上，重建 pod 后仍在。

## 运行环境

| 项目 | 值 |
|---|---|
| Kubernetes namespace | `rlinf` |
| Pod | `rlinf-0` |
| GPU | 单机 4×NVIDIA H20 |
| 源码 | `/workspace/RLinf` |
| 训练 Python | `/opt/venv/openpi/bin/python`（3.11） |
| dashboard venv | `/workspace/dashvenv`（本文第 1 步创建） |
| 前端 bundle | `/opt/rlinf-dashboard/dist`（镜像构建时烘焙） |
| 日志根目录 | `/workspace/runs` |

进容器：

```bash
kubectl exec -it rlinf-0 -n rlinf -- bash
```

报代理错误时先重建隧道：`ssh -fN -D 1080 jumpecs-hl`；隧道也建不起来就先
`kinit --keychain <you>@BYTEDANCE.COM`。

**最短路径**：装（第 1 步）→ 起服务（第 2 步）→ `PORT=8433 bash
dashboard/tests/smoke_server.sh /workspace/dashvenv/bin/python`（第 3.1 步）。
这三步不碰 GPU，约一分钟，能确认这套东西是通的。要看真实曲线再做 3.2。

---

## 1. 装 dashboard

dashboard **不依赖 rlinf**，装在自己的 venv 里，不要装进训练 venv——它的依赖解析
和 openpi/torch 那套是两回事。用镜像自带的 `uv`（`python3 -m venv` 在这个镜像里
建不出 pip）：

```bash
uv venv /workspace/dashvenv --python 3.11
cd /workspace/RLinf
VIRTUAL_ENV=/workspace/dashvenv uv pip install "./dashboard"
```

约 2 秒装完。验证装对了——第二条**应该报错**，那是设计如此：

```bash
/workspace/dashvenv/bin/python -c "import rlinf_dashboard; print('ok')"
/workspace/dashvenv/bin/python -c "import rlinf"   # 期望 ModuleNotFoundError
```

## 2. 启动服务

```bash
mkdir -p /workspace/runs

RLINF_DASHBOARD_FRONTEND_DIST=/opt/rlinf-dashboard/dist \
  setsid /workspace/dashvenv/bin/rlinf-dashboard /workspace/runs \
  --host 0.0.0.0 --port 8420 \
  > /workspace/dashboard.log 2>&1 < /dev/null &
```

**`RLINF_DASHBOARD_FRONTEND_DIST` 这一行不能省。** 服务端按
`<包目录>/../frontend/dist` → `<包目录>/static` 的顺序找前端，非 editable 安装
之后包在 site-packages 里，两个位置都没有 bundle——服务照常启动、API 全部正常、
`GET /` 返回 **404**，而且只在 debug 日志里说一句。这个变量指向镜像里烘焙好的那份。

> 容器里**没有 node**（前端只在镜像构建阶段编译），所以无法在容器内 `npm run build`。
> 要改前端，在容器外构建后把 `dist/` 拷进 `/workspace/RLinf/dashboard/frontend/dist`，
> 并**去掉**上面那个环境变量——它会覆盖查找顺序，否则你会一直看到烘焙的旧版本。

验证：

```bash
curl -s localhost:8420/api/health
curl -s -o /dev/null -w "%{http_code}\n" localhost:8420/     # 期望 200
```

在本机另开终端做端口转发，然后浏览器打开 `http://localhost:8421`：

```bash
kubectl port-forward -n rlinf pod/rlinf-0 8421:8420
```

## 3. Smoke test

### 3.1 只测 dashboard（不需要 GPU）

自带的 smoke 会造一棵假 run 树、起真 uvicorn、把每个端点和 SSE 流都打一遍：

```bash
cd /workspace/RLinf
VIRTUAL_ENV=/workspace/dashvenv uv pip install "./dashboard[test]"
PORT=8433 bash dashboard/tests/smoke_server.sh /workspace/dashvenv/bin/python
```

最后一行应为 `SMOKE PASS`。端口用 8433 而不是默认的 8421，是因为 8421 通常已经被
上面那条 `kubectl port-forward` 占着，撞车时报的是 `[Errno 48] address already in use`。

再跑一遍单元测试（约 5 秒，326 个用例）：

```bash
/workspace/dashvenv/bin/python -m pytest dashboard/tests -q
```

### 3.2 端到端：跑两步真实训练

写一个脚本，每次训练一个独立的 `experiment_name`：

```bash
cat > /workspace/smoke_train.sh <<'EOF'
#!/bin/bash
set -u
REPO=/workspace/RLinf
export EMBODIED_PATH=$REPO/examples/embodiment
export REPO_PATH=$REPO
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export ROBOT_PLATFORM=LIBERO
export LIBERO_TYPE=standard
export PYTHONUNBUFFERED=1

RUN_NAME="${1:-smoke-$(date +%Y%m%d-%H%M%S)}"
# 每个 run 一个 log_path。experiment_name 不会建子目录：指标固定写在
# <log_path>/tensorboard/，控制面写在 <log_path>/_rlinf/runs/<run_id>/，
# 两个 run 共用一个 log_path 就会共用同一个 tensorboard 目录，曲线被拼在一起。
export LOG_DIR=/workspace/runs/$RUN_NAME
mkdir -p "$LOG_DIR"
cd "$REPO" || exit 1

/opt/venv/openpi/bin/python examples/embodiment/train_async.py \
  --config-path "$EMBODIED_PATH/config" \
  --config-name libero_spatial_async_ppo_openpi_pi05 \
  runner.logger.log_path="$LOG_DIR" \
  runner.logger.experiment_name="$RUN_NAME" \
  runner.max_epochs=2 \
  runner.max_steps=2 \
  runner.save_interval=-1 \
  runner.val_check_interval=-1 \
  actor.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT \
  rollout.model.model_path=/workspace/models/RLinf-Pi05-LIBERO-SFT
echo "TRAIN_EXIT=$?"
EOF
chmod +x /workspace/smoke_train.sh

setsid bash /workspace/smoke_train.sh smoke-quickstart \
  > /workspace/smoke_train.log 2>&1 < /dev/null &
```

三个容易踩的点：

- **`${PYTHONPATH:-}` 的冒号减号不能省。** 脚本用了 `set -u`，而全新容器里
  `PYTHONPATH` 没有定义，写成 `$PYTHONPATH` 会直接 `unbound variable` 退出。
- **`runner.max_epochs` 才是步数上限。** `set_max_steps` 把 `num_steps_per_epoch`
  钉成 1，`max_steps` 只能往下压——只设 `max_steps=2` 仍然只跑 1 步。
- **一步的 run 意义不大**：每条曲线只有一个点，趋势无从谈起。2 步是能看到线段的
  最小值。

跟踪进度：

```bash
tail -f /workspace/smoke_train.log
grep "Global Step:" /workspace/runs/*/metrics.log | tail -5
```

首步要等模型加载和 LIBERO 环境初始化，几分钟属正常，别用首步判断性能。

## 4. 在 dashboard 里看这个 run

浏览器刷新 `http://localhost:8421`，run 会自己出现——服务端每隔
`discovery_cache_ttl_s`（默认 5 秒）重扫一次，不用重启。

各页看什么：

| 页面 | 内容 |
|---|---|
| Runs | 所有 run 一行一个：state、health、phase、进度、心跳 |
| Overview | 8 张卡：状态/阶段/进度/时间/ckpt/健康/北极星/异常信号 |
| Metrics | 按模板分组的曲线。右上 `EXPAND TO RANKS` 只在开了 `per_worker_log` 时出现 |
| Media | 录像。卡片滚到视野里才加载，点播放才下载 |
| Events | 生命周期与阶段事件，50 行一页 |

用 API 直接确认也可以：

```bash
curl -s localhost:8420/api/runs | python3 -m json.tool | head -30
curl -s "localhost:8420/api/runs/<RUN_ID>/template" | python3 -c \
  'import json,sys; t=json.load(sys.stdin); print(t["name"], [g["title"] for g in t["groups"]], t.get("unmatched"))'
```

`unmatched` 列出的是这个 run 记了、但模板没有给固定位置的指标。它们仍会显示在
**Other keys** 组里——**非空不代表出错**：async PPO 的 `data_staleness_<n>/ratio`
是按实际观测到的 lag 生成的，出现一个新的 bin 本身就是信息。

## 5. 想看更多曲线时

### 5.1 per-rank 指标

```text
+runner.per_worker_log=true
```

**前面的 `+` 不能省。** 这个 key 只在少数 config 里声明，默认值是 hydra 组装完
之后由 `rlinf/config.py` 填的，直接覆盖会撞 struct 模式，报
`Key 'per_worker_log' is not in struct`——在任何一张 GPU 被碰到之前就退出。

打开后聚合 bundle 会下移到 `tensorboard/all/`，每个 rank 一份写到
`worker_logs/<Group>/rank_<n>/tensorboard/`。**这两个路径里都没有 run id**，所以
两个开了该开关的 run 共用一个 `log_path` 会互相覆盖——各给一个 `log_path`。

### 5.2 录像

`env.<split>.video_cfg.save_video` 控制。注意它默认由 env group 打开，顶层 YAML
里看不到；开着的话每个 rollout epoch 全量录，代价不小。

## 6. 常见问题

| 现象 | 原因 |
|---|---|
| `GET /` 返回 404，API 正常 | 没设 `RLINF_DASHBOARD_FRONTEND_DIST`，见第 2 步 |
| run 列表空 | scan root 路径不对。`curl localhost:8420/api/health` 看 `scan_roots` 的 `exists` |
| run 显示 `state: null` / `health: unknown` | **初始化期间的正常状态**。reporter 在构造时就写了 `manifest.json`，所以 run 在 worker 起来之前就可见；`run.json` 要等训练循环真正开始才写。模型加载 + LIBERO 初始化期间会停在这个状态几分钟。一直不变就去看 `smoke_train.log`——加载阶段崩掉的话不会写终态 |
| 终态 run 显示 `unreachable` | 正常。心跳停了就是停了，终态 run 不会因此变红——若变红说明 state 没写成终态 |
| 列表顶部红色「run id 撞车」 | 不同 run 用了同一个 run id。**点进去看到的可能不是你点的那个**，给每个 run 独立 id |
| `[Errno 48] address already in use` | 8421 被 port-forward 占了，smoke 用 `PORT=8433` |

## 7. 收尾

```bash
pkill -f rlinf-dashboard
pkill -f train_async.py
/opt/venv/openpi/bin/ray stop            # 训练异常退出后残留的 worker
```

日志和 checkpoint 留在 `/workspace/runs/<RUN_NAME>/`，pod 重建后仍在。目录结构：

```text
/workspace/runs/<RUN_NAME>/
├── _rlinf/
│   ├── latest -> runs/<run_id>    # 软链，省得记 run id
│   └── runs/<run_id>/             # 控制面：manifest.json、run.json、heartbeat、
│                                  # events.jsonl、checkpoints.jsonl、media.rank*.jsonl
├── tensorboard/
│   ├── events.out.tfevents.*      # 指标（dashboard 和 tensorboard 读同一份）
│   └── config.yaml                # 本次解析后的完整 hydra 配置
├── metrics.log                    # 人读的表格，第一步完成后才出现
└── checkpoints/                   # save_interval > 0 时才有
```

想直接看控制面：`cat /workspace/runs/<RUN_NAME>/_rlinf/latest/run.json`。

`--port 8420` 指到 `/workspace/runs` 这个父目录即可，它会往下扫；不必逐个 run 指。
