# RLinf Dashboard 快速开始

本文给出三个可复现路径：本地源码、正式 wheel、独立容器。Dashboard 只读
RLinf 写出的 control-plane 文件和 TensorBoard event files，不需要安装
RLinf、PyTorch 或训练环境。

## 1. 从源码启动

在 `dashboard/` 目录执行：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'

cd frontend
npm ci
npm run typecheck
npm run check:scales
npm run check:identity
npm run build
cd ..
```

源码安装会自动使用 `frontend/dist/`，不需要设置
`RLINF_DASHBOARD_FRONTEND_DIST`。

先生成不依赖训练的演示数据：

```bash
.venv/bin/python frontend/scripts/make_demo_runs.py \
  --root /tmp/rlinf-dashboard-demo --clean

.venv/bin/rlinf-dashboard /tmp/rlinf-dashboard-demo \
  --host 127.0.0.1 --port 8420
```

打开 <http://127.0.0.1:8420/>。以下接口可用于快速确认服务与扫描路径：

```bash
curl -fsS http://127.0.0.1:8420/api/health | python3 -m json.tool
curl -fsS http://127.0.0.1:8420/api/runs | python3 -m json.tool
```

需要认证时，同时设置静态用户名和密码；只设置一项或留空会拒绝启动：

```bash
RLINF_DASHBOARD_AUTH_MODE=basic \
RLINF_DASHBOARD_AUTH_USERNAME=operator \
RLINF_DASHBOARD_AUTH_PASSWORD='replace-with-a-secret' \
  .venv/bin/rlinf-dashboard /tmp/rlinf-dashboard-demo \
  --host 127.0.0.1 --port 8420

curl -fsS -u operator \
  http://127.0.0.1:8420/api/health | python3 -m json.tool
```

`curl` 会交互读取密码，避免把密码写进命令行。

认证会覆盖 UI、API、SSE、视频和 OpenAPI。`/healthz` 是唯一无需认证的接口，
只返回进程是否存活，不会暴露扫描目录或 run 数。Basic Auth 本身不加密传输，
非本机部署必须放在 HTTPS ingress 后面。

演示树中的 heartbeat 会随时间过期；重新标记时间而不重建 metrics：

```bash
.venv/bin/python frontend/scripts/make_demo_runs.py \
  --root /tmp/rlinf-dashboard-demo --touch
```

### 手动验证 Media 预览图

演示脚本默认用 ffmpeg 合成一段可播放的 clip，因此 Media 页可以直接验证预览图与
按需加载。先确认本部署能否渲染预览图：

```bash
curl -fsS http://127.0.0.1:8420/api/health | python3 -m json.tool | grep -A 3 posters
```

`available: true` 表示找得到 ffmpeg。随后打开任意一个带 Media 标签的 run，在
浏览器 DevTools 的 Network 面板中逐项确认：

1. **首屏只加载预览图**：过滤 `media/`，应只看到 `media/poster` 请求（每个约
   15KB），没有任何 `media/file` 请求。
2. **点击才加载视频**：点某张卡片后才出现该 clip 的 `media/file` 请求。
3. **切走即断开**：在视频仍在加载时点 Metrics 标签，该 `media/file` 请求应立刻
   变为 canceled，而不是继续占用连接。切换 Split 或 RL iteration 同理。
4. **无 ffmpeg 时降级**：用 `RLINF_DASHBOARD_POSTER_ENABLED=false` 重启，卡片应
   回退为可点击的占位块，视频仍可正常播放。

若要验证解码失败的那条路径（占位块 + poster 404），用 stub clip 重建演示树：

```bash
.venv/bin/python frontend/scripts/make_demo_runs.py \
  --root /tmp/rlinf-dashboard-demo --clean --stub-clips
```

## 2. 验证正式发布物

发布 wheel 会内置 frontend、v2 schema 和 LICENSE，因此安装后同样不需要
`RLINF_DASHBOARD_FRONTEND_DIST`。从源码构建并隔离验证：

```bash
bash tests/smoke_wheel.sh .venv/bin/python
```

该检查会构建 frontend 和 wheel，在全新 venv 中安装，然后验证：

- `GET /` 返回浏览器应用；
- 页面引用的静态资源全部可读取；
- client-side 深链路返回应用 shell；
- 未注册 API 仍返回 404；
- schema 和 LICENSE 已进入 wheel。

若使用外部构建目录，而不是源码 `frontend/dist/` 或 wheel 内置资源，再显式
设置：

```bash
RLINF_DASHBOARD_FRONTEND_DIST=/absolute/path/to/dist \
  rlinf-dashboard /path/to/logs
```

## 3. 使用独立容器

在仓库根目录执行：

```bash
docker build -f dashboard/Dockerfile -t rlinf-dashboard:local .
docker run --rm \
  --read-only --tmpfs /tmp \
  -p 8420:8420 \
  -v /path/to/logs:/runs:ro \
  --env-file /secure/path/dashboard-auth.env \
  rlinf-dashboard:local
```

其中 `dashboard-auth.env` 权限应限制为当前用户读取，内容为：

```text
RLINF_DASHBOARD_AUTH_MODE=basic
RLINF_DASHBOARD_AUTH_USERNAME=operator
RLINF_DASHBOARD_AUTH_PASSWORD=replace-with-a-secret
```

Kubernetes/Helm 部署应从 Secret 通过 `secretKeyRef` 注入这两个环境变量，
不要把密码写进 args、ConfigMap、URL 或 probe。健康探针使用 `/healthz`，不需要
也不应携带认证头。

容器以非 root 用户运行，训练日志建议始终只读挂载。完整 image smoke：

```bash
bash dashboard/tests/smoke_container.sh
```

## 4. 观察真实 RLinf run

Dashboard 扫描的是 `runner.logger.log_path` 或它们的共同父目录。每次 launch
必须使用独立的 `log_path`；目前 TensorBoard 和 per-worker 路径不包含 run ID，
复用同一路径会把不同 run 的曲线合并。

示例原则如下，具体入口和 config 以所运行的 RLinf recipe 为准：

```bash
RUN_NAME="dashboard-check-$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="/path/to/runs/${RUN_NAME}"

# 在训练命令中覆盖：
# runner.logger.experiment_name=$RUN_NAME
# runner.logger.log_path=$RUN_ROOT
```

然后让 Dashboard 扫描 `/path/to/runs`。若需要按 rank 检查 metrics，可在支持
该配置的 recipe 中加入：

```text
+runner.per_worker_log=true
```

实时检查时应看到：

1. `state` 从 `pending` 进入 `running`，正常结束后为 `finished`；
2. `heartbeat_at` 按 manifest 中记录的 interval 前进；
3. `last_progress_at` 随有效 step 前进；
4. `phase` 或 async `components` 反映当前工作；
5. checkpoint 完成后才出现在列表中；
6. 终态 run 的 heartbeat 会停止，但 health 仍应保持 `healthy`。

若 run 进入 `failed`，Events 页的 Exit 信息会展示 reason 和 traceback 末段；
完整诊断仍应以训练日志为准。

不要使用 `pkill -f` 清理训练，它可能终止同一节点上的其他任务。Linux 上可
为验证任务创建独立 process group，并只向该组发送 `SIGINT`，让 lifecycle
有机会写入 `stopped`：

```bash
setsid bash /path/to/run.sh >"${RUN_NAME}.log" 2>&1 < /dev/null &
TRAIN_PGID=$!

# 验证结束后：
kill -INT -- "-${TRAIN_PGID}"
wait "${TRAIN_PGID}" || true
```

Dashboard 服务本身应记录 PID，并只终止该 PID。

## 5. 生命周期边界

manifest 在 runner 构造时写入，并早于 `init_workers()`；模型或环境初始化
期间通常已经能看到 run。默认 600 秒 startup grace 内，manifest-only run
显示为 initializing，而不是故障；超时后仍无 snapshot 才显示为 startup
问题。Cluster、placement 或 worker-group launch 在 runner 构造前失败时，
可能还没有任何 run 目录。若 `init_workers()` 失败，通常只能看到
manifest，health 为 unknown。

这不是“健康运行”，而是 producer lifecycle 仍需继续上移到统一 launcher 的
边界。排查时先看训练日志，不要把 manifest-only 当成 dashboard 故障。

## 6. 常用验证命令

```bash
.venv/bin/python -m pytest tests -q
bash tests/smoke_server.sh .venv/bin/python
bash tests/smoke_wheel.sh .venv/bin/python

cd frontend
npm run typecheck
npm run check:scales
npm run check:identity
npm run build
```

服务 smoke 验证核心 REST、media 和单-run SSE；它不等价于“覆盖所有 API”。
wheel smoke 负责验证真正安装后的浏览器资源。
