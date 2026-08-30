/**
 * Simplified Chinese copy.
 *
 * Typed against `en`, so `tsc` rejects a missing or invented key. Terminology is
 * fixed rather than translated afresh per string:
 *
 * * `run` -> 任务 (never 实验, which is `experiment_name` and appears beside it,
 *   and never 运行, which is reserved for the *state* -- 运行中 / 未在运行)
 * * `step` -> 步, `RL iteration` -> RL 迭代, `minibatch` -> 小批次
 * * `health` -> 健康, `degraded` -> 异常, `unreachable` -> 失联
 * * `north-star metric` -> 核心指标, `checkpoint` -> 检查点, `rank` -> rank
 *
 * Identifiers stay in the original: run id, metric keys, config keys, file
 * names, `events.jsonl`. They are things to grep for on the training side, and a
 * translated one cannot be grepped.
 */

import type { en } from "./en";

export const zh: Record<keyof typeof en, string> = {
  // -- Shell ---------------------------------------------------------------
  "app.title": "RLinf 控制台",
  "app.brandAlt": "RLinf",
  "app.breadcrumb": "面包屑导航",
  "app.runViews": "任务视图",
  "app.streamError": "实时数据流报错",
  "app.loadingRun": "正在加载任务…",
  "app.refresh": "刷新",
  "app.themeToLight": "切换到浅色主题",
  "app.themeToDark": "切换到深色主题",
  "app.themeLight": "浅色",
  "app.themeDark": "深色",
  "app.langToggle": "EN",
  "app.langToggleTitle": "Switch to English（切换到英文）",
  "app.liveTitle": "SSE：{state}",

  "tab.overview": "概览",
  "tab.metrics": "指标",
  "tab.media": "视频",
  "tab.events": "事件",

  "live.connecting": "连接中",
  "live.live": "实时",
  "live.reconnecting": "重连中",
  "live.error": "错误",

  // -- Shared vocabulary ---------------------------------------------------
  "status.healthy": "健康",
  "status.degraded": "异常",
  "status.unreachable": "失联",
  "status.unknown": "未知",
  "status.running": "运行中",
  "status.finished": "已完成",
  "status.failed": "已失败",
  "status.stopped": "已停止",
  "status.pending": "等待中",
  "status.initializing": "初始化中",
  "status.all": "全部",

  "confidence.low": "低置信",
  "confidence.medium": "中置信",
  "confidence.high": "高置信",

  "semantics.rl_iteration": "RL 迭代",
  "semantics.minibatch": "小批次",
  "semantics.optimizer_step": "优化器步",
  "semantics.step": "步",
  "semantics.short.rl_iteration": "迭代",
  "semantics.short.minibatch": "小批",
  "semantics.short.optimizer_step": "步",
  "semantics.short.step": "步",

  "format.justNow": "刚刚",
  "format.secondsAgo": "{n} 秒前",
  "format.minutesAgo": "{n} 分钟前",
  "format.hoursAgo": "{n} 小时前",
  "format.daysAgo": "{n} 天前",

  "healthbar.aria": "任务健康状态：{health}",

  "pager.label": "翻页",
  "pager.page": "第 {current} / {total} 页",
  "pager.range": "{from}–{to} / 共 {total}",
  "pager.first": "第一页",
  "pager.prev": "上一页",
  "pager.next": "下一页",
  "pager.last": "最后一页",
  "progress.noHorizon": "无总步数",

  // -- Server card (run list) ----------------------------------------------
  "server.title": "服务端",
  "server.version": "版本",
  "server.runs": "任务数",
  "server.scanRoot": "扫描根目录",
  "server.missing": "不存在",
  "server.noRunsFound": "未找到任务",
  "server.scanRootChange": "修改",
  "server.scanRootSave": "保存",
  "server.scanRootCancel": "取消",
  "server.scanRootReset": "恢复默认",
  "server.scanRootResetTitle": "回到服务启动时配置的根目录：{path}",
  "server.scanRootLabel": "扫描根目录路径",
  "server.scanRootPlaceholder": "服务端上的一个目录",

  "rollup.none": "尚未发现任何任务",
  "rollup.summary": "{total} 个任务中最差的健康状态：{health}。{bad} 个不健康。",

  // -- Run list ------------------------------------------------------------
  "runlist.runs": "任务",
  "runlist.search": "搜索",
  "runlist.searchPlaceholder": "run id、实验名、任务类型",
  "runlist.state": "状态",
  "runlist.compare": "对比",
  "runlist.compareN": "对比（{count}）",
  "runlist.compareHint": "选中两个及以上的任务才能对比",
  "runlist.selectForCompare": "选中以参与对比",
  "runlist.selectRunForCompare": "选中 {name} 以参与对比",

  "runlist.collided.one": "{count} 个 run id 被不同的任务共用",
  "runlist.collided.other": "{count} 个 run id 被不同的任务共用",
  "runlist.collidedBody":
    "这些任务的名字不同、日志路径不同，run id 却相同，而本面板的每个 URL 和每次 API 调用" +
    "都以 id 定位任务。打开其中任何一个，看到的都是服务端最先找到的那个——{emphasis}——" +
    "对比时同理。",
  "runlist.collidedEmphasis": "未必是你点的那一个",
  "runlist.collidedHint":
    "默认 id 是秒级时间戳加实验名，所以复制来的配置里写死 {code} 时也会这样。给每个任务" +
    "各自的 id 才能区分开。",

  "runlist.attention.one": "1 个任务需要关注",
  "runlist.attention.other": "{count} 个任务需要关注",

  "runlist.col.run": "任务",
  "runlist.col.state": "状态",
  "runlist.col.health": "健康",
  "runlist.col.phase": "阶段",
  "runlist.col.step": "步数",
  "runlist.col.elapsed": "已用时",
  "runlist.col.eta": "预计剩余",
  "runlist.col.ckpt": "检查点",
  "runlist.col.heartbeat": "心跳",

  "runlist.discoveringTitle": "正在发现任务",
  "runlist.discoveringBody": "正在扫描任务。扫描完成前，不对已有内容下任何结论。",
  "runlist.noneTitle": "未发现任何任务",
  "runlist.noneNoRoot": "服务端尚未报告它的扫描根目录。",
  "runlist.noneMissingRoot": "扫描根目录 {path} 不存在。",
  "runlist.noneEmptyRoot":
    "已搜索 {path}，该目录存在但还没有任何任务。一个任务是指含有 " +
    "_rlinf/runs/<id>/manifest.json 的目录，最深可在根目录下六层。",
  "runlist.noMatchTitle": "没有匹配的任务",
  "runlist.noMatchBody": "当前的搜索词或状态过滤把所有已发现的任务都筛掉了。",

  // -- Overview ------------------------------------------------------------
  "overview.startingTitle": "正在启动",
  "overview.startingBody":
    "任务已注册，但还没有发布第一份快照。集群启动、worker 分配和模型加载都发生在这段窗口里。",
  "overview.startingBodyElapsed":
    "任务已注册，但还没有发布第一份快照。集群启动、worker 分配和模型加载都发生在这段窗口里，" +
    "目前已经过去 {elapsed}。",
  "overview.snapshotUnreadable": "快照不可读",

  "overview.state": "状态",
  "overview.components": "组件",
  "overview.phase": "阶段",
  "overview.progress": "进度",
  "overview.timing": "耗时",
  "overview.checkpoint": "最新检查点",
  "overview.health": "健康",
  "overview.northStar": "核心指标",
  "overview.anomalies": "异常信号",

  "overview.noManifest": "无 manifest",
  "overview.started": "启动于 {time}",
  "overview.async": "异步",
  "overview.active": "活跃",
  "overview.idle": "空闲",
  "overview.activeFor": "已活跃 {age}",
  "overview.idleFor": "已空闲 {age}",
  "overview.activeSince": "自 {time} 起活跃",
  "overview.idleSince": "自 {time} 起空闲",
  "overview.notRunning": "未在运行",
  "overview.inPhaseFor": "处于该阶段 {age}",
  "overview.noPhase": "未记录阶段",
  "overview.nodes.one": "{count} 个节点",
  "overview.nodes.other": "{count} 个节点",
  "overview.placementUnknown": "placement 未知",
  "overview.epoch": "第 {epoch} 个 epoch",
  "overview.noEpoch": "未报告 epoch",

  "overview.finished": "结束",
  "overview.eta": "预计剩余",
  "overview.endedAfter": "{state}，历时 {elapsed}",
  "overview.ended": "已结束",
  "overview.etaWithConfidence": "{eta}（{confidence}）",
  "overview.perStep": "每 {unit}",

  "overview.best": "最佳",
  "overview.noCheckpoint": "暂无",
  "overview.saved": "保存于",
  "overview.size": "大小",
  "overview.took": "耗时",
  "overview.noCheckpointHint": "还没有保存过检查点。",
  "overview.noCheckpointTitle": "索引只在保存完成后才追加，所以写了一半的检查点绝不会出现在这里。",

  "overview.heartbeat": "心跳",
  "overview.lastStep": "上次进展",
  "overview.budget": "预算",

  "overview.openMetric": "查看图表",
  "overview.notLogged": "未记录",
  "overview.atStep": "位于第 {step} {unit}",
  "overview.northStarMissing": "该任务没有记录 {key}。{template} 模板期望它存在。",
  "overview.northStarUndeclared": "{template} 模板没有为这类任务声明核心指标。",
  "overview.templateDefault": "默认",

  "overview.derivedFromMetrics": "由指标推导",
  "overview.derivedTitle": "在浏览器里根据指标序列计算得出",
  "overview.anomaliesNone": "无",
  "overview.anomaliesNoneHint":
    "在监控的 {count} 条序列里，没有发现步时间劣化、评测停滞或非有限值。",
  "overview.critical": "严重",
  "overview.warning": "警告",

  // -- Metric signals (computed in the browser) ----------------------------
  "signal.nonFiniteTitle": "指标出现非有限值",
  "signal.nonFiniteDetail": "{key} 在第 {step} 步首次出现 NaN 或 Inf（{total} 个点中有 {count} 个）。",
  "signal.nonFiniteDetailMore":
    "{key} 在第 {step} 步首次出现 NaN 或 Inf（{total} 个点中有 {count} 个），另有 {others} 条序列同样如此。",
  "signal.stepTimeTitle": "步时间劣化",
  "signal.stepTimeDetail":
    "{key} 已达早期基线的 {ratio} 倍（当前 {recent}s，第 {from}-{to} 步为 {baseline}s）。",
  "signal.plateauTitle": "评测连续 {k} 轮无提升",
  "signal.plateauDetail": "{key} 最近 {k} 次评测都没有超过 {best}（近期最好为 {recent}）。",

  // -- Metrics -------------------------------------------------------------
  "metrics.noTemplate": "没有模板",
  "metrics.loadingLayout": "正在加载该任务的图表布局…",
  "metrics.template": "模板",
  "metrics.axis": "{label} 轴",
  "metrics.expandRanks": "展开到各 rank",
  "metrics.sampled": "已采样",
  "metrics.sampledTitle": "服务端对这些序列做了等距抽样，以不超过它的点数上限",
  "metrics.keyCount": "{resolved}/{total} 个指标",
  "metrics.seriesFailed": "指标序列请求失败",
  "metrics.northStarMissingTitle": "核心指标未被记录",
  "metrics.northStarMissingBody":
    "{template} 模板期望 {key}，但该任务没有记录它。下面其余的图表不受影响。",
  "metrics.otherKeys": "其他指标",
  "metrics.groupFallback": "指标",
  "metrics.stackBlockedNote": "堆叠图——只显示聚合值；把每条带子按 N 个 rank 堆叠会把同一段时间累加 N 次",
  "metrics.bundled.one": "另有 {count} 条 rank 曲线未标注",
  "metrics.bundled.other": "另有 {count} 条 rank 曲线未标注",
  "metrics.bundledNamed": "——标注出来的是极值和中位数",
  "metrics.singlePoint": "只有一个点——以标记点绘制",

  // -- Charts --------------------------------------------------------------
  "chart.empty": "该指标暂无数据",
  "chart.total": "合计",
  "chart.zoomReset": "已缩放 · 重置",
  "chart.mean": "均值",
  "chart.stacked": "堆叠",
  "chart.stackedTitle": "各序列以堆叠方式绘制",
  "chart.log": "对数",
  "chart.logNa": "对数不适用",
  "chart.logTitle": "对数坐标",
  "chart.logDroppedTitle": "模板要求对数坐标，但该任务存在零或负值，因此改用线性坐标",
  "chart.percentTitle": "以百分比显示",
  "chart.smoothed": "平滑 {n} 点",
  "chart.smoothedTitle":
    "对 {n} 个点做指数滑动平均，只影响显示，底层数值不变。平滑会削平并延迟尖峰，所以看异常前请先关掉它。",
  "chart.smoothing": "平滑",
  "chart.smoothingAria": "平滑窗口（点数）",
  "chart.smoothingOff": "关",
  "chart.smoothingPoints": "{n} 点",

  // -- Media ---------------------------------------------------------------
  "media.split": "数据划分",
  "split.all": "全部",
  "split.train": "训练",
  "split.eval": "评测",
  "media.allSteps": "全部",
  "media.clips.one": "{count} 段视频",
  "media.clips.other": "{count} 段视频",
  "media.tally": "{success}/{envs} 个环境成功",
  "media.tallyTitle": "对当前显示的视频求和",
  "media.unrecorded": "{count} 段未记录结果",
  "media.unrecordedTitle": "这些视频没有记录结果。它们被排除在统计之外，而不是算作失败。",
  "media.requestFailed": "视频请求失败",
  "media.emptyTitle": "该任务没有视频",
  "media.emptyBody":
    "视频由 env worker 写入分片索引。配置了 {code} 的任务，或录制步数还没轮到的任务，都没有视频。",
  "media.succeeded": "成功",
  "media.succeededTitle": "单环境视频：该 episode 达成了目标。",
  "media.notSucceeded": "未成功",
  "media.notSucceededTitle": "单环境视频：该 episode 没有达成目标。",
  "media.outcomeUnrecorded": "未记录结果",
  "media.outcomeUnrecordedTitle":
    "这段视频没有记录结果。这并不等于失败：环境可能根本没有成功的概念，也可能这段视频早于该字段存在。",
  "media.successCount": "{success}/{envs} 成功",
  "media.successCountTitle": "这段视频里平铺的 {envs} 个环境中，有 {success} 个达成了目标。",
  "media.playAria": "播放第 {step} {unit}的视频",
  "media.decodeFailed": "浏览器无法解码这段视频。",
  "media.noUrl": "服务端没有为这段视频返回 URL。",
  "media.seed": "种子 {seed}",
  "media.shard": "分片 {shard}",
  "media.path": "路径",

  // -- Events --------------------------------------------------------------
  "events.filter": "过滤",
  "events.all": "全部",
  "events.warnError": "警告 + 错误",
  "events.rangeEmpty": "0 / 共 {total}",
  "events.filteredFrom": "（从 {total} 条中筛选）",
  "events.problemCount": "{count} 条警告/错误",
  "events.exit": "退出信息",
  "events.logUnreadable": "事件日志不可读",
  "events.noneWarnTitle": "没有警告或错误",
  "events.noneWarnBody": "该任务的日志里没有警告，也没有错误。",
  "events.noneTitle": "没有事件",
  "events.noneBody":
    "该任务没有写入任何 events.jsonl 记录。这个文件由 runner 在阶段切换、保存检查点和评测时追加，" +
    "所以日志为空通常意味着任务还没走到这些点。",
  "events.truncatedTitle": "更早的事件未显示",
  "events.truncatedBody":
    "这份日志是最近 {limit} 条。在它们之前该任务还写过更多，包括启动那一段，都在任务根目录下的 {code} 里。",
  "events.checkpoints": "检查点",
  "events.best": "最佳",
  "events.col.saved": "保存时间",
  "events.col.size": "大小",
  "events.col.took": "耗时",
  "events.col.path": "路径",

  // -- Compare -------------------------------------------------------------
  "compare.title": "对比（{count}）",
  "compare.metric": "指标",
  "compare.pickMetric": "— 选择一个指标 —",
  "compare.inAllRuns": "{count} 个任务都有",
  "compare.inSomeRuns": "仅部分任务有",
  "compare.runs.one": "{count} 个任务",
  "compare.runs.other": "{count} 个任务",
  "compare.mixedTitle": "同一坐标轴上混了不同的步语义",
  "compare.mixedBody":
    "选中的这些任务对「一步是什么」并不一致（{counts}）。坐标轴标的是 {code}；" +
    "使用其他单位的任务以虚线绘制，它们的 x 值与其余任务不可比。",
  "compare.mixedCount": "{count} 个 {label}",
  "compare.mixedSeparator": "、",
  "compare.nothingTitle": "尚未选中任何任务",
  "compare.nothingBody": "先在任务列表里选中两个及以上的任务，再点{compare}。",
  "compare.noMetric": "未选择指标",
  "compare.col.run": "任务",
  "compare.col.state": "状态",
  "compare.col.step": "步数",
  "compare.col.latest": "最新",
  "compare.col.latestValue": "最新值",
  "compare.col.elapsed": "已用时",
  "compare.col.semantics": "步语义",
};
