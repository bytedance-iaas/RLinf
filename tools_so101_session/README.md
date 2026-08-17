# SO101 工具目录

**每个脚本干什么、要解决什么问题** → `../SO101_TOOLS_RUNBOOK_ZH.md`（按任务分类）。
**照流程复现** → `../SO101_PIPELINE_ZH.md`（阶段 A–G，每步都有命令）。

这里回答另一个问题：**这个脚本还能不能用。** 87 个脚本里有一半属于早期实验或已被证伪的尝试，混在一起会让人照着废弃的路子走。每个文件**头部第一行**都有 `STATUS:` 标记，打开就能看到；这张表是同一份信息的汇总。

```bash
head -1 tools_so101_session/<脚本>                  # 看单个
grep -l 'STATUS: REFUTED' tools_so101_session/*     # 列出所有被证伪的
```

| 状态 | 数量 | 能不能用 |
|---|---|---|
| `ACTIVE` | 26 | 照着 `SO101_PIPELINE_ZH.md` 复现时，用到的就是这些 |
| `TOOL` | 20 | 换机器人、换任务也还用得上 |
| `EVIDENCE` | 7 | 文档里某个结论的出处。不需要重跑，但别删 |
| `SUPERSEDED` | 26 | **不要照着复现**——任务定义已经不同了 |
| `REFUTED` | 8 | 留着是为了别人不再花机时走一遍 |

## 主线：当前流程在用（26）

照着 `SO101_PIPELINE_ZH.md` 复现时，用到的就是这些。

| 脚本 | 说明 |
|---|---|
| `collect_policy_successes.sh` | 阶段 D1/E1，采集策略自己的成功轨迹 |
| `convert_append_region.py` | 阶段 E3，在副本上追加环 1 数据 |
| `convert_cotrain_heldout.py` | 阶段 G，协同训练数据集第二轮（留出 70-86），交付所用 |
| `convert_cotrain_simreal.py` | 阶段 G，协同训练数据集第一轮（真机全 87 集） |
| `convert_expert_iter.py` | 阶段 D2，规划器 h5 + 策略 npz 混合 |
| `convert_fullboard.py` | 阶段 B3，全板示范 -> 数据集，血统统计量在此确立 |
| `convert_narrow_box.py` | 阶段 C2，窄框示范 -> 数据集 |
| `convert_rlinf_to_lerobot.py` | 部署：RLinf 检查点 -> LeRobot 格式 |
| `deploy_policy_server.py` | 部署备用路线：RLinf 直接起 websocket 服务 |
| `gen_demos_annulus.sh` | 阶段 E2，只在环形带补示范 |
| `gen_planner_demos.py` | 阶段 B/C/E 的示范生成器 |
| `night_cpu.sh` | 阶段 G 第二轮：CPU 轨，建留出集数据 |
| `night_gpu.sh` | 阶段 G 第二轮：GPU 轨，三项测量 + 正式训练 |
| `offline_replay_check.py` | 阶段 G 的 sim2real 门，也是上真机前的最后判据 |
| `pipeline_cotrain_simreal.sh` | 阶段 G 第一轮的整段编排 |
| `pipeline_expert_iteration.sh` | 阶段 D 的整段编排 |
| `pipeline_expert_iteration_resume.sh` | 阶段 D 的续跑版（原流水线在 S2 超时后接管） |
| `pipeline_fullboard.sh` | 阶段 B 的整段编排（S0 探针 -> S5 检查点清理） |
| `pipeline_narrow_box.sh` | 阶段 C 的整段编排 |
| `pipeline_region_expand.sh` | 阶段 E 的整段编排 |
| `ppo_freeze_probe.sh` | 阶段 F 启动前的先决条件探针（lr=1e-9 冻结测试） |
| `ppo_train.sh` | 阶段 F，跑通的那次 PPO 启动器（含自动停机守卫） |
| `verify_baseline_control.sh` | 阶段 F 验收的对照：起点同种子同块长 |
| `verify_deploy_horizon.sh` | 部署校验：服务端与离线门的动作块长一致性 |
| `verify_honest_seeds.sh` | 阶段 F 验收，未参与挑选的种子 |
| `verify_lerobot_export.py` | 部署：证明导出没有改变策略 |

## 通用工具：与阶段无关（20）

换机器人、换任务也还用得上。

| 脚本 | 说明 |
|---|---|
| `build_code_appendix.py` | 把脚本正文汇编成文档附录 |
| `calib_test.py` | 标定：逐关节符号候选的真实-仿真对比 |
| `calib_wflex.py` | 标定：wrist_flex 零位偏移扫描 |
| `diag_spawn.sh` | 生成位置与成败的相关性诊断 |
| `export_session_md.py` | 把会话记录导出成 Markdown |
| `measure_chunk_motion.py` | 测相邻动作块之间图像/关节的变化量 |
| `merge_datasets.py` | 合并两个 LeRobot 数据集 |
| `normstats_sub.py` | 子采样版 norm_stats 计算（大数据集上更快） |
| `parse_eval_series.py` | 从训练日志里解析确定性评测序列 |
| `planner_ab.sh` | 规划器改动的 A/B（在它最差的两条带上测） |
| `probe_grasp_only.py` | 只抓取不放置的简化探针，用来隔离失败环节 |
| `render_ready.py` | 渲染当前场景，看资产/位姿对不对 |
| `render_scene_grid.py` | 渲染策略实际看到的画面 —— 找相机缺陷就靠它 |
| `render_so101.py` | 并排渲染两路策略输入 |
| `replay_demo.py` | 把真机录制的动作重放进仿真，逐帧对比 |
| `check_doc_consistency.py` | 把文档里的每个名字按类别对到仓库/磁盘上，并双向核对产物表。人眼复核漏过的表格不一致，它一次查出 |
| `so101_smoke.py` | 冒烟自检：任务注册、注册表条目、环境可用性 |
| `supervisor_v2.sh` | 训练监工：按故障类型分流诊断，而不是一律重启 |
| `verify_calib.py` | 标定验证：烘焙后的参数是否对得上真机 |
| `wrist_remount.py` | 标定改动后重新扫描腕部相机位姿 |
| `wrist_sweep.py` | 腕部相机安装位姿的 3x3 网格扫描 |

## 证据：一次性对照实验（7）

文档里某个结论的出处。不需要重跑，但别删。

| 脚本 | 说明 |
|---|---|
| `bisect_checkpoint_vs_env.sh` | 二分定位：成绩变化来自检查点还是环境 |
| `bisect_rl_settings.sh` | 二分定位：独立评测 57.8% 与 RL 内评测 0.0% 的差异来自哪个设置 |
| `chunk_ab.sh` | 动作块长 5 vs 10 的对照。⚠️ 两个臂当时都是 10（手搓 config 不继承 YAML 插值），结论无效 |
| `ppo_param_search.sh` | PPO 参数探索第二轮 —— 阶段 F 的三个参数由它得出 |
| `verify_env_drift.sh` | 证明环境未漂移：v4 原检查点重测得到与两天前一致的 12.5% |
| `verify_standalone_retry.sh` | 阶段 C 验收的独立重跑（Ray worker 猝死后补测） |
| `verify_warmstart_substitution.sh` | 阶段 B 换热启动来源的对照（10.2% vs 12.5%，阈值内） |

## 已被取代：早期任务规格（26）

**不要照着复现**——任务定义已经不同了。

| 脚本 | 说明 |
|---|---|
| `convert_demos_to_lerobot.py` | 最早的通用转换器，已按阶段拆成四个 |
| `convert_rollouts_early_pp5.py` | pp 时代 rollout 转换 |
| `convert_rollouts_early_pp6.py` | pp 时代 rollout 转换 |
| `convert_rollouts_early_pp7.py` | pp 时代 rollout 转换 |
| `dl_official_sft.sh` | 下载官方 ManiSkill SFT 检查点，该任务线已停 |
| `eval_realdata_sft_in_sim.sh` | 真机 SFT 在仿真里的评测（结论已并入阶段 A） |
| `eval_simdemo_sft.sh` | 上者的评测 |
| `eval_single_ckpt_750.sh` | 早期 RL-750 检查点评测 |
| `eval_single_ckpt_early.sh` | pp 时代单检查点评测 |
| `gen_demos_narrow_box.sh` | 阶段 C 生成的旧启动器，现直接调 gen_planner_demos.py |
| `offline_check_nowrist.sh` | 同上，--no-wrist 变体 |
| `offline_check_run.sh` | 离线检验的早期封装，现直接调 offline_replay_check.py |
| `pipeline_early_phase2.sh` | pp 时代的整段流水线 |
| `pipeline_early_v3.sh` | v3 时代流水线，被阶段 B 取代 |
| `pipeline_early_v3b.sh` | v3 第二轮 |
| `pipeline_early_v3c.sh` | v3 第三轮 |
| `pipeline_early_v3d.sh` | v3d 新血统重训 —— 重算统计量导致 19.5%->9.4%，血统冻结那条规则的来源 |
| `pipeline_gate_resume.sh` | v3d 门评的断点续跑 |
| `ppo_early_v6.sh` | v6 PPO：起点带噪成功率 0.5%，从未起来 |
| `ppo_from_realdata_sft.sh` | 从真机 SFT 直接起 PPO，先决条件不满足 |
| `ppo_resume_early.sh` | 从早期 RL-750 检查点续跑，夹爪已塌缩 |
| `ppo_train_official_recipe.sh` | v11 官方配方首跑：先决条件不满足（带噪 1.0%），失败 |
| `repro_eval.sh` | 复现论文基准数字，该任务线已停 |
| `sft_early_pp.sh` | pp 时代 SFT，任务规格已废弃 |
| `sft_simdemo_standalone.sh` | 独立的仿真示范 SFT，被阶段 B 取代 |
| `supervisor_early_v6.sh` | v6 的监工，被 supervisor_v2.sh 取代 |

## 已被证伪：试过，不行（8）

留着是为了别人不再花机时走一遍。

| 脚本 | 说明 |
|---|---|
| `convert_band_curriculum_refuted.py` | 分带课程学习的数据转换 |
| `gen_planner_demos_finegrid_rejected.py` | 更细的 IK 网格 —— 不是放置误差的根因 |
| `pipeline_band_curriculum_orchestrator.sh` | 分带课程学习的两阶段编排 |
| `pipeline_band_curriculum_refuted.sh` | 分带课程学习：先中带再左带 —— 没有带来提升 |
| `pipeline_selfdistill_refuted.sh` | 纯自蒸馏（真机+仿真混合 SFT）——策略越练越窄，掉 53 点 |
| `ppo_noise_sweep_inert_knob.sh` | 扫 noise_params —— 该参数对 flow_noise 根本不生效，8 组全无变化 |
| `ppo_onlyeval_probe_deprecated.sh` | 用 only_eval=True 当探针 —— 它同时改了三处行为，不是同一条代码路径 |
| `ppo_param_search_inert_knob.sh` | 第一轮 PPO 参数搜索，建立在上面那个失效旋钮上 |

---

## 为什么改名

原来的文件名带着内部实验版本号（`v4`/`v8`/`v9`/`v10`/`pp*`），那是这轮实验的流水编号，对读者没有信息量：
看到 `convert_v8_demos.py` 不知道它转的是什么，也不知道它和 `convert_v9_demos.py` 有什么区别。
现在按**功能**命名。

**没有改的是数据集名、注册表条目名（`pi05_so101_v4` 等）和结果目录名**——它们是磁盘上的真实路径，
而且被写进了已有检查点内部（openpi 按 `<model_path>/<repo_id>/` 查找 norm_stats），改名会让现有产物全部失效。

## 新旧对照（读历史日志时用）

| 旧名 | 新名 |
|---|---|
| `bisect2.sh` | `bisect_rl_settings.sh` |
| `bisect_v10.sh` | `bisect_checkpoint_vs_env.sh` |
| `control_v4_rerun.sh` | `verify_env_drift.sh` |
| `convert_pp5_rollouts.py` | `convert_rollouts_early_pp5.py` |
| `convert_pp6_rollouts.py` | `convert_rollouts_early_pp6.py` |
| `convert_pp7_rollouts.py` | `convert_rollouts_early_pp7.py` |
| `convert_v10_demos.py` | `convert_append_region.py` |
| `convert_v14_cotrain.py` | `convert_cotrain_simreal.py` |
| `convert_v4_demos.py` | `convert_fullboard.py` |
| `convert_v7_demos.py` | `convert_band_curriculum_refuted.py` |
| `convert_v8_demos.py` | `convert_narrow_box.py` |
| `convert_v9_demos.py` | `convert_expert_iter.py` |
| `cotrain_v14.sh` | `pipeline_cotrain_simreal.sh` |
| `eval750_fixed.sh` | `eval_single_ckpt_750.sh` |
| `eval_pp.sh` | `eval_single_ckpt_early.sh` |
| `eval_sft_fixed.sh` | `eval_realdata_sft_in_sim.sh` |
| `eval_simsft.sh` | `eval_simdemo_sft.sh` |
| `freeze_v11.sh` | `ppo_freeze_probe.sh` |
| `gen_pickonly.py` | `probe_grasp_only.py` |
| `gen_so101_demos.py` | `gen_planner_demos.py` |
| `gen_so101_demos_v2.py` | `gen_planner_demos_finegrid_rejected.py` |
| `gen_v8_legacy.sh` | `gen_demos_narrow_box.sh` |
| `merge_mix_v5.py` | `merge_datasets.py` |
| `noise_sweep.sh` | `ppo_noise_sweep_inert_knob.sh` |
| `offline_check.sh` | `offline_check_run.sh` |
| `onlyeval_v11.sh` | `ppo_onlyeval_probe_deprecated.sh` |
| `overnight_v3.sh` | `pipeline_early_v3.sh` |
| `overnight_v3b.sh` | `pipeline_early_v3b.sh` |
| `overnight_v3c.sh` | `pipeline_early_v3c.sh` |
| `overnight_v6_supervisor.sh` | `supervisor_early_v6.sh` |
| `phase2_pipeline.sh` | `pipeline_early_phase2.sh` |
| `ppo_night.sh` | `ppo_param_search_inert_knob.sh` |
| `ppo_night2.sh` | `ppo_param_search.sh` |
| `rl_resume750.sh` | `ppo_resume_early.sh` |
| `rl_sft_restart.sh` | `ppo_from_realdata_sft.sh` |
| `rl_v11.sh` | `ppo_train_official_recipe.sh` |
| `rl_v13.sh` | `ppo_train.sh` |
| `rl_v6.sh` | `ppo_early_v6.sh` |
| `sft_pp.sh` | `sft_early_pp.sh` |
| `sft_sim.sh` | `sft_simdemo_standalone.sh` |
| `v10_collect.sh` | `collect_policy_successes.sh` |
| `v10_gen.sh` | `gen_demos_annulus.sh` |
| `v10_rest.sh` | `pipeline_region_expand.sh` |
| `v13_baseline.sh` | `verify_baseline_control.sh` |
| `v13_verify.sh` | `verify_honest_seeds.sh` |
| `v3d_gate_resume.sh` | `pipeline_gate_resume.sh` |
| `v3d_pipeline.sh` | `pipeline_early_v3d.sh` |
| `v4_pipeline.sh` | `pipeline_fullboard.sh` |
| `v4b_verify.sh` | `verify_warmstart_substitution.sh` |
| `v5_pipeline.sh` | `pipeline_selfdistill_refuted.sh` |
| `v7_curriculum.sh` | `pipeline_band_curriculum_refuted.sh` |
| `v7_orchestrator.sh` | `pipeline_band_curriculum_orchestrator.sh` |
| `v8_pipeline.sh` | `pipeline_narrow_box.sh` |
| `v8_verify.sh` | `verify_standalone_retry.sh` |
| `v9_expert_iter.sh` | `pipeline_expert_iteration.sh` |
| `v9_rest.sh` | `pipeline_expert_iteration_resume.sh` |
