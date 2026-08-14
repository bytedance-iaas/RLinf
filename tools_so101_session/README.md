# SO101 工具目录

**用途索引在 `../SO101_TOOLS_RUNBOOK_ZH.md`** —— 每个脚本干什么、要解决什么问题，按任务分类。
这里只保留一份新旧文件名对照表。

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
