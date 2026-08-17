# STATUS: TOOL — 通用工具，与具体阶段无关。 子采样版 norm_stats 计算（大数据集上更快）
"""Subsampled norm-stats for pi05_so101 (state+actions only).

Reuses toolkits.lerobot.calculate_norm_stats.create_torch_dataloader (so its
module-level RemoveStrings transform is importable/picklable by DataLoader
spawn workers) but passes max_frames to subsample -- norm stats over a
representative subset are statistically ~identical for an 87-episode dataset.
"""
import numpy as np
import tqdm
from openpi.shared import normalize
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from toolkits.lerobot.calculate_norm_stats import create_torch_dataloader

REPO = "henry-guo/so101-pick-place-v2"
MAX_FRAMES = 15000


def main():
    config = get_openpi_config("pi05_so101", repo_id=REPO)
    data_config = config.data.create(config.assets_dirs, config.model)
    print(f"batch_size={config.batch_size} action_horizon={config.model.action_horizon} "
          f"num_workers={config.num_workers} max_frames={MAX_FRAMES}", flush=True)

    data_loader, num_batches = create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        max_frames=MAX_FRAMES,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
    print(f"WROTE_NORM_STATS {output_path}", flush=True)


if __name__ == "__main__":
    main()
