from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from chemberta.data.molnet import TASKS, load_task_splits, to_arrays
from chemberta.training.trainer import train_task
from chemberta.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    task_name = cfg.task.name
    spec = TASKS[task_name]

    _, train, valid, test = load_task_splits(task_name)
    result = train_task(cfg, spec, to_arrays(train), to_arrays(valid), to_arrays(test))

    out = {
        "task": task_name,
        "type": spec.kind,
        "best_val": result.best_val,
        "test_metric": result.test_metric,
    }
    Path("metrics.yaml").write_text(OmegaConf.to_yaml(out), encoding="utf-8")
    print(OmegaConf.to_yaml(out))


if __name__ == "__main__":
    main()
