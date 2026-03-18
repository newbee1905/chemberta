from dataclasses import dataclass
from typing import Callable

import deepchem as dc
import numpy as np


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    num_labels: int


TASKS: dict[str, TaskSpec] = {
    "bace_c": TaskSpec("bace_c", "classification", 1),
    "bbbp": TaskSpec("bbbp", "classification", 1),
    "clintox": TaskSpec("clintox", "classification", 2),
    "hiv": TaskSpec("hiv", "classification", 1),
    "tox21": TaskSpec("tox21", "classification", 12),
    "sider": TaskSpec("sider", "classification", 27),
    "esol": TaskSpec("esol", "regression", 1),
    "bace_r": TaskSpec("bace_r", "regression", 1),
    "lipo": TaskSpec("lipo", "regression", 1),
    "freesolv": TaskSpec("freesolv", "regression", 1),
    "clearance": TaskSpec("clearance", "regression", 1),
}


def _loader_map() -> dict[str, Callable]:
    return {
        "bace_c": dc.molnet.load_bace_classification,
        "bbbp": dc.molnet.load_bbbp,
        "clintox": dc.molnet.load_clintox,
        "hiv": dc.molnet.load_hiv,
        "tox21": dc.molnet.load_tox21,
        "sider": dc.molnet.load_sider,
        "esol": dc.molnet.load_delaney,
        "bace_r": dc.molnet.load_bace_regression,
        "lipo": dc.molnet.load_lipo,
        "freesolv": dc.molnet.load_freesolv,
        "clearance": dc.molnet.load_clearance,
    }


def load_task_splits(task_name: str):
    loader = _loader_map()[task_name]
    tasks, datasets, _ = loader(
        featurizer=dc.feat.DummyFeaturizer(),
        splitter="scaffold",
        transformers=[],
    )
    train, valid, test = datasets
    return tasks, train, valid, test


def to_arrays(dataset):
    smiles = np.asarray(dataset.ids).tolist()
    y = dataset.y.astype(np.float32)
    w = dataset.w.astype(np.float32)
    return smiles, y, w
