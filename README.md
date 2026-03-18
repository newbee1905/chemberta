# ChemBERTa-v3 Downstream Finetuning (Hydra)

This project loads a pretrained ChemBERTa model and finetunes it across the downstream tasks listed in the ChemBERTa-3 benchmarking workflow:

- Classification: `bace_c`, `bbbp`, `clintox`, `hiv`, `tox21`, `sider`
- Regression: `esol`, `bace_r`, `lipo`, `freesolv`, `clearance`

## Requirements

- Python **3.12**
- PyTorch **2.6**

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run all tasks

```bash
python -m chemberta.training.run_all
```

## Run a single task

```bash
python -m chemberta.training.train task=bbbp
```

## Override optimizer/model settings

```bash
python -m chemberta.training.train \
  task=tox21 \
  training.batch_size=16 \
  training.epochs=10 \
  model.name=DeepChem/ChemBERTa-MLM-100M
```

Outputs (checkpoints + metrics) are written under Hydra's run directory (`outputs/...`).
