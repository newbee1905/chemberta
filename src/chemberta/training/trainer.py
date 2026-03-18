from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from chemberta.optim.muon_adamw import MuonAdamW
from chemberta.training.modeling import ChemBertaForDownstream


class SmilesDataset(Dataset):
    def __init__(self, smiles: list[str], y: np.ndarray, w: np.ndarray):
        self.smiles = smiles
        self.y = y
        self.w = w

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx: int):
        return self.smiles[idx], self.y[idx], self.w[idx]


@dataclass
class TrainResult:
    best_val: float
    test_metric: float


def _build_optimizer(model: nn.Module, cfg) -> MuonAdamW:
    matrix_params_by_shape: dict[tuple[int, ...], list[torch.nn.Parameter]] = {}
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Keep potentially unused submodules (e.g. pooler layers in encoder-only usage)
        # on AdamW, because Muon expects all params in a group to have gradients.
        if "pooler" in name:
            other_params.append(p)
        elif p.ndim == 2 and p.shape[0] > 1 and p.shape[1] > 1:
            matrix_params_by_shape.setdefault(tuple(p.shape), []).append(p)
        else:
            other_params.append(p)

    groups = [
        {
            "params": other_params,
            "kind": "adamw",
            "lr": cfg.optimizer.adamw.lr,
            "betas": tuple(cfg.optimizer.adamw.betas),
            "eps": cfg.optimizer.adamw.eps,
            "weight_decay": cfg.optimizer.adamw.weight_decay,
        }
    ]
    for _, shape_group in sorted(matrix_params_by_shape.items(), key=lambda kv: kv[0]):
        groups.append(
            {
                "params": shape_group,
                "kind": "muon",
                "lr": cfg.optimizer.muon.lr,
                "momentum": cfg.optimizer.muon.momentum,
                "ns_steps": cfg.optimizer.muon.ns_steps,
                "beta2": cfg.optimizer.muon.beta2,
                "weight_decay": cfg.optimizer.muon.weight_decay,
            }
        )
    return MuonAdamW(groups)


def _collate(tokenizer, max_length: int):
    def fn(batch):
        smiles, y, w = zip(*batch)
        tok = tokenizer(
            list(smiles),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return tok["input_ids"], tok["attention_mask"], torch.tensor(np.stack(y)), torch.tensor(np.stack(w))

    return fn


def _loss_fn(task_type: str, logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor):
    if task_type == "classification":
        loss_raw = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    else:
        loss_raw = nn.functional.mse_loss(logits, labels, reduction="none")
    masked = loss_raw * weights
    return masked.sum() / weights.sum().clamp_min(1.0)


def _evaluate(task_type: str, model, loader, device):
    model.eval()
    ys, ws, ps = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, y, w in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            ys.append(y.numpy())
            ws.append(w.numpy())
            ps.append(logits.cpu().numpy())

    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    p = np.concatenate(ps, axis=0)

    if task_type == "classification":
        scores = []
        probs = 1.0 / (1.0 + np.exp(-p))
        for i in range(y.shape[1]):
            m = w[:, i] > 0
            if m.sum() < 2 or np.unique(y[m, i]).shape[0] < 2:
                continue
            scores.append(roc_auc_score(y[m, i], probs[m, i]))
        return float(np.mean(scores)) if scores else float("nan")

    rmse = np.sqrt(((p - y) ** 2 * w).sum() / w.sum().clip(min=1.0))
    return float(rmse)


def train_task(cfg, task_spec, train_data, val_data, test_data) -> TrainResult:
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    tokenizer_name = cfg.model.get("tokenizer_name") or cfg.model.name
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load tokenizer/model config from '{tokenizer_name}'. "
            "Check the Hugging Face model id in config (e.g. DeepChem/ChemBERTa-100M-MLM, "
            "DeepChem/ChemBERTa-77M-MTR, DeepChem/ChemBERTa-77M-MLM)."
        ) from exc

    train_ds = SmilesDataset(*train_data)
    val_ds = SmilesDataset(*val_data)
    test_ds = SmilesDataset(*test_data)

    collate = _collate(tokenizer, cfg.model.max_length)
    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.training.num_workers, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=cfg.training.num_workers, collate_fn=collate)

    model = ChemBertaForDownstream(cfg.model.name, task_spec.num_labels, cfg.model.dropout).to(device)
    optimizer = _build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.training.amp and device.type == "cuda"))

    higher_is_better = task_spec.kind == "classification"
    best_val = -float("inf") if higher_is_better else float("inf")
    best_state = None

    for _ in range(cfg.training.epochs):
        model.train()
        for input_ids, attention_mask, y, w in tqdm(train_loader, desc=f"train/{task_spec.name}"):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            w = w.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(cfg.training.amp and device.type == "cuda")):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = _loss_fn(task_spec.kind, logits, y, w)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        val_metric = _evaluate(task_spec.kind, model, val_loader, device)
        improved = val_metric > best_val if higher_is_better else val_metric < best_val
        if improved:
            best_val = val_metric
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metric = _evaluate(task_spec.kind, model, test_loader, device)
    return TrainResult(best_val=best_val, test_metric=test_metric)
