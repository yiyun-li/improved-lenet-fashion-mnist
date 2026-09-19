"""训练、评估与指标统计。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float):
    key = name.lower()
    if key == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
        )
    if key == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if key == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"未知优化器: {name}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion):
    model.eval()
    total_loss = 0.0
    n_correct = 0
    n_total = 0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    if n_total == 0:
        raise RuntimeError("evaluation loader is empty")
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = n_correct / n_total
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return {
        "loss": total_loss / n_total,
        "acc": acc,
        "macro_f1": macro_f1,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    epochs: int,
    ckpt_path: Path,
    class_names: list[str],
) -> dict[str, Any]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(optimizer_name, model, lr, weight_decay)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
    }

    best_val_acc = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_correct = 0
        n_total = 0
        epoch_t0 = time.time()

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            n_correct += (preds == labels).sum().item()
            n_total += labels.size(0)

        train_loss = running_loss / n_total
        train_acc = n_correct / n_total
        val_stats = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_stats["loss"])
        history["val_acc"].append(val_stats["acc"])
        history["val_f1"].append(val_stats["macro_f1"])

        improved = val_stats["acc"] > best_val_acc
        if improved:
            best_val_acc = val_stats["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"  epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val_loss={val_stats['loss']:.4f} acc={val_stats['acc']:.4f}  "
            f"f1={val_stats['macro_f1']:.4f}  "
            f"{time.time() - epoch_t0:.1f}s"
            + ("  *" if improved else "")
        )

    if best_state is not None:
        model.load_state_dict(best_state)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_path)

    test_stats = evaluate(model, test_loader, device, criterion)
    report = classification_report(
        test_stats["y_true"],
        test_stats["y_pred"],
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(test_stats["y_true"], test_stats["y_pred"])

    result = {
        "best_val_acc": best_val_acc,
        "test_acc": float(test_stats["acc"]),
        "test_macro_f1": float(test_stats["macro_f1"]),
        "test_loss": float(test_stats["loss"]),
        "history": history,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "y_true": test_stats["y_true"].tolist(),
        "y_pred": test_stats["y_pred"].tolist(),
        "seconds": time.time() - t0,
        "ckpt": str(ckpt_path),
    }
    print(
        f"  >> test_acc={result['test_acc']:.4f}  "
        f"test_f1={result['test_macro_f1']:.4f}  "
        f"time={result['seconds']:.1f}s"
    )
    return result


def dump_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
