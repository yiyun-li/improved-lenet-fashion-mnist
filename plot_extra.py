#!/usr/bin/env python3
"""补充：最佳模型混淆矩阵、逐类 F1、经典 vs 最佳配置对比。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.models import CLASS_NAMES
import matplotlib.pyplot as plt
from src.plots import (
    FP,
    _apply_style,
    _set_title,
    _set_ylabel,
    _xticks,
    plot_bar_comparison,
    plot_confusion,
    savefig,
)


def main() -> None:
    out = ROOT / "results"
    fig_dir = out / "figures"
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    best = "改进-ReLU-d0.0-Adam"
    classic = "经典LeNet"
    main_name = "改进-ReLU-d0.3-Adam"

    plot_confusion(
        np.array(metrics[best]["confusion_matrix"]),
        fig_dir / "fig17_confusion_best.png",
        "最佳配置（ReLU + BN + MaxPool + Adam，无 Dropout）测试集混淆矩阵",
    )
    plot_bar_comparison(
        [
            (classic, metrics[classic]["test_acc"], metrics[classic]["test_macro_f1"]),
            ("改进-三处改动\n(含 Dropout=0.3)", metrics[main_name]["test_acc"], metrics[main_name]["test_macro_f1"]),
            ("改进-最佳\n(Dropout=0)", metrics[best]["test_acc"], metrics[best]["test_macro_f1"]),
        ],
        "测试集指标",
        "经典 LeNet-5 与两档改进配置的测试性能",
        fig_dir / "fig18_classic_vs_best.png",
    )

    reports = {}
    for name in (classic, main_name, best):
        pred = json.loads((out / "logs" / f"{name}_preds.json").read_text(encoding="utf-8"))
        from sklearn.metrics import f1_score

        y_true = np.array(pred["y_true"])
        y_pred = np.array(pred["y_pred"])
        reports[name] = f1_score(y_true, y_pred, average=None)

    _apply_style()
    x = np.arange(10)
    width = 0.25
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.bar(x - width, reports[classic], width, label="经典 LeNet-5", color="#9ecae1")
    ax.bar(x, reports[main_name], width, label="改进 Dropout=0.3", color="#3182bd")
    ax.bar(x + width, reports[best], width, label="改进 Dropout=0（最佳）", color="#e6550d")
    ax.set_xticks(x)
    _xticks(ax, CLASS_NAMES, rotation=30)
    _set_ylabel(ax, "各类别 F1")
    _set_title(ax, "逐类 F1：上衣类（衬衫/外套/套头衫）是主要误差来源")
    ax.set_ylim(0.55, 1.02)
    ax.legend(prop=FP, frameon=False, ncol=3, loc="lower right")
    savefig(fig, fig_dir / "fig19_per_class_f1.png")
    print("extra figures written")


if __name__ == "__main__":
    main()
