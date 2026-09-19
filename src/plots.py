"""实验可视化：数据集样例、训练曲线、消融对比、混淆矩阵、特征图与 case study。"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

from src.models import CLASS_NAMES, LeNetFamily


FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
]
_CJK_NAME_KEYS = (
    "noto sans cjk",
    "source han sans",
    "simhei",
    "microsoft yahei",
    "pingfang",
    "wenquanyi",
    "droid sans fallback",
)


def _font() -> FontProperties | None:
    for p in FONT_CANDIDATES:
        if Path(p).exists():
            return FontProperties(fname=p)
    for font in fm.fontManager.ttflist:
        if any(k in font.name.lower() for k in _CJK_NAME_KEYS):
            return FontProperties(fname=font.fname)
    return None


FP = _font()


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.unicode_minus": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _set_title(ax, text: str) -> None:
    ax.set_title(text, fontproperties=FP, fontsize=13, pad=8)


def _set_xlabel(ax, text: str) -> None:
    ax.set_xlabel(text, fontproperties=FP)


def _set_ylabel(ax, text: str) -> None:
    ax.set_ylabel(text, fontproperties=FP)


def _set_legend(ax) -> None:
    ax.legend(prop=FP, frameon=False)


def _xticks(ax, labels, rotation=0) -> None:
    ax.set_xticklabels(labels, fontproperties=FP, rotation=rotation)


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dataset_samples(raw_dataset, path: Path, n_per_class: int = 6) -> None:
    _apply_style()
    buckets: dict[int, list] = {i: [] for i in range(10)}
    for img, label in raw_dataset:
        if len(buckets[label]) < n_per_class:
            buckets[label].append(img.squeeze().numpy())
        if all(len(v) >= n_per_class for v in buckets.values()):
            break

    fig, axes = plt.subplots(10, n_per_class, figsize=(n_per_class * 1.15, 11.5))
    for c in range(10):
        for j in range(n_per_class):
            ax = axes[c, j]
            ax.imshow(buckets[c][j], cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 0:
                ax.set_ylabel(CLASS_NAMES[c], fontproperties=FP, fontsize=10)
    fig.suptitle("Fashion-MNIST 各类别样例", fontproperties=FP, fontsize=14, y=0.995)
    savefig(fig, path)


def plot_architecture(path: Path) -> None:
    _apply_style()
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.2))
    classic = [
        ("输入\n1×32×32", "#d9d9d9"),
        ("C1 Conv\n6×5×5", "#9ecae1"),
        ("tanh", "#c6dbef"),
        ("S2 AvgPool", "#9ecae1"),
        ("C3 Conv\n16×5×5", "#9ecae1"),
        ("tanh", "#c6dbef"),
        ("S4 AvgPool", "#9ecae1"),
        ("FC 120\ntanh", "#fdae6b"),
        ("FC 84\ntanh", "#fdae6b"),
        ("输出 10", "#74c476"),
    ]
    improved = [
        ("输入\n1×32×32", "#d9d9d9"),
        ("C1+BN\n6×5×5", "#3182bd"),
        ("ReLU", "#6baed6"),
        ("MaxPool\n+Dropout", "#3182bd"),
        ("C3+BN\n16×5×5", "#3182bd"),
        ("ReLU", "#6baed6"),
        ("MaxPool\n+Dropout", "#3182bd"),
        ("FC 120\nReLU+Drop", "#e6550d"),
        ("FC 84\nReLU", "#e6550d"),
        ("输出 10", "#31a354"),
    ]

    for ax, blocks, title in zip(
        axes,
        [classic, improved],
        ["经典 LeNet-5（对照）", "改进 LeNet-5（本实验主模型，三处改动已标色）"],
    ):
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 3.2)
        ax.axis("off")
        _set_title(ax, title)
        for i, (text, color) in enumerate(blocks):
            x = 0.25 + i * 1.27
            box = FancyBboxPatch(
                (x, 1.0),
                1.12,
                1.35,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=color,
                edgecolor="#333333",
                linewidth=1.0,
            )
            ax.add_patch(box)
            ax.text(
                x + 0.56,
                1.67,
                text,
                ha="center",
                va="center",
                fontsize=8,
                fontproperties=FP,
            )
            if i < len(blocks) - 1:
                ax.annotate(
                    "",
                    xy=(x + 1.18, 1.67),
                    xytext=(x + 1.24, 1.67),
                    arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2),
                )
    savefig(fig, path)


def plot_curves(all_results: dict, path: Path) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for name, res in all_results.items():
        h = res["history"]
        axes[0].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label=name, lw=1.8)
        axes[1].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=name, lw=1.8)
    _set_xlabel(axes[0], "Epoch")
    _set_ylabel(axes[0], "验证集 Loss")
    _set_title(axes[0], "验证损失曲线")
    _set_legend(axes[0])
    _set_xlabel(axes[1], "Epoch")
    _set_ylabel(axes[1], "验证集 Accuracy")
    _set_title(axes[1], "验证准确率曲线")
    _set_legend(axes[1])
    savefig(fig, path)


def plot_group_curves(all_results: dict, names: list[str], title: str, path: Path) -> None:
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for name in names:
        h = all_results[name]["history"]
        axes[0].plot(range(1, len(h["train_acc"]) + 1), h["train_acc"], label=f"{name}-train", lw=1.6, ls="--")
        axes[0].plot(range(1, len(h["val_acc"]) + 1), h["val_acc"], label=f"{name}-val", lw=1.8)
        axes[1].plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], label=name, lw=1.8)
    _set_title(axes[0], f"{title}：训练 / 验证准确率")
    _set_xlabel(axes[0], "Epoch")
    _set_ylabel(axes[0], "Accuracy")
    _set_legend(axes[0])
    _set_title(axes[1], f"{title}：验证损失")
    _set_xlabel(axes[1], "Epoch")
    _set_ylabel(axes[1], "Loss")
    _set_legend(axes[1])
    savefig(fig, path)


def plot_bar_comparison(items: list[tuple[str, float, float]], ylabel: str, title: str, path: Path) -> None:
    _apply_style()
    labels = [x[0] for x in items]
    acc = [x[1] for x in items]
    f1 = [x[2] for x in items]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(7.5, 1.4 * len(labels)), 4.8))
    b1 = ax.bar(x - width / 2, acc, width, label="Test Acc", color="#3182bd")
    b2 = ax.bar(x + width / 2, f1, width, label="Macro-F1", color="#e6550d")
    ax.set_xticks(x)
    _xticks(ax, labels)
    _set_ylabel(ax, ylabel)
    _set_title(ax, title)
    ax.set_ylim(0.75, 1.0)
    ax.legend(prop=FP, frameon=False)
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(
                f"{h:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    savefig(fig, path)


def plot_confusion(cm: np.ndarray, path: Path, title: str = "混淆矩阵（测试集）") -> None:
    _apply_style()
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, fontproperties=FP, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES, fontproperties=FP)
    _set_xlabel(ax, "预测类别")
    _set_ylabel(ax, "真实类别")
    _set_title(ax, title)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    ax.grid(False)
    savefig(fig, path)


def plot_feature_maps(model: LeNetFamily, images: torch.Tensor, labels: torch.Tensor, path: Path) -> None:
    _apply_style()
    model.eval()
    with torch.no_grad():
        feats = model.conv1_feature(images).cpu()
    n = min(4, images.size(0))
    fig, axes = plt.subplots(n, 7, figsize=(10.5, 1.7 * n + 0.8))
    if n == 1:
        axes = np.expand_dims(axes, 0)
    for i in range(n):
        axes[i, 0].imshow(images[i, 0].cpu(), cmap="gray")
        axes[i, 0].axis("off")
        axes[i, 0].set_title(
            f"输入·{CLASS_NAMES[int(labels[i])]}", fontproperties=FP, fontsize=9
        )
        for c in range(6):
            axes[i, c + 1].imshow(feats[i, c], cmap="viridis")
            axes[i, c + 1].axis("off")
            if i == 0:
                axes[i, c + 1].set_title(f"C1-{c}", fontsize=9)
    fig.suptitle("第一层卷积特征图（中间输出）", fontproperties=FP, fontsize=13)
    savefig(fig, path)


def plot_case_study(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    path: Path,
    title: str,
) -> None:
    _apply_style()
    n = images.size(0)
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.85 * rows + 0.6))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        ax.axis("off")
        if k >= n:
            continue
        ax.imshow(images[k, 0].cpu(), cmap="gray")
        y = int(labels[k])
        p = int(preds[k])
        ok = y == p
        color = "#2ca02c" if ok else "#d62728"
        ax.set_title(
            f"真:{CLASS_NAMES[y]}\n预:{CLASS_NAMES[p]}",
            fontproperties=FP,
            fontsize=8,
            color=color,
        )
    fig.suptitle(title, fontproperties=FP, fontsize=13)
    savefig(fig, path)


def plot_training_log_panel(log_text: str, path: Path) -> None:
    """把训练日志排成报告可用的“运行截屏”。"""
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.02),
            0.96,
            0.96,
            boxstyle="round,pad=0.01",
            facecolor="#111111",
            edgecolor="#333333",
        )
    )
    ax.text(
        0.04,
        0.96,
        log_text,
        va="top",
        ha="left",
        fontproperties=FP,
        fontsize=8.2,
        color="#e6e6e6",
        wrap=True,
        transform=ax.transAxes,
    )
    savefig(fig, path)
