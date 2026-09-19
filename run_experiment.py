#!/usr/bin/env python3
"""实验三主入口：改进 LeNet-5 + Fashion-MNIST，含激活函数 / Dropout / 优化器消融。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders
from src.engine import dump_json, set_seed, train_one_run
from src.models import (
    CLASS_NAMES,
    build_classic_lenet,
    build_improved_lenet,
    count_parameters,
)
from src.plots import (
    plot_architecture,
    plot_bar_comparison,
    plot_case_study,
    plot_confusion,
    plot_curves,
    plot_dataset_samples,
    plot_feature_maps,
    plot_group_curves,
    plot_training_log_panel,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=str(ROOT / "data"))
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader workers；默认 GPU 为 2、CPU 为 0",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default="auto",
        help="auto / cpu / cuda / cuda:0 ...；auto 在有 CUDA 时用 GPU",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out = Path(args.out_dir)
    fig_dir = out / "figures"
    ckpt_dir = out / "checkpoints"
    log_dir = out / "logs"
    for d in (fig_dir, ckpt_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，回退到 CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if args.num_workers is None:
        args.num_workers = 2 if device.type == "cuda" else 0
    print(f"device = {device}")
    if device.type == "cuda":
        print(f"GPU = {torch.cuda.get_device_name(device)}")

    train_loader, val_loader, test_loader, raw_train = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(
        f"train={len(train_loader.dataset)}  val={len(val_loader.dataset)}  "
        f"test={len(test_loader.dataset)}"
    )

    # 形状自检：输入 1×32×32，输出 10 类 logits
    _probe = build_improved_lenet("relu", 0.0)
    _y = _probe(torch.zeros(2, 1, 32, 32))
    if tuple(_y.shape) != (2, 10):
        raise RuntimeError(f"unexpected logits shape: {_y.shape}")
    del _probe, _y

    plot_dataset_samples(raw_train, fig_dir / "fig1_dataset_samples.png")
    plot_architecture(fig_dir / "fig2_architecture.png")

    # 学习率按优化器分开：Adam/RMSprop 用 1e-3，SGD 用 0.01（带动量）
    experiments = [
        {
            "name": "经典LeNet",
            "kind": "classic",
            "activation": "tanh",
            "dropout": 0.0,
            "optimizer": "sgd",
            "lr": 0.01,
            "group": "baseline",
        },
        {
            "name": "改进-ReLU-d0.3-Adam",
            "kind": "improved",
            "activation": "relu",
            "dropout": 0.3,
            "optimizer": "adam",
            "lr": 1e-3,
            "group": "main",
        },
        {
            "name": "改进-Tanh-d0.3-Adam",
            "kind": "improved",
            "activation": "tanh",
            "dropout": 0.3,
            "optimizer": "adam",
            "lr": 1e-3,
            "group": "activation",
        },
        {
            "name": "改进-GELU-d0.3-Adam",
            "kind": "improved",
            "activation": "gelu",
            "dropout": 0.3,
            "optimizer": "adam",
            "lr": 1e-3,
            "group": "activation",
        },
        {
            "name": "改进-LeakyReLU-d0.3-Adam",
            "kind": "improved",
            "activation": "leakyrelu",
            "dropout": 0.3,
            "optimizer": "adam",
            "lr": 1e-3,
            "group": "activation",
        },
        {
            "name": "改进-ReLU-d0.0-Adam",
            "kind": "improved",
            "activation": "relu",
            "dropout": 0.0,
            "optimizer": "adam",
            "lr": 1e-3,
            "group": "dropout",
        },
        {
            "name": "改进-ReLU-d0.5-Adam",
            "kind": "improved",
            "activation": "relu",
            "dropout": 0.5,
            "optimizer": "adam",
            "lr": 1e-3,
            "group": "dropout",
        },
        {
            "name": "改进-ReLU-d0.3-SGD",
            "kind": "improved",
            "activation": "relu",
            "dropout": 0.3,
            "optimizer": "sgd",
            "lr": 0.01,
            "group": "optimizer",
        },
        {
            "name": "改进-ReLU-d0.3-RMSprop",
            "kind": "improved",
            "activation": "relu",
            "dropout": 0.3,
            "optimizer": "rmsprop",
            "lr": 1e-3,
            "group": "optimizer",
        },
    ]

    all_results = {}
    log_lines = [f"device={device}  epochs={args.epochs}  batch={args.batch_size}"]

    for cfg in experiments:
        # 每个配置单独重建 DataLoader，避免 shuffle 生成器跨实验漂移，保证消融公平。
        set_seed(args.seed)
        train_loader, val_loader, test_loader, _ = build_dataloaders(
            args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        if cfg["kind"] == "classic":
            model = build_classic_lenet()
        else:
            model = build_improved_lenet(cfg["activation"], cfg["dropout"])
        n_params = count_parameters(model)
        header = (
            f"\n=== {cfg['name']} | act={cfg['activation']}  "
            f"drop={cfg['dropout']}  opt={cfg['optimizer']}  "
            f"lr={cfg['lr']}  params={n_params} ==="
        )
        print(header)
        log_lines.append(header)

        result = train_one_run(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            optimizer_name=cfg["optimizer"],
            lr=cfg["lr"],
            weight_decay=1e-4,
            epochs=args.epochs,
            ckpt_path=ckpt_dir / f"{cfg['name']}.pt",
            class_names=CLASS_NAMES,
        )
        result["config"] = cfg
        result["params"] = n_params
        # 预测向量较大，单独存盘；汇总 json 只留指标
        dump_json(
            {
                "y_true": result["y_true"],
                "y_pred": result["y_pred"],
                "confusion_matrix": result["confusion_matrix"],
                "classification_report": result["classification_report"],
            },
            log_dir / f"{cfg['name']}_preds.json",
        )
        slim = {k: v for k, v in result.items() if k not in ("y_true", "y_pred")}
        all_results[cfg["name"]] = slim
        log_lines.append(
            f"  test_acc={result['test_acc']:.4f}  "
            f"test_f1={result['test_macro_f1']:.4f}  "
            f"val_acc={result['best_val_acc']:.4f}"
        )

    dump_json(all_results, out / "metrics.json")
    dump_json(
        {
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": str(device),
            "torch": torch.__version__,
            "cuda": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "note": "每个配置在 set_seed 后重建 DataLoader；测试指标来自验证集最优检查点。",
        },
        out / "run_manifest.json",
    )

    # ---------- 汇总图 ----------
    plot_curves(all_results, fig_dir / "fig3_all_val_curves.png")

    act_names = [
        "改进-Tanh-d0.3-Adam",
        "改进-ReLU-d0.3-Adam",
        "改进-GELU-d0.3-Adam",
        "改进-LeakyReLU-d0.3-Adam",
    ]
    drop_names = [
        "改进-ReLU-d0.0-Adam",
        "改进-ReLU-d0.3-Adam",
        "改进-ReLU-d0.5-Adam",
    ]
    opt_names = [
        "改进-ReLU-d0.3-SGD",
        "改进-ReLU-d0.3-Adam",
        "改进-ReLU-d0.3-RMSprop",
    ]
    plot_group_curves(all_results, act_names, "激活函数消融", fig_dir / "fig4_activation_curves.png")
    plot_group_curves(all_results, drop_names, "Dropout 消融", fig_dir / "fig5_dropout_curves.png")
    plot_group_curves(all_results, opt_names, "优化器消融", fig_dir / "fig6_optimizer_curves.png")

    plot_bar_comparison(
        [
            (n, all_results[n]["test_acc"], all_results[n]["test_macro_f1"])
            for n in ["经典LeNet", "改进-ReLU-d0.3-Adam"]
        ],
        "测试集指标",
        "改动1–3 相对经典 LeNet-5 的整体提升",
        fig_dir / "fig7_classic_vs_improved.png",
    )
    plot_bar_comparison(
        [(n.replace("改进-", "").replace("-d0.3-Adam", ""), all_results[n]["test_acc"], all_results[n]["test_macro_f1"]) for n in act_names],
        "测试集指标",
        "激活函数选择对测试性能的影响（结构与优化器固定）",
        fig_dir / "fig8_activation_bars.png",
    )
    plot_bar_comparison(
        [
            (
                n.replace("改进-ReLU-", "").replace("-Adam", ""),
                all_results[n]["test_acc"],
                all_results[n]["test_macro_f1"],
            )
            for n in drop_names
        ],
        "测试集指标",
        "Dropout 比例对测试性能的影响（ReLU + Adam + BN）",
        fig_dir / "fig9_dropout_bars.png",
    )
    plot_bar_comparison(
        [
            (
                n.replace("改进-ReLU-d0.3-", ""),
                all_results[n]["test_acc"],
                all_results[n]["test_macro_f1"],
            )
            for n in opt_names
        ],
        "测试集指标",
        "优化器选择对测试性能的影响（ReLU + Dropout=0.3 + BN）",
        fig_dir / "fig10_optimizer_bars.png",
    )

    main_name = "改进-ReLU-d0.3-Adam"
    best_name = max(all_results, key=lambda n: all_results[n]["test_acc"])
    cm = np.array(all_results[main_name]["confusion_matrix"])
    plot_confusion(cm, fig_dir / "fig11_confusion_improved.png", "三处改动同时启用（Dropout=0.3）测试集混淆矩阵")
    plot_confusion(
        np.array(all_results["经典LeNet"]["confusion_matrix"]),
        fig_dir / "fig12_confusion_classic.png",
        "经典 LeNet-5 测试集混淆矩阵",
    )
    plot_confusion(
        np.array(all_results[best_name]["confusion_matrix"]),
        fig_dir / "fig17_confusion_best.png",
        f"测试准确率最高配置（{best_name}）混淆矩阵",
    )

    # 特征图与 case study 使用测试准确率最高的配置
    drop_p = all_results[best_name]["config"]["dropout"]
    act = all_results[best_name]["config"]["activation"]
    kind = all_results[best_name]["config"]["kind"]
    set_seed(args.seed)
    viz_model = build_classic_lenet() if kind == "classic" else build_improved_lenet(act, drop_p)
    viz_model.load_state_dict(
        torch.load(ckpt_dir / f"{best_name}.pt", map_location="cpu", weights_only=True)
    )
    viz_model.to(device)
    viz_model.eval()

    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    plot_feature_maps(viz_model, images[:4], labels[:4], fig_dir / "fig13_feature_maps.png")

    with torch.no_grad():
        all_imgs = []
        all_y = []
        all_p = []
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = viz_model(batch_x)
            pred = logits.argmax(1).cpu()
            all_imgs.append(batch_x.cpu())
            all_y.append(batch_y)
            all_p.append(pred)
        all_imgs = torch.cat(all_imgs)
        all_y = torch.cat(all_y)
        all_p = torch.cat(all_p)

    correct_idx = (all_y == all_p).nonzero(as_tuple=False).view(-1)
    wrong_idx = (all_y != all_p).nonzero(as_tuple=False).view(-1)
    rng = np.random.default_rng(args.seed)
    corr_sel = torch.tensor(rng.choice(correct_idx.numpy(), size=min(8, len(correct_idx)), replace=False))
    wrong_sel = torch.tensor(rng.choice(wrong_idx.numpy(), size=min(8, len(wrong_idx)), replace=False))

    plot_case_study(
        all_imgs[corr_sel],
        all_y[corr_sel],
        all_p[corr_sel],
        fig_dir / "fig14_case_correct.png",
        "Case study：正确分类样例",
    )
    plot_case_study(
        all_imgs[wrong_sel],
        all_y[wrong_sel],
        all_p[wrong_sel],
        fig_dir / "fig15_case_wrong.png",
        "Case study：错误分类样例（衬衫 / 外套 / 套头衫易混）",
    )

    summary_table = ["实验配置                          Acc     Macro-F1   ValAcc"]
    summary_table.append("-" * 62)
    for name, res in all_results.items():
        summary_table.append(
            f"{name:<30s}  {res['test_acc']:.4f}  {res['test_macro_f1']:.4f}    {res['best_val_acc']:.4f}"
        )
    log_text = "\n".join(log_lines + [""] + summary_table)
    (log_dir / "run_summary.txt").write_text(log_text, encoding="utf-8")
    plot_training_log_panel(log_text[-2800:], fig_dir / "fig16_train_log.png")

    print("\n" + "\n".join(summary_table))
    print(f"\n全部结果已写入 {out}")


if __name__ == "__main__":
    main()
