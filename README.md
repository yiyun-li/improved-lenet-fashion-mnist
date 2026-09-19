# 改进 LeNet-5 × Fashion-MNIST

课程实验：用 PyTorch 做服装图像 10 分类，并扫描激活函数、Dropout 和优化器。

相对经典 LeNet-5 的三处改动：

1. 激活：tanh → ReLU（对照 Tanh / GELU / LeakyReLU）
2. 优化器：SGD → Adam（对照 RMSprop）
3. 结构：AvgPool → MaxPool，加 BatchNorm；Dropout 扫描后主结果取 p=0

## 结果（测试集，种子 42）

| 配置 | Acc | Macro-F1 |
|------|-----|----------|
| 经典 LeNet-5 | 88.74% | 0.8869 |
| ReLU + BN + MaxPool + Adam，p=0 | 90.45% | 0.9046 |

报告：[`report/experiment_report.md`](report/experiment_report.md)。图在 `results/figures/`。

## 运行

需要 Python 3.10+，以及 `requirements.txt` 里的包。有 NVIDIA GPU 会自动用 CUDA，没有就走 CPU。

```bash
python -m pip install -r requirements.txt
python run_experiment.py --device auto --epochs 15 --batch-size 128
```

Fashion-MNIST 第一次运行会下载到 `./data/`。指定 GPU：

```bash
export CUDA_VISIBLE_DEVICES=0
python run_experiment.py --device cuda
```

默认路径都相对仓库根目录，不依赖本机绝对路径。

## 目录

```
run_experiment.py      九组消融
src/models.py          LeNet
src/data.py            数据
src/engine.py          训练与评估
src/plots.py           作图
report/                实验报告
results/figures/       报告用图
results/metrics.json   指标
```
