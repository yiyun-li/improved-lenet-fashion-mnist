#!/usr/bin/env python3
"""生成飞书云文档 XML。图片路径相对仓库根目录。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def t(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def p(s: str) -> str:
    return f"<p>{t(s)}</p>"


def img(name: str, caption: str) -> str:
    return f'<img path="@./results/figures/{name}" caption="{t(caption)}"/>'


def th(s: str) -> str:
    return f'<th background-color="light-gray"><p>{t(s)}</p></th>'


def td(s: str) -> str:
    return f"<td><p>{t(s)}</p></td>"


def tr(cells: list[str], header: bool = False) -> str:
    inner = "".join(th(c) if header else td(c) for c in cells)
    return f"<tr>{inner}</tr>"


def table(headers: list[str], rows: list[list[str]]) -> str:
    head = f"<thead>{tr(headers, True)}</thead>"
    body = "<tbody>" + "".join(tr(r) for r in rows) + "</tbody>"
    cols = "<colgroup>" + "".join("<col />" for _ in headers) + "</colgroup>"
    return f"<table>{cols}{head}{body}</table>"


def build() -> str:
    parts: list[str] = []
    parts.append("<title>实验三：基于卷积神经网络的服装图像分类</title>")
    parts.append(
        '<callout emoji="📌" background-color="light-blue" border-color="blue">'
        "<p><b>改动 1｜激活函数</b> tanh 改为 ReLU，并对照 Tanh、GELU、LeakyReLU。</p>"
        "<p><b>改动 2｜优化方法</b> 动量 SGD 改为 Adam，并对照 RMSprop。</p>"
        "<p><b>改动 3｜网络结构</b> AvgPool 改为 MaxPool，加 BatchNorm；Dropout 扫描后主结果取 p=0。</p>"
        "</callout>"
    )
    parts.append(
        p("PyTorch 2.4.1，CUDA 11.8，NVIDIA GeForce RTX 3090。主结果：ReLU + BatchNorm + MaxPool + Adam，Dropout p=0，测试准确率 90.45%，Macro-F1 0.9046。")
    )

    parts.append('<h1 seq="auto">问题描述</h1>')
    parts.append('<h2 seq="auto">待解决问题、背景与应用前景</h2>')
    parts.append(p("图像分类把一张图映射到预定义类别。服装图像的判别主要靠领口、袖长和轮廓。CNN 用局部连接和权值共享提取空间特征。LeNet-5 给出 C1–S2–C3–S4–C5–F6 骨架，最初用于手写数字。"))
    parts.append(p("Fashion-MNIST 保持 MNIST 的规模（28×28 灰度，6 万训练 / 1 万测试，10 类均衡），把数字换成服装。衬衫、T 恤、外套、套头衫在低分辨率下外形接近，比手写数字更能看出激活函数、正则化和优化器有没有起作用。"))

    parts.append('<h2 seq="auto">问题的形式化表述</h2>')
    parts.append(p("输入为 1×32×32：原图 28×28 四周补 2 像素零。标签为 10 类服装。网络输出 10 维 logits，预测取 softmax 的 argmax。训练用带 L2 正则（λ=10⁻⁴）的交叉熵。验证集只按准确率存检查点，测试集训练结束后评估一次。"))
    parts.append("<p>预测：<latex>\\hat{y}=\\arg\\max_{k}\\operatorname{softmax}(f_\\theta(x))_k</latex></p>")

    parts.append('<h2 seq="auto">解决方案与算法</h2>')
    parts.append(p("对照实验保持骨架和宽度不变，每次只改一个因素。经典对照：tanh、AvgPool、无 BN、无 Dropout、动量 SGD。改进模型叠加三处改动，再分别扫激活、Dropout 和优化器。第二层卷积用全连接 C3。"))
    parts.append(img("fig2_architecture.png", "经典 LeNet-5 与改进 LeNet-5 结构对照"))
    parts.append(
        '<pre lang="text" caption="训练伪代码"><code>'
        + t(
            "set_seed(s)，按 s 重建 DataLoader\n"
            "Pad(2)，Normalize(0.5, 0.5)\n"
            "for e = 1 to E:\n"
            "    for (X, y) in D:\n"
            "        θ ← Opt(θ, ∇ CrossEntropy(f_θ(X), y))\n"
            "    验证准确率升高则保存 θ*\n"
            "用 θ* 在测试集上报告指标"
        )
        + "</code></pre>"
    )

    parts.append('<h1 seq="auto">实验设置及分析</h1>')
    parts.append('<h2 seq="auto">数据集、样例、规模与实验环境</h2>')
    parts.append(p("Fashion-MNIST：训练 60,000，测试 10,000。种子 42 划出 10% 验证集，实际训练 54,000、验证 6,000、测试 10,000。像素补到 32×32，映射到 [-1,1]。"))
    parts.append(img("fig1_dataset_samples.png", "Fashion-MNIST 各类别样例"))
    parts.append(p("Python 3.10.13，PyTorch 2.4.1+cu118，RTX 3090。batch 128，15 epoch。SGD 学习率 0.01（动量 0.9），Adam / RMSprop 为 1e-3。每个配置在 set_seed(42) 后重建 DataLoader。测试指标来自验证集最优检查点。改进模型 61,750 参数，经典 61,706。单配置约 1 分钟。代码不绑定机器路径。"))

    parts.append('<h2 seq="auto">实验评价标准</h2>')
    parts.append(p("主指标为测试集 Top-1 准确率。Macro-F1 按 10 类等权。混淆矩阵和逐类 F1 看错类。验证曲线看收敛。"))

    parts.append('<h2 seq="auto">实验结果</h2>')
    parts.append(p("主结果是 ReLU + BatchNorm + MaxPool + Adam，Dropout p=0：测试准确率 90.45%，Macro-F1 0.9046。经典 LeNet-5 为 88.74% / 0.8869，相差 1.71 个百分点。验证准确率也是该配置最高（90.77%）。"))
    parts.append(img("fig16_train_log.png", "训练日志摘要"))
    parts.append(
        table(
            ["配置", "实验角色", "Test Acc", "Macro-F1", "最佳 Val Acc"],
            [
                ["经典LeNet", "基线", "0.8874", "0.8869", "0.8935"],
                ["改进-ReLU-d0.3-Adam", "三处改动 + Dropout 0.3", "0.8848", "0.8839", "0.8917"],
                ["改进-Tanh-d0.3-Adam", "激活消融", "0.8922", "0.8914", "0.8933"],
                ["改进-GELU-d0.3-Adam", "激活消融", "0.8932", "0.8929", "0.8963"],
                ["改进-LeakyReLU-d0.3-Adam", "激活消融", "0.8889", "0.8885", "0.8933"],
                ["改进-ReLU-d0.0-Adam", "主模型", "0.9045", "0.9046", "0.9077"],
                ["改进-ReLU-d0.5-Adam", "Dropout 扫描", "0.8602", "0.8556", "0.8720"],
                ["改进-ReLU-d0.3-SGD", "优化器消融", "0.8836", "0.8838", "0.8905"],
                ["改进-ReLU-d0.3-RMSprop", "优化器消融", "0.8874", "0.8858", "0.8907"],
            ],
        )
    )
    parts.append(img("fig3_all_val_curves.png", "九组配置验证曲线"))
    parts.append(img("fig18_classic_vs_best.png", "经典 LeNet-5 与主模型"))
    parts.append(p("改动 1。固定 BN、MaxPool、Dropout=0.3 和 Adam 后，四种激活测试准确率在 88.48%–89.32%。这一档 GELU 最高。主模型用 ReLU，因为它和 p=0 组成全局最好的一组。"))
    parts.append(img("fig8_activation_bars.png", "激活函数"))
    parts.append(p("改动 3。p 取 0 / 0.3 / 0.5，测试准确率分别为 90.45%、88.48%、86.02%。约 6.2 万参数、5.4 万训练样本，工作点在 p=0。"))
    parts.append(img("fig9_dropout_bars.png", "Dropout 比例"))
    parts.append(img("fig5_dropout_curves.png", "Dropout 曲线"))
    parts.append(p("改动 2。15 epoch、常用学习率下 Adam 和 RMSprop 验证曲线升得更快。主模型用 Adam，p=0 时测到 90.45%。"))
    parts.append(img("fig10_optimizer_bars.png", "优化器"))
    parts.append(img("fig13_feature_maps.png", "第一层卷积特征图"))
    parts.append(p("主模型混淆矩阵对角占优。裤子、凉鞋、包、短靴几乎不分错。剩余错误集中在 T 恤、衬衫、套头衫、外套。"))
    parts.append(img("fig17_confusion_best.png", "主模型测试集混淆矩阵"))

    parts.append('<h1 seq="auto">实验讨论与 Case Study</h1>')
    parts.append(p("改进后测试准确率从 88.74% 到 90.45%。衬衫 F1 0.692 → 0.735，套头衫 0.805 → 0.856，T 恤 0.829 → 0.851。ReLU、BN 和 MaxPool 出局部特征，Adam 在 15 轮里把损失压下来，Dropout 按这个网络的容量取 p=0。"))
    parts.append(img("fig19_per_class_f1.png", "逐类 F1"))
    parts.append(p("错例几乎都是上装互认。领口和门襟在 28×28 里只有几个像素。对的例子多是裤子、短靴、包、凉鞋。"))
    parts.append(img("fig15_case_wrong.png", "错误分类样例"))
    parts.append(img("fig14_case_correct.png", "正确分类样例"))
    parts.append(
        table(
            ["编号", "相对原版 LeNet-5", "证据"],
            [
                ["改动 1", "tanh → ReLU", "激活消融；与 p=0 组成 90.45%"],
                ["改动 2", "SGD → Adam", "优化器消融；15 epoch"],
                ["改动 3", "MaxPool + BN，Dropout p=0", "结构对照 + Dropout 扫描"],
            ],
        )
    )

    parts.append('<h1 seq="auto">参考文献</h1>')
    refs = [
        (
            "LeCun Y, et al. Gradient-based learning applied to document recognition. Proc. IEEE, 1998.",
            "https://doi.org/10.1109/5.726791",
        ),
        (
            "Xiao H, Rasul K, Vollgraf R. Fashion-MNIST. arXiv:1708.07747, 2017.",
            "https://arxiv.org/abs/1708.07747",
        ),
        (
            "Nair V, Hinton G E. Rectified linear units improve restricted Boltzmann machines. ICML, 2010.",
            "https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf",
        ),
        (
            "Hendrycks D, Gimpel K. GELUs. arXiv:1606.08415, 2016.",
            "https://arxiv.org/abs/1606.08415",
        ),
        (
            "Maas A L, et al. Rectifier nonlinearities improve neural network acoustic models. ICML Workshop, 2013.",
            "https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf",
        ),
        (
            "Srivastava N, et al. Dropout. JMLR, 2014, 15(1): 1929–1958.",
            "https://jmlr.org/papers/v15/srivastava14a.html",
        ),
        (
            "Ioffe S, Szegedy C. Batch normalization. ICML, 2015.",
            "https://proceedings.mlr.press/v37/ioffe15.html",
        ),
        (
            "Kingma D P, Ba J. Adam. ICLR, 2015. arXiv:1412.6980.",
            "https://arxiv.org/abs/1412.6980",
        ),
        (
            "Tieleman T, Hinton G. RMSProp. COURSERA, 2012.",
            "https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf",
        ),
        (
            "Krizhevsky A, Sutskever I, Hinton G E. ImageNet classification with deep convolutional neural networks. NeurIPS, 2012.",
            "https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html",
        ),
    ]
    lis = "".join(
        f'<li>{t(text)} <a href="{t(url)}">{t(url)}</a></li>' for text, url in refs
    )
    parts.append(f"<ol>{lis}</ol>")

    parts.append('<h1 seq="auto">附录 源代码说明</h1>')
    parts.append(p("代码：https://github.com/yiyun-li/improved-lenet-fashion-mnist 。入口 run_experiment.py。默认相对仓库根目录读写 data/ 与 results/，--device auto 在有 CUDA 时用 GPU。"))
    parts.append(
        '<pre lang="bash" caption="复现"><code>'
        + t("python -m pip install -r requirements.txt\npython run_experiment.py --device auto --epochs 15 --batch-size 128")
        + "</code></pre>"
    )
    return "\n".join(parts)


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "report" / "feishu_draft.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
