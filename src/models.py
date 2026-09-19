"""LeNet-5 及其改进变体。

经典 LeNet-5（LeCun et al., 1998）使用 tanh 激活、平均池化，不含 BN / Dropout。
本实验在保持 C1-S2-C3-S4-C5-F6 骨架的前提下，通过配置开关实现三处核心改动：
  改动1 激活函数：tanh -> ReLU / GELU / LeakyReLU
  改动2 优化器：在训练脚本中由 SGD 换为 Adam / RMSprop
  改动3 网络结构：AvgPool -> MaxPool，并加入 BatchNorm 与 Dropout
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


CLASS_NAMES = [
    "T恤",
    "裤子",
    "套头衫",
    "连衣裙",
    "外套",
    "凉鞋",
    "衬衫",
    "运动鞋",
    "包",
    "短靴",
]

CLASS_NAMES_EN = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def get_activation(name: str) -> Callable[[], nn.Module]:
    """按名称构造激活层。name 大小写不敏感。"""
    key = name.lower()
    mapping = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "leakyrelu": lambda: nn.LeakyReLU(0.1),
        "leaky_relu": lambda: nn.LeakyReLU(0.1),
        "sigmoid": nn.Sigmoid,
    }
    if key not in mapping:
        raise ValueError(f"未知激活函数: {name}")
    return mapping[key]


class LeNetFamily(nn.Module):
    """可配置的 LeNet-5 家族。

    输入约定：1 x 32 x 32（Fashion-MNIST 28x28 经零填充得到，与原始论文一致）。
    前向尺寸变化：
        conv1 5x5: 32 -> 28
        pool  2x2: 28 -> 14
        conv2 5x5: 14 -> 10
        pool  2x2: 10 -> 5
        flatten:   16 * 5 * 5 = 400
        fc: 400 -> 120 -> 84 -> 10
    """

    def __init__(
        self,
        activation: str = "relu",
        pooling: str = "max",
        use_bn: bool = True,
        dropout: float = 0.3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.activation_name = activation
        self.pooling = pooling
        self.use_bn = use_bn
        self.dropout_p = dropout

        act_ctor = get_activation(activation)
        pool_cls = nn.MaxPool2d if pooling == "max" else nn.AvgPool2d

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6) if use_bn else nn.Identity()
        self.act1 = act_ctor()
        self.pool1 = pool_cls(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 现代实现为全连接 C3；原论文 C3 为稀疏连接
        self.bn2 = nn.BatchNorm2d(16) if use_bn else nn.Identity()
        self.act2 = act_ctor()
        self.pool2 = pool_cls(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.act3 = act_ctor()
        self.drop3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(120, 84)
        self.act4 = act_ctor()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.act1(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(self.act2(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = torch.flatten(x, 1)
        x = self.drop3(self.act3(self.fc1(x)))
        x = self.act4(self.fc2(x))
        return self.fc3(x)

    def conv1_feature(self, x: torch.Tensor) -> torch.Tensor:
        """返回第一层卷积后的特征图，用于中间结果可视化。"""
        return self.act1(self.bn1(self.conv1(x)))


def build_classic_lenet() -> LeNetFamily:
    """复现经典 LeNet-5：tanh + AvgPool，无 BN / Dropout。"""
    return LeNetFamily(
        activation="tanh",
        pooling="avg",
        use_bn=False,
        dropout=0.0,
    )


def build_improved_lenet(
    activation: str = "relu",
    dropout: float = 0.3,
) -> LeNetFamily:
    """改进 LeNet-5：ReLU + MaxPool + BN + Dropout。"""
    return LeNetFamily(
        activation=activation,
        pooling="max",
        use_bn=True,
        dropout=dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
