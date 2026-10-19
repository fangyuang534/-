"""image_encoder.py —— 图像编码器：把一张图变成一个向量（或一组 token）。

（本文件基于参考文件 `原/03_image_encoder(1).py`，补全了注释与维度说明）

教学要点
--------
1. ResNet 的最后两层 avgpool + fc 去掉后，输出的是 [B, C, h, w] 特征图：
       输入 224x224 -> layer4 输出 [B, 512, 7, 7]
   它同时支持两种用法：
       * 全局平均池化 -> [B, 512]         给 Early / Late / Gated 融合用
       * 展平成 49 个 token -> [B, 49, 512]
         给 Cross-Attention 融合用（图像被切成 49 个"视觉单词"，
         才能和文本 token 做 attention 交互）

   维度变换（这是手册里那个问题的答案）：
       [B, 512, 7, 7] --flatten(2)--> [B, 512, 49] --transpose(1, 2)--> [B, 49, 512]

2. freeze=True 时主干不训练：只训练融合层 + 分类头，显存/时间都省，
   适合课堂上快速做消融对比。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

# backbone 名称 -> (构造函数名, ImageNet 预训练权重枚举名, 输出通道数说明)
_RESNET_ZOO = {
    "resnet18": ("resnet18", "ResNet18_Weights", 512),
    "resnet34": ("resnet34", "ResNet34_Weights", 512),
    "resnet50": ("resnet50", "ResNet50_Weights", 2048),
}


def _build_resnet(backbone: str, pretrained: bool):
    """兼容新旧 torchvision 的预训练参数写法。

    torchvision >= 0.13 用 `weights=...`；老版本用 `pretrained=True/False`。
    注意：pretrained=True 需要联网下载 ImageNet 权重（约 45 MB / resnet18）。
    """
    if backbone not in _RESNET_ZOO:
        raise ValueError(f"目前支持 {list(_RESNET_ZOO)}，收到：{backbone}")
    ctor_name, weights_name, _ = _RESNET_ZOO[backbone]
    ctor = getattr(models, ctor_name)
    try:
        weights_cls = getattr(models, weights_name, None)
        weights = None
        if pretrained and weights_cls is not None:
            weights = weights_cls.IMAGENET1K_V1
        return ctor(weights=weights)
    except TypeError:  # 老版本 torchvision
        return ctor(pretrained=pretrained)


class ImageEncoder(nn.Module):
    """ResNet 图像编码器。

    Args:
        backbone:   resnet18 / resnet34 / resnet50
        pretrained: 是否加载 ImageNet 预训练权重（强烈建议 True）
        freeze:     是否冻结主干（只训练后面的投影层与融合层）
        out_dim:    输出维度。None 表示保持主干原生维度（resnet18 -> 512）
        dropout:    投影层 dropout
    """

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, freeze: bool = False,
                 out_dim: int = None, dropout: float = 0.0):
        super().__init__()
        net = _build_resnet(backbone, pretrained)
        # 去掉最后的 avgpool 与 fc，只保留到 layer4 -> [B, C, h, w]
        self.body = nn.Sequential(*list(net.children())[:-2])
        self.backbone = backbone
        self.feature_dim = net.fc.in_features          # resnet18 -> 512, resnet50 -> 2048
        self.out_dim = self.feature_dim if out_dim is None else out_dim
        self.pool = nn.AdaptiveAvgPool2d(1)

        if out_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.feature_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )

        self.freeze = freeze
        if freeze:
            for p in self.body.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "ImageEncoder":
        """冻结时把主干固定在 eval 模式：BatchNorm 的 running stats 不再更新，
        否则即使 requires_grad=False，BN 仍会在训练中被改写，导致"冻结"名不副实。"""
        super().train(mode)
        if self.freeze:
            self.body.eval()
        return self

    def forward(self, x: torch.Tensor, return_tokens: bool = False) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]，已经过 ImageNet 标准化
            return_tokens: True  -> [B, h*w, out_dim]（空间 token，h*w = 49）
                           False -> [B, out_dim]（全局特征）
        """
        fmap = self.body(x)                                  # [B, C, h, w]
        if return_tokens:
            tokens = fmap.flatten(2).transpose(1, 2)         # [B, h*w, C]
            return self.proj(tokens)                         # Linear 作用在最后一维
        pooled = self.pool(fmap).flatten(1)                  # [B, C]
        return self.proj(pooled)

    def forward_with_maps(self, x: torch.Tensor) -> dict:
        """调试/可视化用：一次返回中间所有形状，方便在报告里贴"模型拆解"。"""
        fmap = self.body(x)
        tokens = fmap.flatten(2).transpose(1, 2)
        pooled = self.pool(fmap).flatten(1)
        return {
            "input": x,
            "feature_map": fmap,                              # [B, C, 7, 7]
            "tokens": self.proj(tokens),                      # [B, 49, out_dim]
            "pooled": self.proj(pooled),                      # [B, out_dim]
            "pooled_from_map": pooled,                        # [B, C]（未投影）
        }

    def extra_repr(self) -> str:
        return (f"backbone={self.backbone}, feature_dim={self.feature_dim}, "
                f"out_dim={self.out_dim}, frozen={self.freeze}")
