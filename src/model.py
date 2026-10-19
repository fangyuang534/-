"""model.py —— 三个模型的组装：Image-Only、Text-Only、多模态。

    Image-Only :  ImageEncoder(resnet18)                -> Linear -> logits
    Text-Only  :  TextEncoder(bert-base-uncased, mean)   -> Linear -> logits
    Multimodal :  ImageEncoder + TextEncoder -> fusion   -> logits

多模态模型的两个关键设置（都通过配置控制）：
    freeze_image / freeze_text
        True  -> 主干冻结，只训练"投影层 + 融合层 + 分类头"
        好处：显存小、速度快、能把注意力集中在"融合方式"这个实验变量上
"""
from __future__ import annotations

import torch
import torch.nn as nn

from fusion import FUSION_DESCRIPTIONS, build_fusion
from image_encoder import ImageEncoder
from text_encoder import TextEncoder
from utils import count_parameters


# ----------------------------------------------------------------------------
# 单模态基线
# ----------------------------------------------------------------------------
class ImageOnlyModel(nn.Module):
    """Image-Only：ResNet18 提特征 + 一个全连接分类头（等价于把 resnet 的 fc 换成 100 类）。"""

    def __init__(self, num_classes: int, backbone: str = "resnet18", pretrained: bool = True,
                 freeze_backbone: bool = False, dropout: float = 0.2):
        super().__init__()
        self.encoder = ImageEncoder(backbone=backbone, pretrained=pretrained,
                                    freeze=freeze_backbone, out_dim=None, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.feature_dim, num_classes),
        )

    def forward(self, images, **kwargs):
        return self.head(self.encoder(images))


class TextOnlyModel(nn.Module):
    """Text-Only：BERT 句向量 + 全连接分类头（与参考脚本 02 等价，这里补了 dropout）。"""

    def __init__(self, num_classes: int, model_name: str = "bert-base-uncased",
                 pretrained: bool = True, freeze_backbone: bool = False,
                 pooling: str = "mean", dropout: float = 0.2):
        super().__init__()
        self.encoder = TextEncoder(model_name=model_name, pretrained=pretrained,
                                   freeze=freeze_backbone, out_dim=None,
                                   pooling=pooling, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.encoder.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        sentence = self.encoder(input_ids, attention_mask)      # [B, 768]
        return self.head(sentence)


# ----------------------------------------------------------------------------
# 多模态模型
# ----------------------------------------------------------------------------
class MultimodalBirdModel(nn.Module):
    """图像 + 文本 -> 融合 -> 分类。

    统一返回 dict：{"logits": [B, C], 以及融合模块附加的可视化字段}
    """

    def __init__(self, num_classes: int, fusion: str = "attention",
                 image_backbone: str = "resnet18", image_pretrained: bool = True,
                 freeze_image: bool = False,
                 text_model: str = "bert-base-uncased", text_pretrained: bool = True,
                 freeze_text: bool = True, text_pooling: str = "mean",
                 proj_dim: int = 256, dropout: float = 0.2, num_heads: int = 8):
        super().__init__()
        self.fusion_name = fusion
        self.image_encoder = ImageEncoder(backbone=image_backbone, pretrained=image_pretrained,
                                          freeze=freeze_image, out_dim=proj_dim,
                                          dropout=dropout)
        self.text_encoder = TextEncoder(model_name=text_model, pretrained=text_pretrained,
                                        freeze=freeze_text, out_dim=proj_dim,
                                        pooling=text_pooling, dropout=dropout)
        self.fusion = build_fusion(fusion, dim=proj_dim, num_classes=num_classes,
                                   dropout=dropout, num_heads=num_heads)
        # 图像与文本各自先归一化到单位球面再融合：两个模态的量纲/范数才可比
        self.image_norm = nn.LayerNorm(proj_dim)
        self.text_norm = nn.LayerNorm(proj_dim)

    def forward(self, images, input_ids, attention_mask=None, **kwargs):
        need_tokens = self.fusion.needs_tokens

        # 图像：一次 forward 同时拿到 [B,d] 全局向量和 [B,49,d] token（不重复算主干）
        maps = self.image_encoder.forward_with_maps(images)
        image_vec = self.image_norm(maps["pooled"])                    # [B, d]
        image_tokens = self.image_norm(maps["tokens"]) if need_tokens else None

        # 文本：一次 forward 拿到 [B,L,H]，再分别 pool 和投影
        hidden, mask = self.text_encoder.encode_hidden(input_ids, attention_mask)
        text_vec = self.text_norm(self.text_encoder.proj(self.text_encoder.pool(hidden, mask)))
        text_tokens = self.text_norm(self.text_encoder.proj(hidden)) if need_tokens else None

        return self.fusion(image_vec, text_vec, attention_mask=mask,
                           image_tokens=image_tokens, text_tokens=text_tokens)


# ----------------------------------------------------------------------------
# 工厂 + 统计
# ----------------------------------------------------------------------------
def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """根据配置字典构建模型（三个训练脚本共用）。"""
    modality = cfg.get("modality", "multimodal")

    if modality == "image":
        return ImageOnlyModel(
            num_classes=num_classes,
            backbone=cfg.get("image_backbone", "resnet18"),
            pretrained=bool(cfg.get("image_pretrained", True)),
            freeze_backbone=bool(cfg.get("freeze_image", False)),
            dropout=float(cfg.get("dropout", 0.2)),
        )

    if modality == "text":
        return TextOnlyModel(
            num_classes=num_classes,
            model_name=cfg.get("text_model", "bert-base-uncased"),
            pretrained=bool(cfg.get("text_pretrained", True)),
            freeze_backbone=bool(cfg.get("freeze_text", False)),
            pooling=cfg.get("text_pooling", "mean"),
            dropout=float(cfg.get("dropout", 0.2)),
        )

    return MultimodalBirdModel(
        num_classes=num_classes,
        fusion=cfg.get("fusion", "attention"),
        image_backbone=cfg.get("image_backbone", "resnet18"),
        image_pretrained=bool(cfg.get("image_pretrained", True)),
        freeze_image=bool(cfg.get("freeze_image", False)),
        text_model=cfg.get("text_model", "bert-base-uncased"),
        text_pretrained=bool(cfg.get("text_pretrained", True)),
        freeze_text=bool(cfg.get("freeze_text", True)),
        text_pooling=cfg.get("text_pooling", "mean"),
        proj_dim=int(cfg.get("proj_dim", 256)),
        dropout=float(cfg.get("dropout", 0.2)),
        num_heads=int(cfg.get("num_heads", 8)),
    )


def describe_model(model: nn.Module, logger=None) -> dict:
    """打印各子模块参数量，报告里"模型规模"一栏直接引用。"""
    lines = [f"model: {type(model).__name__}"]
    total, trainable = count_parameters(model)
    for name, child in model.named_children():
        c_total, c_train = count_parameters(child)
        lines.append(f"  {name:16s} params={c_total / 1e6:8.2f} M  "
                     f"trainable={c_train / 1e6:8.2f} M")
    lines.append(f"  {'TOTAL':16s} params={total / 1e6:8.2f} M  "
                 f"trainable={trainable / 1e6:8.2f} M")
    if hasattr(model, "fusion_name"):
        name = model.fusion_name.lower()
        lines.append(f"  fusion: {FUSION_DESCRIPTIONS.get(name, name)}")
    text = "\n".join(lines)
    if logger is not None:
        logger.info(text)
    else:
        print(text)
    return {"total": total, "trainable": trainable}
