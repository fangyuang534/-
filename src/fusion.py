"""fusion.py —— 多模态融合模块（本实验的核心）。

四种融合方式的本质区别
--------------------
1. Early Fusion (ConcatFusion)
      [图像向量, 文本向量] -> 拼接 -> MLP -> 分类
   优点：简单、参数少；缺点：两个模态从头到尾被当成一串数字，没有显式交互。

2. Late Fusion (LateFusion)
      图像 -> 图像分类头 -> 概率
      文本 -> 文本分类头 -> 概率
      两者加权平均 -> 最终概率
   优点：任一模态缺失都能工作、可解释（能看到 w）；缺点：模态之间从不交流。

3. Cross-Attention Fusion (CrossAttentionFusion)  ← 推荐
      图像 token 作为 Query，文本 token 作为 Key/Value，
      让"图像的每个空间位置"主动去文本里找相关描述（"这里是不是 blue？"）。
   优点：显式建模跨模态对齐；可导出注意力权重 → "模型在看哪个词"。

4. Gated Fusion (GatedFusion)
      一个门控网络逐个样本决定"这次更信图像还是更信文本"：
          a = sigmoid(W [img ; txt])      fusion = a * img + (1 - a) * txt
   优点：对"描述没提颜色"这类不可靠文本有天然的抑制能力。

统一接口
--------
所有融合模块的 forward 签名完全一致，因此换融合方式只需要改配置里的 `fusion:` 一个字段：

    forward(image_feat, text_feat, attention_mask=None,
            image_tokens=None, text_tokens=None) -> dict

返回的 dict 至少含 "logits"，另外可能含：
    "image_logits" / "text_logits"  —— 仅 Late Fusion
    "attn_weights"                  —— 仅 Cross-Attention Fusion，[B, T_img, L_text]
    "gate"                          —— 仅 Gated Fusion，[B, 1]
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseFusion(nn.Module):
    """统一接口的基类。needs_tokens=True 表示该融合需要完整的 token 序列。"""

    needs_tokens: bool = False

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.dropout = dropout

    def forward(self, image_feat, text_feat, attention_mask=None,
                image_tokens=None, text_tokens=None) -> dict:
        raise NotImplementedError


# ----------------------------------------------------------------------------
# ① Early Fusion
# ----------------------------------------------------------------------------
class ConcatFusion(BaseFusion):
    """早融合：先拼接再一起过 MLP。"""

    needs_tokens = False

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1, hidden: int = None):
        super().__init__(dim, num_classes, dropout)
        hidden = hidden or dim * 2
        self.norm = nn.LayerNorm(dim * 2)          # 两个模态量纲不同，先归一化更稳
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, image_feat, text_feat, attention_mask=None,
                image_tokens=None, text_tokens=None) -> dict:
        fused = torch.cat([image_feat, text_feat], dim=-1)     # [B, 2*dim]
        logits = self.net(self.norm(fused))
        return {"logits": logits, "fused": fused}


# ----------------------------------------------------------------------------
# ② Late Fusion
# ----------------------------------------------------------------------------
class LateFusion(BaseFusion):
    """晚融合：两个独立分类头各出概率，再按可学习权重 w 加权。"""

    needs_tokens = False

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__(dim, num_classes, dropout)
        self.image_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))
        self.text_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))
        # w 是 sigmoid 后的标量：w -> 更信图像，1-w -> 更信文本
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, image_feat, text_feat, attention_mask=None,
                image_tokens=None, text_tokens=None) -> dict:
        image_logits = self.image_head(image_feat)             # [B, C]
        text_logits = self.text_head(text_feat)                # [B, C]
        w = torch.sigmoid(self.w)                              # 标量 [1]
        p_img = F.softmax(image_logits, dim=-1)
        p_txt = F.softmax(text_logits, dim=-1)
        p = w * p_img + (1.0 - w) * p_txt                      # 概率空间加权平均
        logits = torch.log(p.clamp_min(1e-8))                  # 变回 logit 便于用 CE
        return {
            "logits": logits,
            "image_logits": image_logits,
            "text_logits": text_logits,
            "gate": w.expand(image_feat.size(0), 1),
        }


# ----------------------------------------------------------------------------
# ③ Cross-Attention Fusion（推荐）
# ----------------------------------------------------------------------------
class CrossAttentionFusion(BaseFusion):
    """跨模态注意力：图像 token 作 Query，文本 token 作 Key/Value。

        Q = W_q * image_tokens     [B, 49, d]
        K = V = W_k * text_tokens  [B, L, d]
        out = MultiHeadAttention(Q, K, V, key_padding_mask=~mask)
        logits = Linear(mean_pool(LN(out)))
    """

    needs_tokens = True

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1,
                 num_heads: int = 8, ffn_ratio: int = 2):
        super().__init__(dim, num_classes, dropout)
        if dim % num_heads != 0:
            raise ValueError(f"dim({dim}) 必须能被 num_heads({num_heads}) 整除")
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_ratio, dim),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(dim, num_classes))

    def forward(self, image_feat, text_feat, attention_mask=None,
                image_tokens=None, text_tokens=None) -> dict:
        if image_tokens is None or text_tokens is None:
            raise ValueError("CrossAttentionFusion 需要 image_tokens / text_tokens")
        key_padding_mask = None
        if attention_mask is not None:
            # MultiheadAttention 的 key_padding_mask：True 表示"这个位置要屏蔽"
            key_padding_mask = attention_mask == 0

        attn_out, attn_weights = self.attn(
            query=image_tokens,          # [B, 49, d]
            key=text_tokens,             # [B, L, d]
            value=text_tokens,           # [B, L, d]
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,   # 多头求平均 -> [B, 49, L]，便于画词级注意力
        )
        x = self.norm1(image_tokens + attn_out)
        x = self.norm2(x + self.ffn(x))                    # [B, 49, d]
        pooled = x.mean(dim=1)                             # 49 个位置求平均 -> [B, d]
        logits = self.head(pooled)
        return {"logits": logits, "attn_weights": attn_weights, "fused": pooled}


# ----------------------------------------------------------------------------
# ④ Gated Fusion
# ----------------------------------------------------------------------------
class GatedFusion(BaseFusion):
    """门控融合：逐样本学习一个标量权重 a，决定"这次更信谁"。"""

    needs_tokens = False

    def __init__(self, dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__(dim, num_classes, dropout)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes),
        )

    def forward(self, image_feat, text_feat, attention_mask=None,
                image_tokens=None, text_tokens=None) -> dict:
        a = torch.sigmoid(self.gate(torch.cat([image_feat, text_feat], dim=-1)))   # [B, 1]
        fused = a * image_feat + (1.0 - a) * text_feat                            # [B, dim]
        return {"logits": self.head(fused), "gate": a, "fused": fused}


# ----------------------------------------------------------------------------
# 工厂
# ----------------------------------------------------------------------------
FUSION_REGISTRY = {
    "early": ConcatFusion,          # 别名：concat
    "concat": ConcatFusion,
    "late": LateFusion,
    "attention": CrossAttentionFusion,   # 别名：cross_attention
    "cross_attention": CrossAttentionFusion,
    "gated": GatedFusion,           # 别名：gate
    "gate": GatedFusion,
}

FUSION_DESCRIPTIONS = {
    "early": "Early Fusion  : concat([img, txt]) -> MLP",
    "late": "Late Fusion   : w * p_img + (1-w) * p_txt",
    "attention": "Cross-Attn    : img tokens (Q) x text tokens (K/V)",
    "gated": "Gated Fusion  : a * img + (1-a) * txt, a = sigmoid(W[img;txt])",
}


def build_fusion(name: str, dim: int, num_classes: int, dropout: float = 0.1,
                 num_heads: int = 8) -> BaseFusion:
    key = (name or "attention").lower()
    if key not in FUSION_REGISTRY:
        raise ValueError(f"未知融合方式 {name!r}，可选：{sorted(set(FUSION_REGISTRY))}")
    cls = FUSION_REGISTRY[key]
    if cls is CrossAttentionFusion:
        return cls(dim, num_classes, dropout=dropout, num_heads=num_heads)
    return cls(dim, num_classes, dropout=dropout)
