"""text_encoder.py —— 文本编码器：把一句话变成一个向量（或一组 token）。

（本文件基于参考文件 `原/03_text_encoder(2).py`，补全了注释与维度说明）

一条 caption 在 BERT 里走过的四步
--------------------------------
1. Tokenizer：切成 WordPiece 子词并查词表（30522 个 token），
   "a small blue bird" -> [CLS] a small blue bird [SEP] -> id 序列 [B, L]
2. Embedding：查 Token Embedding 表（30522 x 768），再叠加
   Position Embedding + Segment Embedding  -> [B, L, 768]
3. Transformer：12 层自注意力，输出每个 token 的上下文向量 [B, L, 768]
4. Pooling：整句要一个向量 -> [B, 768]
   - cls  ：取 hidden[:, 0]，[CLS] 在自注意力里汇总了整句信息
   - mean ：对**真实 token** 求平均（用 attention_mask 过滤 padding），更稳

为什么 mean pooling 一定要带 mask？
--------------------------------
batch 里短句要补 [PAD] 到同一长度。若把 PAD 的向量也平均进去，
真正有意义的 5 个词会被 20 个零向量稀释，向量范数被拉低、方向也被拉偏。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def load_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """加载 tokenizer。

    国内网络不通时先设镜像（Windows PowerShell）：
        $env:HF_ENDPOINT = "https://hf-mirror.com"
    或离线时把模型放到本地目录，然后 --text_model D:/models/bert-base-uncased
    """
    return AutoTokenizer.from_pretrained(model_name)


def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """只对真实 token 求平均（把 padding 位置屏蔽掉）。

    hidden: [B, L, H]；attention_mask: [B, L]（1=真实 token，0=padding）
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)      # [B, L, 1]
    summed = (hidden * mask).sum(dim=1)                       # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-6)                  # [B, 1]
    return summed / counts


class TextEncoder(nn.Module):
    """BERT 文本编码器。

    Args:
        model_name: HuggingFace 模型名，如 bert-base-uncased / distilbert-base-uncased
        pretrained: 是否加载预训练权重（False 则随机初始化，只用于形状调试）
        freeze:     是否冻结 BERT 主干
        out_dim:    输出维度。None 表示保持 BERT 原生维度（bert-base = 768）
        pooling:    "cls" 或 "mean"
        dropout:    投影层 dropout
    """

    def __init__(self, model_name: str = "bert-base-uncased", pretrained: bool = True,
                 freeze: bool = False, out_dim: int = None,
                 pooling: str = "mean", dropout: float = 0.0):
        super().__init__()
        if pretrained:
            self.bert = AutoModel.from_pretrained(model_name)
        else:
            from transformers import AutoConfig
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        self.model_name = model_name
        self.hidden_size = self.bert.config.hidden_size   # bert-base -> 768
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling 只能是 'cls' 或 'mean'，收到：{pooling}")
        self.pooling = pooling
        self.out_dim = self.hidden_size if out_dim is None else out_dim
        if out_dim is None:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.hidden_size, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        self.freeze = freeze
        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def train(self, mode: bool = True) -> "TextEncoder":
        super().train(mode)
        if self.freeze:
            self.bert.eval()
        return self

    def encode_hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """返回 (每个 token 的上下文向量 [B, L, H], attention_mask)。"""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state, attention_mask

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """[B, L, H] -> [B, H]"""
        if self.pooling == "cls":
            return hidden[:, 0]
        if attention_mask is None:
            return hidden.mean(dim=1)
        return masked_mean(hidden, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                return_tokens: bool = False) -> torch.Tensor:
        """
        Args:
            input_ids:      [B, L]
            attention_mask: [B, L]
            return_tokens:  True  -> [B, L, out_dim]（给 Cross-Attention 融合当 Key/Value）
                            False -> [B, out_dim]（句向量）
        """
        hidden, attention_mask = self.encode_hidden(input_ids, attention_mask)   # [B, L, H]
        if return_tokens:
            return self.proj(hidden)
        return self.proj(self.pool(hidden, attention_mask))

    def extra_repr(self) -> str:
        return (f"hidden_size={self.hidden_size}, out_dim={self.out_dim}, "
                f"pooling={self.pooling}, frozen={'(bert)' if self.freeze else 'no'}")
