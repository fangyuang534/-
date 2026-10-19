"""inspect_features.py —— 把模型拆开：一步步看清图像与文本是怎么变成向量的（手册 §3）

对应参考文件 `原/03_image_encoder(1).py` 与 `原/03_text_encoder(2).py` 的教学目标：
    "分析模型对图像和文本处理后的特征"

本脚本按顺序打印 6 段内容，每一段都对应报告里的一小节：
    ① Tokenizer：一句话怎么变成 id 序列（+ 词表/特殊标记）
    ② Embedding：id 怎么查表变成向量（Token + Position + Segment）
    ③ BERT 输出 & 池化：[B, L, 768] 怎么变成整句向量；mask 关不关掉到底差多少
    ④ 图像编码器：[B,3,224,224] -> [B,512,7,7] -> [B,49,512] / [B,512]
    ⑤ 相似度：同模态内"同一类更近"，跨模态必须先投影到同一空间
    ⑥ 特征可视化：对图像/文本特征做 PCA，保存散点图

    python src/inspect_features.py                 # 全部
    python src/inspect_features.py --num_plot 300  # 可视化用多少样本
    python src/inspect_features.py --image_pretrained 0

产物：outputs/inspect/features.log, pca_image.png, pca_text.png
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F

from data import BirdDataset, build_transform, load_annotations
from image_encoder import ImageEncoder
from text_encoder import TextEncoder, load_tokenizer, masked_mean
from utils import (Logger, build_config, describe_device, ensure_dir, get_device,
                   save_json, set_seed, setup_console)

DEFAULTS = {
    "data_root": "data/cub200_subset",
    "ann_file": "data/cub200_subset/annotations.json",
    "out_dir": "outputs/inspect",
    "text_model": "bert-base-uncased",
    "image_backbone": "resnet18",
    "image_pretrained": True,
    "max_length": 48,
    "image_size": 224,
    "num_plot": 300,
    "seed": 42,
    "device": "cpu",
}


def shape(t) -> str:
    return "x".join(str(s) for s in tuple(t.shape))


# ----------------------------------------------------------------------------
# ① Tokenizer
# ----------------------------------------------------------------------------
def demo_tokenizer(tokenizer, logger) -> None:
    logger.info("=" * 78)
    logger.info("[1/6] Tokenizer: text -> token -> id")
    logger.info("=" * 78)
    text = "a small blue bird"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    encoded = tokenizer(text, return_tensors="pt")
    logger.info(f"text        : {text!r}")
    logger.info(f"vocab size  : {tokenizer.vocab_size}  (bert-base-uncased 共 30522 个子词)")
    logger.info(f"tokens      : {tokens}")
    logger.info(f"ids         : {ids}")
    logger.info(f"with special: {tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist())}")
    logger.info(f"input_ids   : {encoded['input_ids'][0].tolist()}")
    logger.info(f"attention_mask: {encoded['attention_mask'][0].tolist()}")
    logger.info("special ids : [PAD]=%d [UNK]=%d [CLS]=%d [SEP]=%d [MASK]=%d" % (
        tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.cls_token_id,
        tokenizer.sep_token_id, tokenizer.mask_token_id))
    logger.info("note        : 词表有限，但子词组合 + 多模态语义对齐 -> 可处理开放世界的说法")


# ----------------------------------------------------------------------------
# ② Embedding
# ----------------------------------------------------------------------------
def demo_embedding(text_encoder: TextEncoder, tokenizer, logger) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("[2/6] Embedding: 查表 + 位置 + 片段 三者相加")
    logger.info("=" * 78)
    emb = text_encoder.bert.embeddings
    weight = emb.word_embeddings.weight
    logger.info(f"Token Embedding Matrix : [{weight.size(0)}, {weight.size(1)}]  "
                f"(行=词表大小, 列=每个词的维度)")
    encoded = tokenizer("a small blue bird", return_tensors="pt")
    input_ids = encoded["input_ids"]
    token_emb = emb.word_embeddings(input_ids)
    pos_emb = emb.position_embeddings(torch.arange(input_ids.size(1)).unsqueeze(0))
    seg_emb = emb.token_type_embeddings(torch.zeros_like(input_ids))
    total = token_emb + pos_emb + seg_emb
    logger.info(f"token_emb   : {shape(token_emb)}   <- 查 1037/2235/... 这类 id 得到")
    logger.info(f"position_emb: {shape(pos_emb)}   <- 让模型知道词的先后顺序")
    logger.info(f"segment_emb : {shape(seg_emb)}   <- 区分句子 A / 句子 B")
    logger.info(f"sum         : {shape(total)}   <- 送进 Transformer 的初值")
    same = torch.allclose(emb(input_ids, torch.zeros_like(input_ids)),
                          emb(input_ids, torch.zeros_like(input_ids)))
    logger.info(f"deterministic: {same}")
    logger.info(f"word 'bird' id={tokenizer.convert_tokens_to_ids('bird')}, "
                f"norm={token_emb[0, -2].norm().item():.4f}")


# ----------------------------------------------------------------------------
# ③ BERT 输出 + 池化
# ----------------------------------------------------------------------------
def demo_pooling(text_encoder: TextEncoder, tokenizer, logger) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("[3/6] BERT 输出 [B, L, H] -> 整句向量 [B, H]")
    logger.info("=" * 78)
    texts = [
        "a small blue bird with a long beak",
        "this bird has blue feathers and long wings",
    ]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=48,
                        return_tensors="pt")
    input_ids, mask = encoded["input_ids"], encoded["attention_mask"]
    with torch.no_grad():
        hidden, _ = text_encoder.encode_hidden(input_ids, mask)
    logger.info(f"input_ids   : {shape(input_ids)}   (batch x 序列长度)")
    logger.info(f"attention_mask: {shape(mask)}   (1=真实词, 0=padding)")
    logger.info(f"last_hidden_state: {shape(hidden)}   (每个 token 一个 768 维上下文向量)")
    logger.info(f"真实 token 数: {mask.sum(dim=1).tolist()}, 补齐到 L={input_ids.size(1)}")

    cls_vec = hidden[:, 0]
    mean_masked = masked_mean(hidden, mask)
    mean_naive = hidden.mean(dim=1)
    logger.info(f"[CLS] 向量        : {shape(cls_vec)}  norm={cls_vec.norm(dim=-1).tolist()}")
    logger.info(f"mask mean pooling : {shape(mean_masked)}  norm={mean_masked.norm(dim=-1).tolist()}")
    logger.info(f"naive mean (错误) : {shape(mean_naive)}  norm={mean_naive.norm(dim=-1).tolist()}")
    diff = (mean_masked - mean_naive).norm(dim=-1)
    logger.info(f"两者差异 ||mask_mean - naive_mean|| = {diff.tolist()}")
    logger.info("结论: padding 越多，naive mean 被稀释得越厉害；必须用 attention_mask 过滤")


# ----------------------------------------------------------------------------
# ④ 图像编码器
# ----------------------------------------------------------------------------
def demo_image_encoder(image_encoder: ImageEncoder, batch: dict, logger) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("[4/6] Image Encoder: [B,3,224,224] -> [B,512,7,7] -> [B,49,512] / [B,512]")
    logger.info("=" * 78)
    images = batch["images"]
    with torch.no_grad():
        out = image_encoder.forward_with_maps(images)
    for key in ("input", "feature_map", "tokens", "pooled_from_map", "pooled"):
        logger.info(f"{key:16s}: {shape(out[key])}")
    logger.info("")
    logger.info("维度变换（手册里那个问题的答案）：")
    logger.info("    [B, 512, 7, 7] --flatten(2)--> [B, 512, 49] --transpose(1,2)--> [B, 49, 512]")
    logger.info("    49 = 7 x 7，相当于把图片切成 49 个'视觉单词'，才能和文本 token 做注意力")

    # 数值验证：flatten+transpose 与手写 reshape 是否一致
    fmap = out["feature_map"]
    manual = fmap.reshape(fmap.size(0), fmap.size(1), -1).transpose(1, 2)
    logger.info(f"check flatten(2).transpose(1,2) == reshape().transpose(1,2): "
                f"{torch.allclose(manual, fmap.flatten(2).transpose(1, 2))}")
    # 全局平均池化 vs token 平均：projection 里有 ReLU，两者不严格相等
    logger.info(f"pooled norm={out['pooled'].norm(dim=-1).tolist()[:4]}  "
                f"tokens mean norm={out['tokens'].mean(dim=1).norm(dim=-1).tolist()[:4]}")
    logger.info("note: 融合时图像侧用 pooled 向量；Attention 融合用 49 个 token")


# ----------------------------------------------------------------------------
# ⑤ 相似度
# ----------------------------------------------------------------------------
def demo_similarity(image_encoder, text_encoder, tokenizer, ann, batch, logger) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("[5/6] 相似度：同一个空间里才能比较")
    logger.info("=" * 78)
    with torch.no_grad():
        img_vec = image_encoder(batch["images"])
        txt_vec = text_encoder(batch["input_ids"], batch["attention_mask"])
    logger.info(f"image vectors: {shape(img_vec)}   (resnet18 -> 512 维)")
    logger.info(f"text  vectors: {shape(txt_vec)}   (bert-base -> 768 维)")
    logger.info("维度都不一样，所以直接做内积/余弦是**没有意义**的：")
    logger.info("    这就是为什么多模态要先各自投影到一个公共维度，再融合/对齐")

    img_sim = F.normalize(img_vec, dim=-1) @ F.normalize(img_vec, dim=-1).t()
    txt_sim = F.normalize(txt_vec, dim=-1) @ F.normalize(txt_vec, dim=-1).t()
    logger.info("image-image 余弦相似度矩阵:")
    for row in img_sim.tolist():
        logger.info("    " + "  ".join(f"{v:+.3f}" for v in row))
    logger.info("text-text  余弦相似度矩阵:")
    for row in txt_sim.tolist():
        logger.info("    " + "  ".join(f"{v:+.3f}" for v in row))
    labels = batch["label"].tolist()
    logger.info(f"本 batch 的 label: {labels}")
    logger.info("观察: 同类的语义/视觉特征更接近 —— 这正是分类与检索能工作的原因")


# ----------------------------------------------------------------------------
# ⑥ PCA 可视化
# ----------------------------------------------------------------------------
def pca_2d(features: np.ndarray) -> np.ndarray:
    x = features - features.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def demo_visualization(image_encoder, text_encoder, ann, cfg, logger) -> None:
    logger.info("")
    logger.info("=" * 78)
    logger.info("[6/6] 特征可视化（PCA 降到 2 维）")
    logger.info("=" * 78)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("[skip] matplotlib 未安装，跳过可视化（pip install matplotlib）")
        return

    num_plot = int(cfg["num_plot"])
    samples = ann["splits"]["train"][:num_plot]
    dataset = BirdDataset(cfg["data_root"], samples, tokenizer=None, train=False,
                          image_size=int(cfg["image_size"]), load_image=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    img_feats, txt_feats, labels = [], [], []
    tokenizer = load_tokenizer(cfg["text_model"])
    for batch in loader:
        with torch.no_grad():
            img_feats.append(image_encoder(batch["images"]).cpu().numpy())
        enc = tokenizer([s["captions"][0] for s in
                         [samples[i] for i in batch["index"].tolist()]],
                        padding=True, truncation=True, max_length=int(cfg["max_length"]),
                        return_tensors="pt")
        with torch.no_grad():
            txt_feats.append(text_encoder(enc["input_ids"], enc["attention_mask"]).cpu().numpy())
        labels.append(batch["label"].numpy())

    img_feats = np.concatenate(img_feats, axis=0)
    txt_feats = np.concatenate(txt_feats, axis=0)
    labels = np.concatenate(labels, axis=0)

    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    for name, feats in (("image", img_feats), ("text", txt_feats)):
        coords = pca_2d(feats)
        colors = [palette[int(v) % len(palette)] for v in labels]
        plt.figure(figsize=(6.4, 5.2), dpi=150)
        plt.scatter(coords[:, 0], coords[:, 1], s=14, c=colors, alpha=0.85, linewidths=0)
        plt.title(f"PCA of {name} features ({len(labels)} samples, color = class id % 10)")
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.tight_layout()
        out_path = ensure_dir(cfg["out_dir"]) / f"pca_{name}.png"
        plt.savefig(out_path)
        plt.close()
        logger.info(f"[write] {out_path}")
    logger.info("观察: 同一颜色的点倾向于聚成簇 —— 说明编码器已经把'语义/外观相似的图'放到相近位置")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Inspect image/text encoders step by step")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--ann_file", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--text_model", default=None)
    parser.add_argument("--image_backbone", default=None)
    parser.add_argument("--image_pretrained", type=int, default=None, choices=[0, 1])
    parser.add_argument("--num_plot", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config", None)
    cfg = build_config(DEFAULTS, config_path, args)
    set_seed(int(cfg["seed"]))
    if cfg.get("image_pretrained") is not None:
        cfg["image_pretrained"] = bool(cfg["image_pretrained"])

    device = get_device(str(cfg.get("device", "cpu")))
    out_dir = ensure_dir(cfg["out_dir"])
    logger = Logger(out_dir / "features.log")
    logger.info("=" * 78)
    logger.info("Model dissection: how image and text become vectors")
    logger.info(f"device: {describe_device(device)}")
    logger.info("=" * 78)

    ann = load_annotations(cfg["ann_file"])
    tokenizer = load_tokenizer(cfg["text_model"])
    text_encoder = TextEncoder(model_name=cfg["text_model"], pretrained=True,
                               freeze=True, out_dim=None, pooling="mean").to(device).eval()
    image_encoder = ImageEncoder(backbone=cfg["image_backbone"],
                                 pretrained=cfg["image_pretrained"],
                                 freeze=True, out_dim=None).to(device).eval()

    batch_samples = ann["splits"]["val"][:4]
    dataset = BirdDataset(cfg["data_root"], batch_samples, tokenizer=tokenizer,
                          max_length=int(cfg["max_length"]), train=False,
                          image_size=int(cfg["image_size"]), load_image=True)
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    logger.info(f"[data] inspection batch: {len(batch_samples)} samples from val split")
    for i, (path, label) in enumerate(zip(batch["image_path"], batch["label"].tolist())):
        logger.info(f"  [{i}] label={label:3d} class={ann['classes'][label]}  image={path}")

    demo_tokenizer(tokenizer, logger)
    demo_embedding(text_encoder, tokenizer, logger)
    demo_pooling(text_encoder, tokenizer, logger)
    demo_image_encoder(image_encoder, batch, logger)
    demo_similarity(image_encoder, text_encoder, tokenizer, ann, batch, logger)

    save_json({"num_classes": len(ann["classes"]),
               "image_feature_dim": image_encoder.feature_dim,
               "text_hidden_size": text_encoder.hidden_size,
               "tokens_per_image": 7 * 7}, out_dir / "shapes.json")
    demo_visualization(image_encoder, text_encoder, ann, cfg, logger)

    logger.info("")
    logger.info("[done] 建议把 features.log 里的 6 段内容按顺序贴进报告 §3（模型拆解）")
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
