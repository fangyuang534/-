"""zeroshot_clip.py —— CLIP 思想实验：从"分类器"到"共同语义空间"（手册 §5）

它验证三件事
-----------
1. 零样本分类：类别名写成提示词（"a photo of a black footed albatross"），
   图像向量与 100 个文本向量比余弦相似度，取最大者 -> 不需要任何训练。
2. 图文检索：R@1 / R@5（image->text 与 text->image），
   证明"同一个空间里能互相检索"。
3. 与监督模型的对照：零样本 vs 训练好的融合模型，差距在哪里。

    python src/zeroshot_clip.py
    python src/zeroshot_clip.py --limit 300            # 只跑 300 张，快速看效果
    python src/zeroshot_clip.py --linear_probe         # 额外做 CLIP 特征 + 线性分类器

产物：outputs/zeroshot_clip/results.json, zeroshot_predictions.npz
注意：openai/clip-vit-base-patch32 需要下载约 600 MB 权重。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from build_annotations import class_display_name
from data import get_num_classes, load_annotations
from metrics import compute_metrics, print_metrics
from utils import (Logger, build_config, describe_device, ensure_dir, get_device,
                   save_json, set_seed, setup_console)

DEFAULTS = {
    "data_root": "data/cub200_subset",
    "ann_file": "data/cub200_subset/annotations.json",
    "out_dir": "outputs/zeroshot_clip",
    "clip_model": "openai/clip-vit-base-patch32",
    "batch_size": 32,
    "num_workers": 0,
    "limit": 0,
    "linear_probe": False,
    "seed": 42,
    "device": "cuda",
}


def build_prompts(ann: dict, classes: list) -> list:
    """给每个类别构造一组提示词（prompt ensemble），提升零样本稳定性。"""
    prompts = []
    raw = ann.get("class_prompts", {})
    for name in classes:
        pretty = class_display_name(name)
        base = raw.get(name) or f"a photo of a {pretty}"
        prompts.append([
            base,
            f"a photo of a {pretty}",
            f"a photo of a {pretty}, a type of bird",
            f"a photo of the bird species {pretty}",
        ])
    return prompts


@torch.no_grad()
def encode_texts(model, processor, device, texts_per_class: list, batch_size: int = 64) -> torch.Tensor:
    """每个类别的多条提示词分别编码后取平均，再 L2 归一化。"""
    class_vectors = []
    for texts in texts_per_class:
        embs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            inputs = processor(text=chunk, return_tensors="pt", padding=True,
                               truncation=True, max_length=77).to(device)
            emb = model.get_text_features(**inputs)
            embs.append(F.normalize(emb.float(), dim=-1))
        class_vectors.append(torch.cat(embs, dim=0).mean(dim=0))
    matrix = torch.stack(class_vectors, dim=0)                 # [C, D]
    return F.normalize(matrix, dim=-1)


@torch.no_grad()
def encode_images(model, processor, device, data_root: Path, samples: list,
                  batch_size: int, limit: int = 0) -> tuple:
    if limit:
        samples = samples[:limit]
    feats, labels = [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        images = []
        for rec in chunk:
            with Image.open(data_root / rec["image_path"]) as img:
                images.append(img.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt").to(device)
        emb = model.get_image_features(**inputs)
        feats.append(F.normalize(emb.float(), dim=-1))
        labels.extend(rec["label"] for rec in chunk)
        if (start // batch_size) % 10 == 0:
            print(f"  encoded {min(start + batch_size, len(samples))}/{len(samples)} images", flush=True)
    return torch.cat(feats, dim=0), np.array(labels, dtype=np.int64), samples


@torch.no_grad()
def encode_captions(model, processor, device, captions: list, batch_size: int = 64) -> torch.Tensor:
    feats = []
    for start in range(0, len(captions), batch_size):
        chunk = captions[start:start + batch_size]
        inputs = processor(text=chunk, return_tensors="pt", padding=True,
                           truncation=True, max_length=77).to(device)
        emb = model.get_text_features(**inputs)
        feats.append(F.normalize(emb.float(), dim=-1))
    return torch.cat(feats, dim=0)


def retrieval_recall(image_feats: torch.Tensor, text_feats: torch.Tensor, ks=(1, 5)) -> dict:
    """i2t / t2i R@k：测试集里每张图找与它配对的那条描述（对角线位置）。"""
    sim = image_feats @ text_feats.t()                          # [N, N]
    n = sim.size(0)
    target = torch.arange(n, device=sim.device)
    result = {}
    for k in ks:
        k = min(k, n)
        i2t = sim.topk(k, dim=1).indices                         # 每行前 k 个文本
        t2i = sim.t().topk(k, dim=1).indices                     # 每列前 k 个图像
        result[f"i2t_R@{k}"] = float((i2t == target[:, None]).any(dim=1).float().mean())
        result[f"t2i_R@{k}"] = float((t2i == target[:, None]).any(dim=1).float().mean())
    return result


def main(argv=None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="CLIP zero-shot + retrieval")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--ann_file", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--clip_model", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="只用前 N 张测试图（0=全部）")
    parser.add_argument("--linear_probe", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config", None)
    args["linear_probe"] = args["linear_probe"] or None
    cfg = build_config(DEFAULTS, config_path, args)
    set_seed(int(cfg["seed"]))

    device = get_device(str(cfg.get("device", "cuda")))
    out_dir = ensure_dir(cfg["out_dir"])
    logger = Logger(out_dir / "zeroshot.log")
    logger.info("=" * 74)
    logger.info(f"CLIP zero-shot  model={cfg['clip_model']}  device={describe_device(device)}")
    logger.info("=" * 74)

    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(cfg["clip_model"]).to(device).eval()
    processor = CLIPProcessor.from_pretrained(cfg["clip_model"])

    ann = load_annotations(cfg["ann_file"])
    classes = ann["classes"]
    num_classes = get_num_classes(ann)
    data_root = Path(cfg["data_root"])
    logger.info(f"[data] {num_classes} classes, test images = {len(ann['splits']['test'])}")

    # ---------- 1. 文本端：100 个类别的提示词向量 ----------
    prompts = build_prompts(ann, classes)
    logger.info(f"[text] prompt ensemble: {len(prompts[0])} prompts / class")
    text_proto = encode_texts(model, processor, device, prompts)     # [C, D]

    # ---------- 2. 图像端 ----------
    logger.info("[image] encoding test images ...")
    image_feats, labels, used_samples = encode_images(
        model, processor, device, data_root, ann["splits"]["test"],
        batch_size=int(cfg["batch_size"]), limit=int(cfg.get("limit") or 0))

    # ---------- 3. 零样本分类 ----------
    logit_scale = float(model.logit_scale.exp().item()) if hasattr(model, "logit_scale") else 100.0
    sim = (image_feats @ text_proto.t()) * logit_scale              # [N, C]
    probs = sim.softmax(dim=-1).cpu().numpy()
    preds = sim.argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(labels, preds, probs, num_classes=num_classes, classes=classes)
    print_metrics(metrics, title="CLIP ZERO-SHOT / test", logger=logger)

    # ---------- 4. 图文检索 ----------
    captions = [rec["captions"][0] for rec in used_samples]
    logger.info("[retrieval] encoding test captions ...")
    caption_feats = encode_captions(model, processor, device, captions,
                                    batch_size=int(cfg["batch_size"]))
    recall = retrieval_recall(image_feats, caption_feats)
    for key, value in recall.items():
        logger.info(f"  {key}: {value:.4f}")

    # ---------- 5. 可选：CLIP 特征 + 线性分类器（探针）----------
    probe_acc = None
    if cfg.get("linear_probe"):
        try:
            from sklearn.linear_model import LogisticRegression
            logger.info("[probe] encoding TRAIN images for linear probe ...")
            train_feats, train_labels, _ = encode_images(
                model, processor, device, data_root, ann["splits"]["train"],
                batch_size=int(cfg["batch_size"]), limit=0)
            clf = LogisticRegression(max_iter=300, n_jobs=-1, C=1.0)
            logger.info(f"[probe] fitting logistic regression on {len(train_labels)} samples ...")
            clf.fit(train_feats.cpu().numpy(), train_labels)
            probe_preds = clf.predict(image_feats.cpu().numpy())
            probe_acc = float(np.mean(probe_preds == labels))
            logger.info(f"[probe] linear probe accuracy = {probe_acc:.4f}")
        except ImportError:
            logger.info("[probe] scikit-learn not installed, skipped.")

    results = {
        "clip_model": cfg["clip_model"],
        "num_test_samples": int(len(labels)),
        "zero_shot": {k: v for k, v in metrics.items() if k != "per_class_accuracy"},
        "retrieval": recall,
        "linear_probe_acc": probe_acc,
        "note": "零样本：无任何训练；线性探针：冻结 CLIP 图像特征 + 逻辑回归",
    }
    save_json(results, out_dir / "results.json")
    np.savez_compressed(out_dir / "zeroshot_predictions.npz",
                        y_true=labels, y_pred=preds, probs=probs.astype(np.float32),
                        paths=np.array([rec["image_path"] for rec in used_samples]))
    logger.info(f"[write] {out_dir / 'results.json'}")
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
