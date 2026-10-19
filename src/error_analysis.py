"""error_analysis.py —— 错误分析：模型错在哪里、为什么错、注意力在看哪个词

输入是训练脚本留下的 test_predictions.npz（含 y_true / y_pred / probs / paths），
所以本脚本**不需要重新训练**。

会产出
-----
    top_confusions.csv      最容易混淆的类别对（按次数排序）
    worst_classes.csv       准确率最低的类别
    misclassified.png       16 张被分错的图（标注 true -> pred 与置信度）
    gate_distribution.png   门控/晚融合权重分布（"更信图像还是更信文本"）
    per_class_accuracy.csv  100 类的逐类准确率
    attention/              若提供 --viz_ckpt 且模型是 attention 融合：
      attn_<i>.png            每张图的 49 个视觉位置对每个词的注意力（词级柱状图）
      attn_words.csv          每个词的注意力均值

    python src/error_analysis.py --pred outputs/fusion_attention/test_predictions.npz
    python src/error_analysis.py --pred outputs/fusion_attention/test_predictions.npz \\
        --viz_ckpt outputs/fusion_attention/checkpoints/best.pt --viz_samples 3
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from data import load_annotations
from metrics import compute_metrics, hardest_classes, print_metrics, top_confusion_pairs
from utils import (Logger, build_config, ensure_dir, get_device, project_path,
                   setup_console)

DEFAULTS = {
    "data_root": "data/cub200_subset",
    "ann_file": "data/cub200_subset/annotations.json",
    "out_dir": "outputs/error_analysis",
    "pred": "outputs/fusion_attention/test_predictions.npz",
    "viz_ckpt": "",
    "viz_samples": 2,
    "num_grid": 16,
    "seed": 42,
    "device": "cuda",
}


def short_name(class_name: str) -> str:
    """"001.Black_footed_Albatross" -> "Black_footed_Albatross"（作图时省空间）"""
    parts = str(class_name).split(".", 1)
    return parts[1] if len(parts) == 2 and parts[0].isdigit() else str(class_name)


def write_csv(path, header: list, rows: list) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_misclassified(npz, classes, data_root: Path, out_path: Path, num: int, logger) -> None:
    """把被分错的样本拼成一张网格图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.info("[skip] matplotlib not installed.")
        return

    wrong = np.where(npz["y_true"] != npz["y_pred"])[0][:num]
    if len(wrong) == 0:
        logger.info("[viz] no misclassified sample (100% accuracy!)")
        return
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.3), dpi=130)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, wrong):
        path = data_root / str(npz["paths"][idx])
        try:
            with Image.open(path) as img:
                arr = np.asarray(img.convert("RGB").resize((160, 160)))
        except OSError:
            continue
        ax.imshow(arr)
        true_name = short_name(classes[int(npz["y_true"][idx])])
        pred_name = short_name(classes[int(npz["y_pred"][idx])])
        conf = float(npz["probs"][idx].max())
        ax.set_title(f"true: {true_name}\npred: {pred_name} ({conf:.2f})", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"[write] {out_path}")


def plot_gate(npz, out_path: Path, logger) -> None:
    if "gate" not in npz.files:
        return
    gate = np.asarray(npz["gate"]).reshape(-1)
    if gate.size == 0:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=140)
    ax.hist(gate, bins=30, color="#4c72b0", alpha=0.85)
    ax.set_xlabel("fusion weight on IMAGE  (a = sigmoid(W[img;txt]))")
    ax.set_ylabel("#samples")
    ax.set_title(f"Per-sample gate distribution (mean={gate.mean():.3f})")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"[write] {out_path}")


def visualize_attention(cfg, npz, classes, logger) -> None:
    """重建模型，导出"图像每个位置 -> 文本每个词"的平均注意力。"""
    ckpt_path = project_path(cfg["viz_ckpt"]) if cfg.get("viz_ckpt") else None
    if not ckpt_path or not Path(ckpt_path).is_file():
        logger.info("[attn] --viz_ckpt not given, skip word-level attention.")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    import torch
    from data import BirdDataset
    from model import build_model
    from text_encoder import load_tokenizer

    device = get_device(str(cfg.get("device", "cuda")))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_cfg = dict(ckpt.get("cfg") or {})
    model_cfg["device"] = str(device)
    if model_cfg.get("fusion", "").lower() not in ("attention", "cross_attention"):
        logger.info(f"[attn] fusion={model_cfg.get('fusion')} has no word-level attention, skip.")
        return
    model = build_model(model_cfg, int(ckpt["num_classes"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = load_tokenizer(model_cfg.get("text_model", "bert-base-uncased"))
    ann = load_annotations(cfg["ann_file"])
    samples = ann["splits"]["test"]
    # 挑几张"分错了"的样本来观察注意力（更有分析价值）
    wrong = np.where(npz["y_true"] != npz["y_pred"])[0]
    pick = wrong[:int(cfg["viz_samples"])] if len(wrong) else np.arange(int(cfg["viz_samples"]))
    pick = [int(i) for i in pick if int(i) < len(samples)]
    if not pick:
        logger.info("[attn] nothing to visualize.")
        return

    attn_dir = ensure_dir(Path(cfg["out_dir"]) / "attention")
    dataset = BirdDataset(cfg["data_root"], samples, tokenizer=tokenizer, train=False,
                          image_size=224, load_image=True)
    rows = []
    for idx in pick:
        item = dataset[idx]
        images = item["images"].unsqueeze(0).to(device)
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        mask = item["attention_mask"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(images, input_ids, mask)
        weights = output.get("attn_weights")
        if weights is None:
            logger.info("[attn] model output has no attn_weights, skip.")
            return
        weights = weights[0].float().cpu().numpy()          # [49, L]
        word_importance = weights.mean(axis=0)              # [L] 每个词被"看"的平均强度
        length = int(mask[0].sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:length].tolist())
        importance = word_importance[:length]

        labels = classes[int(npz["y_true"][idx])], classes[int(npz["y_pred"][idx])]
        caption = item["caption"]
        logger.info(f"[attn] sample {idx}: true={short_name(labels[0])} "
                    f"pred={short_name(labels[1])}\n       caption={caption!r}")
        top = np.argsort(-importance)[:5]
        logger.info("       top words: " + ", ".join(
            f"{tokens[i]}({importance[i]:.3f})" for i in top))
        for token, value in zip(tokens, importance):
            rows.append([idx, caption, token, f"{float(value):.6f}"])

        fig, ax = plt.subplots(figsize=(max(5.0, 0.34 * length), 3.4), dpi=140)
        pos = np.arange(length)
        ax.bar(pos, importance, color="#dd8452")
        ax.set_xticks(pos)
        ax.set_xticklabels(tokens, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("mean attention over 49 image tokens")
        ax.set_title(f"sample {idx}: true={short_name(labels[0])} / pred={short_name(labels[1])}",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(attn_dir / f"attn_{idx}.png")
        plt.close(fig)
        logger.info(f"[write] {attn_dir / f'attn_{idx}.png'}")

    write_csv(attn_dir / "attn_words.csv", ["sample_index", "caption", "token", "attention"], rows)


def main(argv=None) -> int:
    setup_console()
    parser = argparse.ArgumentParser(description="Error analysis of a trained run")
    parser.add_argument("--config", default=None)
    parser.add_argument("--pred", default=None, help="test_predictions.npz 路径")
    parser.add_argument("--ann_file", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--viz_ckpt", default=None, help="best.pt（用于词级注意力可视化）")
    parser.add_argument("--viz_samples", type=int, default=None)
    parser.add_argument("--num_grid", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = vars(parser.parse_args(argv))
    config_path = args.pop("config", None)
    cfg = build_config(DEFAULTS, config_path, args)

    out_dir = ensure_dir(cfg["out_dir"])
    logger = Logger(out_dir / "error_analysis.log")
    logger.info("=" * 74)
    logger.info(f"Error analysis: {cfg['pred']}")
    logger.info("=" * 74)

    pred_path = project_path(cfg["pred"])
    if not Path(pred_path).is_file():
        logger.info(f"[error] not found: {pred_path}")
        logger.info("        run a training script first, e.g. python src/train_multimodal.py --fusion attention")
        logger.close()
        return 1
    npz = np.load(pred_path, allow_pickle=True)
    ann = load_annotations(cfg["ann_file"])
    classes = ann["classes"]
    num_classes = len(classes)
    data_root = Path(cfg["data_root"])

    metrics = compute_metrics(npz["y_true"], npz["y_pred"], npz["probs"],
                             num_classes=num_classes, classes=classes)
    print_metrics(metrics, title="TEST (from saved predictions)", logger=logger)

    # ---- 1. 逐类准确率 ----
    rows = []
    for idx, (acc, support) in enumerate(zip(metrics["per_class_accuracy"],
                                             metrics["per_class_support"])):
        rows.append([classes[idx], short_name(classes[idx]), f"{acc:.4f}", support])
    rows.sort(key=lambda r: float(r[2]))
    write_csv(out_dir / "per_class_accuracy.csv", ["class", "short_name", "accuracy", "support"], rows)
    logger.info(f"[write] {out_dir / 'per_class_accuracy.csv'}")

    # ---- 2. 最容易混淆的类别对 ----
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (npz["y_true"], npz["y_pred"]), 1)
    pairs = top_confusion_pairs(cm, classes, topk=30)
    write_csv(out_dir / "top_confusions.csv",
              ["true_class", "pred_class", "count"],
              [[short_name(p["true"]), short_name(p["pred"]), p["count"]] for p in pairs])
    logger.info("[confusion] top-5 confused pairs:")
    for p in pairs[:5]:
        logger.info(f"    {short_name(p['true']):32s} -> {short_name(p['pred']):32s} "
                    f"x{p['count']}")

    # ---- 3. 最难分的类别 ----
    worst = hardest_classes(metrics, topk=15)
    write_csv(out_dir / "worst_classes.csv", ["class", "accuracy", "support"],
              [[name, f"{acc:.4f}", support] for name, acc, support in worst])
    logger.info("[worst] lowest per-class accuracy:")
    for name, acc, support in worst[:5]:
        logger.info(f"    {short_name(name):32s} acc={acc:.3f} (n={support})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [short_name(n) for n, _, _ in worst]
        values = [acc for _, acc, _ in worst]
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=140)
        ax.barh(range(len(values))[::-1], values, color="#c44e52", alpha=0.9)
        ax.set_yticks(range(len(values))[::-1])
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("per-class accuracy")
        ax.set_xlim(0, 1.0)
        ax.set_title("Hardest classes (lowest accuracy)")
        fig.tight_layout()
        fig.savefig(out_dir / "worst_classes.png")
        plt.close(fig)
        logger.info(f"[write] {out_dir / 'worst_classes.png'}")
    except ImportError:
        logger.info("[skip] matplotlib not installed.")

    # ---- 4. 错分样本网格 + 门控分布 ----
    plot_misclassified(npz, classes, data_root, out_dir / "misclassified.png",
                       int(cfg["num_grid"]), logger)
    plot_gate(npz, out_dir / "gate_distribution.png", logger)

    # ---- 5. 词级注意力 ----
    visualize_attention(cfg, npz, classes, logger)

    logger.info("[done] 建议在报告的'错误分析'一节里放：misclassified.png、"
                "worst_classes.png、top_confusions.csv 前几行、attn_*.png")
    logger.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
